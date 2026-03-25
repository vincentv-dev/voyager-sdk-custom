# SuperPoint CPU decoder for Axelera Voyager SDK
#
# The ONNX model already handles:
#   - Softmax + pixel-shuffle   → full-resolution score map [1,1,H,W]
#   - Max-pool NMS              → suppressed score map
#   - L2-normalisation          → unit-norm descriptor map [1,256,H//8,W//8]
#
# This decoder only needs to:
#   1. Border removal + threshold → candidate keypoints
#   2. Top-k selection           → final keypoints
#   3. Bilinear descriptor sampling at keypoint locations
#
# Results are stored as SuperPointMeta (torch path) or TensorMeta (GStreamer path).
# In GStreamer mode use --pipe=torch for visual validation.

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import torch
import torch.nn.functional as F

from axelera import types
from axelera.app import display, gst_builder, logging_utils
from axelera.app.meta import TensorMeta
from axelera.app.meta.base import AxTaskMeta
from axelera.app.operators import AxOperator, PipelineContext

if TYPE_CHECKING:
    from axelera.app.pipe import graph

LOG = logging_utils.getLogger(__name__)

_SUPERPOINT_STRIDE = 8   # VGG backbone has 3 max-pools → stride 8
_DESC_DIM = 256

_GREEN = (0, 220, 0, 255)


# ---------------------------------------------------------------------------
# Result meta with display support (torch path)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class SuperPointMeta(AxTaskMeta):
    """Holds SuperPoint keypoints, scores and descriptors; supports draw()."""

    keypoints: np.ndarray     # [N, 2] float32 (x, y) pixel coords
    scores: np.ndarray        # [N]    float32
    descriptors: np.ndarray   # [N, 256] float32 L2-normalised

    def __len__(self):
        return len(self.keypoints)

    def to_evaluation(self):
        from axelera.app import exceptions
        raise exceptions.NotSupportedForTask("SuperPointMeta", "to_evaluation")

    def draw(self, draw: display.Draw):
        for x, y in self.keypoints:
            draw.keypoint((float(x), float(y)), _GREEN, size=3)


# ---------------------------------------------------------------------------
# Pure-torch helpers
# ---------------------------------------------------------------------------

def _sample_descriptors(
    keypoints: torch.Tensor,          # [N, 2]  (x, y) pixel coordinates
    desc_map: torch.Tensor,           # [1, C, Hf, Wf]  L2-normalised dense map
    stride: int = _SUPERPOINT_STRIDE,
) -> torch.Tensor:
    """Bilinear interpolation of descriptors at keypoint locations.

    Follows the original SuperPoint sampling convention: keypoint (x, y) is
    in full-image pixel space; desc_map is at 1/stride resolution.

    Returns:
        [N, C] L2-normalised descriptors
    """
    _, c, h, w = desc_map.shape
    # Normalise coordinates to [-1, 1] for grid_sample
    norm_kpts = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * stride)
    norm_kpts = norm_kpts * 2 - 1                    # [-1, 1]

    descs = F.grid_sample(
        desc_map,
        norm_kpts.view(1, 1, -1, 2),
        mode="bilinear",
        align_corners=False,
    )                                                # [1, C, 1, N]
    descs = descs.reshape(1, c, -1)                 # [1, C, N]
    descs = F.normalize(descs, p=2, dim=1)          # unit-norm
    return descs.squeeze(0).T                       # [N, C]


# ---------------------------------------------------------------------------
# AxOperator decoder
# ---------------------------------------------------------------------------

class SuperPointDecoder(AxOperator):
    """Post-process SuperPoint AIPU outputs on the host CPU.

    The ONNX model already performs softmax, pixel-shuffle, NMS, and
    L2-normalisation.  This operator only does keypoint extraction and
    descriptor sampling.

    Parameters (all configurable from YAML):
        detection_threshold Score threshold (default: 0.005)
        remove_borders      Pixels to suppress at image border (default: 4)
        max_keypoints       Cap on returned keypoints; -1 = unlimited (default: 1024)
    """

    detection_threshold: float = 0.005
    remove_borders: int = 4
    max_keypoints: int = 1024

    def _post_init(self):
        pass

    # ------------------------------------------------------------------
    # GStreamer on-device pipeline
    # ------------------------------------------------------------------

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_superpoint.so',
            mode='read',
            options=(
                f'meta_key:{self.task_name};'
                f'detection_threshold:{self.detection_threshold};'
                f'remove_borders:{self.remove_borders};'
                f'max_keypoints:{self.max_keypoints};'
            ),
        )

    # ------------------------------------------------------------------
    # Software / calibration path
    # ------------------------------------------------------------------

    def exec_torch(
        self,
        image,
        predict: Union[tuple, list],
        axmeta,
    ):
        """Process AIPU outputs into keypoints + descriptors.

        Args:
            image:   Input PIL Image (passed through unchanged)
            predict: Tuple (score_map [1,1,H,W], desc_norm [1,256,Hf,Wf])
                     Softmax, pixel-shuffle, NMS, and L2-norm are already
                     applied by the ONNX model (handle_all=true dequantizes).
            axmeta:  Pipeline metadata container

        Returns:
            (image, predict, axmeta) — axmeta enriched with a TensorMeta entry.
        """
        if not isinstance(predict, (list, tuple)) or len(predict) < 2:
            LOG.warning(
                "SuperPointDecoder: expected a 2-element tuple (score_map, desc_norm), "
                f"got {type(predict)}. Skipping."
            )
            return image, predict, axmeta

        # Identify outputs by channel count — the AIPU may return them in any order.
        score_map = desc_norm = None
        for t in predict:
            if t.ndim == 4 and t.shape[1] == 1:
                score_map = t    # [1, 1, H, W]  NMS-suppressed
            elif t.ndim == 4 and t.shape[1] == 256:
                desc_norm = t    # [1, 256, Hf, Wf]  L2-normalised

        if score_map is None or desc_norm is None:
            shapes = [tuple(t.shape) for t in predict]
            LOG.warning(f"SuperPointDecoder: could not identify outputs by shape: {shapes}")
            return image, predict, axmeta

        scores_2d = score_map.squeeze(0).squeeze(0)   # [H, W]

        # ---- 1. Border removal ----
        pad = self.remove_borders
        if pad > 0:
            scores_2d = scores_2d.clone()
            scores_2d[:pad, :] = 0
            scores_2d[:, :pad] = 0
            scores_2d[-pad:, :] = 0
            scores_2d[:, -pad:] = 0

        # ---- 2. Threshold + extract keypoints ----
        row_idx, col_idx = torch.where(scores_2d > self.detection_threshold)
        # Stack as (x, y) = (col, row)
        kpts = torch.stack([col_idx, row_idx], dim=-1).float()   # [N, 2]
        kpt_scores = scores_2d[row_idx, col_idx]                 # [N]

        # ---- 3. Top-k ----
        if self.max_keypoints > 0 and len(kpts) > self.max_keypoints:
            kpt_scores, top_idx = torch.topk(kpt_scores, self.max_keypoints)
            kpts = kpts[top_idx]

        n_kpts = len(kpts)
        LOG.debug(f"SuperPoint: {n_kpts} keypoints detected")

        # ---- 4. Sample descriptors ----
        if n_kpts > 0:
            descs = _sample_descriptors(kpts, desc_norm, stride=_SUPERPOINT_STRIDE)
            descs_np = descs.cpu().detach().numpy().astype(np.float32)   # [N, 256]
        else:
            descs_np = np.zeros((0, _DESC_DIM), dtype=np.float32)

        kpts_np = kpts.cpu().detach().numpy().astype(np.float32)        # [N, 2]
        scores_np = kpt_scores.cpu().detach().numpy().astype(np.float32) # [N]

        # ---- 5. Store results ----
        model_meta = SuperPointMeta(
            keypoints=kpts_np, scores=scores_np, descriptors=descs_np
        )
        axmeta.add_instance(self.task_name, model_meta)

        return image, predict, axmeta
