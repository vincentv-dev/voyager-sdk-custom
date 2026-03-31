# EdgePoint2 CPU decoder for Axelera Voyager SDK
#
# The ONNX model outputs padded backbone feature maps only:
#   x1_feat: [1, 64, H/2,  W/2 ]
#   x2_feat: [1, 64, H/8,  W/8 ]
#   x3_feat: [1, 64, H/32, W/32]
#
# This decoder reconstructs the original EdgePoint2 descriptor and detector
# heads on the host CPU, then performs:
#   1. Max-pool NMS
#   2. Border removal + raw-logit threshold → candidate keypoints
#   3. Top-k selection
#   4. Bilinear descriptor sampling at keypoint locations + L2-normalisation
#
# Results are stored as EdgePointMeta (torch path) or TensorMeta (GStreamer path).

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from axelera import types
from axelera.app import display, gst_builder, logging_utils
from axelera.app.meta import TensorMeta
from axelera.app.meta.base import AxTaskMeta
from axelera.app.operators import AxOperator, PipelineContext

if TYPE_CHECKING:
    from axelera.app.pipe import graph

LOG = logging_utils.getLogger(__name__)

_DESC_UPSCALE = 4   # descriptor map spatial stride (H/4, W/4)
_DESC_DIM     = 64
_C2_DIM       = 16
_C3_DIM       = 48
_C4_DIM       = 64

_GREEN = (0, 220, 0, 255)


# ---------------------------------------------------------------------------
# Result meta with display support (torch path)
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class EdgePointMeta(AxTaskMeta):
    """Holds EdgePoint2 keypoints, scores and descriptors; supports draw().

    Works for both the torch path (built via exec_torch) and the GStreamer AIPU
    path (decoded from AxMetaRawTensor via the classmethod decode()).
    """

    # Register as the TensorMeta decoder so the GStreamer/AIPU path gets draw()
    # support via -o output.mp4.  Safe because the edgepoint pipeline is
    # single-model; only override in a multi-model pipeline with caution.
    META_TYPE = 'TensorMeta'

    keypoints:   np.ndarray   # [N, 2]  float32  (x, y) pixel coords
    scores:      np.ndarray   # [N]     float32
    descriptors: np.ndarray   # [N, 64] float32  L2-normalised

    def __len__(self):
        return len(self.keypoints)

    def to_evaluation(self):
        from axelera.app import exceptions
        raise exceptions.NotSupportedForTask('EdgePointMeta', 'to_evaluation')

    def draw(self, draw: display.Draw):
        for x, y in self.keypoints:
            draw.keypoint((float(x), float(y)), _GREEN, size=3)

    @classmethod
    def decode(cls, data: dict) -> 'EdgePointMeta':
        """Decode AxMetaRawTensor byte dict from the GStreamer C++ decoder.

        The C++ libdecode_edgepoint.so stores three tensors:
            tensors[0]  keypoints    [N, 2]     float32
            tensors[1]  scores       [N]        float32
            tensors[2]  descriptors  [N, DESC]  float32
        """
        tensors = []
        i = 0
        while f'data_{i}' in data and f'dims_{i}' in data:
            raw   = data[f'data_{i}']
            dims  = np.frombuffer(data[f'dims_{i}'], dtype=np.int64)
            dtype = np.dtype(data[f'dtype_{i}'].decode() if f'dtype_{i}' in data else 'f4')
            arr   = np.frombuffer(raw, dtype=dtype).reshape(dims) if dims.size else np.array([], dtype=dtype)
            tensors.append(arr)
            i += 1

        empty_kpts = np.zeros((0, 2), dtype=np.float32)
        empty_scr  = np.zeros((0,),   dtype=np.float32)
        empty_desc = np.zeros((0, _DESC_DIM), dtype=np.float32)

        kpts  = tensors[0].reshape(-1, 2) if len(tensors) > 0 and tensors[0].size else empty_kpts
        scrs  = tensors[1].reshape(-1)    if len(tensors) > 1 and tensors[1].size else empty_scr
        descs = tensors[2].reshape(-1, _DESC_DIM) if len(tensors) > 2 and tensors[2].size else empty_desc

        return cls(keypoints=kpts, scores=scrs, descriptors=descs)


# ---------------------------------------------------------------------------
# Pure-torch helpers
# ---------------------------------------------------------------------------

def _sample_descriptors(
    keypoints: torch.Tensor,   # [N, 2]  (x, y) pixel coordinates
    desc_map: torch.Tensor,    # [1, 64, Hd, Wd]  raw dense map
    stride: int = _DESC_UPSCALE,
) -> torch.Tensor:
    """Bilinear interpolation of descriptors at keypoint locations.

    Follows the EdgePoint2 sampling convention: coord = (kpt + 0.5) / size * 2 - 1
    where size = [W, H] in full-image pixel space.

    Returns:
        [N, 64] L2-normalised descriptors
    """
    _, c, h, w = desc_map.shape
    # Normalise (x, y) pixel coords to [-1, 1] using full-image dimensions.
    full_wh = keypoints.new_tensor([w * stride, h * stride])
    norm_kpts = (keypoints + 0.5) / full_wh * 2 - 1   # [-1, 1]

    descs = F.grid_sample(
        desc_map,
        norm_kpts.view(1, 1, -1, 2),
        mode='bilinear',
        align_corners=False,
    )                                                   # [1, 64, 1, N]
    descs = descs.reshape(1, c, -1)                    # [1, 64, N]
    descs = F.normalize(descs, p=2, dim=1)             # unit-norm
    return descs.squeeze(0).T                          # [N, 64]


def _load_edgepoint_heads(edgepoint_repo: str, weights_path: str) -> nn.Module:
    """Load the original EdgePoint2 model; the decoder uses its heads on CPU."""
    repo_path = str(Path(edgepoint_repo).expanduser().resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from model.model import EdgePoint2

    cfg = {'c1': 16, 'c2': 16, 'c3': 48, 'c4': 64, 'cdesc': 64, 'cdetect': 16}
    model = EdgePoint2(**cfg)
    state = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# AxOperator decoder
# ---------------------------------------------------------------------------

class EdgePointDecoder(AxOperator):
    """Post-process EdgePoint2 AIPU outputs on the host CPU.

    The ONNX model outputs padded backbone feature maps. This operator
    reconstructs the original EdgePoint2 score and descriptor heads on the
    host CPU, then performs NMS, keypoint extraction, and descriptor sampling.

    Parameters (all configurable from YAML):
        detection_threshold  Raw logit threshold (default: 0.0)
        remove_borders       Pixels to suppress at image border (default: 4)
        max_keypoints        Cap on returned keypoints; -1 = unlimited (default: 1024)
        nms_radius           Max-pool NMS radius in pixels (default: 2)
    """

    detection_threshold: float = 0.0
    remove_borders: int = 4
    max_keypoints: int = 1024
    nms_radius: int = 2
    weights_path: str = '$AXELERA_FRAMEWORK/ax_models/custom/edgepoint/weights/E64.pth'
    head_weights_path: str = '$AXELERA_FRAMEWORK/ax_models/custom/edgepoint/weights/edgepoint_heads.bin'
    edgepoint_repo: str = '/home/Vydar/EdgePoint2'

    def _post_init(self):
        self._cpu_model = None

    def _get_cpu_model(self) -> nn.Module:
        if self._cpu_model is None:
            framework_root = str(Path(__file__).resolve().parents[3])
            weights_path = os.path.expandvars(
                self.weights_path.replace('$AXELERA_FRAMEWORK', framework_root)
            )
            repo_path = os.path.expandvars(
                self.edgepoint_repo.replace('$AXELERA_FRAMEWORK', framework_root)
            )
            self._cpu_model = _load_edgepoint_heads(repo_path, weights_path)
        return self._cpu_model

    # ------------------------------------------------------------------
    # GStreamer on-device pipeline
    # ------------------------------------------------------------------

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        framework_root = str(Path(__file__).resolve().parents[3])
        head_weights_path = os.path.expandvars(
            self.head_weights_path.replace('$AXELERA_FRAMEWORK', framework_root)
        )
        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_edgepoint.so',
            mode='read',
            options=(
                f'meta_key:{self.task_name};'
                f'detection_threshold:{self.detection_threshold};'
                f'remove_borders:{self.remove_borders};'
                f'max_keypoints:{self.max_keypoints};'
                f'nms_radius:{self.nms_radius};'
                f'head_weights_path:{head_weights_path};'
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
            predict: Tuple (x1_feat [1,64,H/2,W/2], x2_feat [1,64,H/8,W/8],
                     x3_feat [1,64,H/32,W/32]) with padded channels.
                     Raw NCHW float32 outputs — handle_all dequantizes before here.
            axmeta:  Pipeline metadata container

        Returns:
            (image, predict, axmeta) — axmeta enriched with a TensorMeta entry.
        """
        if not isinstance(predict, (list, tuple)) or len(predict) < 3:
            LOG.warning(
                'EdgePointDecoder: expected a 3-element tuple '
                '(x1_feat, x2_feat, x3_feat), '
                f'got {type(predict)}. Skipping.'
            )
            return image, predict, axmeta

        # Identify backbone stages by spatial resolution.
        feats = [t for t in predict if t.ndim == 4 and t.shape[1] == 64]
        if len(feats) < 3:
            shapes = [tuple(t.shape) for t in predict]
            LOG.warning(f'EdgePointDecoder: could not identify outputs by shape: {shapes}')
            return image, predict, axmeta
        x1_pad, x2_pad, x3_pad = sorted(feats, key=lambda t: int(t.shape[2]), reverse=True)[:3]

        # Recover the original channel counts from the padded backbone features.
        x1 = x1_pad[:, :_C2_DIM, :, :]
        x2 = x2_pad[:, :_C3_DIM, :, :]
        x3 = x3_pad[:, :_C4_DIM, :, :]

        model = self._get_cpu_model()
        with torch.no_grad():
            _x2 = F.avg_pool2d(x1, 2, 2)
            desc_map_raw = model.desc_head(
                torch.cat(
                    [
                        _x2,
                        F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False),
                        F.interpolate(x3, scale_factor=8, mode='bilinear', align_corners=False),
                    ],
                    dim=1,
                )
            )
            score_map = model.score_head(
                model.conv1(x1)
                + F.interpolate(model.conv2(x2), scale_factor=4, mode='bilinear', align_corners=False)
                + F.interpolate(model.conv3(x3), scale_factor=16, mode='bilinear', align_corners=False)
            )

        # ---- 1. Max-pool NMS ----
        k = 2 * self.nms_radius + 1
        max_pool  = F.max_pool2d(score_map, kernel_size=k, stride=1,
                                 padding=self.nms_radius)
        score_map = score_map * (score_map == max_pool).float()

        # ---- 2. Border removal + threshold ----
        scores_2d = score_map.squeeze(0).squeeze(0)   # [H, W]
        pad = self.remove_borders
        if pad > 0:
            scores_2d = scores_2d.clone()
            scores_2d[:pad,  :]  = 0
            scores_2d[:,  :pad]  = 0
            scores_2d[-pad:, :]  = 0
            scores_2d[:,  -pad:] = 0

        row_idx, col_idx = torch.where(scores_2d > self.detection_threshold)
        kpts       = torch.stack([col_idx, row_idx], dim=-1).float()  # [N, 2] (x,y)
        kpt_scores = scores_2d[row_idx, col_idx]                      # [N]

        # ---- 3. Top-k ----
        if self.max_keypoints > 0 and len(kpts) > self.max_keypoints:
            kpt_scores, top_idx = torch.topk(kpt_scores, self.max_keypoints)
            kpts = kpts[top_idx]

        n_kpts = len(kpts)
        LOG.debug(f'EdgePoint2: {n_kpts} keypoints detected')

        # ---- 4. Bilinear descriptor sampling ----
        if n_kpts > 0:
            descs    = _sample_descriptors(kpts, desc_map_raw, stride=_DESC_UPSCALE)
            descs_np = descs.cpu().detach().numpy().astype(np.float32)   # [N, 64]
        else:
            descs_np = np.zeros((0, _DESC_DIM), dtype=np.float32)

        kpts_np   = kpts.cpu().detach().numpy().astype(np.float32)         # [N, 2]
        scores_np = kpt_scores.cpu().detach().numpy().astype(np.float32)   # [N]

        # ---- 5. Store results ----
        model_meta = EdgePointMeta(
            keypoints=kpts_np, scores=scores_np, descriptors=descs_np
        )
        axmeta.add_instance(self.task_name, model_meta)

        return image, predict, axmeta
