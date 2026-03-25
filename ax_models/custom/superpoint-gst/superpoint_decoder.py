# SuperPoint CPU decoder for Axelera Voyager SDK
#
# GStreamer path (handle_all: false):
#   Receives raw INT8 NHWC tensors from the AIPU; all post-processing
#   (channel-max, NMS, TopK, softmax scoring, bilinear desc sampling,
#   L2-normalise) is performed in the C++ plugin libdecode_superpoint.so.
#   This Python class only injects runtime parameters into the GStreamer
#   options string (including per-tensor dequantize params from manifest.json).
#
# Python / ORT path (exec_torch):
#   ORT runs the full ONNX graph in float32, outputting pre-processed
#   keypoints [1,K,2], scores [1,K], descriptors [1,K,256].
#   This method applies detection_threshold and unpacks the results.
#
# Results are stored as SuperPointMeta (torch path) or TensorMeta (GStreamer path).
#
# NOTE: max_keypoints and remove_borders are runtime parameters for the C++
#       decoder (passed via options string) as well as for exec_torch.

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import torch

from axelera import types
from axelera.app import display, gst_builder, logging_utils
from axelera.app.meta import TensorMeta
from axelera.app.meta.base import AxTaskMeta
from axelera.app.operators import AxOperator, PipelineContext

if TYPE_CHECKING:
    from axelera.app.pipe import graph

LOG = logging_utils.getLogger(__name__)

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
# AxOperator decoder
# ---------------------------------------------------------------------------

class SuperPointDecoder(AxOperator):
    """Post-process SuperPoint AIPU outputs on the host CPU.

    GStreamer path (handle_all: false):
        Receives raw INT8 NHWC tensors directly from the AIPU:
            desc_map   [1, Hf, Wf, 256]  int8  — raw descriptor features
            logits     [1, Hf, Wf, 128]  int8  — detector logits (65 real + 63 padding)
        Performs entirely in fixed-point:
            1. Channel-max over 64 spatial channels  →  per-cell peak
            2. 3×3 spatial NMS at backbone resolution (INT8 comparisons)
            3. TopK selection
            4. Softmax score computation for K selected cells only
            5. Bilinear descriptor sampling + dequantize + L2-normalise

    Python exec_torch path (ORT runs full ONNX graph, outputs float32):
        Receives keypoints [1,K,2], scores [1,K], descriptors [1,K,256].
        Applies detection_threshold and unpacks.

    Parameters (all configurable from YAML):
        detection_threshold  Score threshold in probability space (default: 0.005)
        remove_borders       Border pixels to suppress (default: 4)
        max_keypoints        Maximum keypoints returned (default: 1024)
    """

    detection_threshold: float = 0.005
    remove_borders: int = 4
    max_keypoints: int = 512

    def _post_init(self):
        pass

    # ------------------------------------------------------------------
    # GStreamer on-device pipeline
    # ------------------------------------------------------------------

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        # Read per-tensor dequantize params from the compiled model manifest so the
        # C++ decoder can selectively dequantize only the keypoints it selects.
        scales      = "1.0,1.0"
        zero_points = "0.0,0.0"
        if self._compiled_model_dir is not None:
            import json
            from pathlib import Path
            manifest_path = Path(self._compiled_model_dir) / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                    deq = manifest.get("dequantize_params", [])
                    if len(deq) >= 2:
                        # deq[0] = [scale, zp] for desc [1,Hf,Wf,256]
                        # deq[1] = [scale, zp] for logits [1,Hf,Wf,128]
                        scales      = f"{deq[0][0]},{deq[1][0]}"
                        zero_points = f"{deq[0][1]},{deq[1][1]}"
                except Exception as exc:
                    LOG.warning(f"SuperPoint: could not read manifest dequant params: {exc}")

        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_superpoint.so',
            mode='read',
            options=(
                f'meta_key:{self.task_name};'
                f'detection_threshold:{self.detection_threshold};'
                f'remove_borders:{self.remove_borders};'
                f'max_keypoints:{self.max_keypoints};'
                f'scales:{scales};'
                f'zero_points:{zero_points};'
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
        """Unpack pre-computed ONNX outputs into keypoints + descriptors.

        Args:
            image:   Input PIL Image (passed through unchanged)
            predict: Tuple of three tensors output by the ONNX model:
                       keypoints    [1, K, 2]    (x, y) pixel coords, score-sorted
                       scores       [1, K]       detector confidence
                       descriptors  [1, K, 256]  L2-normalised
                     handle_all=true ensures they arrive as float32.
            axmeta:  Pipeline metadata container

        Returns:
            (image, predict, axmeta) — axmeta enriched with a TensorMeta entry.
        """
        if not isinstance(predict, (list, tuple)) or len(predict) < 3:
            LOG.warning(
                "SuperPointDecoder: expected 3 tensors (keypoints, scores, descriptors), "
                f"got {len(predict) if isinstance(predict, (list, tuple)) else type(predict)}. "
                "Skipping."
            )
            return image, predict, axmeta

        # Identify outputs by shape — AIPU may return them in any order.
        keypoints = scores = descriptors = None
        for t in predict:
            if t.ndim == 3 and t.shape[-1] == 2:
                keypoints = t        # [1, K, 2]
            elif t.ndim == 3 and t.shape[-1] == _DESC_DIM:
                descriptors = t      # [1, K, 256]
            elif t.ndim == 2:
                scores = t           # [1, K]

        if keypoints is None or scores is None or descriptors is None:
            shapes = [tuple(t.shape) for t in predict]
            LOG.warning(f"SuperPointDecoder: could not identify outputs by shape: {shapes}")
            return image, predict, axmeta

        # Squeeze batch dim
        kpts  = keypoints[0]     # [K, 2]
        scr   = scores[0]        # [K]
        descs = descriptors[0]   # [K, 256]

        # ---- Threshold filter ----
        # TopK always returns K slots; slots where no real keypoint exists have
        # score ≈ 0 (NMS zero-suppressed areas).  Drop them here.
        if self.detection_threshold > 0:
            mask  = scr > self.detection_threshold
            kpts  = kpts[mask]
            scr   = scr[mask]
            descs = descs[mask]

        n_kpts = len(kpts)
        LOG.debug(f"SuperPoint: {n_kpts} keypoints detected")

        kpts_np  = kpts.cpu().detach().numpy().astype(np.float32)   # [N, 2]
        scr_np   = scr.cpu().detach().numpy().astype(np.float32)    # [N]
        descs_np = descs.cpu().detach().numpy().astype(np.float32)  # [N, 256]

        # ---- Store results ----
        model_meta = SuperPointMeta(
            keypoints=kpts_np, scores=scr_np, descriptors=descs_np
        )
        axmeta.add_instance(self.task_name, model_meta)

        return image, predict, axmeta
