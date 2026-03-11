# SuperPoint CPU post-processing decoder for Axelera AI Voyager SDK
#
# The hardware outputs two tensors (via TensorMeta):
#   tensors[0]: scores       [1, H, W]          raw score map
#   tensors[1]: descriptors  [1, 256, H/8, W/8] dense descriptor map
#
# This operator runs on the CPU host and performs:
#   1. NMS on the score map
#   2. Border masking
#   3. Threshold + top-k keypoint extraction
#   4. Descriptor sampling at keypoint locations
#   5. Stores results in a custom AxTaskMeta for downstream use / display

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from axelera import types
from axelera.app import gst_builder, logging_utils, meta
from axelera.app.meta.base import AxTaskMeta
from axelera.app.operators import AxOperator, PipelineContext

LOG = logging_utils.getLogger(__name__)


# ---------------------------------------------------------------------------
# CPU-side NMS (mirrors superpoint_pytorch.batched_nms)
# ---------------------------------------------------------------------------

def _max_pool2d(x: torch.Tensor, radius: int) -> torch.Tensor:
    return F.max_pool2d(
        x, kernel_size=radius * 2 + 1, stride=1, padding=radius
    )


def batched_nms_cpu(scores: torch.Tensor, nms_radius: int) -> torch.Tensor:
    """Non-maximum suppression on a [H, W] score map."""
    assert nms_radius >= 0
    scores = scores.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W] for pool2d
    zeros = torch.zeros_like(scores)
    max_mask = scores == _max_pool2d(scores, nms_radius)
    for _ in range(2):
        supp_mask = _max_pool2d(max_mask.float(), nms_radius) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == _max_pool2d(supp_scores, nms_radius)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    result = torch.where(max_mask, scores, zeros)
    return result.squeeze(0).squeeze(0)  # [H, W]


def sample_descriptors(
    keypoints: torch.Tensor,  # [N, 2] (x, y)
    desc_map: torch.Tensor,   # [1, 256, H/8, W/8]
    stride: int = 8,
) -> torch.Tensor:
    """Bilinear interpolation of descriptors at keypoint locations → [N, 256]."""
    b, c, h, w = desc_map.shape
    # Normalise keypoints to [-1, 1]
    kp = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * stride)
    kp = kp * 2 - 1
    descs = F.grid_sample(
        desc_map, kp.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )                                   # [1, 256, 1, N]
    descs = F.normalize(descs.reshape(b, c, -1), p=2, dim=1)  # [1, 256, N]
    return descs.squeeze(0).T           # [N, 256]


# ---------------------------------------------------------------------------
# Custom AxTaskMeta to carry SuperPoint results through the pipeline
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuperPointMeta(AxTaskMeta):
    """Holds keypoints, scores, and descriptors for one frame."""

    keypoints: np.ndarray   # [N, 2] float32  (x, y)
    kp_scores: np.ndarray   # [N]    float32
    descriptors: np.ndarray  # [N, 256] float32

    def to_evaluation(self):
        return {"keypoints": self.keypoints, "scores": self.kp_scores}

    def draw(self, draw: Any):
        """Draw keypoints as small circles scaled by confidence."""
        try:
            import cv2

            img = draw.image  # underlying numpy image (H, W, 3) BGR
            for (x, y), s in zip(self.keypoints, self.kp_scores):
                radius = max(1, int(s * 4))
                cv2.circle(img, (int(x), int(y)), radius, (0, 255, 0), -1)
        except Exception:
            pass  # visualisation is best-effort


# ---------------------------------------------------------------------------
# Operator
# ---------------------------------------------------------------------------

class SuperPointDecoder(AxOperator):
    """
    CPU decoder for the SuperPoint ONNX model.

    YAML parameters (all optional, with sensible defaults):
      nms_radius:           int   = 4      NMS suppression radius
      detection_threshold:  float = 0.005  minimum keypoint score
      remove_borders:       int   = 4      border width to suppress
      max_num_keypoints:    int   = 1000   cap on returned keypoints (0 = no cap)
    """

    nms_radius:          int   = 4
    detection_threshold: float = 0.005
    remove_borders:      int   = 4
    max_num_keypoints:   int   = 1000

    # AxOperator lifecycle -------------------------------------------------

    def _post_init(self):
        pass

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, compiled_model_dir, task_graph
        )

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        # Decode hardware tensor outputs into TensorMeta; all keypoint
        # post-processing runs in exec_torch on the host CPU.
        master_meta_option = f'master_meta:{self._where};' if self._where else ''
        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_to_raw_tensor.so',
            options=f'meta_key:{str(self.task_name)};{master_meta_option}',
        )

    def exec_torch(self, image, predict, axmeta):
        """
        Called for every frame on the CPU host.

        predict:  list/tuple of tensors from the hardware output
                  [0] → scores       [1, H, W]
                  [1] → descriptors  [1, 256, H/8, W/8]
        """
        if predict is None:
            return image, predict, axmeta

        # Unpack hardware outputs.
        # postprocess_graph.onnx outputs descriptors first, then scores.
        if isinstance(predict, (list, tuple)):
            desc_map   = predict[0]
            scores_raw = predict[1]
        elif hasattr(predict, "tensors"):                 # TensorMeta path
            tensors    = predict.tensors
            desc_map   = torch.from_numpy(tensors[0])
            scores_raw = torch.from_numpy(tensors[1])
        else:
            LOG.warning(f"SuperPointDecoder: unexpected predict type {type(predict)}")
            return image, predict, axmeta

        # Ensure float32 tensors on CPU
        scores_raw = scores_raw.float().squeeze(0)  # [H, W]
        desc_map   = desc_map.float()               # [1, 256, H/8, W/8]
        if desc_map.dim() == 3:
            desc_map = desc_map.unsqueeze(0)

        # 1. NMS
        scores = batched_nms_cpu(scores_raw, self.nms_radius)

        # 2. Border masking
        pad = self.remove_borders
        if pad > 0:
            scores[:pad,  :] = 0
            scores[-pad:, :] = 0
            scores[:,  :pad] = 0
            scores[:, -pad:] = 0

        # 3. Threshold
        mask = scores > self.detection_threshold
        yx   = torch.stack(torch.where(mask), dim=-1).float()  # [N, 2]
        kp_scores = scores[mask]                               # [N]

        if yx.shape[0] == 0:
            kp_np   = np.zeros((0, 2), dtype=np.float32)
            sc_np   = np.zeros((0,),   dtype=np.float32)
            desc_np = np.zeros((0, 256), dtype=np.float32)
        else:
            # Convert (row, col) → (x, y)
            keypoints = yx.flip(1)  # [N, 2] (x, y)

            # 4. Top-k
            if self.max_num_keypoints > 0 and keypoints.shape[0] > self.max_num_keypoints:
                top_scores, top_idx = torch.topk(
                    kp_scores, self.max_num_keypoints, dim=0, sorted=True
                )
                keypoints = keypoints[top_idx]
                kp_scores = top_scores

            # 5. Sample descriptors
            desc = sample_descriptors(keypoints, desc_map, stride=8)  # [N, 256]

            kp_np   = keypoints.cpu().numpy().astype(np.float32)
            sc_np   = kp_scores.cpu().numpy().astype(np.float32)
            desc_np = desc.cpu().numpy().astype(np.float32)

        sp_meta = SuperPointMeta(
            keypoints=kp_np,
            kp_scores=sc_np,
            descriptors=desc_np,
        )
        axmeta.add_instance(self.task_name, sp_meta, self._where)

        LOG.debug(
            f"SuperPoint: {kp_np.shape[0]} keypoints detected "
            f"(max_score={sc_np.max():.3f})" if sc_np.size else "0 keypoints"
        )
        return image, predict, axmeta
