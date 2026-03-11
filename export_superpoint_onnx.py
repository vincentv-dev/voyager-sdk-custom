"""
Export SuperPoint PyTorch model to ONNX for Axelera AI Metis compilation.

The exported model outputs:
  - scores:       [1, H, W]       raw score map after softmax + pixel-shuffle (before NMS)
  - descriptors:  [1, 256, H/8, W/8]  dense L2-normalized descriptor map

All post-processing (NMS, border masking, keypoint extraction, descriptor sampling)
is handled in the CPU decoder inside the voyager-sdk pipeline.

Usage:
  python export_superpoint_onnx.py \
      --weights weights/superpoint_v6_from_tf.pth \
      --output  weights/superpoint.onnx \
      --height  480 --width 640
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from superpoint_pytorch import SuperPoint


class SuperPointExportable(nn.Module):
    """
    Simplified SuperPoint wrapper that accepts a plain grayscale tensor
    [1, 1, H, W] and returns two dense output maps suitable for ONNX export
    to Axelera AI hardware.

    Operations included (all static, hardware-friendly):
      - VGG backbone
      - Detector head → softmax → pixel-shuffle → score map [1, H, W]
      - Descriptor head → L2 normalize → descriptor map [1, 256, H/8, W/8]

    Operations intentionally excluded (dynamic, run on CPU in the decoder):
      - NMS (batched_nms) — uses complex conditional logic
      - Border masking — trivial CPU op
      - Keypoint extraction (torch.where + topk) — variable-size output
      - Descriptor sampling (grid_sample at keypoints) — depends on keypoints
    """

    def __init__(self, superpoint: SuperPoint):
        super().__init__()
        self.backbone   = superpoint.backbone
        self.detector   = superpoint.detector
        self.descriptor = superpoint.descriptor
        self.stride     = superpoint.stride  # 8

    def forward(self, image: torch.Tensor) -> tuple:
        """
        Args:
            image: [1, 1, H, W]  float32 in [0, 1], grayscale
                   The pipeline handles color conversion before this point.

        Returns:
            scores:      [1, H, W]       float32 score map
            descriptors: [1, 256, H/8, W/8] float32 descriptor map
        """
        features = self.backbone(image)   # [1, 128, H/8, W/8]

        # --- Descriptor branch ---
        desc_dense = F.normalize(self.descriptor(features), p=2, dim=1)  # [1, 256, H/8, W/8]

        # --- Detector branch ---
        scores = self.detector(features)                      # [1, 65, H/8, W/8]
        scores = F.softmax(scores, dim=1)[:, :-1]             # [1, 64, H/8, W/8]  drop dustbin

        # Pixel-shuffle: fold 8×8 spatial cells back into full-res score map
        b, _, h, w = scores.shape
        s = self.stride                                        # 8
        scores = scores.permute(0, 2, 3, 1)                  # [1, H/8, W/8, 64]
        scores = scores.reshape(b, h, w, s, s)               # [1, H/8, W/8, 8, 8]
        scores = scores.permute(0, 1, 3, 2, 4)               # [1, H/8, 8, W/8, 8]
        scores = scores.reshape(b, h * s, w * s)             # [1, H, W]

        return scores, desc_dense


def export(weights_path: str, output_path: str, height: int, width: int):
    # ------------------------------------------------------------------
    # Load weights
    # ------------------------------------------------------------------
    print(f"Loading weights from {weights_path} ...")
    sp = SuperPoint()
    state_dict = torch.load(weights_path, map_location="cpu")
    sp.load_state_dict(state_dict, strict=True)
    sp.eval()

    # ------------------------------------------------------------------
    # Wrap in exportable module
    # ------------------------------------------------------------------
    model = SuperPointExportable(sp)
    model.eval()

    # ------------------------------------------------------------------
    # Sanity-check forward pass
    # ------------------------------------------------------------------
    dummy = torch.randn(1, 1, height, width)
    with torch.no_grad():
        scores, descs = model(dummy)
    print(f"Sanity check passed:")
    print(f"  scores shape      : {tuple(scores.shape)}")   # (1, H, W)
    print(f"  descriptors shape : {tuple(descs.shape)}")    # (1, 256, H/8, W/8)

    # ------------------------------------------------------------------
    # Export to ONNX
    # ------------------------------------------------------------------
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to ONNX → {output_path} ...")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=11,
        input_names=["image"],
        output_names=["scores", "descriptors"],
        dynamic_axes=None,  # fixed shapes for Axelera compilation
        do_constant_folding=True,
        verbose=False,
    )
    print("Export complete.")

    # ------------------------------------------------------------------
    # Verify with onnxruntime
    # ------------------------------------------------------------------
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        inp = np.random.randn(1, 1, height, width).astype(np.float32)
        ort_out = sess.run(None, {"image": inp})
        print(f"ONNX Runtime verification:")
        print(f"  scores shape      : {ort_out[0].shape}")
        print(f"  descriptors shape : {ort_out[1].shape}")
        print("ONNX model verified successfully.")
    except ImportError:
        print("onnxruntime not installed — skipping verification.")
    except Exception as e:
        print(f"ONNX verification failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SuperPoint to ONNX")
    parser.add_argument(
        "--weights",
        default="weights/superpoint_v6_from_tf.pth",
        help="Path to .pth weights file",
    )
    parser.add_argument(
        "--output",
        default="weights/superpoint.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument("--height", type=int, default=480, help="Input height (must be div by 8)")
    parser.add_argument("--width",  type=int, default=640, help="Input width  (must be div by 8)")
    args = parser.parse_args()

    assert args.height % 8 == 0, "height must be divisible by 8"
    assert args.width  % 8 == 0, "width must be divisible by 8"

    export(args.weights, args.output, args.height, args.width)
