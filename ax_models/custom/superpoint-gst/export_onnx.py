#!/usr/bin/env python3
"""Export SuperPoint encoder (backbone + both heads, raw outputs only) to ONNX.

The exported model includes ONLY ops supported by the Axelera Metis M2 AIPU:
  - backbone (Conv + ReLU + MaxPool)
  - detector head  → scores_logits [1, 65, H//8, W//8]  (raw, pre-softmax)
  - descriptor head → desc_map     [1, 256, H//8, W//8]  (raw, pre-L2-norm)

All post-processing (Softmax, PixelShuffle/Reshape/Transpose, Equal+Cast for NMS,
and ReduceL2/Div for L2-norm) is intentionally left out because these ops are NOT
supported by the Metis M2 compiler and would be silently pushed to the IMX8MP CPU
build path.  The SuperPointDecoder handles them on the host instead.

Usage (from voyager-sdk root):
    python ax_models/superpoint/export_onnx.py

Or with custom paths/resolution:
    python ax_models/superpoint/export_onnx.py \\
        --weights /path/to/superpoint_v6_from_tf.pth \\
        --output  ax_models/superpoint/weights/superpoint_encoder.onnx \\
        --height 480 --width 640
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Locate the SuperPoint repo next to the voyager-sdk checkout
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_SUPERPOINT_DIR = _THIS_DIR.parent.parent.parent.parent / "SuperPoint-custom"

if not _SUPERPOINT_DIR.exists():
    raise RuntimeError(
        f"SuperPoint repo not found at {_SUPERPOINT_DIR}.\n"
        "Clone it with: git clone https://github.com/rpautrat/SuperPoint.git"
    )

sys.path.insert(0, str(_SUPERPOINT_DIR))
from superpoint_pytorch import SuperPoint  # noqa: E402


# ---------------------------------------------------------------------------
# Encoder with pixel-level outputs
# ---------------------------------------------------------------------------
class SuperPointEncoder(nn.Module):
    """SuperPoint CNN encoder — backbone + heads, raw outputs only.

    Softmax, PixelShuffle, NMS, and L2-norm are NOT included here because
    those ops (Softmax general, Reshape, Transpose, Equal, Cast, ReduceL2)
    are unsupported by the Metis M2 AIPU compiler and would fall back to the
    IMX8MP CPU build path.  They are handled in SuperPointDecoder instead.

    Outputs:
        scores_logits: [1, 65, H//8, W//8]  raw detector logits (pre-softmax)
        desc_map:      [1, 256, H//8, W//8]  raw descriptor map  (pre-L2-norm)
    """

    def __init__(self, superpoint: SuperPoint):
        super().__init__()
        self.backbone = superpoint.backbone
        self.detector = superpoint.detector
        self.descriptor = superpoint.descriptor

    def forward(self, image: torch.Tensor):
        """
        Args:
            image: [B, 1, H, W] float32 grayscale, pixel values in [0, 1].
        """
        features = self.backbone(image)
        scores_logits = self.detector(features)   # [B, 65, H/8, W/8]
        desc_map = self.descriptor(features)      # [B, 256, H/8, W/8]
        return scores_logits, desc_map


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def main():
    default_weights = str(_SUPERPOINT_DIR / "weights" / "superpoint_v6_from_tf.pth")
    default_output = str(_THIS_DIR / "weights" / "superpoint_encoder.onnx")

    parser = argparse.ArgumentParser(description="Export SuperPoint encoder to ONNX")
    parser.add_argument(
        "--weights", default=default_weights,
        help=f"Path to SuperPoint .pth weights (default: {default_weights})",
    )
    parser.add_argument(
        "--output", default=default_output,
        help=f"Output ONNX path (default: {default_output})",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Input height in pixels, must be divisible by 8 (default: 480)",
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Input width in pixels, must be divisible by 8 (default: 640)",
    )
    args = parser.parse_args()

    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError(
            f"Height ({args.height}) and width ({args.width}) must both be "
            "divisible by 8 (SuperPoint stride = 8)."
        )

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    print(f"Loading SuperPoint weights from {weights_path} ...")
    sp = SuperPoint()
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    sp.load_state_dict(state)
    sp.eval()

    encoder = SuperPointEncoder(sp).eval()

    dummy = torch.zeros(1, 1, args.height, args.width)

    with torch.no_grad():
        scores_logits, desc_map = encoder(dummy)
    print(
        f"Encoder outputs:\n"
        f"  scores_logits : {tuple(scores_logits.shape)}\n"
        f"  desc_map      : {tuple(desc_map.shape)}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        encoder,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["image"],
        output_names=["scores_logits", "desc_map"],
        dynamic_axes=None,  # fixed shapes required for Axelera AIPU compilation
    )
    print(f"ONNX encoder written to {output_path}")
    print(
        "\nNext step — compile for Metis M2:\n"
        f"  python deploy.py ax_models/custom/superpoint-gst/superpoint.yaml"
    )


if __name__ == "__main__":
    main()
