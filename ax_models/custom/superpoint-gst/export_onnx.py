#!/usr/bin/env python3
"""Export SuperPoint encoder (backbone + both heads + pixel-level post-processing) to ONNX.

The exported model includes:
  - backbone + detector + descriptor heads
  - softmax over 65-channel logits, dustbin removal, pixel-shuffle → [1,1,H,W] score map
  - max-pool NMS with configurable radius → suppressed score map
  - L2-normalisation of the descriptor map → [1,256,H//8,W//8]

This moves the expensive softmax (~5 ms) and NMS (~4 ms) from the CPU decoder to the AIPU,
leaving only keypoint extraction and bilinear descriptor sampling on the host CPU.

Usage (from voyager-sdk root):
    python ax_models/superpoint/export_onnx.py

Or with custom paths/resolution/NMS:
    python ax_models/superpoint/export_onnx.py \\
        --weights /path/to/superpoint_v6_from_tf.pth \\
        --output  ax_models/superpoint/weights/superpoint_encoder.onnx \\
        --height 480 --width 640 --nms-radius 4
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """SuperPoint CNN encoder with pixel-level score map output.

    Outputs:
        score_map:  [1, 1, H, W]         NMS-suppressed softmax scores
        desc_norm:  [1, 256, H//8, W//8]  L2-normalised descriptor map
    """

    def __init__(self, superpoint: SuperPoint, nms_radius: int = 4):
        super().__init__()
        self.backbone = superpoint.backbone
        self.detector = superpoint.detector
        self.descriptor = superpoint.descriptor
        self.nms_radius = nms_radius

    def forward(self, image: torch.Tensor):
        """
        Args:
            image: [B, 1, H, W] float32 grayscale, pixel values in [0, 1].
        """
        features = self.backbone(image)
        scores_logits = self.detector(features)   # [B, 65, H/8, W/8]
        desc_map = self.descriptor(features)      # [B, 256, H/8, W/8]

        # 1. Softmax over 65 channels, drop dustbin, pixel-shuffle → [B, 1, H, W]
        scores_sm = torch.softmax(scores_logits, dim=1)
        scores_no_dustbin = scores_sm[:, :-1, :, :]            # [B, 64, H/8, W/8]
        score_map = F.pixel_shuffle(scores_no_dustbin, 8)      # [B, 1, H, W]

        # 2. Max-pool NMS: zero out non-local-maxima within nms_radius
        k = 2 * self.nms_radius + 1
        max_pool = F.max_pool2d(score_map, kernel_size=k, stride=1, padding=self.nms_radius)
        score_map = score_map * (score_map == max_pool).float()

        # 3. L2-normalise descriptor map (sampled descriptors are ~unit-norm)
        desc_norm = F.normalize(desc_map, p=2, dim=1)

        return score_map, desc_norm


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
    parser.add_argument(
        "--nms-radius", type=int, default=4,
        help="NMS suppression radius in pixels, baked into the ONNX graph (default: 4)",
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

    encoder = SuperPointEncoder(sp, nms_radius=args.nms_radius).eval()

    dummy = torch.zeros(1, 1, args.height, args.width)

    with torch.no_grad():
        score_map, desc_norm = encoder(dummy)
    print(
        f"Encoder outputs (nms_radius={args.nms_radius}):\n"
        f"  score_map : {tuple(score_map.shape)}\n"
        f"  desc_norm : {tuple(desc_norm.shape)}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        encoder,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["image"],
        output_names=["score_map", "desc_norm"],
        dynamic_axes=None,  # fixed shapes required for Axelera AIPU compilation
    )
    print(f"ONNX encoder written to {output_path}")
    print(
        "\nNext step — compile for Metis M2:\n"
        f"  python deploy.py ax_models/custom/superpoint-gst/superpoint.yaml"
    )


if __name__ == "__main__":
    main()
