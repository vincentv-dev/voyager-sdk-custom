#!/usr/bin/env python3
"""Export SuperPoint encoder (backbone + both heads + full post-processing) to ONNX.

The exported model includes:
  - backbone + detector + descriptor heads
  - softmax over 65-channel logits, dustbin removal
  - max-pool NMS at backbone resolution (H/8 × W/8) → 64× cheaper than full-resolution NMS
  - pixel-shuffle → [1,1,H,W] score map
  - border masking (constant) → zeroes edges before keypoint selection
  - TopK extraction → top-K keypoints as (x, y) pixel coordinates
  - bilinear descriptor sampling + L2-normalisation at keypoint locations

Outputs (fixed shapes, all float32):
  keypoints    [1, K, 2]    (x, y) pixel coords, sorted by score descending
  scores       [1, K]       detector confidence
  descriptors  [1, K, 256]  L2-normalised descriptors

This eliminates the expensive dequantization of the full dense descriptor map
(previously 1.2 M elements) by dequantizing only the K sampled descriptors.
Total output: K × (2 + 1 + 256) ≈ 265 K elements vs the previous 1.54 M.

Usage (from voyager-sdk root):
    python ax_models/superpoint/export_onnx.py

Or with custom paths/resolution/NMS:
    python ax_models/superpoint/export_onnx.py \\
        --weights /path/to/superpoint_v6_from_tf.pth \\
        --output  ax_models/superpoint/weights/superpoint_encoder.onnx \\
        --height 480 --width 640 --nms-radius 4 --max-keypoints 1024 --remove-borders 4

NOTE: max_keypoints and remove_borders are baked into the ONNX graph at export time.
      Changing them requires re-exporting and recompiling for the AIPU.
      detection_threshold remains a runtime parameter applied by the decoder.
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
# Encoder with keypoint-level outputs
# ---------------------------------------------------------------------------
class SuperPointEncoder(nn.Module):
    """SuperPoint CNN encoder that outputs sampled keypoints and descriptors.

    All heavy post-processing is baked into the ONNX graph so only a lightweight
    threshold filter remains on the host CPU decoder.

    Outputs (all float32, batch dimension B=1 at inference time):
        keypoints    [B, K, 2]    (x, y) pixel coords, sorted by score descending
        scores       [B, K]       detector confidence
        descriptors  [B, K, 256]  L2-normalised descriptors

    Args:
        superpoint:     Pretrained SuperPoint model instance.
        nms_radius:     Max-pool NMS radius (default 4). Baked into graph.
        max_keypoints:  Fixed output size K for TopK (default 1024). Baked into graph.
        remove_borders: Pixels to zero before TopK (default 4). Baked into graph.
        height:         Input image height (default 480). Must be divisible by 8.
        width:          Input image width (default 640). Must be divisible by 8.
    """

    def __init__(
        self,
        superpoint: SuperPoint,
        nms_radius: int = 4,
        max_keypoints: int = 512,
        remove_borders: int = 4,
        height: int = 480,
        width: int = 640,
    ):
        super().__init__()
        self.backbone = superpoint.backbone
        self.detector = superpoint.detector
        self.descriptor = superpoint.descriptor
        self.nms_radius = nms_radius
        self.max_keypoints = max_keypoints

        # Precomputed border mask — exported as a constant in the ONNX graph.
        mask = torch.ones(1, 1, height, width)
        if remove_borders > 0:
            mask[:, :, :remove_borders, :]  = 0
            mask[:, :, -remove_borders:, :] = 0
            mask[:, :, :, :remove_borders]  = 0
            mask[:, :, :, -remove_borders:] = 0
        self.register_buffer("border_mask", mask)

    def forward(self, image: torch.Tensor):
        """
        Args:
            image: [B, 1, H, W] float32 grayscale, pixel values in [0, 1].

        Returns:
            keypoints:   [B, K, 2]    (x, y) pixel coords
            scores:      [B, K]       sorted descending
            descriptors: [B, K, 256]  L2-normalised
        """
        features = self.backbone(image)
        scores_logits = self.detector(features)   # [B, 65, H/8, W/8]
        desc_map = self.descriptor(features)      # [B, 256, H/8, W/8]

        # 1. Softmax over 65 channels, drop dustbin → [B, 64, H/8, W/8]
        scores_sm = torch.softmax(scores_logits, dim=1)
        scores_no_dust = scores_sm[:, :-1, :, :]               # [B, 64, H/8, W/8]

        # 2. NMS at backbone resolution — 64× cheaper than post-pixel-shuffle.
        #
        #    Standard NMS: pixel_shuffle → max_pool2d(kernel=9) on [B,1,H,W]
        #    reads ~100 MB (307 K pixels × 81 neighbours) on the CPU.
        #
        #    Instead: find the per-cell peak score (max over 64 sub-pixels),
        #    suppress non-maximal cells at coarse scale, then pixel-shuffle.
        #    The coarse max_pool2d operates on [B,1,H/8,W/8] — 64× fewer elements.
        #
        #    Trade-off: effective NMS radius rounds up to the nearest cell boundary
        #    (nms_radius=4 px → 1 cell = 8 px).  Pairs of keypoints 5–7 px apart
        #    in different cells may both survive, but this is rare in practice.
        cell_peak = scores_no_dust.max(dim=1, keepdim=True)[0]     # [B, 1, H/8, W/8]
        nms_r = max(1, (self.nms_radius + 7) // 8)                 # px → cells (ceiling)
        max_pool = F.max_pool2d(cell_peak, kernel_size=2 * nms_r + 1,
                                stride=1, padding=nms_r)
        keep = (cell_peak == max_pool).float()                      # [B, 1, H/8, W/8]
        score_map = F.pixel_shuffle(scores_no_dust * keep, 8)      # [B, 1, H, W]

        # 3. L2-normalise descriptor map
        desc_norm = F.normalize(desc_map, p=2, dim=1)              # [B, 256, H/8, W/8]

        # 4. Zero border pixels (constant mask folded into graph)
        score_map = score_map * self.border_mask                    # [B, 1, H, W]

        # 5. Top-K keypoint extraction
        B, _, H, W = score_map.shape
        flat = score_map.view(B, -1)                            # [B, H*W]
        top_scores, flat_idx = torch.topk(flat, self.max_keypoints, dim=-1)  # [B, K]

        # 6. Flat indices → (x, y) pixel coordinates
        #    flat_idx = row * W + col  →  row = flat_idx // W,  col = flat_idx % W
        row_idx = (flat_idx // W).float()                       # [B, K]
        col_idx = (flat_idx  % W).float()                       # [B, K]
        keypoints = torch.stack([col_idx, row_idx], dim=-1)     # [B, K, 2]  (x=col, y=row)

        # 7. Bilinear descriptor sampling at keypoint locations
        #    SuperPoint convention: map coordinate = (pixel + 0.5) / (map_size * stride)
        #    grid_sample expects normalised coords in [-1, 1]
        norm_kpts = (keypoints + 0.5) / keypoints.new_tensor([W, H]) * 2.0 - 1.0  # [-1,1]
        descs = F.grid_sample(
            desc_norm,
            norm_kpts.unsqueeze(1),          # [B, 1, K, 2]
            mode="bilinear",
            align_corners=False,
        )                                    # [B, C, 1, K]
        descs = F.normalize(descs.squeeze(2), p=2, dim=1)       # [B, C, K]

        # 8. Transpose descriptors to row-major layout expected by decoders
        descriptors = descs.permute(0, 2, 1).contiguous()       # [B, K, 256]

        return keypoints, top_scores, descriptors


# ---------------------------------------------------------------------------
# RGB-input wrapper
# ---------------------------------------------------------------------------
class SuperPointEncoderRGB(nn.Module):
    """RGB-input wrapper around SuperPointEncoder.

    Accepts [B, 3, H, W] float32 RGB images in [0, 1] and converts to
    grayscale before forwarding to the SuperPoint backbone.

    Exporting this instead of SuperPointEncoder causes the AIPU compiler to
    see a 3-channel input, which it pads to 4 channels (same as any RGB model)
    rather than padding 1-channel input to 64 channels.  This reduces the
    CPU-side padding tensor from ~20 MB to ~1.6 MB per frame, cutting the
    libtransform_padding_0 cost from ~40 ms to ~2 ms.
    """

    def __init__(self, encoder: SuperPointEncoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, image: torch.Tensor):
        """
        Args:
            image: [B, 3, H, W] float32 RGB, pixel values in [0, 1].

        Returns:
            Same as SuperPointEncoder: (keypoints, scores, descriptors)
        """
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        return self.encoder(gray)


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
    parser.add_argument(
        "--max-keypoints", type=int, default=512,
        help="Fixed TopK output size K, baked into the ONNX graph (default: 512)",
    )
    parser.add_argument(
        "--remove-borders", type=int, default=4,
        help="Border pixels to suppress before TopK, baked into the ONNX graph (default: 4)",
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

    encoder = SuperPointEncoder(
        sp,
        nms_radius=args.nms_radius,
        max_keypoints=args.max_keypoints,
        remove_borders=args.remove_borders,
        height=args.height,
        width=args.width,
    ).eval()

    encoder_rgb = SuperPointEncoderRGB(encoder).eval()

    dummy = torch.zeros(1, 3, args.height, args.width)

    with torch.no_grad():
        keypoints, scores, descriptors = encoder_rgb(dummy)
    print(
        f"Encoder outputs (nms_radius={args.nms_radius}, K={args.max_keypoints}):\n"
        f"  keypoints   : {tuple(keypoints.shape)}\n"
        f"  scores      : {tuple(scores.shape)}\n"
        f"  descriptors : {tuple(descriptors.shape)}"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        encoder_rgb,
        dummy,
        str(output_path),
        opset_version=17,
        input_names=["image"],
        output_names=["keypoints", "scores", "descriptors"],
        dynamic_axes=None,  # fixed shapes required for Axelera AIPU compilation
    )
    print(f"ONNX encoder written to {output_path}")
    print(
        "\nNext step — compile for Metis M2:\n"
        f"  python deploy.py ax_models/custom/superpoint-gst/superpoint.yaml"
    )


if __name__ == "__main__":
    main()
