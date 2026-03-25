#!/usr/bin/env python3
"""Validate SuperPoint decoder: compare new model post-processing against old.

Runs two pipelines on the same input image:
  OLD: raw CNN outputs → manual softmax + batched-NMS (3-round) in Python
  NEW: ONNX model with softmax + single-pass max-pool NMS baked in

Reports keypoint statistics and descriptor quality metrics, then saves a
side-by-side visualisation.

Usage:
    python ax_models/superpoint/validate_decoder.py <image_or_video> [options]

Examples:
    python ax_models/superpoint/validate_decoder.py media/traffic1_480p.mp4
    python ax_models/superpoint/validate_decoder.py media/traffic1_1080p.mp4 --output kpts.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_SDK_ROOT = _THIS_DIR.parent.parent
_SP_DIR = _SDK_ROOT.parent.parent / "SuperPoint-custom"

if not _SP_DIR.exists():
    raise RuntimeError(
        f"SuperPoint repo not found at {_SP_DIR}.\n"
        "Clone: git clone https://github.com/rpautrat/SuperPoint.git"
    )
sys.path.insert(0, str(_SP_DIR))
from superpoint_pytorch import SuperPoint  # noqa: E402


# ---------------------------------------------------------------------------
# Old encoder (backbone + heads only, no post-processing)
# ---------------------------------------------------------------------------
class _RawEncoder(nn.Module):
    """Returns raw logits, same as the original export before this change."""
    def __init__(self, sp):
        super().__init__()
        self.backbone = sp.backbone
        self.detector = sp.detector
        self.descriptor = sp.descriptor

    def forward(self, x):
        f = self.backbone(x)
        return self.detector(f), self.descriptor(f)   # [1,65,Hf,Wf], [1,256,Hf,Wf]


# ---------------------------------------------------------------------------
# Old post-processing (matches the original superpoint_decoder.py)
# ---------------------------------------------------------------------------
def _batched_nms(scores: torch.Tensor, nms_radius: int) -> torch.Tensor:
    """3-round max-pool NMS, same as original _batched_nms."""
    def mp(x):
        return F.max_pool2d(x, kernel_size=nms_radius * 2 + 1,
                            stride=1, padding=nms_radius)
    s = scores.unsqueeze(0).unsqueeze(0)
    zeros = torch.zeros_like(s)
    max_mask = s == mp(s)
    for _ in range(2):
        supp = mp(max_mask.float()) > 0
        supp_s = torch.where(supp, zeros, s)
        new_max = supp_s == mp(supp_s)
        max_mask = max_mask | (new_max & ~supp)
    return torch.where(max_mask, s, zeros).squeeze(0).squeeze(0)


def _old_postprocess(scores_logits, desc_map_raw,
                     nms_radius=4, threshold=0.005,
                     remove_borders=4, max_keypoints=512):
    """Full old pipeline: softmax → pixel-shuffle → batched NMS → kpts → descs."""
    s = F.softmax(scores_logits, dim=1)[:, :-1]   # [1, 64, Hf, Wf]
    b, _, hf, wf = s.shape
    stride = 8
    scores_2d = (
        s.permute(0, 2, 3, 1)
        .reshape(b, hf, wf, stride, stride)
        .permute(0, 1, 3, 2, 4)
        .reshape(b, hf * stride, wf * stride)
        .squeeze(0)
    )                                              # [H, W]

    scores_2d = _batched_nms(scores_2d, nms_radius)

    pad = remove_borders
    if pad > 0:
        scores_2d = scores_2d.clone()
        scores_2d[:pad] = 0; scores_2d[-pad:] = 0
        scores_2d[:, :pad] = 0; scores_2d[:, -pad:] = 0

    row, col = torch.where(scores_2d > threshold)
    kpts = torch.stack([col, row], dim=-1).float()
    kpt_scores = scores_2d[row, col]

    if max_keypoints > 0 and len(kpts) > max_keypoints:
        kpt_scores, idx = torch.topk(kpt_scores, max_keypoints)
        kpts = kpts[idx]

    desc_norm = F.normalize(desc_map_raw, p=2, dim=1)
    _, c, hf, wf = desc_norm.shape
    norm_kpts = (kpts + 0.5) / (kpts.new_tensor([wf, hf]) * stride)
    norm_kpts = norm_kpts * 2 - 1
    descs = F.grid_sample(desc_norm, norm_kpts.view(1, 1, -1, 2),
                          mode='bilinear', align_corners=False)
    descs = F.normalize(descs.reshape(1, c, -1), p=2, dim=1).squeeze(0).T

    return kpts.cpu().numpy(), kpt_scores.cpu().numpy(), descs.cpu().numpy()


def _new_postprocess(keypoints, scores, descriptors, threshold=0.005):
    """New pipeline: unpack pre-computed ONNX outputs and threshold-filter."""
    # keypoints:   [1, K, 2]   (x, y) pixel coords, sorted by score descending
    # scores:      [1, K]
    # descriptors: [1, K, 256]
    kpts  = keypoints[0]     # [K, 2]
    scr   = scores[0]        # [K]
    descs = descriptors[0]   # [K, 256]

    if threshold > 0:
        mask  = scr > threshold
        kpts  = kpts[mask]
        scr   = scr[mask]
        descs = descs[mask]

    return kpts.cpu().numpy(), scr.cpu().numpy(), descs.cpu().numpy()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _load_frame(path, height=480, width=640):
    path = Path(path)
    if path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}:
        import cv2
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Cannot read frame from {path}")
        img = Image.fromarray(frame[:, :, ::-1])   # BGR → RGB
    else:
        img = Image.open(path).convert('RGB')
    img = img.resize((width, height), Image.LANCZOS)
    gray = np.array(img.convert('L'), dtype=np.float32) / 255.0
    t_gray = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)       # [1,1,H,W]
    rgb = np.array(img, dtype=np.float32) / 255.0                   # [H,W,3]
    t_rgb = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)    # [1,3,H,W]
    return t_gray, t_rgb, img


def _draw_kpts(img_rgb: Image.Image, kpts: np.ndarray, color) -> np.ndarray:
    import cv2
    vis = np.array(img_rgb)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    for x, y in kpts.astype(int):
        cv2.circle(vis, (x, y), 2, color, -1)
    return vis


def _nearest_keypoints(kpts_a, kpts_b, radius=4):
    """Fraction of kpts_a that have a kpts_b match within `radius` pixels."""
    if len(kpts_a) == 0 or len(kpts_b) == 0:
        return 0.0
    dists = np.linalg.norm(kpts_a[:, None] - kpts_b[None], axis=2)   # [Na, Nb]
    return float((dists.min(axis=1) <= radius).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Validate SuperPoint decoder")
    parser.add_argument("input", help="Image or video file")
    parser.add_argument(
        "--weights",
        default=str(_SP_DIR / "weights" / "superpoint_v6_from_tf.pth"),
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--nms-radius", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--max-keypoints", type=int, default=512)
    parser.add_argument("--remove-borders", type=int, default=4)
    parser.add_argument(
        "--output", default="validation_output.png",
        help="Path for side-by-side visualisation (default: validation_output.png)"
    )
    args = parser.parse_args()

    # ---- Load weights ----
    print(f"Loading weights from {args.weights} ...")
    sp = SuperPoint()
    sp.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    sp.eval()

    raw_enc = _RawEncoder(sp).eval()

    from export_onnx import SuperPointEncoder, SuperPointEncoderRGB
    new_enc = SuperPointEncoderRGB(
        SuperPointEncoder(
            sp,
            nms_radius=args.nms_radius,
            max_keypoints=args.max_keypoints,
            remove_borders=args.remove_borders,
            height=args.height,
            width=args.width,
        )
    ).eval()

    # ---- Load image ----
    image_t, image_rgb_t, orig_img = _load_frame(args.input, args.height, args.width)
    print(f"Input: {args.input}  →  {args.width}×{args.height}")

    # ---- Run both pipelines ----
    with torch.no_grad():
        scores_logits, desc_map_raw = raw_enc(image_t)
        new_keypoints, new_scores, new_descriptors = new_enc(image_rgb_t)

    # ---- Sanity-check model outputs ----
    assert new_scores.min() >= 0, "scores have negative values"
    assert new_scores.max() <= 1 + 1e-4, f"scores max {new_scores.max():.5f} > 1"
    desc_norms = new_descriptors.norm(dim=-1)
    # Zero-padded slots (score == 0) will have zero-norm descriptors; skip them.
    valid_mask = new_scores[0] > 0
    if valid_mask.any():
        valid_norms = desc_norms[0][valid_mask]
        assert valid_norms.min() > 0.99, f"descriptors not unit-norm: min={valid_norms.min():.4f}"

    kw = dict(nms_radius=args.nms_radius, threshold=args.threshold,
              max_keypoints=args.max_keypoints)
    old_kpts, old_scr, old_desc = _old_postprocess(scores_logits, desc_map_raw, **kw)
    new_kpts, new_scr, new_desc = _new_postprocess(
        new_keypoints, new_scores, new_descriptors, threshold=args.threshold
    )

    # ---- Report ----
    print(f"\n{'':=<62}")
    print(f"  {'Metric':<34}  {'OLD':>10}  {'NEW':>10}")
    print(f"{'':=<62}")
    print(f"  {'Keypoints detected':<34}  {len(old_kpts):>10}  {len(new_kpts):>10}")

    if len(old_kpts) and len(new_kpts):
        match_o2n = _nearest_keypoints(old_kpts, new_kpts, radius=args.nms_radius)
        match_n2o = _nearest_keypoints(new_kpts, old_kpts, radius=args.nms_radius)
        print(f"  {'OLD kpts matched in NEW  (r=nms)':<34}  {match_o2n:>10.1%}")
        print(f"  {'NEW kpts matched in OLD  (r=nms)':<34}  {match_n2o:>10.1%}")

        # Score distribution
        print(f"  {'Score median':<34}  {np.median(old_scr):>10.4f}  {np.median(new_scr):>10.4f}")
        print(f"  {'Score max':<34}  {old_scr.max():>10.4f}  {new_scr.max():>10.4f}")

        # Descriptor similarity at matched keypoints
        # For each old kpt, find nearest new kpt and compute descriptor cosine sim
        dists = np.linalg.norm(old_kpts[:, None] - new_kpts[None], axis=2)   # [No, Nn]
        matched = dists.min(axis=1) <= args.nms_radius
        if matched.sum() > 0:
            nn_idx = dists.argmin(axis=1)[matched]
            cos_sim = (old_desc[matched] * new_desc[nn_idx]).sum(axis=1)
            print(f"  {'Descriptor cos-sim (matched kpts)':<34}  {'n/a':>10}  {cos_sim.mean():>10.4f}")
            print(f"  {'  (1.0 = identical, >0.95 = good)':<34}")

    print(f"{'':=<62}")

    # ---- NMS check on new output ----
    if len(new_kpts) >= 2:
        diffs = np.abs(new_kpts[:, None] - new_kpts[None])       # [N,N,2]
        too_close = (diffs[..., 0] <= args.nms_radius) & \
                    (diffs[..., 1] <= args.nms_radius)
        np.fill_diagonal(too_close, False)
        violations = too_close.any(axis=1).sum()
        if violations:
            print(f"[WARNING] NMS check: {violations} keypoints within radius={args.nms_radius}")
        else:
            print(f"[OK] NMS: no two keypoints within radius={args.nms_radius}")

    # ---- Save side-by-side visualisation ----
    import cv2
    left  = _draw_kpts(orig_img, old_kpts, (0, 0, 255))   # red = old
    right = _draw_kpts(orig_img, new_kpts, (0, 200, 0))   # green = new

    h, w = left.shape[:2]
    sep = np.full((h, 4, 3), 200, dtype=np.uint8)
    combined = np.hstack([left, sep, right])

    cv2.putText(combined, f"OLD  N={len(old_kpts)}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(combined, f"NEW  N={len(new_kpts)}", (w + 12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    cv2.imwrite(args.output, combined)
    print(f"\nSaved: {args.output}  (red=old batched-NMS, green=new single-pass NMS)")


if __name__ == "__main__":
    main()
