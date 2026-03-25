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
                     remove_borders=4, max_keypoints=1024):
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


def _new_postprocess(score_map, desc_norm,
                     threshold=0.005, remove_borders=4, max_keypoints=1024):
    """New pipeline: score_map already NMS-suppressed; just extract kpts."""
    scores_2d = score_map.squeeze(0).squeeze(0).clone()

    pad = remove_borders
    if pad > 0:
        scores_2d[:pad] = 0; scores_2d[-pad:] = 0
        scores_2d[:, :pad] = 0; scores_2d[:, -pad:] = 0

    row, col = torch.where(scores_2d > threshold)
    kpts = torch.stack([col, row], dim=-1).float()
    kpt_scores = scores_2d[row, col]

    if max_keypoints > 0 and len(kpts) > max_keypoints:
        kpt_scores, idx = torch.topk(kpt_scores, max_keypoints)
        kpts = kpts[idx]

    _, c, hf, wf = desc_norm.shape
    stride = score_map.shape[2] // hf
    norm_kpts = (kpts + 0.5) / (kpts.new_tensor([wf, hf]) * stride)
    norm_kpts = norm_kpts * 2 - 1
    descs = F.grid_sample(desc_norm, norm_kpts.view(1, 1, -1, 2),
                          mode='bilinear', align_corners=False)
    descs = F.normalize(descs.reshape(1, c, -1), p=2, dim=1).squeeze(0).T

    return kpts.cpu().numpy(), kpt_scores.cpu().numpy(), descs.cpu().numpy()


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
    t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
    return t, img


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
    parser.add_argument("--max-keypoints", type=int, default=1024)
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

    from export_onnx import SuperPointEncoder
    new_enc = SuperPointEncoder(sp, nms_radius=args.nms_radius).eval()

    # ---- Load image ----
    image_t, orig_img = _load_frame(args.input, args.height, args.width)
    print(f"Input: {args.input}  →  {args.width}×{args.height} grayscale")

    # ---- Run both pipelines ----
    with torch.no_grad():
        scores_logits, desc_map_raw = raw_enc(image_t)
        score_map, desc_norm = new_enc(image_t)

    # ---- Sanity-check model outputs ----
    assert score_map.min() >= 0, "score_map has negative values"
    assert score_map.max() <= 1 + 1e-4, f"score_map max {score_map.max():.5f} > 1"
    norms = desc_norm.norm(dim=1)
    assert norms.min() > 0.99, f"desc_norm not unit: min={norms.min():.4f}"

    kw = dict(nms_radius=args.nms_radius, threshold=args.threshold,
              max_keypoints=args.max_keypoints)
    old_kpts, old_scr, old_desc = _old_postprocess(scores_logits, desc_map_raw, **kw)
    new_kw = dict(threshold=args.threshold, max_keypoints=args.max_keypoints)
    new_kpts, new_scr, new_desc = _new_postprocess(score_map, desc_norm, **new_kw)

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
