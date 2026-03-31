#!/usr/bin/env python3
"""Validate EdgePoint decoder and produce an annotated MP4.

Runs the Python decoder (CPU/ONNX path) on a video or image, draws detected
keypoints as green circles, and saves the result.

Usage:
    python ax_models/custom/edgepoint/validate_edgepoint.py <input> [options]

Examples:
    python ax_models/custom/edgepoint/validate_edgepoint.py media/traffic1_480p.mp4
    python ax_models/custom/edgepoint/validate_edgepoint.py media/traffic1_480p.mp4 \\
        --output /tmp/edgepoint_vis.mp4 --frames 300 --threshold 0.0
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_THIS_DIR  = Path(__file__).resolve().parent
_SDK_ROOT  = _THIS_DIR.parents[2]           # voyager-sdk-custom root
_EP_REPO   = Path('/home/Vydar/EdgePoint2')
_WEIGHTS   = _THIS_DIR / 'weights' / 'E64.pth'
_GREEN_BGR = (0, 220, 0)

_DESC_UPSCALE = 4
_DESC_DIM     = 64
_C2_DIM       = 16
_C3_DIM       = 48
_C4_DIM       = 64


def _load_model(edgepoint_repo: Path, weights: Path):
    if not edgepoint_repo.exists():
        sys.exit(f'EdgePoint2 repo not found at {edgepoint_repo}')
    if not weights.exists():
        sys.exit(f'Weights not found at {weights}')
    if str(edgepoint_repo) not in sys.path:
        sys.path.insert(0, str(edgepoint_repo))
    from model.model import EdgePoint2
    cfg   = {'c1': 16, 'c2': 16, 'c3': 48, 'c4': 64, 'cdesc': 64, 'cdetect': 16}
    model = EdgePoint2(**cfg)
    state = torch.load(str(weights), map_location='cpu')
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _load_encoder(edgepoint_repo: Path, onnx_path: Path):
    """Load the ONNX backbone encoder via ONNXRuntime for speed."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    return sess


def _preprocess(frame_bgr: np.ndarray, size=(512, 512)):
    """BGR frame → float32 NCHW tensor in [0, 1]."""
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resz  = cv2.resize(gray, size)
    t     = torch.from_numpy(resz).float().div(255.0)
    return t.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]


def _run_heads(model, x1_pad, x2_pad, x3_pad):
    x1 = x1_pad[:, :_C2_DIM]
    x2 = x2_pad[:, :_C3_DIM]
    x3 = x3_pad[:, :_C4_DIM]
    with torch.no_grad():
        _x2       = F.avg_pool2d(x1, 2, 2)
        desc_raw  = model.desc_head(
            torch.cat([
                _x2,
                F.interpolate(x2, scale_factor=2,  mode='bilinear', align_corners=False),
                F.interpolate(x3, scale_factor=8,  mode='bilinear', align_corners=False),
            ], dim=1)
        )
        score_map = model.score_head(
            model.conv1(x1)
            + F.interpolate(model.conv2(x2), scale_factor=4,  mode='bilinear', align_corners=False)
            + F.interpolate(model.conv3(x3), scale_factor=16, mode='bilinear', align_corners=False)
        )
    return score_map, desc_raw


def _detect(score_map, desc_raw, threshold, nms_radius, remove_borders, max_kpts):
    k = 2 * nms_radius + 1
    mp = F.max_pool2d(score_map, k, stride=1, padding=nms_radius)
    s  = score_map * (score_map == mp).float()
    s2 = s.squeeze(0).squeeze(0)
    p  = remove_borders
    if p > 0:
        s2 = s2.clone()
        s2[:p, :] = s2[-p:, :] = s2[:, :p] = s2[:, -p:] = 0
    row_idx, col_idx = torch.where(s2 > threshold)
    kpts   = torch.stack([col_idx, row_idx], dim=-1).float()
    scores = s2[row_idx, col_idx]
    if max_kpts > 0 and len(kpts) > max_kpts:
        scores, top = torch.topk(scores, max_kpts)
        kpts = kpts[top]
    return kpts.numpy(), scores.numpy()


def _draw(frame_bgr, kpts, input_size, output_size):
    """Draw keypoints scaled from input_size back to output_size."""
    h_out, w_out = output_size
    h_in, w_in   = input_size
    sx = w_out / w_in
    sy = h_out / h_in
    for x, y in kpts:
        cx, cy = int(x * sx + 0.5), int(y * sy + 0.5)
        cv2.circle(frame_bgr, (cx, cy), 3, _GREEN_BGR, -1)
        cv2.circle(frame_bgr, (cx, cy), 3, (0, 0, 0),  1)
    cv2.putText(frame_bgr, f'{len(kpts)} keypoints', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    return frame_bgr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input',  help='input video or image')
    ap.add_argument('-o', '--output', default='/tmp/edgepoint_validated.mp4',
                    help='output annotated video (default: /tmp/edgepoint_validated.mp4)')
    ap.add_argument('--frames',    type=int,   default=0,
                    help='max frames to process (0 = all)')
    ap.add_argument('--threshold', type=float, default=0.0,
                    help='keypoint score threshold (default: 0.0)')
    ap.add_argument('--nms-radius',      type=int, default=2)
    ap.add_argument('--remove-borders',  type=int, default=4)
    ap.add_argument('--max-keypoints',   type=int, default=1024)
    ap.add_argument('--repo',    default=str(_EP_REPO), help='path to EdgePoint2 repo')
    ap.add_argument('--weights', default=str(_WEIGHTS), help='path to E64.pth')
    ap.add_argument('--size',    default='512x512',
                    help='inference resolution WxH (default: 512x512)')
    args = ap.parse_args()

    W_in, H_in = (int(x) for x in args.size.lower().split('x'))

    print(f'Loading EdgePoint2 model from {args.repo} ...')
    model = _load_model(Path(args.repo), Path(args.weights))

    # Try to use the ONNX encoder for the backbone (faster)
    onnx_path = _THIS_DIR / 'weights' / 'edgepoint_encoder.onnx'
    ort_sess  = _load_encoder(Path(args.repo), onnx_path) if onnx_path.exists() else None
    if ort_sess:
        print(f'Using ONNX backbone: {onnx_path.name}')
    else:
        print('ONNX backbone not found, using PyTorch backbone (slower)')

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f'Cannot open: {args.input}')

    fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W_out = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_out = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit = args.frames if args.frames > 0 else total

    out = cv2.VideoWriter(args.output,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (W_out, H_out))
    if not out.isOpened():
        sys.exit(f'Cannot create output: {args.output}')

    print(f'Processing {limit} frames → {args.output}')
    processed = 0
    kpt_counts = []

    while processed < limit:
        ok, frame = cap.read()
        if not ok:
            break

        inp = _preprocess(frame, (W_in, H_in))  # [1,1,H,W]

        if ort_sess:
            inp_np = inp.numpy()
            names  = [x.name for x in ort_sess.get_inputs()]
            outs   = ort_sess.run(None, {names[0]: inp_np})
            # Sort outputs by spatial area (largest first = x1, x2, x3)
            feats  = sorted(outs, key=lambda t: t.shape[2] * t.shape[3], reverse=True)
            x1_pad = torch.from_numpy(feats[0])
            x2_pad = torch.from_numpy(feats[1])
            x3_pad = torch.from_numpy(feats[2])
        else:
            with torch.no_grad():
                x1_pad, x2_pad, x3_pad = model.encoder(inp)

        score_map, desc_raw = _run_heads(model, x1_pad, x2_pad, x3_pad)
        kpts, scores = _detect(score_map, desc_raw,
                               args.threshold, args.nms_radius,
                               args.remove_borders, args.max_keypoints)

        frame = _draw(frame, kpts, (H_in, W_in), (H_out, W_out))
        out.write(frame)
        kpt_counts.append(len(kpts))
        processed += 1
        if processed % 50 == 0:
            print(f'  {processed}/{limit}  avg keypoints: {np.mean(kpt_counts[-50:]):.0f}')

    cap.release()
    out.release()

    if kpt_counts:
        print(f'\nDone. {processed} frames written to {args.output}')
        print(f'Keypoints per frame — mean: {np.mean(kpt_counts):.0f}  '
              f'min: {min(kpt_counts)}  max: {max(kpt_counts)}')
    else:
        print('No frames processed.')


if __name__ == '__main__':
    main()
