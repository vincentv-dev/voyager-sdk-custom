#!/usr/bin/env python3
"""
SuperPoint inference using axelera.runtime directly (bypasses GStreamer).

Uses 4 parallel AIPU workers + pipelining to approach hardware throughput.
Must be run from /home/vverkoren/Documents/Vydar/voyager-sdk with venv active:

    cd /home/vverkoren/Documents/Vydar/voyager-sdk
    source venv/bin/activate
    python3 ../SuperPoint/inference_superpoint_fast.py --video media/traffic1_1080p.mp4 --no-display
"""

import argparse
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

FRAMEWORK = os.environ.get("AXELERA_FRAMEWORK", str(Path(__file__).parent.parent / "voyager-sdk"))
sys.path.insert(0, FRAMEWORK)

from axelera.runtime import Context

MODEL_PATH = Path(FRAMEWORK) / "build/superpoint-onnx/superpoint-onnx/1/model.json"

# From manifest.json
DEQUANT_SCALE_DESC  = 0.005005544982850552
DEQUANT_ZP_DESC     = 0
DEQUANT_SCALE_SCORE = 0.11909855902194977
DEQUANT_ZP_SCORE    = 16
SCORE_REAL_CH       = 65   # output 1 has 128 channels; only first 65 are real

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → quantised int8 [480, 640] (only the real grayscale channel).

    The full padded input [1,482,642,64] is 20MB (mostly -128 padding).
    Each Worker holds a pre-filled buffer and only writes this 0.3MB channel,
    cutting per-frame memory bandwidth by ~98%.
    """
    gray = cv2.cvtColor(cv2.resize(bgr, (640, 480)), cv2.COLOR_BGR2GRAY)  # [480,640] uint8
    # quant_scale = 1/255, quant_zeropoint = -128  =>  q = pixel - 128
    return (gray.astype(np.int16) - 128).astype(np.int8)                  # [480,640]


# ── Postprocessing ─────────────────────────────────────────────────────────────

def dequant_and_postprocess(out0: np.ndarray, out1: np.ndarray):
    """
    out0: [1,60,80,256] int8 NHWC descriptors
    out1: [1,60,80,128] int8 NHWC scores (first 65 channels real)
    Returns: desc [256,60,80] float32, score_map [480,640] float32

    Replaces postprocess_graph.onnx with equivalent numpy ops:
      descriptors : L2-normalise over the channel axis
      scores      : softmax over 65 channels → drop dustbin → pixel-shuffle 8×
    """
    s = (out1[0, :, :, :SCORE_REAL_CH].astype(np.float32) - DEQUANT_ZP_SCORE) * DEQUANT_SCALE_SCORE  # [60,80,65]
    s = s.transpose(2, 0, 1)   # [65,60,80]

    # Numerically-stable softmax over 65 score channels
    s -= s.max(axis=0, keepdims=True)
    np.exp(s, out=s)
    s /= s.sum(axis=0, keepdims=True)

    # Drop dustbin channel, then pixel-shuffle 8× → [480,640]
    score_map = s[:64].reshape(8, 8, 60, 80).transpose(2, 0, 3, 1).reshape(480, 640)

    return score_map


# ── SuperPoint CPU decoder ─────────────────────────────────────────────────────

def _dilate(x: np.ndarray, radius: int) -> np.ndarray:
    k = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.dilate(x, k)


def nms(scores: np.ndarray, radius: int) -> np.ndarray:
    k = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return np.where(scores == cv2.dilate(scores, k), scores, 0.0)


def sample_descriptors(kp: np.ndarray, desc_map: np.ndarray, stride: int = 8) -> np.ndarray:
    c, h, w = desc_map.shape
    if len(kp) == 0:
        return np.zeros((0, c), dtype=np.float32)
    n = (kp + 0.5) / (np.array([w, h], np.float32) * stride) * 2.0 - 1.0
    fx = (n[:, 0] + 1.0) / 2.0 * (w - 1)
    fy = (n[:, 1] + 1.0) / 2.0 * (h - 1)
    x0 = np.clip(np.floor(fx).astype(np.int32), 0, w - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.clip(np.floor(fy).astype(np.int32), 0, h - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (fx - x0)[:, np.newaxis].astype(np.float32)
    wy = (fy - y0)[:, np.newaxis].astype(np.float32)
    d = (desc_map[:, y0, x0].T * (1 - wx) * (1 - wy) +
         desc_map[:, y0, x1].T * wx       * (1 - wy) +
         desc_map[:, y1, x0].T * (1 - wx) * wy +
         desc_map[:, y1, x1].T * wx       * wy)
    nrm = np.linalg.norm(d, axis=1, keepdims=True)
    return d / np.where(nrm < 1e-8, 1.0, nrm)


def decode(desc_map, score_map, nms_radius=4, threshold=0.005, borders=4, max_kp=1000,
           compute_descriptors=False):
    scores = nms(score_map, nms_radius)
    if borders > 0:
        p = borders
        scores[:p, :] = scores[-p:, :] = scores[:, :p] = scores[:, -p:] = 0
    ys, xs = np.where(scores > threshold)
    kp_sc = scores[ys, xs]
    kp    = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(kp) == 0:
        return kp, kp_sc, np.zeros((0, 256), np.float32)
    if max_kp > 0 and len(kp) > max_kp:
        idx = np.argsort(kp_sc)[::-1][:max_kp]
        kp, kp_sc = kp[idx], kp_sc[idx]
    if not compute_descriptors:
        return kp, kp_sc, np.zeros((0, 256), np.float32)
    return kp, kp_sc, sample_descriptors(kp, desc_map)


# ── AIPU Worker thread ─────────────────────────────────────────────────────────

class Worker(threading.Thread):
    def __init__(self, instance, in_shapes, out_shapes):
        super().__init__(daemon=True)
        self.instance = instance
        self.inq  = queue.Queue(maxsize=2)
        self.outq = queue.Queue(maxsize=2)
        # Pre-fill the input buffer with -128 (the padding value).
        # Only channel 0 of the spatial region [1:481, 1:641] will ever change.
        self.inputs  = [np.full(s, -128, np.int8) for s in in_shapes]
        self.outputs = [np.zeros(s, np.int8) for s in out_shapes]
        self.start()

    def run(self):
        while True:
            item = self.inq.get()
            if item is None:
                break
            frame_id, gray_q = item           # gray_q: [480,640] int8
            # Update only the real channel; all padding stays -128
            self.inputs[0][0, 1:481, 1:641, 0] = gray_q
            try:
                self.instance.run(self.inputs, self.outputs)
            except Exception as e:
                self.outq.put(e)
                return
            self.outq.put((frame_id, [o.copy() for o in self.outputs]))

    def submit(self, frame_id, data):
        self.inq.put((frame_id, data))

    def collect(self):
        result = self.outq.get()
        if isinstance(result, Exception):
            raise result
        return result

    def stop(self):
        self.inq.put(None)


# ── Visualisation ──────────────────────────────────────────────────────────────

def draw(frame: np.ndarray, kp, kp_sc, fps, n_kp, idx, frame_ms) -> np.ndarray:
    vis = frame.copy()
    if kp is not None and len(kp):
        mx = float(kp_sc.max()) if kp_sc.size else 1.0
        brightness = (80 + 175 * kp_sc / max(mx, 1e-8)).clip(0, 255).astype(np.uint8)
        xs = kp[:, 0].astype(np.int32)
        ys = kp[:, 1].astype(np.int32)
        fh, fw = vis.shape[:2]
        valid = (xs >= 1) & (xs < fw - 1) & (ys >= 1) & (ys < fh - 1)
        xs, ys, brightness = xs[valid], ys[valid], brightness[valid]
        colors = np.zeros((len(xs), 3), dtype=np.uint8)
        colors[:, 1] = brightness  # green channel only
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                vis[ys + dy, xs + dx] = colors
    ov = vis.copy()
    cv2.rectangle(ov, (0, 0), (360, 110), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.45, vis, 0.55, 0, vis)
    put = lambda t, y, c=(255, 255, 255): cv2.putText(
        vis, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 1, cv2.LINE_AA)
    put("SuperPoint  [Axelera Metis M2]", 22, (100, 220, 100))
    put(f"Frame      : {idx}", 44)
    put(f"Keypoints  : {n_kp}", 64)
    put(f"Frame time : {frame_ms:.1f} ms", 84)
    put(f"FPS        : {fps:.1f}", 104)
    return vis


# ── Main ───────────────────────────────────────────────────────────────────────

def run(video_path, output_path, no_display, max_frames,
        nms_radius=4, threshold=0.005, borders=4, max_kp=1000):

    print(f"Model      : {MODEL_PATH}")
    print(f"Video      : {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Resolution : {src_w}x{src_h}  FPS: {src_fps:.1f}  Frames: {n_total}")

    writer = None
    if output_path:
        writer = cv2.VideoWriter(
            output_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (src_w, src_h))
        print(f"Output     : {output_path}")

    # Queue between AIPU collection thread and postprocessing thread
    postproc_queue = queue.Queue(maxsize=8)
    display_queue  = queue.Queue(maxsize=8)
    stop_event     = threading.Event()

    fps_win  = deque(maxlen=30)
    t_start  = [time.perf_counter()]
    counters = {"collected": 0, "displayed": 0}

    def postproc_thread():
        """Runs numpy postprocess + keypoint decode in a dedicated thread."""
        t_prev = [time.perf_counter()]
        _prof = {"dequant_onnx": 0.0, "decode": 0.0, "draw": 0.0, "total": 0.0, "n": 0}
        while True:
            item = postproc_queue.get()
            if item is None:
                display_queue.put(None)
                if _prof["n"]:
                    n = _prof["n"]
                    print(f"\n[PROFILE] postproc avg over {n} frames:")
                    print(f"  dequant+ONNX : {_prof['dequant_onnx']/n*1000:.2f} ms")
                    print(f"  decode (NMS) : {_prof['decode']/n*1000:.2f} ms")
                    print(f"  draw         : {_prof['draw']/n*1000:.2f} ms")
                    print(f"  total        : {_prof['total']/n*1000:.2f} ms  →  {n/_prof['total']:.1f} fps max\n")
                break
            fid, outputs, bgr = item
            t_now    = time.perf_counter()
            frame_ms = (t_now - t_prev[0]) * 1000.0
            fps_win.append(1.0 / max(t_now - t_prev[0], 1e-6))
            t_prev[0] = t_now

            t0 = time.perf_counter()
            score_map = dequant_and_postprocess(outputs[0], outputs[1])
            t1 = time.perf_counter()
            kp, kp_sc, _ = decode(None, score_map, nms_radius, threshold, borders, max_kp)
            t2 = time.perf_counter()

            # Scale keypoints from model space (640x480) to source frame resolution
            h, w = bgr.shape[:2]
            if len(kp) and (w != 640 or h != 480):
                kp_vis = kp * np.array([w / 640.0, h / 480.0], np.float32)
            else:
                kp_vis = kp

            avg_fps = float(np.mean(fps_win)) if fps_win else 0.0
            vis = draw(bgr, kp_vis, kp_sc, avg_fps, len(kp), fid, frame_ms)
            t3 = time.perf_counter()
            _prof["dequant_onnx"] += t1 - t0
            _prof["decode"]       += t2 - t1
            _prof["draw"]         += t3 - t2
            _prof["total"]        += t3 - t0
            _prof["n"]            += 1
            display_queue.put((fid, vis, len(kp), frame_ms, avg_fps))

    t_postproc = threading.Thread(target=postproc_thread, daemon=True)
    t_postproc.start()

    with Context() as ctx:
        model    = ctx.load_model(str(MODEL_PATH))
        in_info  = model.inputs()
        out_info = model.outputs()
        in_shapes  = [i.shape for i in in_info]
        out_shapes = [o.shape for o in out_info]
        print(f"Input      : {in_shapes}")
        print(f"Outputs    : {out_shapes}")

        NUM_WORKERS = 4
        connections = [ctx.device_connect(None, num_sub_devices=1) for _ in range(NUM_WORKERS)]
        instances   = [c.load_model_instance(model, num_sub_devices=1, aipu_cores=1)
                       for c in connections]
        workers = [Worker(inst, in_shapes, out_shapes) for inst in instances]

        frame_idx   = 0
        collect_idx = 0
        in_flight   = 0
        pending_bgr = {}

        print("\nRunning — press Q to quit.\n")

        WIN_TITLE = "SuperPoint \u2014 Axelera Metis M2"
        if not no_display:
            cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN_TITLE, src_w, src_h)

        quit_flag = False
        while not quit_flag:
            # Fill workers
            while in_flight < NUM_WORKERS:
                if max_frames and frame_idx >= max_frames:
                    break
                ret, bgr = cap.read()
                if not ret:
                    break
                pending_bgr[frame_idx] = bgr
                workers[frame_idx % NUM_WORKERS].submit(frame_idx, preprocess(bgr))
                frame_idx += 1
                in_flight += 1

            if in_flight == 0:
                break

            # Collect AIPU result and hand off to postproc thread
            fid, outputs = workers[collect_idx % NUM_WORKERS].collect()
            in_flight  -= 1
            bgr = pending_bgr.pop(fid)
            collect_idx += 1
            postproc_queue.put((fid, outputs, bgr))

            # Drain display queue (non-blocking)
            while True:
                try:
                    item = display_queue.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    quit_flag = True
                    break
                fid2, vis, n_kp, frame_ms, avg_fps = item
                counters["displayed"] += 1
                if writer:
                    if vis.shape[:2] != (src_h, src_w):
                        vis = cv2.resize(vis, (src_w, src_h))
                    writer.write(vis)
                if not no_display:
                    cv2.imshow(WIN_TITLE, vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        quit_flag = True
                        break
                if fid2 % 30 == 0:
                    print(f"  frame {fid2:5d} | {avg_fps:6.1f} fps | {n_kp:5d} kp | {frame_ms:.1f}ms")

    # Signal postproc thread to stop.
    # Drain display_queue WHILE joining — postproc blocks on a full queue otherwise.
    postproc_queue.put(None)
    while True:
        try:
            item = display_queue.get(timeout=0.1)
        except queue.Empty:
            if not t_postproc.is_alive():
                break
            continue
        if item is None:
            break
        fid2, vis, n_kp, frame_ms, avg_fps = item
        counters["displayed"] += 1
        if writer:
            if vis.shape[:2] != (src_h, src_w):
                vis = cv2.resize(vis, (src_w, src_h))
            writer.write(vis)
        if fid2 % 30 == 0:
            print(f"  frame {fid2:5d} | {avg_fps:6.1f} fps | {n_kp:5d} kp | {frame_ms:.1f}ms")
    t_postproc.join()

    for w in workers:
        w.stop()
    cap.release()
    if writer:
        writer.release()
    if not no_display:
        cv2.destroyAllWindows()

    elapsed    = time.perf_counter() - t_start[0]
    total_done = collect_idx
    print(f"\n{'='*50}")
    print(f"  SuperPoint — axelera.runtime direct")
    print(f"{'='*50}")
    print(f"  Frames     : {total_done}")
    print(f"  Time       : {elapsed:.2f}s")
    if elapsed > 0 and total_done > 0:
        print(f"  Avg FPS    : {total_done / elapsed:.1f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SuperPoint on MP4 via axelera.runtime (fast)")
    p.add_argument("--video",               required=True)
    p.add_argument("--output",              default=None)
    p.add_argument("--no-display",          action="store_true")
    p.add_argument("--max-frames",          type=int,   default=0)
    p.add_argument("--nms-radius",          type=int,   default=4)
    p.add_argument("--detection-threshold", type=float, default=0.005)
    p.add_argument("--remove-borders",      type=int,   default=4)
    p.add_argument("--max-keypoints",       type=int,   default=1000)
    a = p.parse_args()
    run(a.video, a.output, a.no_display, a.max_frames,
        a.nms_radius, a.detection_threshold, a.remove_borders, a.max_keypoints)
