// SuperPoint post-processing GStreamer decoder for Axelera Metis M2
//
// The ONNX model outputs raw tensors only (Conv/ReLU/MaxPool backbone + heads).
// Post-processing ops (Softmax, PixelShuffle/Reshape/Transpose, Equal+Cast for
// NMS, ReduceL2/Div) are unsupported by the Metis M2 AIPU compiler and are
// handled here on the host CPU:
//
//   1. Softmax over 65 channels + dustbin removal + pixel-shuffle → score_map [H, W]
//   2. Separable max-pool NMS (two 1-D passes, O(H·W·2k) vs naïve O(H·W·k²))
//   3. Border removal + threshold → candidate keypoints
//   4. Top-k selection → final N keypoints
//   5. Bilinear sampling + per-descriptor L2-normalise
//
// Input tensor layout: NHWC float32  (handle_transpose: false in the YAML)
//
//   The SDK's dequantize transform outputs NHWC float32 without transposing.
//   NHWC data is already channel-contiguous per spatial cell — logits_ptr[hw*65+c]
//   and desc_ptr[hw*256+c] — so no CHW→HWC scatter-gather is needed.
//   This eliminates two full-tensor transpose passes vs the old NCHW path:
//     - old: NHWC int8 → NCHW float32 (SDK) → HWC float32 (decoder) — 2 passes
//     - new: NHWC int8 → NHWC float32 (SDK) — direct use, 0 extra passes
//
// Performance notes (480×640, nms_radius=4, N=1024):
//   Bottleneck        naïve          optimised       speedup
//   NMS               ~25 M cmp      ~5.5 M cmp      ~4.5×
//   CHW→HWC passes    2 × ~6 MB      eliminated       ∞
//   Heap alloc        ~7 MB/frame    once (Props)     zero per-frame
//
// Build flags that help further:  -O3 -ffast-math -mfpu=neon-vfpv4
// For more cores:                 add -fopenmp and uncomment the pragmas below.
//
// Output stored as AxMetaRawTensor → deserialized by Python TensorMeta:
//   tensors[0]  keypoints    [N, 2]    float32  (x, y) pixel coords
//   tensors[1]  scores       [N]       float32  detector confidence
//   tensors[2]  descriptors  [N, 256]  float32  L2-normalised

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaRawTensor.hpp"
#include "AxOpUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

// ---- Constants matching the SuperPoint architecture ----
static constexpr int DESC_DIM       = 256;
static constexpr int LOGIT_CHANNELS = 65;   // 64 sub-pixel + 1 dustbin
static constexpr int UPSCALE        = 8;    // VGG backbone stride

// ---- Properties + pre-allocated frame buffers ----
struct Props {
  std::string meta_name        = "superpoint";
  float       det_threshold    = 0.005f;
  int         remove_borders   = 4;
  int         max_keypoints    = 1024;   // ≤0 → unlimited
  int         nms_radius       = 4;

  // Working buffers resized once on first inference call (mutable because
  // decode_to_meta receives a const Props*).
  // logits_hwc and desc_hwc are no longer needed: NHWC tensors arrive already
  // in channel-contiguous (HWC) order from the dequantize transform.
  mutable int buf_H = 0, buf_W = 0, buf_Hf = 0, buf_Wf = 0;
  mutable std::vector<float> score_map;   // [H, W]   pixel-shuffled scores
  mutable std::vector<float> row_max;     // [H, W]   NMS horizontal pass
  mutable std::vector<float> maxpool;     // [H, W]   NMS result / van Herk suffix temp
  mutable std::vector<float> nms_prefix;  // [H, W]   van Herk prefix buffer
};

// ============================================================
//  Helpers
// ============================================================

// L2-normalise a float vector in-place (safe for zero-norm).
static void
l2_normalize(float* v, int n)
{
  float sq = 0.f;
  for (int i = 0; i < n; ++i) sq += v[i] * v[i];
  float inv = 1.f / std::sqrt(sq + 1e-10f);
  for (int i = 0; i < n; ++i) v[i] *= inv;
}

// ============================================================
//  Plugin entry points (extern "C")
// ============================================================
extern "C" {

std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string>& input, Ax::Logger& log)
{
  auto p = std::make_shared<Props>();
  auto get = [&](const char* k, auto& dst) {
    if (auto it = input.find(k); it != input.end()) {
      if constexpr (std::is_same_v<std::decay_t<decltype(dst)>, std::string>)
        dst = it->second;
      else if constexpr (std::is_same_v<std::decay_t<decltype(dst)>, int>)
        dst = std::stoi(it->second);
      else
        dst = std::stof(it->second);
    }
  };
  get("meta_key",           p->meta_name);
  get("detection_threshold",p->det_threshold);
  get("remove_borders",     p->remove_borders);
  get("max_keypoints",      p->max_keypoints);
  get("nms_radius",         p->nms_radius);

  log(AX_INFO) << "SuperPoint decoder init: threshold=" << p->det_threshold
               << " borders=" << p->remove_borders
               << " max_kpts=" << p->max_keypoints
               << " nms_radius=" << p->nms_radius;
  return p;
}

const std::unordered_set<std::string>&
allowed_properties()
{
  static const std::unordered_set<std::string> s{
    "meta_key", "detection_threshold", "remove_borders", "max_keypoints", "nms_radius"
  };
  return s;
}

void
set_dynamic_properties(
    const std::unordered_map<std::string, std::string>&, void*, Ax::Logger&) {}

void
decode_to_meta(
    const AxTensorsInterface&  in_tensors,
    const Props*               prop,
    unsigned int               subframe_index,
    unsigned int               subframe_number,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>>& map,
    const AxDataInterface&     /*video*/,
    Ax::Logger&                log)
try {
  // --- Identify tensors by channel count (AIPU may reorder outputs) ---
  // Tensors arrive as NHWC float32 (handle_transpose: false):
  // scores_logits: [1, Hf, Wf, 65]   raw detector logits (pre-softmax)
  // desc_map:      [1, Hf, Wf, 256]  raw descriptor map  (pre-L2-norm)
  const float* logits_ptr = nullptr;
  const float* desc_ptr   = nullptr;
  int logits_Hf = 0, logits_Wf = 0;
  int Hf = 0, Wf = 0;

  for (const auto& t : in_tensors) {
    if (t.sizes.size() != 4 || t.bytes != 4) continue;
    if (!t.data) continue;
    int C = t.sizes[3];   // NHWC: channel is the last dimension
    if (C == LOGIT_CHANNELS) {
      logits_ptr = static_cast<const float*>(t.data);
      logits_Hf  = t.sizes[1];
      logits_Wf  = t.sizes[2];
    } else if (C == DESC_DIM) {
      desc_ptr = static_cast<const float*>(t.data);
      Hf = t.sizes[1];
      Wf = t.sizes[2];
    }
  }

  if (!logits_ptr || !desc_ptr || logits_Hf == 0 || Hf == 0) {
    log(AX_ERROR) << "SuperPoint: could not identify output tensors "
                     "(need C=65 logits and C=256 desc_map)";
    return;
  }

  const int H    = logits_Hf * UPSCALE;
  const int W    = logits_Wf * UPSCALE;
  const int HfWf = logits_Hf * logits_Wf;

  // --- Resize working buffers (once on first frame or on resolution change) ---
  if (prop->buf_H != H || prop->buf_W != W ||
      prop->buf_Hf != Hf || prop->buf_Wf != Wf) {
    prop->buf_H = H;  prop->buf_W = W;
    prop->buf_Hf = Hf; prop->buf_Wf = Wf;
    prop->score_map.resize(H * W);
    prop->row_max.resize(H * W);
    prop->maxpool.resize(H * W);
    prop->nms_prefix.resize(H * W);
  }

  // NHWC data is already channel-contiguous per cell: ptr[hw * C + c].
  // No CHW→HWC transpose needed — alias the tensor pointers directly.
  const float* logits_hwc = logits_ptr;   // [Hf·Wf, 65]
  float*       score_map  = prop->score_map.data();
  float*       row_max    = prop->row_max.data();
  float*       maxpool    = prop->maxpool.data();
  const float* desc_hwc   = desc_ptr;     // [Hf·Wf, 256]

  // =========================================================================
  // Step 1: Softmax + pixel-shuffle → score_map [H, W]
  //
  // For each cell hw in [Hf·Wf]:
  //   - Read 65 contiguous logits from logits_hwc[hw * 65 ..]
  //   - Numerically-stable softmax (max subtraction)
  //   - Drop dustbin (channel 64); scatter 64 values via pixel-shuffle
  // =========================================================================
  #pragma omp parallel for schedule(static)
  for (int hw = 0; hw < HfWf; ++hw) {
    const float* lv_in = logits_hwc + hw * LOGIT_CHANNELS;
    const int    h     = hw / logits_Wf;
    const int    w     = hw % logits_Wf;

    // Softmax with max subtraction
    float max_val = lv_in[0];
    for (int c = 1; c < LOGIT_CHANNELS; ++c)
      if (lv_in[c] > max_val) max_val = lv_in[c];

    float lv[LOGIT_CHANNELS];
    float sum_exp = 0.f;
    for (int c = 0; c < LOGIT_CHANNELS; ++c) {
      lv[c]    = std::exp(lv_in[c] - max_val);
      sum_exp += lv[c];
    }
    const float inv_sum = 1.f / sum_exp;

    // Pixel-shuffle: channel c → full-res offset (oh = c/8, ow = c%8)
    for (int c = 0; c < LOGIT_CHANNELS - 1; ++c) {
      score_map[(h * UPSCALE + c / UPSCALE) * W + (w * UPSCALE + c % UPSCALE)]
          = lv[c] * inv_sum;
    }
  }

  // =========================================================================
  // Step 2: Separable max-pool NMS
  //
  // Pass 1 uses the van Herk / Gil-Werman O(n) sliding-window maximum,
  // replacing the naive O(n·k) inner loop.  For k=9 this saves ~3× work.
  //
  // Algorithm: divide each row into blocks of k.  Compute prefix-max (left→
  // right within each block) into nms_prefix, and suffix-max (right→left)
  // into maxpool (safe temp: overwritten in Pass 2).  Combine:
  //   row_max[x] = max(suffix[max(0,x−r)], prefix[min(W−1,x+r)])
  //
  // Pass 2 keeps the naive O(k) vertical sweep (column-strided reads are
  // cache-unfriendly regardless of algorithm, so the win is smaller).
  // =========================================================================
  const int r = prop->nms_radius;
  const int k = 2 * r + 1;

  // Pass 1 — horizontal (van Herk, O(n)):
  // maxpool is borrowed as suffix temp; it is fully overwritten in Pass 2.
  float* pre = prop->nms_prefix.data();
  float* suf = maxpool;
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < H; ++y) {
    const float* src = score_map + y * W;
    float*       dst = row_max   + y * W;
    float*       p   = pre + y * W;
    float*       s   = suf + y * W;

    // forward: prefix-max within each block of k
    for (int x = 0; x < W; ++x)
      p[x] = (x % k == 0) ? src[x] : std::max(p[x-1], src[x]);

    // backward: suffix-max within each block of k
    for (int x = W-1; x >= 0; --x)
      s[x] = ((x+1) % k == 0 || x == W-1) ? src[x] : std::max(src[x], s[x+1]);

    // combine into row_max
    for (int x = 0; x < W; ++x) {
      const int l  = x - r < 0     ? 0     : x - r;
      const int rr = x + r >= W    ? W - 1 : x + r;
      dst[x] = std::max(s[l], p[rr]);
    }
  }

  // Pass 2 — vertical: maxpool[y][x] = max(row_max[y-r .. y+r][x])
  #pragma omp parallel for schedule(static)
  for (int y = 0; y < H; ++y) {
    float*    dst = maxpool + y * W;
    const int y0  = y - r < 0     ? 0     : y - r;
    const int y1  = y + r >= H    ? H - 1 : y + r;
    for (int x = 0; x < W; ++x) {
      float m = 0.f;
      for (int yy = y0; yy <= y1; ++yy) {
        float v = row_max[yy * W + x];
        if (v > m) m = v;
      }
      dst[x] = m;
    }
  }

  // Suppress non-maxima
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < H * W; ++i)
    score_map[i] = score_map[i] >= maxpool[i] ? score_map[i] : 0.f;

  // =========================================================================
  // Step 3: Border removal + threshold → candidates
  // =========================================================================
  struct Candidate { float score; int x, y; };
  std::vector<Candidate> cands;
  cands.reserve(4096);

  const int pad = prop->remove_borders;
  for (int y = pad; y < H - pad; ++y) {
    const float* row = score_map + y * W;
    for (int x = pad; x < W - pad; ++x) {
      float s = row[x];
      if (s > prop->det_threshold)
        cands.push_back({ s, x, y });
    }
  }

  // =========================================================================
  // Step 4: Top-k selection
  // =========================================================================
  int N = static_cast<int>(cands.size());
  if (prop->max_keypoints > 0 && N > prop->max_keypoints) {
    std::partial_sort(cands.begin(), cands.begin() + prop->max_keypoints, cands.end(),
        [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    cands.resize(prop->max_keypoints);
    N = prop->max_keypoints;
  }
  log(AX_DEBUG) << "SuperPoint: " << N << " keypoints";

  // =========================================================================
  // Step 5: Bilinear descriptor sampling + per-descriptor L2-normalise
  //
  // desc_hwc aliases desc_ptr (NHWC): all 256 channels of each cell are
  // already contiguous, so the 4-neighbour bilinear fetch is cache-friendly
  // without any prior transpose.
  //
  // Descriptor-map coordinate for keypoint (x, y) in full-image pixel space:
  //   desc_x = (x + 0.5) / stride - 0.5   (SuperPoint grid_sample convention)
  // =========================================================================
  std::vector<float> kpts(N * 2);
  std::vector<float> scores_out(N);
  std::vector<float> descs(N * DESC_DIM);

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N; ++i) {
    kpts[i * 2 + 0] = static_cast<float>(cands[i].x);
    kpts[i * 2 + 1] = static_cast<float>(cands[i].y);
    scores_out[i]   = cands[i].score;

    // Map to descriptor-map coordinates
    const float px = (cands[i].x + 0.5f) / UPSCALE - 0.5f;
    const float py = (cands[i].y + 0.5f) / UPSCALE - 0.5f;

    const int   x0 = std::max(0,      std::min(Wf - 1, (int)std::floor(px)));
    const int   y0 = std::max(0,      std::min(Hf - 1, (int)std::floor(py)));
    const int   x1 = std::min(Wf - 1, x0 + 1);
    const int   y1 = std::min(Hf - 1, y0 + 1);
    const float wx = px - std::floor(px);
    const float wy = py - std::floor(py);

    const float w00 = (1.f - wx) * (1.f - wy);
    const float w10 = wx         * (1.f - wy);
    const float w01 = (1.f - wx) * wy;
    const float w11 = wx         * wy;

    // All four neighbour vectors are contiguous in HWC memory (256 floats each).
    const float* p00 = desc_hwc + (y0 * Wf + x0) * DESC_DIM;
    const float* p10 = desc_hwc + (y0 * Wf + x1) * DESC_DIM;
    const float* p01 = desc_hwc + (y1 * Wf + x0) * DESC_DIM;
    const float* p11 = desc_hwc + (y1 * Wf + x1) * DESC_DIM;

    float* out = descs.data() + i * DESC_DIM;
    for (int c = 0; c < DESC_DIM; ++c)
      out[c] = w00 * p00[c] + w10 * p10[c] + w01 * p01[c] + w11 * p11[c];

    l2_normalize(out, DESC_DIM);
  }

  // =========================================================================
  // Store as AxMetaRawTensor (→ Python TensorMeta)
  // =========================================================================
  auto* meta = ax_utils::insert_meta<AxMetaRawTensor>(
      map, prop->meta_name, std::string{}, subframe_index, subframe_number);
  if (!meta) {
    log(AX_ERROR) << "SuperPoint: failed to create AxMetaRawTensor";
    return;
  }

  // Guard: std::vector::data() may return nullptr when empty (N==0).
  // FORTIFY_SOURCE aborts on memcpy(dst, nullptr, 0), so provide a
  // valid fallback pointer when there are no keypoints.
  static constexpr float kEmpty = 0.f;
  const float* kpts_data  = N > 0 ? kpts.data()      : &kEmpty;
  const float* scr_data   = N > 0 ? scores_out.data(): &kEmpty;
  const float* desc_data  = N > 0 ? descs.data()     : &kEmpty;

  meta->add_tensor(kpts_data,  N * 2,        sizeof(float), { N, 2 });
  meta->add_tensor(scr_data,   N,             sizeof(float), { N });
  meta->add_tensor(desc_data,  N * DESC_DIM, sizeof(float), { N, DESC_DIM });
}
catch (const std::exception& e) {
  log(AX_ERROR) << "SuperPoint decoder caught exception: " << e.what();
}
catch (...) {
  log(AX_ERROR) << "SuperPoint decoder caught unknown exception";
}

}  // extern "C"
