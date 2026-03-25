// SuperPoint post-processing GStreamer decoder for Axelera Metis M2
//
// The ONNX model already handles:
//   - Softmax + pixel-shuffle  → score_map  [1, 1, H, W]  (NMS-suppressed)
//   - Max-pool NMS             → suppressed score map
//   - L2-normalisation         → desc_norm  [1, 256, Hf, Wf]
//
// This decoder only needs to run on the host CPU:
//   1. Border removal + threshold → candidate keypoints
//   2. Top-k selection            → final N keypoints
//   3. Bilinear sampling of desc_norm at keypoint locations → [N, 256] descriptors
//   4. Per-descriptor L2-normalise (bilinear blend of unit vectors isn't unit-norm)
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
static constexpr int DESC_DIM = 256;

// ---- Properties parsed from YAML options string ----
struct Props {
  std::string meta_name        = "superpoint";
  float       det_threshold    = 0.005f;
  int         remove_borders   = 4;
  int         max_keypoints    = 1024;   // ≤0 → unlimited
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

// Bilinear sample from a single 2-D channel [Hf × Wf].
// Coordinates (px, py) are in descriptor-map pixel space [0, Wf) × [0, Hf).
static float
bilinear(const float* ch, int Hf, int Wf, float px, float py)
{
  int x0 = std::max(0, std::min(Wf - 1, (int)std::floor(px)));
  int y0 = std::max(0, std::min(Hf - 1, (int)std::floor(py)));
  int x1 = std::min(Wf - 1, x0 + 1);
  int y1 = std::min(Hf - 1, y0 + 1);
  float wx = px - std::floor(px), wy = py - std::floor(py);
  return (1 - wx) * (1 - wy) * ch[y0 * Wf + x0]
       +      wx  * (1 - wy) * ch[y0 * Wf + x1]
       + (1 - wx) *      wy  * ch[y1 * Wf + x0]
       +      wx  *      wy  * ch[y1 * Wf + x1];
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

  log(AX_INFO) << "SuperPoint decoder init: threshold=" << p->det_threshold
               << " borders=" << p->remove_borders
               << " max_kpts=" << p->max_keypoints;
  return p;
}

const std::unordered_set<std::string>&
allowed_properties()
{
  static const std::unordered_set<std::string> s{
    "meta_key", "detection_threshold", "remove_borders", "max_keypoints"
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
  // score_map:  [1, 1,   H,  W ]  — NMS-suppressed softmax scores
  // desc_norm:  [1, 256, Hf, Wf]  — L2-normalised descriptor map
  const float* score_ptr = nullptr;
  const float* desc_ptr  = nullptr;
  int H = 0, W = 0, Hf = 0, Wf = 0;

  for (const auto& t : in_tensors) {
    if (t.sizes.size() != 4 || t.bytes != 4) continue;  // expect float32 NCHW
    if (!t.data) continue;                               // unmapped DMA buffer
    int C = t.sizes[1];
    if (C == 1) {
      score_ptr = static_cast<const float*>(t.data);
      H = t.sizes[2];
      W = t.sizes[3];
    } else if (C == DESC_DIM) {
      desc_ptr = static_cast<const float*>(t.data);
      Hf = t.sizes[2];
      Wf = t.sizes[3];
    }
  }

  if (!score_ptr || !desc_ptr || H == 0 || Hf == 0) {
    log(AX_ERROR) << "SuperPoint: could not identify output tensors";
    return;
  }

  // Stride (H / Hf = 8 for SuperPoint VGG backbone)
  const float stride = static_cast<float>(H) / static_cast<float>(Hf);

  // ---- 1. Border removal + threshold → candidates ----
  struct Candidate { float score; int x, y; };
  std::vector<Candidate> cands;
  cands.reserve(4096);

  const int pad = prop->remove_borders;
  for (int h = 0; h < H; ++h) {
    if (h < pad || h >= H - pad) continue;
    for (int w = 0; w < W; ++w) {
      if (w < pad || w >= W - pad) continue;
      float s = score_ptr[h * W + w];
      if (s > prop->det_threshold)
        cands.push_back({ s, w, h });
    }
  }

  // ---- 2. Top-k selection ----
  int N = static_cast<int>(cands.size());
  if (prop->max_keypoints > 0 && N > prop->max_keypoints) {
    std::partial_sort(cands.begin(), cands.begin() + prop->max_keypoints, cands.end(),
        [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    cands.resize(prop->max_keypoints);
    N = prop->max_keypoints;
  }
  log(AX_DEBUG) << "SuperPoint: " << N << " keypoints";

  // ---- 3. Sample descriptors ----
  // Descriptor-map coordinate for keypoint (x, y):
  //   desc_x = (x + 0.5) / stride - 0.5   (follows SuperPoint grid_sample convention)
  std::vector<float> kpts(N * 2);
  std::vector<float> scores_out(N);
  std::vector<float> descs(N * DESC_DIM, 0.f);

  for (int i = 0; i < N; ++i) {
    kpts[i * 2 + 0] = static_cast<float>(cands[i].x);
    kpts[i * 2 + 1] = static_cast<float>(cands[i].y);
    scores_out[i]   = cands[i].score;

    float dx = (cands[i].x + 0.5f) / stride - 0.5f;
    float dy = (cands[i].y + 0.5f) / stride - 0.5f;

    for (int c = 0; c < DESC_DIM; ++c)
      descs[i * DESC_DIM + c] = bilinear(desc_ptr + c * Hf * Wf, Hf, Wf, dx, dy);

    // 4. Per-descriptor L2-normalise (bilinear blend of unit vectors isn't unit-norm)
    l2_normalize(&descs[i * DESC_DIM], DESC_DIM);
  }

  // ---- 5. Store as AxMetaRawTensor (→ Python TensorMeta) ----
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

  // tensors[0]: keypoints  [N, 2]
  meta->add_tensor(kpts_data,  N * 2,        sizeof(float), { N, 2 });
  // tensors[1]: scores     [N]
  meta->add_tensor(scr_data,   N,             sizeof(float), { N });
  // tensors[2]: descriptors [N, 256]
  meta->add_tensor(desc_data,  N * DESC_DIM, sizeof(float), { N, DESC_DIM });
}
catch (const std::exception& e) {
  log(AX_ERROR) << "SuperPoint decoder caught exception: " << e.what();
}
catch (...) {
  log(AX_ERROR) << "SuperPoint decoder caught unknown exception";
}

}  // extern "C"
