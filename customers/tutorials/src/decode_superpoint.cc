// SuperPoint INT8 GStreamer decoder for Axelera Metis M2
//
// Receives raw INT8 NHWC tensors directly from the AIPU (handle_all: false):
//   desc_tensor   [1, Hf, Wf, 256]  int8  — raw descriptor features
//   logit_tensor  [1, Hf, Wf, 128]  int8  — detector logits (65 real + 63 padding)
//
// Algorithm — all ordering decisions use INT8 arithmetic until after TopK:
//   1. Channel-max over channels 0..63 per backbone cell
//      → per-cell peak score (int8) + sub-pixel channel index (uint8)
//   2. 3×3 spatial NMS at backbone resolution (INT8 comparisons)
//      + border masking (ceil(remove_borders / 8) cells excluded)
//   3. Sort surviving cells by INT8 score descending, take top max_keypoints
//   4. For each selected cell:
//        a. Dequantize 65 logit channels, compute softmax probability
//        b. Discard if prob <= detection_threshold
//        c. Bilinear-sample desc map at sub-pixel location, dequantize, L2-normalise
//   5. Store [N,2] keypoints, [N] scores, [N,256] descriptors as AxMetaRawTensor
//
// Parameters injected via YAML options string:
//   meta_key            — AxMeta dictionary key (default: "superpoint")
//   detection_threshold — Score threshold in probability space (default: 0.005)
//   remove_borders      — Border pixels to suppress (default: 4)
//   max_keypoints       — Maximum keypoints returned (default: 1024)
//   scales              — Comma-separated dequant scales: desc,logit
//   zero_points         — Comma-separated dequant zero points: desc,logit
//                         Formula: float = scale * (int8 - zero_point)

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaRawTensor.hpp"
#include "AxOpUtils.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

static constexpr int DESC_DIM      = 256;
static constexpr int LOGIT_CH_REAL = 65;   // 64 non-dustbin + 1 dustbin
static constexpr int LOGIT_CH_PAD  = 128;  // NHWC stride (63 zero-padded channels)
static constexpr int STRIDE        = 8;    // backbone stride (pixel-shuffle cell size)

// ---- Properties parsed from YAML options string ----
struct Props {
  std::string meta_name      = "superpoint";
  float       det_threshold  = 0.005f;
  int         remove_borders = 4;
  int         max_keypoints  = 512;
  float       scale_desc     = 1.0f;  // desc dequant scale
  float       zp_desc        = 0.0f;  // desc dequant zero point
  float       scale_logit    = 1.0f;  // logit dequant scale
  float       zp_logit       = 0.0f;  // logit dequant zero point
};

static std::vector<float>
parse_floats(const std::string& s)
{
  std::vector<float> v;
  std::istringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    try { v.push_back(std::stof(tok)); } catch (...) {}
  }
  return v;
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

  if (auto it = input.find("meta_key");            it != input.end()) p->meta_name      = it->second;
  if (auto it = input.find("detection_threshold"); it != input.end()) p->det_threshold  = std::stof(it->second);
  if (auto it = input.find("remove_borders");      it != input.end()) p->remove_borders = std::stoi(it->second);
  if (auto it = input.find("max_keypoints");       it != input.end()) p->max_keypoints  = std::stoi(it->second);

  if (auto it = input.find("scales"); it != input.end()) {
    auto v = parse_floats(it->second);
    if (v.size() >= 1) p->scale_desc  = v[0];
    if (v.size() >= 2) p->scale_logit = v[1];
  }
  if (auto it = input.find("zero_points"); it != input.end()) {
    auto v = parse_floats(it->second);
    if (v.size() >= 1) p->zp_desc  = v[0];
    if (v.size() >= 2) p->zp_logit = v[1];
  }

  log(AX_INFO) << "SuperPoint INT8 decoder init:"
               << " threshold=" << p->det_threshold
               << " max_kpts="  << p->max_keypoints
               << " borders="   << p->remove_borders
               << " scale_desc="   << p->scale_desc  << " zp_desc="  << p->zp_desc
               << " scale_logit="  << p->scale_logit << " zp_logit=" << p->zp_logit;
  return p;
}

const std::unordered_set<std::string>&
allowed_properties()
{
  static const std::unordered_set<std::string> s{
    "meta_key", "detection_threshold", "remove_borders",
    "max_keypoints", "scales", "zero_points"
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
  // ---- Identify tensors by last dimension ----
  // desc:  NHWC [1, Hf, Wf, 256]  int8
  // logit: NHWC [1, Hf, Wf, 128]  int8  (65 real + 63 padding)
  const int8_t* desc_ptr  = nullptr;
  const int8_t* logit_ptr = nullptr;
  int Hf = 0, Wf = 0;

  for (const auto& t : in_tensors) {
    if (!t.data || t.bytes != 1) continue;
    const auto& sz = t.sizes;
    if (sz.size() == 4 && sz[3] == DESC_DIM && !desc_ptr) {
      desc_ptr = static_cast<const int8_t*>(t.data);
      Hf = sz[1]; Wf = sz[2];
    } else if (sz.size() == 4 && sz[3] == LOGIT_CH_PAD && !logit_ptr) {
      logit_ptr = static_cast<const int8_t*>(t.data);
      if (Hf == 0) { Hf = sz[1]; Wf = sz[2]; }
    }
  }

  if (!desc_ptr || !logit_ptr || Hf == 0 || Wf == 0) {
    log(AX_ERROR) << "SuperPoint: could not identify INT8 NHWC output tensors";
    return;
  }

  // ---- Step 1: Channel-max over 64 non-dustbin channels per backbone cell ----
  // Pixel-shuffle convention: channel c at backbone cell (h,w) maps to
  //   full-resolution pixel (w*8 + c%8, h*8 + c//8)
  const int Ncells = Hf * Wf;
  std::vector<int8_t>  cell_best_val(Ncells, INT8_MIN);
  std::vector<uint8_t> cell_best_ch (Ncells, 0);

  for (int h = 0; h < Hf; ++h) {
    for (int w = 0; w < Wf; ++w) {
      const int8_t* row = logit_ptr + (h * Wf + w) * LOGIT_CH_PAD;
      int8_t  best_val = INT8_MIN;
      uint8_t best_ch  = 0;
      for (int c = 0; c < 64; ++c) {   // 0..63: non-dustbin channels
        if (row[c] > best_val) {
          best_val = row[c];
          best_ch  = static_cast<uint8_t>(c);
        }
      }
      const int idx = h * Wf + w;
      cell_best_val[idx] = best_val;
      cell_best_ch [idx] = best_ch;
    }
  }

  // ---- Step 2: 3×3 spatial NMS + border masking (all INT8 comparisons) ----
  const int border_cells = (prop->remove_borders + STRIDE - 1) / STRIDE;
  std::vector<bool> nms_mask(Ncells, false);

  for (int h = border_cells; h < Hf - border_cells; ++h) {
    for (int w = border_cells; w < Wf - border_cells; ++w) {
      const int8_t v = cell_best_val[h * Wf + w];
      bool is_local_max = true;
      for (int dh = -1; dh <= 1 && is_local_max; ++dh) {
        for (int dw = -1; dw <= 1 && is_local_max; ++dw) {
          if (dh == 0 && dw == 0) continue;
          const int nh = h + dh, nw = w + dw;
          if (nh < 0 || nh >= Hf || nw < 0 || nw >= Wf) continue;
          if (cell_best_val[nh * Wf + nw] > v) is_local_max = false;
        }
      }
      nms_mask[h * Wf + w] = is_local_max;
    }
  }

  // ---- Step 3: Collect surviving cells, sort by INT8 score, take TopK ----
  struct Candidate {
    int8_t  score_int8;
    uint8_t ch;
    int16_t h, w;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(512);

  for (int h = 0; h < Hf; ++h)
    for (int w = 0; w < Wf; ++w)
      if (nms_mask[h * Wf + w])
        candidates.push_back({
          cell_best_val[h * Wf + w],
          cell_best_ch [h * Wf + w],
          static_cast<int16_t>(h),
          static_cast<int16_t>(w)
        });

  std::sort(candidates.begin(), candidates.end(),
      [](const Candidate& a, const Candidate& b) { return a.score_int8 > b.score_int8; });

  if (static_cast<int>(candidates.size()) > prop->max_keypoints)
    candidates.resize(prop->max_keypoints);

  // ---- Steps 4–7: Softmax score, threshold, bilinear descriptor sampling ----
  const int K = static_cast<int>(candidates.size());
  std::vector<float> out_kpts;
  std::vector<float> out_scores;
  std::vector<float> out_descs;
  out_kpts.reserve(K * 2);
  out_scores.reserve(K);
  out_descs.reserve(K * DESC_DIM);

  std::array<float, LOGIT_CH_REAL> logits_f;
  std::array<float, DESC_DIM>      desc_f;

  for (const auto& cand : candidates) {
    const int h = cand.h, w = cand.w;
    const int8_t* logit_row = logit_ptr + (h * Wf + w) * LOGIT_CH_PAD;

    // -- Softmax score for the winning sub-pixel channel --
    // Dequantize all 65 channels (0..63 non-dustbin + channel 64 dustbin)
    for (int ch = 0; ch < LOGIT_CH_REAL; ++ch)
      logits_f[ch] = prop->scale_logit * (static_cast<float>(logit_row[ch]) - prop->zp_logit);

    // Numerically-stable softmax
    float max_l = logits_f[0];
    for (int ch = 1; ch < LOGIT_CH_REAL; ++ch)
      if (logits_f[ch] > max_l) max_l = logits_f[ch];

    float sum_exp = 0.0f;
    for (int ch = 0; ch < LOGIT_CH_REAL; ++ch) {
      logits_f[ch] = std::exp(logits_f[ch] - max_l);
      sum_exp += logits_f[ch];
    }
    const float prob = logits_f[cand.ch] / sum_exp;
    if (prob <= prop->det_threshold) continue;

    // -- Bilinear descriptor sampling at sub-pixel location --
    // Keypoint pixel coords (full resolution):
    //   px = w*8 + ch%8,  py = h*8 + ch//8
    // Sampling position in desc map [Hf, Wf]:
    //   fx = px/8 = w + (ch%8)/8  →  floor=w, frac_x = (ch%8)/8
    //   fy = py/8 = h + (ch//8)/8 →  floor=h, frac_y = (ch//8)/8
    const float px = static_cast<float>(w * STRIDE + (cand.ch % STRIDE));
    const float py = static_cast<float>(h * STRIDE + (cand.ch / STRIDE));

    const float dx = static_cast<float>(cand.ch % STRIDE) / static_cast<float>(STRIDE);
    const float dy = static_cast<float>(cand.ch / STRIDE) / static_cast<float>(STRIDE);
    const int w1 = std::min(w + 1, Wf - 1);
    const int h1 = std::min(h + 1, Hf - 1);

    const float wa = (1.f - dy) * (1.f - dx);
    const float wb = (1.f - dy) * dx;
    const float wc = dy         * (1.f - dx);
    const float wd = dy         * dx;

    const int8_t* d00 = desc_ptr + (h  * Wf + w ) * DESC_DIM;
    const int8_t* d01 = desc_ptr + (h  * Wf + w1) * DESC_DIM;
    const int8_t* d10 = desc_ptr + (h1 * Wf + w ) * DESC_DIM;
    const int8_t* d11 = desc_ptr + (h1 * Wf + w1) * DESC_DIM;

    // Dequantize and bilinear-blend each descriptor channel, then L2-normalise
    // float = scale * (blended_int8 - zp)   [wa+wb+wc+wd == 1 ensures zp cancels]
    float norm_sq = 0.0f;
    for (int d = 0; d < DESC_DIM; ++d) {
      const float blended =
          wa * static_cast<float>(d00[d]) +
          wb * static_cast<float>(d01[d]) +
          wc * static_cast<float>(d10[d]) +
          wd * static_cast<float>(d11[d]);
      const float val = prop->scale_desc * (blended - prop->zp_desc);
      desc_f[d] = val;
      norm_sq  += val * val;
    }
    const float inv_norm = (norm_sq > 1e-12f) ? (1.0f / std::sqrt(norm_sq)) : 0.0f;
    for (int d = 0; d < DESC_DIM; ++d)
      desc_f[d] *= inv_norm;

    out_kpts.push_back(px);
    out_kpts.push_back(py);
    out_scores.push_back(prob);
    out_descs.insert(out_descs.end(), desc_f.begin(), desc_f.end());
  }

  const int N = static_cast<int>(out_scores.size());
  log(AX_DEBUG) << "SuperPoint: " << N << " keypoints (K=" << K << " candidates=" << candidates.size() << ")";

  // ---- Store as AxMetaRawTensor (→ Python TensorMeta / SuperPointMeta) ----
  auto* meta = ax_utils::insert_meta<AxMetaRawTensor>(
      map, prop->meta_name, std::string{}, subframe_index, subframe_number);
  if (!meta) {
    log(AX_ERROR) << "SuperPoint: failed to create AxMetaRawTensor";
    return;
  }

  static constexpr float kEmpty = 0.f;
  const float* kpts_data = N > 0 ? out_kpts.data()   : &kEmpty;
  const float* scr_data  = N > 0 ? out_scores.data() : &kEmpty;
  const float* desc_data = N > 0 ? out_descs.data()  : &kEmpty;

  // tensors[0]: keypoints    [N, 2]
  // tensors[1]: scores       [N]
  // tensors[2]: descriptors  [N, 256]
  meta->add_tensor(kpts_data, N * 2,        sizeof(float), { N, 2 });
  meta->add_tensor(scr_data,  N,            sizeof(float), { N });
  meta->add_tensor(desc_data, N * DESC_DIM, sizeof(float), { N, DESC_DIM });
}
catch (const std::exception& e) {
  log(AX_ERROR) << "SuperPoint decoder caught exception: " << e.what();
}
catch (...) {
  log(AX_ERROR) << "SuperPoint decoder caught unknown exception";
}

}  // extern "C"
