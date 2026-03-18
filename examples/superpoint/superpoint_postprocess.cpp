// SuperPoint C++ postprocessor — see superpoint_postprocess.hpp for overview.

#include "superpoint_postprocess.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>

// ── Fast exp approximation (Schraudolph 1999 IEEE-754 bit trick) ─────────────
// ~4% max relative error — sufficient for softmax thresholding at 0.005.
// Replaces scalar glibc expf with a multiply+add+clip+reinterpret, which the
// ARM Cortex-A55 compiler auto-vectorises with NEON at -O3 -march=armv8-a+simd.
static inline float fast_exp(float x) noexcept
{
  union { float f; int32_t i; } u;
  u.f = x * 12102203.161f + 1065353216.f;
  if (u.i < 0)          u.i = 0;          // clamp underflow
  if (u.i > 2139095040) u.i = 2139095040; // clamp overflow  (max finite f32)
  return u.f;
}

// ── Constructor ──────────────────────────────────────────────────────────────

SuperPointDecoder::SuperPointDecoder(const Config& cfg)
  : cfg_(cfg)
{
  const int ksz = cfg_.nms_radius * 2 + 1;
  nms_kernel_  = cv::Mat::ones(ksz, ksz, CV_8U);

  s_f32_.resize(CELL_ROWS * CELL_COLS * SCORE_REAL_CH);
  score_map_.resize(MAP_H * MAP_W);
  smap16_.resize(MAP_H * MAP_W);

  smap16_mat_  = cv::Mat(MAP_H, MAP_W, CV_16U, smap16_.data());
  dilated_mat_ = cv::Mat(MAP_H, MAP_W, CV_16U);
}

// ── Step 1: dequantise int8 scores → float32[60,80,65] ──────────────────────

void SuperPointDecoder::dequant_scores(const int8_t* src) const
{
  // src layout: [1, CELL_ROWS, CELL_COLS, 128] int8 NHWC
  // We only need the first SCORE_REAL_CH=65 channels per cell.
  const int in_stride  = 128;           // total channels in raw tensor
  const int out_stride = SCORE_REAL_CH; // channels we keep
  float* dst = s_f32_.data();

  for (int rc = 0; rc < CELL_ROWS * CELL_COLS; ++rc) {
    const int8_t* in_cell  = src + rc * in_stride;
    float*        out_cell = dst + rc * out_stride;
    for (int k = 0; k < SCORE_REAL_CH; ++k) {
      out_cell[k] = (static_cast<float>(in_cell[k]) - SCORE_ZP) * SCORE_SCALE;
    }
  }
}

// ── Step 2: numerically-stable softmax over SCORE_REAL_CH channels ───────────
// Inner loops over 65 elements auto-vectorise well at -O3.

void SuperPointDecoder::softmax_cells() const
{
  const int n_cells = CELL_ROWS * CELL_COLS;
  float* data = s_f32_.data();

  for (int i = 0; i < n_cells; ++i) {
    float* cell = data + i * SCORE_REAL_CH;

    // Find max for numerical stability
    float vmax = cell[0];
    for (int k = 1; k < SCORE_REAL_CH; ++k)
      vmax = (cell[k] > vmax) ? cell[k] : vmax;

    // Subtract max, apply fast exp, accumulate sum
    float sum = 0.f;
    for (int k = 0; k < SCORE_REAL_CH; ++k) {
      cell[k] = fast_exp(cell[k] - vmax);
      sum += cell[k];
    }

    // Normalise
    const float inv_sum = 1.f / sum;
    for (int k = 0; k < SCORE_REAL_CH; ++k)
      cell[k] *= inv_sum;
  }
}

// ── Step 3: pixel-shuffle 8× — [60,80,64] → [480,640] ───────────────────────
// score_map_[r*8+dr, c*8+dc] = s_f32_[r, c, dr*8+dc]
// Written for cache efficiency: inner dc loop is 8 contiguous float reads
// and scattered writes with stride 640 — NEON can vectorise the 8-wide copy.

void SuperPointDecoder::pixel_shuffle() const
{
  const float* s   = s_f32_.data();
  float*       out = score_map_.data();

  for (int r = 0; r < CELL_ROWS; ++r) {
    for (int c = 0; c < CELL_COLS; ++c) {
      const float* cell = s + (r * CELL_COLS + c) * SCORE_REAL_CH;
      for (int dr = 0; dr < CELL_CH; ++dr) {
        // 8 contiguous source elements → row (r*8+dr) of the output
        const float* src_row = cell + dr * CELL_CH;
        float*       dst_row = out + (r * CELL_CH + dr) * MAP_W + c * CELL_CH;
        std::copy(src_row, src_row + CELL_CH, dst_row);
      }
    }
  }
}

// ── Step 4: uint16 NMS — morphological dilation on CV_16U ────────────────────
// cv::dilate on uint16 is ~4× faster than float32 on ARM (NEON SIMD path).

cv::Mat SuperPointDecoder::nms_mask() const
{
  // Convert float score_map to uint16 (values in [0,1] → [0,65535])
  const float* sm = score_map_.data();
  uint16_t*    u  = smap16_.data();
  for (int i = 0; i < MAP_H * MAP_W; ++i)
    u[i] = static_cast<uint16_t>(sm[i] * 65535.f);

  // Morphological dilation: local max in (2r+1)² neighbourhood
  cv::dilate(smap16_mat_, dilated_mat_, nms_kernel_);

  // Local maxima above threshold
  const uint16_t thresh_u16 = static_cast<uint16_t>(cfg_.detection_threshold * 65535.f);
  cv::Mat mask = (smap16_mat_ == dilated_mat_) & (smap16_mat_ > thresh_u16);

  // Zero out borders
  if (cfg_.borders > 0) {
    const int b = cfg_.borders;
    mask.rowRange(0, b).setTo(0);
    mask.rowRange(MAP_H - b, MAP_H).setTo(0);
    mask.colRange(0, b).setTo(0);
    mask.colRange(MAP_W - b, MAP_W).setTo(0);
  }

  return mask;
}

// ── Step 5: extract and rank keypoints ───────────────────────────────────────

std::vector<SPKeypoint>
SuperPointDecoder::extract_keypoints(const cv::Mat& mask) const
{
  std::vector<cv::Point> pts;
  cv::findNonZero(mask, pts);

  std::vector<SPKeypoint> kps;
  kps.reserve(pts.size());
  for (const auto& p : pts) {
    kps.push_back({ static_cast<float>(p.x),
                    static_cast<float>(p.y),
                    score_map_[p.y * MAP_W + p.x] });
  }

  // Top-k by score (partial sort — O(N log K) instead of O(N log N))
  const int max_kp = cfg_.max_keypoints;
  if (max_kp > 0 && static_cast<int>(kps.size()) > max_kp) {
    std::partial_sort(kps.begin(), kps.begin() + max_kp, kps.end(),
        [](const SPKeypoint& a, const SPKeypoint& b) { return a.score > b.score; });
    kps.resize(max_kp);
  } else {
    std::sort(kps.begin(), kps.end(),
        [](const SPKeypoint& a, const SPKeypoint& b) { return a.score > b.score; });
  }

  return kps;
}

// ── Step 6: descriptor bilinear sampling ─────────────────────────────────────
// Mirrors the Python sample_descriptors — bilinear interpolation over the
// [CELL_ROWS, CELL_COLS, 256] descriptor map (dequantised from int8).
// Layout of descs_raw: [1, CELL_ROWS, CELL_COLS, 256] int8 NHWC.

void SuperPointDecoder::sample_descriptors(
    std::vector<SPKeypoint>&              kps,
    std::vector<std::array<float, 256>>&  descs,
    const int8_t*                         descs_raw) const
{
  descs.resize(kps.size());
  constexpr int D = 256;

  const float norm_x = 1.f / (static_cast<float>(CELL_COLS) * CELL_CH);
  const float norm_y = 1.f / (static_cast<float>(CELL_ROWS) * CELL_CH);

  for (size_t n = 0; n < kps.size(); ++n) {
    // Normalise keypoint to [-1, 1] grid (matches torch grid_sample)
    const float fx = ((kps[n].x + 0.5f) * norm_x) * 2.f - 1.f;
    const float fy = ((kps[n].y + 0.5f) * norm_y) * 2.f - 1.f;

    // Map from [-1,1] to [0, CELL_COLS-1]
    const float gx = (fx + 1.f) * 0.5f * (CELL_COLS - 1);
    const float gy = (fy + 1.f) * 0.5f * (CELL_ROWS - 1);

    const int x0 = std::clamp(static_cast<int>(gx),     0, CELL_COLS - 1);
    const int x1 = std::clamp(x0 + 1,                   0, CELL_COLS - 1);
    const int y0 = std::clamp(static_cast<int>(gy),     0, CELL_ROWS - 1);
    const int y1 = std::clamp(y0 + 1,                   0, CELL_ROWS - 1);

    const float wx = gx - x0;
    const float wy = gy - y0;

    auto cell_ptr = [&](int r, int c) -> const int8_t* {
      return descs_raw + (r * CELL_COLS + c) * D;
    };

    // Bilinear blend + dequant
    auto& d = descs[n];
    float norm_sq = 0.f;
    for (int k = 0; k < D; ++k) {
      d[k] = ((static_cast<float>(cell_ptr(y0,x0)[k]) * DESC_SCALE) * (1-wx) * (1-wy)
            + (static_cast<float>(cell_ptr(y0,x1)[k]) * DESC_SCALE) * wx     * (1-wy)
            + (static_cast<float>(cell_ptr(y1,x0)[k]) * DESC_SCALE) * (1-wx) * wy
            + (static_cast<float>(cell_ptr(y1,x1)[k]) * DESC_SCALE) * wx     * wy);
      norm_sq += d[k] * d[k];
    }
    // L2 normalise
    const float inv_norm = (norm_sq > 1e-8f) ? 1.f / std::sqrt(norm_sq) : 1.f;
    for (float& v : d) v *= inv_norm;
  }
}

// ── Public entry point ────────────────────────────────────────────────────────

SPResult SuperPointDecoder::decode(const int8_t* scores_raw,
                                   const int8_t* descs_raw) const
{
  dequant_scores(scores_raw);
  softmax_cells();
  pixel_shuffle();

  const cv::Mat mask = nms_mask();
  auto kps = extract_keypoints(mask);

  SPResult result;
  result.keypoints = std::move(kps);

  if (cfg_.compute_descriptors && descs_raw != nullptr)
    sample_descriptors(result.keypoints, result.descriptors, descs_raw);

  return result;
}
