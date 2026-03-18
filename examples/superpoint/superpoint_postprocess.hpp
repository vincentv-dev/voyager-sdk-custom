// SuperPoint C++ postprocessor for Axelera Metis M2 (iMX8MP)
//
// Receives raw int8 NHWC tensors from AxInferenceNet (handle_all: False):
//   tensor 0 — descriptors  [1, 60, 80, 256] int8
//   tensor 1 — scores       [1, 60, 80, 128] int8  (first 65 channels real)
//
// Pipeline:
//   1. Dequant scores  int8 → float32
//   2. Softmax (fast Schraudolph exp)  [60,80,65] per cell
//   3. Pixel-shuffle 8×                [60,80,64] → [480,640]
//   4. uint16 NMS dilation             cv::dilate on CV_16U (~4× vs float32)
//   5. Top-k keypoint extraction
//   6. Descriptor bilinear sampling    [N, 256] float32  (for LightGlue)

#pragma once

#include <cstdint>
#include <vector>
#include "opencv2/opencv.hpp"

// ── Output types ─────────────────────────────────────────────────────────────

struct SPKeypoint {
  float x, y;    // pixel coordinates in model space (640×480)
  float score;   // softmax probability after NMS
};

struct SPResult {
  std::vector<SPKeypoint>             keypoints;    // [N]
  std::vector<std::array<float, 256>> descriptors;  // [N, 256], empty if not requested
};

// ── Decoder ──────────────────────────────────────────────────────────────────

class SuperPointDecoder
{
  public:
  // Dequantisation constants from the compiled model manifest
  static constexpr float SCORE_SCALE    = 0.11909855902194977f;
  static constexpr int   SCORE_ZP       = 16;
  static constexpr int   SCORE_REAL_CH  = 65;   // 64 spatial + 1 dustbin
  static constexpr float DESC_SCALE     = 0.005005544982850552f;
  static constexpr int   DESC_ZP        = 0;

  // Model geometry
  static constexpr int CELL_ROWS  = 60;
  static constexpr int CELL_COLS  = 80;
  static constexpr int CELL_CH    = 8;   // pixel-shuffle stride
  static constexpr int MAP_H      = CELL_ROWS * CELL_CH;  // 480
  static constexpr int MAP_W      = CELL_COLS * CELL_CH;  // 640

  struct Config {
    int   nms_radius          = 4;
    float detection_threshold = 0.005f;
    int   borders             = 4;
    int   max_keypoints       = 1000;
    bool  compute_descriptors = false;
  };

  explicit SuperPointDecoder(const Config& cfg = {});

  // Main entry point.  Call once per frame.
  // scores_raw : int8 pointer to tensor 1  shape [1,60,80,128]
  // descs_raw  : int8 pointer to tensor 0  shape [1,60,80,256]  (may be nullptr)
  SPResult decode(const int8_t* scores_raw, const int8_t* descs_raw) const;

  private:
  Config  cfg_;
  cv::Mat nms_kernel_;   // (2*nms_radius+1)² ones, uint8

  // Pre-allocated working buffers (reused across frames to avoid malloc)
  mutable std::vector<float>    s_f32_;     // [60*80*65]  dequantised scores
  mutable std::vector<float>    score_map_; // [480*640]   after pixel-shuffle
  mutable std::vector<uint16_t> smap16_;    // [480*640]   uint16 for NMS
  mutable cv::Mat               smap16_mat_;
  mutable cv::Mat               dilated_mat_;

  // Internal steps
  void   dequant_scores(const int8_t* src) const;
  void   softmax_cells() const;
  void   pixel_shuffle() const;
  cv::Mat nms_mask() const;
  std::vector<SPKeypoint> extract_keypoints(const cv::Mat& mask) const;
  void   sample_descriptors(std::vector<SPKeypoint>& kps,
                             std::vector<std::array<float,256>>& descs,
                             const int8_t* descs_raw) const;
};
