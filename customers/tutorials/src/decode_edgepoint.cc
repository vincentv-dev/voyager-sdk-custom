// EdgePoint2 E64 post-processing GStreamer decoder for Axelera Metis M2
//
// The ONNX model now exports padded backbone feature maps only:
//   x1_feat: [1, H/2,  W/2,  64]
//   x2_feat: [1, H/8,  W/8,  64]
//   x3_feat: [1, H/32, W/32, 64]
//
// This decoder reconstructs the original EdgePoint2 descriptor and detector
// heads on the host CPU using a compact exported weight blob:
//   1. Crop padded backbone channels back to [16, 48, 64]
//   2. Rebuild detector head on CPU
//   3. Rebuild descriptor head on CPU
//   4. Max-pool NMS
//   5. Border removal + raw-logit threshold
//   6. Top-k selection
//   7. Bilinear descriptor sampling + per-descriptor L2-normalise
//
// Input tensor layout: NHWC float32  (handle_transpose: false in the YAML)
//
// Output stored as AxMetaRawTensor → deserialized by Python TensorMeta:
//   tensors[0]  keypoints    [N, 2]   float32  (x, y) pixel coords
//   tensors[1]  scores       [N]      float32  raw logit detector confidence
//   tensors[2]  descriptors  [N, 64]  float32  L2-normalised

#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMetaRawTensor.hpp"
#include "AxOpUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr int PADDED_C     = 64;
static constexpr int X1_C         = 16;
static constexpr int X2_C         = 48;
static constexpr int X3_C         = 64;
static constexpr int DETECT_C     = 16;
static constexpr int DESC_CAT_C   = 128;
static constexpr int DESC_DIM     = 64;
static constexpr int SCORE_CH     = 4;
static constexpr int SCORE_UP     = 2;
static constexpr int DESC_UP      = 4;
static constexpr int DESC_GROUPS  = 8;
static constexpr int DESC_GCH     = DESC_CAT_C / DESC_GROUPS;

struct HeadWeights {
  std::vector<float> conv1_w, conv1_b;
  std::vector<float> conv2_w, conv2_b;
  std::vector<float> conv3_w, conv3_b;

  std::vector<float> score0_w, score0_b;
  std::vector<float> score1_w, score1_b;
  std::vector<float> score2_w, score2_b;

  std::vector<float> desc0_w, desc0_b;
  std::vector<float> desc1_w, desc1_b;
  std::vector<float> desc2_w, desc2_b;
};

struct Props {
  std::string meta_name         = "edgepoint";
  std::string head_weights_path = "ax_models/custom/edgepoint/weights/edgepoint_heads.bin";
  float       det_threshold     = 0.0f;
  int         remove_borders    = 4;
  int         max_keypoints     = 1024;
  int         nms_radius        = 2;

  mutable bool weights_loaded = false;
  mutable HeadWeights weights;

  mutable int buf_H = 0, buf_W = 0;
  mutable std::vector<float> score_map;
  mutable std::vector<float> row_max;
  mutable std::vector<float> maxpool;
};

static inline int
hwc_idx(int y, int x, int c, int W, int C)
{
  return (y * W + x) * C + c;
}

static void
l2_normalize(float* v, int n)
{
  float sq = 0.f;
  for (int i = 0; i < n; ++i) sq += v[i] * v[i];
  float inv = 1.f / std::sqrt(sq + 1e-10f);
  for (int i = 0; i < n; ++i) v[i] *= inv;
}

static void
relu_inplace(std::vector<float>& v)
{
  for (float& x : v) {
    if (x < 0.f) x = 0.f;
  }
}

static uint32_t
read_u32(std::ifstream& in)
{
  uint32_t value = 0;
  in.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!in) throw std::runtime_error("unexpected EOF while reading u32");
  return value;
}

static std::vector<float>
read_tensor(std::ifstream& in, const std::vector<int>& expected_shape)
{
  const uint32_t ndim = read_u32(in);
  if (ndim != expected_shape.size()) {
    throw std::runtime_error("unexpected tensor rank in EdgePoint weight blob");
  }

  std::vector<int> shape(ndim);
  size_t total = 1;
  for (uint32_t i = 0; i < ndim; ++i) {
    shape[i] = static_cast<int>(read_u32(in));
    total *= static_cast<size_t>(shape[i]);
  }

  if (shape != expected_shape) {
    throw std::runtime_error("unexpected tensor shape in EdgePoint weight blob");
  }

  std::vector<float> tensor(total);
  in.read(reinterpret_cast<char*>(tensor.data()), static_cast<std::streamsize>(total * sizeof(float)));
  if (!in) throw std::runtime_error("unexpected EOF while reading tensor payload");
  return tensor;
}

static void
load_weights_if_needed(const Props* prop, Ax::Logger& log)
{
  if (prop->weights_loaded) return;

  std::ifstream in(prop->head_weights_path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open EdgePoint head weights: " + prop->head_weights_path);
  }

  char magic[4] = {};
  in.read(magic, sizeof(magic));
  if (!in || std::memcmp(magic, "EPH1", 4) != 0) {
    throw std::runtime_error("invalid EdgePoint head weight blob header");
  }

  const uint32_t tensor_count = read_u32(in);
  if (tensor_count != 18) {
    throw std::runtime_error("unexpected EdgePoint head tensor count");
  }

  auto& w = prop->weights;
  w.conv1_w  = read_tensor(in, {DETECT_C, X1_C, 1, 1});
  w.conv1_b  = read_tensor(in, {DETECT_C});
  w.conv2_w  = read_tensor(in, {DETECT_C, X2_C, 1, 1});
  w.conv2_b  = read_tensor(in, {DETECT_C});
  w.conv3_w  = read_tensor(in, {DETECT_C, X3_C, 1, 1});
  w.conv3_b  = read_tensor(in, {DETECT_C});

  w.score0_w = read_tensor(in, {DETECT_C, DETECT_C, 3, 3});
  w.score0_b = read_tensor(in, {DETECT_C});
  w.score1_w = read_tensor(in, {DETECT_C, DETECT_C, 3, 3});
  w.score1_b = read_tensor(in, {DETECT_C});
  w.score2_w = read_tensor(in, {SCORE_CH, DETECT_C, 3, 3});
  w.score2_b = read_tensor(in, {SCORE_CH});

  w.desc0_w  = read_tensor(in, {DESC_CAT_C, DESC_CAT_C, 1, 1});
  w.desc0_b  = read_tensor(in, {DESC_CAT_C});
  w.desc1_w  = read_tensor(in, {DESC_CAT_C, DESC_GCH, 3, 3});
  w.desc1_b  = read_tensor(in, {DESC_CAT_C});
  w.desc2_w  = read_tensor(in, {DESC_DIM, DESC_CAT_C, 1, 1});
  w.desc2_b  = read_tensor(in, {DESC_DIM});

  prop->weights_loaded = true;
  log(AX_INFO) << "EdgePoint decoder loaded head weights from " << prop->head_weights_path;
}

static std::vector<float>
crop_channels(const float* src, int H, int W, int srcC, int dstC)
{
  std::vector<float> out(static_cast<size_t>(H) * W * dstC);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const float* in_ptr = src + hwc_idx(y, x, 0, W, srcC);
      float* out_ptr = out.data() + hwc_idx(y, x, 0, W, dstC);
      std::memcpy(out_ptr, in_ptr, static_cast<size_t>(dstC) * sizeof(float));
    }
  }
  return out;
}

static std::vector<float>
avg_pool2x2_stride2(const std::vector<float>& src, int H, int W, int C)
{
  const int outH = H / 2;
  const int outW = W / 2;
  std::vector<float> out(static_cast<size_t>(outH) * outW * C, 0.f);

  for (int y = 0; y < outH; ++y) {
    for (int x = 0; x < outW; ++x) {
      float* out_ptr = out.data() + hwc_idx(y, x, 0, outW, C);
      for (int ky = 0; ky < 2; ++ky) {
        for (int kx = 0; kx < 2; ++kx) {
          const float* in_ptr = src.data() + hwc_idx(y * 2 + ky, x * 2 + kx, 0, W, C);
          for (int c = 0; c < C; ++c) out_ptr[c] += in_ptr[c] * 0.25f;
        }
      }
    }
  }
  return out;
}

static std::vector<float>
bilinear_resize(const std::vector<float>& src, int inH, int inW, int C, int outH, int outW)
{
  std::vector<float> out(static_cast<size_t>(outH) * outW * C);
  const float scale_y = static_cast<float>(inH) / static_cast<float>(outH);
  const float scale_x = static_cast<float>(inW) / static_cast<float>(outW);

#pragma omp parallel for schedule(static)
  for (int oy = 0; oy < outH; ++oy) {
    const float py = (static_cast<float>(oy) + 0.5f) * scale_y - 0.5f;
    const float fy = std::floor(py);
    const int y0 = std::max(0, std::min(inH - 1, static_cast<int>(fy)));
    const int y1 = std::min(inH - 1, y0 + 1);
    const float wy = py - fy;

    for (int ox = 0; ox < outW; ++ox) {
      const float px = (static_cast<float>(ox) + 0.5f) * scale_x - 0.5f;
      const float fx = std::floor(px);
      const int x0 = std::max(0, std::min(inW - 1, static_cast<int>(fx)));
      const int x1 = std::min(inW - 1, x0 + 1);
      const float wx = px - fx;

      const float w00 = (1.f - wx) * (1.f - wy);
      const float w10 = wx * (1.f - wy);
      const float w01 = (1.f - wx) * wy;
      const float w11 = wx * wy;

      const float* p00 = src.data() + hwc_idx(y0, x0, 0, inW, C);
      const float* p10 = src.data() + hwc_idx(y0, x1, 0, inW, C);
      const float* p01 = src.data() + hwc_idx(y1, x0, 0, inW, C);
      const float* p11 = src.data() + hwc_idx(y1, x1, 0, inW, C);
      float* dst = out.data() + hwc_idx(oy, ox, 0, outW, C);

      for (int c = 0; c < C; ++c) {
        dst[c] = w00 * p00[c] + w10 * p10[c] + w01 * p01[c] + w11 * p11[c];
      }
    }
  }
  return out;
}

static std::vector<float>
conv1x1_nhwc(
    const std::vector<float>& src,
    int H,
    int W,
    int inC,
    int outC,
    const std::vector<float>& weights,
    const std::vector<float>& bias)
{
  std::vector<float> out(static_cast<size_t>(H) * W * outC);
#pragma omp parallel for schedule(static)
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const float* in_ptr = src.data() + hwc_idx(y, x, 0, W, inC);
      float* out_ptr = out.data() + hwc_idx(y, x, 0, W, outC);
      for (int oc = 0; oc < outC; ++oc) {
        float acc = bias[oc];
        const float* w_ptr = weights.data() + static_cast<size_t>(oc) * inC;
        for (int ic = 0; ic < inC; ++ic) acc += in_ptr[ic] * w_ptr[ic];
        out_ptr[oc] = acc;
      }
    }
  }
  return out;
}

static std::vector<float>
conv3x3_nhwc(
    const std::vector<float>& src,
    int H,
    int W,
    int inC,
    int outC,
    const std::vector<float>& weights,
    const std::vector<float>& bias)
{
  std::vector<float> out(static_cast<size_t>(H) * W * outC);
#pragma omp parallel for schedule(static)
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      float* out_ptr = out.data() + hwc_idx(y, x, 0, W, outC);
      for (int oc = 0; oc < outC; ++oc) {
        float acc = bias[oc];
        const float* w_oc = weights.data() + static_cast<size_t>(oc) * inC * 9;
        for (int ky = 0; ky < 3; ++ky) {
          const int iy = y + ky - 1;
          if (iy < 0 || iy >= H) continue;
          for (int kx = 0; kx < 3; ++kx) {
            const int ix = x + kx - 1;
            if (ix < 0 || ix >= W) continue;
            const float* in_ptr = src.data() + hwc_idx(iy, ix, 0, W, inC);
            const float* w_ptr = w_oc + (ky * 3 + kx) * inC;
            for (int ic = 0; ic < inC; ++ic) acc += in_ptr[ic] * w_ptr[ic];
          }
        }
        out_ptr[oc] = acc;
      }
    }
  }
  return out;
}

static std::vector<float>
grouped_conv3x3_nhwc(
    const std::vector<float>& src,
    int H,
    int W,
    const std::vector<float>& weights,
    const std::vector<float>& bias)
{
  std::vector<float> out(static_cast<size_t>(H) * W * DESC_CAT_C);
#pragma omp parallel for schedule(static)
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      float* out_ptr = out.data() + hwc_idx(y, x, 0, W, DESC_CAT_C);
      for (int oc = 0; oc < DESC_CAT_C; ++oc) {
        const int group = oc / DESC_GCH;
        const int in_base = group * DESC_GCH;
        float acc = bias[oc];
        const float* w_oc = weights.data() + static_cast<size_t>(oc) * DESC_GCH * 9;
        for (int ky = 0; ky < 3; ++ky) {
          const int iy = y + ky - 1;
          if (iy < 0 || iy >= H) continue;
          for (int kx = 0; kx < 3; ++kx) {
            const int ix = x + kx - 1;
            if (ix < 0 || ix >= W) continue;
            const float* in_ptr = src.data() + hwc_idx(iy, ix, in_base, W, DESC_CAT_C);
            const float* w_ptr = w_oc + (ky * 3 + kx) * DESC_GCH;
            for (int ic = 0; ic < DESC_GCH; ++ic) acc += in_ptr[ic] * w_ptr[ic];
          }
        }
        out_ptr[oc] = acc;
      }
    }
  }
  return out;
}

static void
add_inplace(std::vector<float>& dst, const std::vector<float>& src)
{
  for (size_t i = 0; i < dst.size(); ++i) dst[i] += src[i];
}

static std::vector<float>
concat_desc(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const std::vector<float>& c,
    int H,
    int W)
{
  std::vector<float> out(static_cast<size_t>(H) * W * DESC_CAT_C);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      float* dst = out.data() + hwc_idx(y, x, 0, W, DESC_CAT_C);
      const float* a_ptr = a.data() + hwc_idx(y, x, 0, W, X1_C);
      const float* b_ptr = b.data() + hwc_idx(y, x, 0, W, X2_C);
      const float* c_ptr = c.data() + hwc_idx(y, x, 0, W, X3_C);
      std::memcpy(dst, a_ptr, static_cast<size_t>(X1_C) * sizeof(float));
      std::memcpy(dst + X1_C, b_ptr, static_cast<size_t>(X2_C) * sizeof(float));
      std::memcpy(dst + X1_C + X2_C, c_ptr, static_cast<size_t>(X3_C) * sizeof(float));
    }
  }
  return out;
}

static std::vector<float>
pixel_shuffle2_score(const std::vector<float>& src, int Hs, int Ws)
{
  const int H = Hs * SCORE_UP;
  const int W = Ws * SCORE_UP;
  std::vector<float> out(static_cast<size_t>(H) * W);
  for (int y = 0; y < Hs; ++y) {
    for (int x = 0; x < Ws; ++x) {
      const float* in_ptr = src.data() + hwc_idx(y, x, 0, Ws, SCORE_CH);
      out[(y * 2 + 0) * W + (x * 2 + 0)] = in_ptr[0];
      out[(y * 2 + 0) * W + (x * 2 + 1)] = in_ptr[1];
      out[(y * 2 + 1) * W + (x * 2 + 0)] = in_ptr[2];
      out[(y * 2 + 1) * W + (x * 2 + 1)] = in_ptr[3];
    }
  }
  return out;
}

extern "C" {

std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string>& input, Ax::Logger& log)
{
  auto p = std::make_shared<Props>();
  auto get = [&](const char* k, auto& dst) {
    if (auto it = input.find(k); it != input.end()) {
      if constexpr (std::is_same_v<std::decay_t<decltype(dst)>, std::string>) {
        dst = it->second;
      } else if constexpr (std::is_same_v<std::decay_t<decltype(dst)>, int>) {
        dst = std::stoi(it->second);
      } else {
        dst = std::stof(it->second);
      }
    }
  };

  get("meta_key", p->meta_name);
  get("head_weights_path", p->head_weights_path);
  get("detection_threshold", p->det_threshold);
  get("remove_borders", p->remove_borders);
  get("max_keypoints", p->max_keypoints);
  get("nms_radius", p->nms_radius);

  log(AX_INFO) << "EdgePoint decoder init: threshold=" << p->det_threshold
               << " borders=" << p->remove_borders
               << " max_kpts=" << p->max_keypoints
               << " nms_radius=" << p->nms_radius
               << " head_weights=" << p->head_weights_path;
  return p;
}

const std::unordered_set<std::string>&
allowed_properties()
{
  static const std::unordered_set<std::string> s{
    "meta_key", "head_weights_path", "detection_threshold",
    "remove_borders", "max_keypoints", "nms_radius"
  };
  return s;
}

void
set_dynamic_properties(
    const std::unordered_map<std::string, std::string>&, void*, Ax::Logger&) {}

void
decode_to_meta(
    const AxTensorsInterface& in_tensors,
    const Props* prop,
    unsigned int subframe_index,
    unsigned int subframe_number,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>>& map,
    const AxDataInterface& /*video*/,
    Ax::Logger& log)
try {
  load_weights_if_needed(prop, log);

  struct TensorRef {
    const float* data = nullptr;
    int H = 0;
    int W = 0;
  };
  std::vector<TensorRef> feats;
  feats.reserve(3);

  for (const auto& t : in_tensors) {
    if (t.sizes.size() != 4 || t.bytes != 4 || !t.data) continue;
    if (t.sizes[3] != PADDED_C) continue;
    feats.push_back({static_cast<const float*>(t.data), t.sizes[1], t.sizes[2]});
  }

  if (feats.size() < 3) {
    log(AX_ERROR) << "EdgePoint: could not identify backbone outputs (need three NHWC tensors with C=64)";
    return;
  }

  std::sort(feats.begin(), feats.end(), [](const TensorRef& a, const TensorRef& b) {
    return static_cast<long long>(a.H) * a.W > static_cast<long long>(b.H) * b.W;
  });

  const TensorRef& x1_pad = feats[0];
  const TensorRef& x2_pad = feats[1];
  const TensorRef& x3_pad = feats[2];

  auto x1 = crop_channels(x1_pad.data, x1_pad.H, x1_pad.W, PADDED_C, X1_C);
  auto x2 = crop_channels(x2_pad.data, x2_pad.H, x2_pad.W, PADDED_C, X2_C);
  auto x3 = crop_channels(x3_pad.data, x3_pad.H, x3_pad.W, PADDED_C, X3_C);

  const auto& w = prop->weights;

  auto score = conv1x1_nhwc(x1, x1_pad.H, x1_pad.W, X1_C, DETECT_C, w.conv1_w, w.conv1_b);
  add_inplace(score, bilinear_resize(
      conv1x1_nhwc(x2, x2_pad.H, x2_pad.W, X2_C, DETECT_C, w.conv2_w, w.conv2_b),
      x2_pad.H, x2_pad.W, DETECT_C, x1_pad.H, x1_pad.W));
  add_inplace(score, bilinear_resize(
      conv1x1_nhwc(x3, x3_pad.H, x3_pad.W, X3_C, DETECT_C, w.conv3_w, w.conv3_b),
      x3_pad.H, x3_pad.W, DETECT_C, x1_pad.H, x1_pad.W));

  score = conv3x3_nhwc(score, x1_pad.H, x1_pad.W, DETECT_C, DETECT_C, w.score0_w, w.score0_b);
  relu_inplace(score);
  score = conv3x3_nhwc(score, x1_pad.H, x1_pad.W, DETECT_C, DETECT_C, w.score1_w, w.score1_b);
  relu_inplace(score);
  score = conv3x3_nhwc(score, x1_pad.H, x1_pad.W, DETECT_C, SCORE_CH, w.score2_w, w.score2_b);

  auto score_map_local = pixel_shuffle2_score(score, x1_pad.H, x1_pad.W);
  const int H = x1_pad.H * SCORE_UP;
  const int W = x1_pad.W * SCORE_UP;

  auto x1_half = avg_pool2x2_stride2(x1, x1_pad.H, x1_pad.W, X1_C);
  auto x2_up = bilinear_resize(x2, x2_pad.H, x2_pad.W, X2_C, x1_pad.H / 2, x1_pad.W / 2);
  auto x3_up = bilinear_resize(x3, x3_pad.H, x3_pad.W, X3_C, x1_pad.H / 2, x1_pad.W / 2);
  auto desc_cat = concat_desc(x1_half, x2_up, x3_up, x1_pad.H / 2, x1_pad.W / 2);

  auto desc = conv1x1_nhwc(desc_cat, x1_pad.H / 2, x1_pad.W / 2, DESC_CAT_C, DESC_CAT_C, w.desc0_w, w.desc0_b);
  desc = grouped_conv3x3_nhwc(desc, x1_pad.H / 2, x1_pad.W / 2, w.desc1_w, w.desc1_b);
  relu_inplace(desc);
  desc = conv1x1_nhwc(desc, x1_pad.H / 2, x1_pad.W / 2, DESC_CAT_C, DESC_DIM, w.desc2_w, w.desc2_b);
  const int Hd = x1_pad.H / 2;
  const int Wd = x1_pad.W / 2;

  if (prop->buf_H != H || prop->buf_W != W) {
    prop->buf_H = H;
    prop->buf_W = W;
    prop->score_map.resize(static_cast<size_t>(H) * W);
    prop->row_max.resize(static_cast<size_t>(H) * W);
    prop->maxpool.resize(static_cast<size_t>(H) * W);
  }

  float* score_map = prop->score_map.data();
  float* row_max = prop->row_max.data();
  float* maxpool = prop->maxpool.data();
  std::copy(score_map_local.begin(), score_map_local.end(), score_map);

  const int r = prop->nms_radius;
  for (int y = 0; y < H; ++y) {
    const float* src = score_map + y * W;
    float* dst = row_max + y * W;
    for (int x = 0; x < W; ++x) {
      float m = src[x];
      const int x0 = x - r < 0 ? 0 : x - r;
      const int x1 = x + r >= W ? W - 1 : x + r;
      for (int xx = x0; xx <= x1; ++xx) {
        if (src[xx] > m) m = src[xx];
      }
      dst[x] = m;
    }
  }

  for (int y = 0; y < H; ++y) {
    float* dst = maxpool + y * W;
    const int y0 = y - r < 0 ? 0 : y - r;
    const int y1 = y + r >= H ? H - 1 : y + r;
    for (int x = 0; x < W; ++x) {
      float m = row_max[y * W + x];
      for (int yy = y0; yy <= y1; ++yy) {
        float v = row_max[yy * W + x];
        if (v > m) m = v;
      }
      dst[x] = m;
    }
  }

  for (int i = 0; i < H * W; ++i) {
    score_map[i] = score_map[i] >= maxpool[i] ? score_map[i] : 0.f;
  }

  struct Candidate { float score; int x; int y; };
  std::vector<Candidate> cands;
  cands.reserve(4096);

  const int pad = prop->remove_borders;
  for (int y = pad; y < H - pad; ++y) {
    const float* row = score_map + y * W;
    for (int x = pad; x < W - pad; ++x) {
      float s = row[x];
      if (s > prop->det_threshold) cands.push_back({s, x, y});
    }
  }

  int N = static_cast<int>(cands.size());
  if (prop->max_keypoints > 0 && N > prop->max_keypoints) {
    std::partial_sort(
        cands.begin(), cands.begin() + prop->max_keypoints, cands.end(),
        [](const Candidate& a, const Candidate& b) { return a.score > b.score; });
    cands.resize(prop->max_keypoints);
    N = prop->max_keypoints;
  }
  log(AX_DEBUG) << "EdgePoint: " << N << " keypoints";

  std::vector<float> kpts(static_cast<size_t>(N) * 2);
  std::vector<float> scores_out(N);
  std::vector<float> descs(static_cast<size_t>(N) * DESC_DIM);

  for (int i = 0; i < N; ++i) {
    kpts[i * 2 + 0] = static_cast<float>(cands[i].x);
    kpts[i * 2 + 1] = static_cast<float>(cands[i].y);
    scores_out[i] = cands[i].score;

    const float px = (cands[i].x + 0.5f) / DESC_UP - 0.5f;
    const float py = (cands[i].y + 0.5f) / DESC_UP - 0.5f;

    const int x0 = std::max(0, std::min(Wd - 1, static_cast<int>(std::floor(px))));
    const int y0 = std::max(0, std::min(Hd - 1, static_cast<int>(std::floor(py))));
    const int x1 = std::min(Wd - 1, x0 + 1);
    const int y1 = std::min(Hd - 1, y0 + 1);
    const float wx = px - std::floor(px);
    const float wy = py - std::floor(py);

    const float w00 = (1.f - wx) * (1.f - wy);
    const float w10 = wx * (1.f - wy);
    const float w01 = (1.f - wx) * wy;
    const float w11 = wx * wy;

    const float* p00 = desc.data() + hwc_idx(y0, x0, 0, Wd, DESC_DIM);
    const float* p10 = desc.data() + hwc_idx(y0, x1, 0, Wd, DESC_DIM);
    const float* p01 = desc.data() + hwc_idx(y1, x0, 0, Wd, DESC_DIM);
    const float* p11 = desc.data() + hwc_idx(y1, x1, 0, Wd, DESC_DIM);

    float* out = descs.data() + static_cast<size_t>(i) * DESC_DIM;
    for (int c = 0; c < DESC_DIM; ++c) {
      out[c] = w00 * p00[c] + w10 * p10[c] + w01 * p01[c] + w11 * p11[c];
    }
    l2_normalize(out, DESC_DIM);
  }

  auto* meta = ax_utils::insert_meta<AxMetaRawTensor>(
      map, prop->meta_name, std::string{}, subframe_index, subframe_number);
  if (!meta) {
    log(AX_ERROR) << "EdgePoint: failed to create AxMetaRawTensor";
    return;
  }

  static constexpr float kEmpty = 0.f;
  const float* kpts_data = N > 0 ? kpts.data() : &kEmpty;
  const float* scr_data = N > 0 ? scores_out.data() : &kEmpty;
  const float* desc_data = N > 0 ? descs.data() : &kEmpty;

  meta->add_tensor(kpts_data, N * 2, sizeof(float), {N, 2});
  meta->add_tensor(scr_data, N, sizeof(float), {N});
  meta->add_tensor(desc_data, N * DESC_DIM, sizeof(float), {N, DESC_DIM});
}
catch (const std::exception& e) {
  log(AX_ERROR) << "EdgePoint decoder caught exception: " << e.what();
}
catch (...) {
  log(AX_ERROR) << "EdgePoint decoder caught unknown exception";
}

}  // extern "C"
