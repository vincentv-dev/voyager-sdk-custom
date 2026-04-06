// Copyright Axelera AI, 2025
#include <unordered_map>
#include <unordered_set>
#include "AxDataInterface.h"
#include "AxLog.hpp"
#include "AxMeta.hpp"
#include "AxOpUtils.hpp"

#include <cstring>
#include <optional>

#include <arm_neon.h>
#include <opencv2/core/ocl.hpp>

namespace
{
struct padding_properties {
  std::vector<std::vector<int>> paddings;
  std::optional<int8_t> fill{};
  std::vector<int> in_shape{};
  std::vector<int> out_shape{};
};


} // namespace

extern "C" const std::unordered_set<std::string> &
allowed_properties()
{
  static const std::unordered_set<std::string> allowed_properties{ "padding",
    "fill", "input_shape", "output_shape" };
  return allowed_properties;
}

extern "C" std::shared_ptr<void>
init_and_set_static_properties(
    const std::unordered_map<std::string, std::string> &input, Ax::Logger &logger)
{
  std::shared_ptr<padding_properties> prop = std::make_shared<padding_properties>();
  prop->paddings = Ax::get_property(
      input, "padding", "padding_properties", std::vector<std::vector<int>>{});
  // For backward compatibility - if no paddings were specified but there's a single padding vector, convert it
  if (prop->paddings.empty()) {
    std::vector<int> single_padding = Ax::get_property(
        input, "padding", "padding_properties", std::vector<int>{});
    if (!single_padding.empty()) {
      prop->paddings.push_back(single_padding);
    }
  }
  prop->fill = Ax::get_property(input, "fill", "padding_properties", prop->fill);
  prop->in_shape
      = Ax::get_property(input, "input_shape", "padding_properties", prop->in_shape);
  prop->out_shape
      = Ax::get_property(input, "output_shape", "padding_properties", prop->out_shape);
  return prop;
}

extern "C" AxDataInterface
set_output_interface(const AxDataInterface &interface,
    const padding_properties *prop, Ax::Logger &logger)
{
  if (!std::holds_alternative<AxTensorsInterface>(interface)) {
    throw std::runtime_error("transform_padding requires tensor input");
  }
  auto input = std::get<AxTensorsInterface>(interface);


  // Make sure we have at least one tensor
  if (input.empty() || input[0].bytes != 1) {
    throw std::runtime_error("transform_padding requires at least one int8 tensor input");
  }

  // Make sure we have paddings for each tensor or at least one default padding
  if (prop->paddings.empty()) {
    throw std::runtime_error("transform_padding: no padding configurations provided");
  }

  if (prop->paddings.size() < input.size()) {
    throw std::runtime_error("transform_padding: fewer padding configurations than tensors, expected "
                             + std::to_string(input.size()) + " but got "
                             + std::to_string(prop->paddings.size()));
  }

  // Validate each padding configuration
  for (size_t i = 0; i < std::min(prop->paddings.size(), input.size()); ++i) {
    const auto &padding = prop->paddings[i];
    if ((padding.size() % 2) != 0) {
      throw std::runtime_error("transform_padding: padding must be a multiple of 2:"
                               + ax_utils::sizes_to_string(padding));
    }
    if (padding.size() / 2 > input[i].sizes.size()) {
      throw std::runtime_error("transform_padding: padding "
                               + ax_utils::sizes_to_string(padding) + " too long for input tensor "
                               + ax_utils::sizes_to_string(input[i].sizes));
    }
  }

  if (!ax_utils::validate_shape(prop->in_shape, input[0].sizes)) {
    throw std::runtime_error("transform_padding: input_shape "
                             + ax_utils::sizes_to_string(prop->in_shape) + " does not match input tensor "
                             + ax_utils::sizes_to_string(input[0].sizes));
  }

  auto output = input;

  // Calculate output sizes for each tensor based on its padding
  for (size_t i = 0; i < input.size(); ++i) {
    // Use the appropriate padding for this tensor (or the last one if we have fewer paddings than tensors)
    const auto &padding
        = i < prop->paddings.size() ? prop->paddings[i] : prop->paddings.back();

    auto in_sizes = prop->in_shape.empty() ? input[i].sizes : prop->in_shape;
    const auto info = ax_utils::get_transfer_info(in_sizes, padding);

    if (i == 0 && !ax_utils::validate_shape(prop->out_shape, info.out_sizes)) {
      throw std::runtime_error("transform_padding: output_shape "
                               + ax_utils::sizes_to_string(prop->out_shape) + " does not match calculated output tensor "
                               + ax_utils::sizes_to_string(info.out_sizes));
    }

    output[i].sizes = prop->out_shape.empty() ? info.out_sizes : prop->out_shape;
  }


  return { output };
}

extern "C" void
transform(const AxDataInterface &input, const AxDataInterface &output,
    const padding_properties *prop, unsigned int, unsigned int,
    std::unordered_map<std::string, std::unique_ptr<AxMetaBase>> &, Ax::Logger &logger)
{
  static const bool opencl_disabled = (cv::ocl::setUseOpenCL(false), true);
  (void)opencl_disabled;

  auto input_tensors = std::get<AxTensorsInterface>(input);
  auto output_tensors = std::get<AxTensorsInterface>(output);

  for (size_t i = 0; i < input_tensors.size(); ++i) {
    // Use the appropriate padding for this tensor (or the last one if we have fewer paddings than tensors)
    const auto &padding
        = i < prop->paddings.size() ? prop->paddings[i] : prop->paddings.back();

    auto in_shape = prop->in_shape.empty() ? input_tensors[i].sizes : prop->in_shape;
    const auto info = ax_utils::get_transfer_info(in_shape, padding);

    cv::Mat input_mat(info.in_sizes, CV_8UC1, input_tensors[i].data);
    cv::Mat output_mat(info.out_sizes, CV_8UC1, output_tensors[i].data);

    if (info.is_crop) {
      input_mat(info.ranges).copyTo(output_mat);
    } else {
      // Fast path: single-channel input expanded to N≥16 channels via padding.
      // Replaces memset+strided-copyTo (two passes, write-allocate per pixel)
      // with one pass writing full 64-byte-aligned cache lines via NEON.
      // Condition: no reshape (in/out_shape not set), C_in==1, C_out≥16 and
      // divisible by 16, non-negative fill, at least 3 effective dimensions.
      const int ndim = static_cast<int>(info.out_sizes.size());
      const bool use_neon_fast_path =
          prop->fill &&
          prop->in_shape.empty() &&
          prop->out_shape.empty() &&
          ndim >= 3 &&
          info.in_sizes.back() == 1 &&
          info.out_sizes.back() >= 16 &&
          (info.out_sizes.back() % 16) == 0;

      if (use_neon_fast_path) {
        const int C_out  = info.out_sizes[ndim - 1];
        const int W_out  = info.out_sizes[ndim - 2];
        const int H_out  = info.out_sizes[ndim - 3];
        const int H_in   = info.in_sizes [ndim - 3];
        const int W_in   = info.in_sizes [ndim - 2];
        const int h_pad  = info.ranges   [ndim - 3].start;
        const int w_pad  = info.ranges   [ndim - 2].start;

        // Leading batch dimensions (product of all dims before H,W,C).
        int n_batch = 1;
        for (int d = 0; d < ndim - 3; ++d) n_batch *= info.out_sizes[d];

        const uint8_t fill_u8 = static_cast<uint8_t>(
            static_cast<int8_t>(*prop->fill));
        const uint8x16_t fill16 = vdupq_n_u8(fill_u8);
        const int neon_vecs     = C_out / 16;   // NEON vectors per cell

        const uint8_t* src_base = static_cast<const uint8_t*>(input_tensors[i].data);
        uint8_t*       dst_base = static_cast<uint8_t*>(output_tensors[i].data);
        const std::size_t row_stride = static_cast<std::size_t>(W_out) * C_out;

        for (int b = 0; b < n_batch; ++b) {
          const uint8_t* src_b = src_base + b * H_in * W_in;
          uint8_t*       dst_b = dst_base + b * H_out * W_out * C_out;

          // Top border rows — fill only
          if (h_pad > 0)
            std::memset(dst_b, fill_u8, static_cast<std::size_t>(h_pad) * row_stride);

          for (int h = 0; h < H_in; ++h) {
            const uint8_t* src_row = src_b + h * W_in;
            uint8_t* dst_row = dst_b + static_cast<std::size_t>(h + h_pad) * row_stride;

            // Left border cols — fill only
            if (w_pad > 0)
              std::memset(dst_row, fill_u8, static_cast<std::size_t>(w_pad) * C_out);

            // Image cells: write [pixel, fill×(C_out-1)] per cell in one pass.
            // Each cell is exactly C_out bytes; writing all bytes in one shot
            // avoids the read-for-ownership penalty of the single-byte copyTo.
            uint8_t* dst_img = dst_row + static_cast<std::size_t>(w_pad) * C_out;
            for (int w = 0; w < W_in; ++w) {
              uint8_t* cell = dst_img + w * C_out;
              uint8x16_t v0 = fill16;
              v0 = vsetq_lane_u8(src_row[w], v0, 0);  // channel 0 = pixel
              vst1q_u8(cell, v0);
              for (int v = 1; v < neon_vecs; ++v)
                vst1q_u8(cell + v * 16, fill16);
            }

            // Right border cols — fill only
            const int right_pad = W_out - w_pad - W_in;
            if (right_pad > 0)
              std::memset(dst_img + static_cast<std::size_t>(W_in) * C_out,
                          fill_u8,
                          static_cast<std::size_t>(right_pad) * C_out);
          }

          // Bottom border rows — fill only
          const int bot_pad = H_out - h_pad - H_in;
          if (bot_pad > 0)
            std::memset(dst_b + static_cast<std::size_t>(h_pad + H_in) * row_stride,
                        fill_u8,
                        static_cast<std::size_t>(bot_pad) * row_stride);
        }
      } else {
        if (prop->fill) {
          std::memset(output_tensors[i].data,
              static_cast<unsigned char>(static_cast<uint8_t>(*prop->fill)),
              output_mat.total() * output_mat.elemSize());
        }
        input_mat.copyTo(output_mat(info.ranges));
      }
    }
  }
}
