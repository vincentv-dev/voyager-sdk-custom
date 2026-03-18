// SuperPoint C++ inference on Axelera Metis M2 (iMX8MP)
//
// Uses AxInferenceNet to run the AIPU backbone and receives raw int8 tensors.
// All postprocessing (softmax, pixel-shuffle, NMS, top-k) runs in C++ with
// NEON auto-vectorisation.
//
// Usage:
//   superpoint_inference <model>.axnet <video-or-camera>
//
// Generate the .axnet with (auto-discovers YAML by name field):
//   ./inference.py superpoint-cpp fakevideo --frames 1 --no-display

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "AxInferenceNet.hpp"
#include "AxMetaRawTensor.hpp"
#include "AxStreamerUtils.hpp"
#include "AxUtils.hpp"
#include "AxFFMpegVideoDecoder.hpp"
#include "opencv2/opencv.hpp"

#include "superpoint_postprocess.hpp"

using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;
using Ms        = std::chrono::duration<double, std::milli>;

// ── Frame container ──────────────────────────────────────────────────────────

struct Frame {
  cv::Mat      bgr;
  Ax::MetaMap  meta;
};

// ── Stats accumulator ─────────────────────────────────────────────────────────

struct Stats {
  std::atomic<uint64_t> n_frames{0};
  std::atomic<uint64_t> total_kp{0};
  TimePoint             t_start{Clock::now()};

  void update(size_t n_kp) {
    ++n_frames;
    total_kp += n_kp;
  }

  void print() const {
    const auto elapsed = Ms(Clock::now() - t_start).count() / 1000.0;
    const auto n = n_frames.load();
    std::cout << "\n══════════════════════════════════════════════\n"
              << "  SuperPoint C++ — Axelera Metis M2\n"
              << "══════════════════════════════════════════════\n"
              << "  Frames      : " << n << "\n"
              << "  Time        : " << std::fixed << std::setprecision(2)
                                   << elapsed << " s\n"
              << "  Avg FPS     : " << (n / elapsed) << "\n"
              << "  Avg kp/frame: " << (n > 0 ? total_kp.load() / n : 0) << "\n"
              << "══════════════════════════════════════════════\n";
  }
};

// ── Argument parsing ──────────────────────────────────────────────────────────

struct Args {
  std::string axnet;
  std::string video;
  int         max_frames       = 0;
  bool        compute_descs    = false;
  int         max_keypoints    = 1000;
  float       threshold        = 0.005f;
};

static void print_usage(const char* argv0)
{
  std::cerr
    << "Usage: " << argv0
    << " <model>.axnet <video> [--max-frames N] [--descriptors] [--max-kp N] [--threshold F]\n"
    << "\n"
    << "  --max-frames N    stop after N frames (0 = run until end)\n"
    << "  --descriptors     also sample 256-D descriptors (for LightGlue)\n"
    << "  --max-kp N        top-k keypoints per frame  (default: 1000)\n"
    << "  --threshold F     detection threshold        (default: 0.005)\n";
}

static Args parse_args(int argc, char** argv)
{
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s.ends_with(".axnet"))        a.axnet = s;
    else if (s == "--descriptors")    a.compute_descs = true;
    else if (s == "--max-frames" && i+1 < argc) a.max_frames = std::stoi(argv[++i]);
    else if (s == "--max-kp"     && i+1 < argc) a.max_keypoints = std::stoi(argv[++i]);
    else if (s == "--threshold"  && i+1 < argc) a.threshold = std::stof(argv[++i]);
    else if (a.video.empty())         a.video = s;
  }
  if (a.axnet.empty() || a.video.empty()) {
    print_usage(argv[0]);
    std::exit(1);
  }
  return a;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
  const auto args = parse_args(argc, argv);

  SuperPointDecoder::Config dec_cfg;
  dec_cfg.detection_threshold = args.threshold;
  dec_cfg.max_keypoints       = args.max_keypoints;
  dec_cfg.compute_descriptors = args.compute_descs;
  SuperPointDecoder decoder(dec_cfg);

  Ax::Logger logger;
  Ax::BlockingQueue<std::shared_ptr<Frame>> ready;

  auto props = Ax::read_inferencenet_properties(args.axnet, logger);
  auto net   = Ax::create_inference_net(props, logger, Ax::forward_to(ready));

  // Per-frame postproc timing
  std::atomic<uint64_t> total_postproc_us{0};
  Stats stats;
  std::atomic<int> frame_count{0};

  auto frame_callback = [&](cv::Mat frame) {
    if (frame.empty()) {
      net->end_of_input();
      return;
    }
    if (args.max_frames > 0 && frame_count.load() >= args.max_frames) {
      net->end_of_input();
      return;
    }
    ++frame_count;
    auto fd = std::make_shared<Frame>();
    fd->bgr = std::move(frame);
    auto video = Ax::video_from_cvmat(fd->bgr, AxVideoFormat::BGR);
    net->push_new_frame(fd, video, fd->meta);
  };

  auto decoder_inst = Ax::FFMpegVideoDecoder(args.video, frame_callback, AxVideoFormat::BGR);
  decoder_inst.start_decoding();

  // ── Main result loop ────────────────────────────────────────────────────────
  uint64_t n = 0;
  TimePoint t_prev = Clock::now();

  while (true) {
    auto frame = ready.wait_one();
    if (!frame) break;

    // Locate the raw tensor meta (key = pipeline step name, here "superpoint")
    const std::string meta_key = "superpoint";
    auto it = frame->meta.find(meta_key);
    if (it == frame->meta.end()) {
      std::cerr << "Warning: meta key '" << meta_key << "' not found\n";
      continue;
    }

    try {
      auto& tensor_meta = dynamic_cast<AxMetaRawTensor&>(*it->second);
      const auto* tc = tensor_meta.get_tensor();
      if (!tc || tc->num_tensors() < 2) {
        std::cerr << "Warning: expected 2 tensors, got "
                  << (tc ? tc->num_tensors() : 0) << "\n";
        continue;
      }

      // tensor 0 = descriptors [1,60,80,256], tensor 1 = scores [1,60,80,128]
      const auto* scores_raw = get_tensor_data<int8_t>(tensor_meta, 1);
      const auto* descs_raw  = args.compute_descs
                               ? get_tensor_data<int8_t>(tensor_meta, 0)
                               : nullptr;

      const auto t0 = Clock::now();
      const SPResult result = decoder.decode(scores_raw, descs_raw);
      const auto dt_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           Clock::now() - t0).count();

      total_postproc_us += dt_us;
      stats.update(result.keypoints.size());

      // Per-30-frame progress line
      ++n;
      if (n % 30 == 0) {
        const double fps = 30.0 / Ms(Clock::now() - t_prev).count() * 1000.0;
        t_prev = Clock::now();
        const double avg_pp = static_cast<double>(total_postproc_us.load()) / n / 1000.0;
        std::cout << "  frame " << std::setw(5) << n
                  << " | " << std::fixed << std::setprecision(1) << fps << " fps"
                  << " | " << result.keypoints.size() << " kp"
                  << " | postproc " << std::setprecision(2) << avg_pp << " ms avg\n";
      }
    } catch (const std::bad_cast&) {
      std::cerr << "Error: expected AxMetaRawTensor for '" << meta_key << "'\n";
    }
  }

  net->stop();
  stats.print();

  // Final postproc breakdown
  if (n > 0) {
    std::cout << "  C++ postproc avg : "
              << std::fixed << std::setprecision(2)
              << static_cast<double>(total_postproc_us.load()) / n / 1000.0
              << " ms/frame\n\n";
  }

  return 0;
}
