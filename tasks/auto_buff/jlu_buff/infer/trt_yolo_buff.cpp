#include "tasks/auto_buff/jlu_buff/infer/trt_yolo_buff.hpp"

#include <algorithm>
#include <chrono>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_buff::jlu
{
namespace
{
template <typename T>
T read_or(const YAML::Node & n, const std::string & key, const T & fallback)
{
  return n && n[key] ? n[key].as<T>() : fallback;
}

float iou(const cv::Rect2f & a, const cv::Rect2f & b)
{
  const float inter = static_cast<float>((a & b).area());
  const float uni = static_cast<float>(a.area() + b.area() - inter);
  return uni <= 0.0F ? 0.0F : inter / uni;
}
}  // namespace

TrtBuffDetector::TrtBuffDetector(const std::string & config_path) : TrtBuffDetector(loadConfig(config_path)) {}

TrtBuffDetector::TrtBuffDetector(TrtBuffDetectorConfig config) : config_(std::move(config))
{
  engine_ = std::make_unique<TrtEngine>(config_.trt);
  if (!engine_->loadOrBuild()) {
    tools::logger()->error("[JLU-Buff] TensorRT detector is not ready; detection will return empty results.");
  }
}

TrtBuffDetectorConfig TrtBuffDetector::loadConfig(const std::string & config_path)
{
  TrtBuffDetectorConfig cfg;
  auto yaml = YAML::LoadFile(config_path);
  auto detector = yaml["jlu_buff"] && yaml["jlu_buff"]["detector"] ? yaml["jlu_buff"]["detector"] : YAML::Node{};
  cfg.trt.onnx_path = read_or<std::string>(detector, "onnx_path", "models/buff/buff.onnx");
  cfg.trt.engine_path = read_or<std::string>(detector, "engine_path", "models/buff/buff_trt10_3.engine");
  cfg.trt.input_width = read_or<int>(detector, "input_width", 640);
  cfg.trt.input_height = read_or<int>(detector, "input_height", 640);
  cfg.trt.fp16 = read_or<bool>(detector, "fp16", true);
  cfg.trt.device_id = read_or<int>(detector, "device_id", 0);
  cfg.trt.input_tensor_name = read_or<std::string>(detector, "input_tensor_name", "");
  cfg.trt.output_tensor_name = read_or<std::string>(detector, "output_tensor_name", "");
  cfg.confidence_threshold = read_or<float>(detector, "confidence_threshold", 0.5F);
  cfg.nms_threshold = read_or<float>(detector, "nms_threshold", 0.45F);
  cfg.num_points = read_or<int>(detector, "num_points", 5);
  return cfg;
}

std::vector<BuffBlade> TrtBuffDetector::detect(const cv::Mat & image)
{
  if (!ready() || image.empty()) return {};
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<float> raw;
  LetterboxInfo lb;
  if (!engine_->infer(image, raw, &lb)) return {};
  const auto t1 = std::chrono::steady_clock::now();
  auto blades = postprocess(raw, engine_->outputDims(), lb, image.size());
  const auto t2 = std::chrono::steady_clock::now();
  tools::logger()->debug(
    "[JLU-Buff] TensorRT infer={:.3f}ms postprocess={:.3f}ms blades={}",
    tools::delta_time(t1, t0) * 1000.0, tools::delta_time(t2, t1) * 1000.0, blades.size());
  return blades;
}

cv::Point2f TrtBuffDetector::unletterbox(float x, float y, const LetterboxInfo & lb, const cv::Size & image_size) const
{
  const float px = std::clamp((x - lb.pad_x) / lb.scale, 0.0F, static_cast<float>(image_size.width - 1));
  const float py = std::clamp((y - lb.pad_y) / lb.scale, 0.0F, static_cast<float>(image_size.height - 1));
  return {px, py};
}
std::vector<BuffBlade> TrtBuffDetector::postprocess(
  const std::vector<float> & output, const nvinfer1::Dims & dims, const LetterboxInfo & lb,
  const cv::Size & image_size) const
{
  if (output.empty()) return {};

  int rows = 0;
  int cols = 0;
  bool channel_major = false;

  if (dims.nbDims == 3) {
    const int d1 = dims.d[1];
    const int d2 = dims.d[2];

    if (d1 <= 128 && d2 > d1) {
      rows = d2;
      cols = d1;
      channel_major = true;
    } else {
      rows = d1;
      cols = d2;
      channel_major = false;
    }
  } else if (dims.nbDims == 2) {
    rows = dims.d[0];
    cols = dims.d[1];
    channel_major = false;
  } else {
    tools::logger()->warn(
      "[JLU-Buff] unsupported output dims nbDims={} output_size={}", dims.nbDims, output.size());
    return {};
  }

  if (rows <= 0 || cols < 15) {
    tools::logger()->warn(
      "[JLU-Buff] invalid output shape: nbDims={} rows={} cols={} output_size={}",
      dims.nbDims, rows, cols, output.size());
    return {};
  }

  auto at = [&](int r, int c) -> float {
    return channel_major ? output[static_cast<size_t>(c) * rows + r]
                         : output[static_cast<size_t>(r) * cols + c];
  };

  // JLU YOLOX rune output:
  // [point0_x, point0_y, point1_x, point1_y, point2_x, point2_y,
  //  point3_x, point3_y, point4_x, point4_y,
  //  confidence, color0, color1, class0, class1]
  //
  // rows = 4725 when input is 480:
  // 60*60 + 30*30 + 15*15, strides = 8,16,32.
  static bool printed_once = false;
  if (!printed_once) {
    printed_once = true;

    std::string dims_str = "[";
    for (int i = 0; i < dims.nbDims; ++i) {
      dims_str += std::to_string(dims.d[i]);
      if (i + 1 < dims.nbDims) dims_str += ", ";
    }
    dims_str += "]";

    tools::logger()->warn(
      "[JLU-Buff-Debug] output dims={} nbDims={} rows={} cols={} channel_major={} output_size={}",
      dims_str, dims.nbDims, rows, cols, channel_major, output.size());

    const int dump_rows = std::min(rows, 5);
    const int dump_cols = std::min(cols, 15);
    for (int r = 0; r < dump_rows; ++r) {
      std::string line = "[JLU-Buff-Debug] row " + std::to_string(r) + ":";
      for (int c = 0; c < dump_cols; ++c) {
        line += " " + std::to_string(at(r, c));
      }
      tools::logger()->warn("{}", line);
    }

    for (int c = 0; c < dump_cols; ++c) {
      float mn = 1e9F;
      float mx = -1e9F;
      for (int r = 0; r < rows; ++r) {
        const float v = at(r, c);
        mn = std::min(mn, v);
        mx = std::max(mx, v);
      }
      tools::logger()->warn("[JLU-Buff-Debug] col {} min={} max={}", c, mn, mx);
    }
  }

  std::vector<int> strides = {8, 16, 32};
  struct GridStride
  {
    int grid0 = 0;
    int grid1 = 0;
    int stride = 0;
  };

  std::vector<GridStride> grid_strides;
  grid_strides.reserve(static_cast<size_t>(rows));

  for (const int stride : strides) {
    const int grid_w = config_.trt.input_width / stride;
    const int grid_h = config_.trt.input_height / stride;
    for (int g1 = 0; g1 < grid_h; ++g1) {
      for (int g0 = 0; g0 < grid_w; ++g0) {
        grid_strides.push_back({g0, g1, stride});
      }
    }
  }

  if (static_cast<int>(grid_strides.size()) != rows) {
    tools::logger()->warn(
      "[JLU-Buff] grid count mismatch: rows={} grid_strides={} input={}x{}. "
      "Check detector input_width/input_height. JLU yolox_rune usually needs 480x480.",
      rows, grid_strides.size(), config_.trt.input_width, config_.trt.input_height);
    return {};
  }

  std::vector<BuffBlade> candidates;
  candidates.reserve(static_cast<size_t>(rows));

  for (int anchor_idx = 0; anchor_idx < rows; ++anchor_idx) {
    const float confidence = at(anchor_idx, 10);
    if (!std::isfinite(confidence) || confidence < config_.confidence_threshold) continue;

    const int grid0 = grid_strides[anchor_idx].grid0;
    const int grid1 = grid_strides[anchor_idx].grid1;
    const int stride = grid_strides[anchor_idx].stride;

    std::array<cv::Point2f, kBuffBladePointCount> model_points;

    for (int i = 0; i < kBuffBladePointCount; ++i) {
      const float ox = at(anchor_idx, i * 2);
      const float oy = at(anchor_idx, i * 2 + 1);

      if (!std::isfinite(ox) || !std::isfinite(oy)) continue;

      const float x = (ox + static_cast<float>(grid0)) * static_cast<float>(stride);
      const float y = (oy + static_cast<float>(grid1)) * static_cast<float>(stride);

      model_points[i] = unletterbox(x, y, lb, image_size);
    }

    // 原 JLU RunePoints 顺序：
    // 0 center, 1 bottom_left, 2 top_left, 3 top_right, 4 bottom_right
    //
    // 当前 jlu_buff 需要：
    // 0 r_center, 1 bottom_right, 2 top_right, 3 top_left, 4 bottom_left
    BuffBlade blade;
    blade.confidence = confidence;
    blade.points.image[0] = model_points[0];  // center / r_center
    blade.points.image[1] = model_points[4];  // bottom_right
    blade.points.image[2] = model_points[3];  // top_right
    blade.points.image[3] = model_points[2];  // top_left
    blade.points.image[4] = model_points[1];  // bottom_left

    std::vector<cv::Point2f> rect_points;
    rect_points.reserve(kBuffBladePointCount);
    for (const auto & p : blade.points.image) rect_points.push_back(p);
    blade.rect = cv::boundingRect(rect_points);

    if (blade.rect.width <= 1.0F || blade.rect.height <= 1.0F) continue;

    // class: col13 class0, col14 class1
    // JLU 原版：class_id.x ? Activated : Inactivated
    const float class0 = at(anchor_idx, 13);
    const float class1 = at(anchor_idx, 14);
    const int class_id = class1 > class0 ? 1 : 0;
    blade.state = class_id ? BladeState::ACTIVATED : BladeState::UNACTIVATED;

    blade.roll = blade_roll_from_points(blade.points);
    candidates.push_back(blade);
  }

  std::sort(candidates.begin(), candidates.end(), [](const auto & a, const auto & b) {
    return a.confidence > b.confidence;
  });

  constexpr size_t kPreNmsTopK = 128;
  if (candidates.size() > kPreNmsTopK) candidates.resize(kPreNmsTopK);

  std::vector<BuffBlade> result;
  result.reserve(candidates.size());

  for (const auto & c : candidates) {
    bool keep = true;
    for (const auto & r : result) {
      if (iou(c.rect, r.rect) > config_.nms_threshold) {
        keep = false;
        break;
      }
    }

    if (keep) result.push_back(c);
  }

  constexpr size_t kPostNmsTopK = 20;
  if (result.size() > kPostNmsTopK) result.resize(kPostNmsTopK);

  return result;
}
}