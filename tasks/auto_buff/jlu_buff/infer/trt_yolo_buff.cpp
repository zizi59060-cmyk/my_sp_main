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

  // Supported layouts:
  // 1) [1, N, 4 + 1 + 10(+classes)] row-major.
  // 2) [1, C, N] YOLOv8/11-style channel-major; C >= 15.
  int rows = 0;
  int cols = 0;
  bool channel_major = false;
  if (dims.nbDims == 3) {
    const int d1 = dims.d[1];
    const int d2 = dims.d[2];
    if (d1 <= 64 && d2 > d1) {
      rows = d2;
      cols = d1;
      channel_major = true;
    } else {
      rows = d1;
      cols = d2;
    }
  } else if (dims.nbDims == 2) {
    rows = dims.d[0];
    cols = dims.d[1];
  } else {
    const int stride = 4 + 1 + config_.num_points * 2;
    cols = stride;
    rows = static_cast<int>(output.size() / stride);
  }
  if (rows <= 0 || cols < 4 + 1 + config_.num_points * 2) return {};

  auto at = [&](int r, int c) -> float {
    return channel_major ? output[static_cast<size_t>(c) * rows + r] : output[static_cast<size_t>(r) * cols + c];
  };

  std::vector<BuffBlade> candidates;
  for (int r = 0; r < rows; ++r) {
    const float conf = at(r, 4);
    if (conf < config_.confidence_threshold) continue;
    const float cx = at(r, 0);
    const float cy = at(r, 1);
    const float w = at(r, 2);
    const float h = at(r, 3);
    BuffBlade blade;
    blade.confidence = conf;
    const auto tl = unletterbox(cx - 0.5F * w, cy - 0.5F * h, lb, image_size);
    const auto br = unletterbox(cx + 0.5F * w, cy + 0.5F * h, lb, image_size);
    blade.rect = cv::Rect2f(tl, br);

    // Raw detector keypoint order is assumed to be the exported JLU model order:
    // r_center, bottom_right, top_right, top_left, bottom_left. If a future model emits a
    // different order, change only this adapter and keep BuffBladePoints in JLU order.
    for (int i = 0; i < kBuffBladePointCount; ++i) {
      blade.points.image[i] = unletterbox(at(r, 5 + i * 2), at(r, 5 + i * 2 + 1), lb, image_size);
    }
    blade.roll = blade_roll_from_points(blade.points);
    candidates.push_back(blade);
  }

  std::sort(candidates.begin(), candidates.end(), [](const auto & a, const auto & b) { return a.confidence > b.confidence; });
  std::vector<BuffBlade> result;
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
  return result;
}
}  // namespace auto_buff::jlu
