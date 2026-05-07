#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "tasks/auto_buff/jlu_buff/infer/trt_engine.hpp"
#include "tasks/auto_buff/jlu_buff/types.hpp"

namespace auto_buff::jlu
{
struct TrtBuffDetectorConfig
{
  TrtConfig trt;
  float confidence_threshold = 0.5F;
  float nms_threshold = 0.45F;
  int num_points = 5;
};

class TrtBuffDetector
{
public:
  explicit TrtBuffDetector(const std::string & config_path);
  explicit TrtBuffDetector(TrtBuffDetectorConfig config);

  bool ready() const { return engine_ && engine_->ready(); }
  std::vector<BuffBlade> detect(const cv::Mat & image);

private:
  static TrtBuffDetectorConfig loadConfig(const std::string & config_path);
  std::vector<BuffBlade> postprocess(
    const std::vector<float> & output, const nvinfer1::Dims & dims, const LetterboxInfo & letterbox,
    const cv::Size & image_size) const;
  cv::Point2f unletterbox(float x, float y, const LetterboxInfo & letterbox, const cv::Size & image_size) const;

  TrtBuffDetectorConfig config_;
  std::unique_ptr<TrtEngine> engine_;
};
}  // namespace auto_buff::jlu
