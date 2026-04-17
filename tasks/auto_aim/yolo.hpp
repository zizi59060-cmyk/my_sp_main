#pragma once

#include <list>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>

#include "armor.hpp"
// #include "yolos/yolo11.hpp" // 暂时禁用
#include "yolos/yolov5.hpp"
// #include "yolos/yolov8.hpp" // 暂时禁用

namespace auto_aim
{
class YOLO
{
public:
  explicit YOLO(const std::string & config_path);
  std::list<Armor> detect(const cv::Mat & raw_img, int frame_count);

private:
  std::string yolo_type_;
  // std::unique_ptr<YOLO11> yolo11_; // 暂时禁用
  std::unique_ptr<YOLOV5> yolov5_;
  // std::unique_ptr<YOLOV8> yolov8_; // 暂时禁用
};
}  // namespace auto_aim