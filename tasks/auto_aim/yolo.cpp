#include "yolo.hpp"

#include <yaml-cpp/yaml.h>

namespace auto_aim
{
YOLO::YOLO(const std::string & config_path)
: // yolo11_(std::make_unique<YOLO11>(config_path)), // 暂时禁用
  yolov5_(std::make_unique<YOLOV5>(config_path, false))
  // yolov8_(std::make_unique<YOLOV8>(config_path, false)) // 暂时禁用
{
  auto yaml = YAML::LoadFile(config_path);
  yolo_type_ = yaml["yolo_type"].as<std::string>();
}

std::list<Armor> YOLO::detect(const cv::Mat & raw_img, int frame_count)
{
  if (yolo_type_ == "yolov5")
    return yolov5_->detect(raw_img, frame_count);
  else if (yolo_type_ == "yolov8")
    // return yolov8_->detect(raw_img, frame_count); // 暂时禁用
    return std::list<Armor>(); // 如果配置错误, 返回空
  else if (yolo_type_ == "yolo11")
    // return yolo11_->detect(raw_img, frame_count); // 暂时禁用
    return std::list<Armor>(); // 如果配置错误, 返回空
  else
    return std::list<Armor>();
}
}  // namespace auto_aim