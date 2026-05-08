#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/opencv.hpp>

#include "tasks/auto_buff/jlu_buff/infer/trt_yolo_buff.hpp"

namespace
{
cv::Scalar colorForPoint(int idx)
{
  static const cv::Scalar colors[] = {
    {0, 255, 255},  // r_center
    {0, 0, 255},    // bottom_right
    {0, 128, 255},  // top_right
    {255, 0, 0},    // top_left
    {255, 0, 255},  // bottom_left
  };
  return colors[idx % 5];
}

void drawBlade(cv::Mat & image, const auto_buff::jlu::BuffBlade & blade, int index)
{
  cv::rectangle(image, blade.rect, {0, 255, 0}, 2);

  std::ostringstream label;
  label << "#" << index << " conf=" << std::fixed << std::setprecision(2) << blade.confidence
        << " roll=" << std::setprecision(2) << blade.roll;

  const int base_x = static_cast<int>(blade.rect.x);
  const int base_y = std::max(20, static_cast<int>(blade.rect.y) - 6);
  cv::putText(
    image, label.str(), {base_x, base_y}, cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 255, 0}, 2);

  static const char * names[] = {"R", "BR", "TR", "TL", "BL"};

  for (int i = 0; i < auto_buff::jlu::kBuffBladePointCount; ++i) {
    const auto & p = blade.points.image[i];
    cv::circle(image, p, 4, colorForPoint(i), -1);
    cv::putText(
      image, names[i], p + cv::Point2f(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.45,
      colorForPoint(i), 1);
  }

  // 按 JLU 点序连线：BR -> TR -> TL -> BL -> BR，R 点单独画
  const auto & pts = blade.points.image;
  cv::line(image, pts[1], pts[2], {255, 255, 0}, 2);
  cv::line(image, pts[2], pts[3], {255, 255, 0}, 2);
  cv::line(image, pts[3], pts[4], {255, 255, 0}, 2);
  cv::line(image, pts[4], pts[1], {255, 255, 0}, 2);
  cv::line(image, pts[0], (pts[1] + pts[4]) * 0.5F, {0, 255, 255}, 2);
}
}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::cerr << "Usage:\n"
              << "  " << argv[0] << " <config.yaml> <input_video> [output_video]\n\n"
              << "Example:\n"
              << "  " << argv[0] << " configs/standard3.yaml big_buff.mp4 "
              << "big_buff_detect_result.mp4\n";
    return 1;
  }

  const std::string config_path = argv[1];
  const std::string input_video = argv[2];
  const std::string output_video = argc >= 4 ? argv[3] : "jlu_buff_detect_result.mp4";

  auto_buff::jlu::TrtBuffDetector detector(config_path);
  if (!detector.ready()) {
    std::cerr << "[ERROR] TrtBuffDetector is not ready. Check jlu_buff.detector config, "
              << "ONNX path, engine path, TensorRT, and CUDA.\n";
    return 2;
  }

  cv::VideoCapture cap(input_video);
  if (!cap.isOpened()) {
    std::cerr << "[ERROR] Failed to open video: " << input_video << "\n";
    return 3;
  }

  const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);
  if (fps <= 1.0 || !std::isfinite(fps)) fps = 30.0;

  cv::VideoWriter writer;
  writer.open(
    output_video,
    cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
    fps,
    {width, height});

  if (!writer.isOpened()) {
    std::cerr << "[ERROR] Failed to open output video: " << output_video << "\n";
    return 4;
  }

  std::cout << "[INFO] input:  " << input_video << "\n";
  std::cout << "[INFO] output: " << output_video << "\n";
  std::cout << "[INFO] size:   " << width << "x" << height << " fps=" << fps << "\n";

  cv::Mat frame;
  int frame_id = 0;
  double total_ms = 0.0;
  int total_detections = 0;

  while (cap.read(frame)) {
    const auto t0 = std::chrono::steady_clock::now();
    auto blades = detector.detect(frame);
    const auto t1 = std::chrono::steady_clock::now();

    const double infer_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();

    total_ms += infer_ms;
    total_detections += static_cast<int>(blades.size());

    cv::Mat vis = frame.clone();

    for (int i = 0; i < static_cast<int>(blades.size()); ++i) {
      drawBlade(vis, blades[i], i);
    }

    std::ostringstream status;
    status << "frame=" << frame_id
           << " blades=" << blades.size()
           << " detect_ms=" << std::fixed << std::setprecision(2) << infer_ms;

    cv::putText(
      vis, status.str(), {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.9, {0, 255, 0}, 2);

    writer.write(vis);

    if (frame_id % 30 == 0) {
      std::cout << "[FRAME " << frame_id << "] blades=" << blades.size()
                << " detect_ms=" << infer_ms << "\n";
    }

    ++frame_id;
  }

  writer.release();
  cap.release();

  const double avg_ms = frame_id > 0 ? total_ms / frame_id : 0.0;
  const double avg_det = frame_id > 0 ? static_cast<double>(total_detections) / frame_id : 0.0;

  std::cout << "\n[DONE]\n";
  std::cout << "frames: " << frame_id << "\n";
  std::cout << "avg_detect_ms: " << avg_ms << "\n";
  std::cout << "avg_blades_per_frame: " << avg_det << "\n";
  std::cout << "saved: " << output_video << "\n";

  return 0;
}