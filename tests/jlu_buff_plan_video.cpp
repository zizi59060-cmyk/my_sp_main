#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_buff/jlu_buff/jlu_buff.hpp"

namespace
{
std::string modeToString(auto_buff::jlu::BuffMode mode)
{
  return mode == auto_buff::jlu::BuffMode::BIG ? "BIG" : "SMALL";
}

void drawPlan(
  cv::Mat & image, int frame_id, double time_sec, auto_buff::jlu::BuffMode mode,
  const auto_buff::jlu::JluBuffPlan & plan)
{
  const auto green = cv::Scalar(0, 255, 0);
  const auto red = cv::Scalar(0, 0, 255);
  const auto yellow = cv::Scalar(0, 255, 255);
  const auto white = cv::Scalar(255, 255, 255);

  int y = 35;
  const int dy = 32;

  auto put_color = [&](const std::string & text, const cv::Scalar & color) {
    cv::putText(image, text, {20, y}, cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
    y += dy;
  };

  auto put = [&](const std::string & text) {
    put_color(text, white);
  };

  std::ostringstream line;

  line << "frame=" << frame_id << " time=" << std::fixed << std::setprecision(3) << time_sec
       << " mode=" << modeToString(mode);
  put_color(line.str(), yellow);

  line.str("");
  line.clear();
  line << "track_state=" << auto_buff::jlu::to_string(plan.track_state)
       << " control=" << plan.control << " fire=" << plan.fire;
  put_color(line.str(), plan.control ? green : red);

  line.str("");
  line.clear();
  line << "yaw=" << std::fixed << std::setprecision(4) << plan.yaw
       << " pitch=" << plan.pitch;
  put(line.str());

  line.str("");
  line.clear();
  line << "blade_index=" << plan.blade_index
       << " fly_time=" << std::fixed << std::setprecision(4) << plan.fly_time;
  put(line.str());

  // 画一个简单的屏幕中心十字，方便看输出是否连续；不是实际瞄点投影。
  const cv::Point center(image.cols / 2, image.rows / 2);
  cv::line(image, center + cv::Point(-20, 0), center + cv::Point(20, 0), green, 1);
  cv::line(image, center + cv::Point(0, -20), center + cv::Point(0, 20), green, 1);
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 5) {
    std::cerr
      << "Usage:\n"
      << "  " << argv[0]
      << " <config.yaml> <input_video> <small|big> <output.csv> [output_video]\n\n"
      << "Example:\n"
      << "  " << argv[0]
      << " configs/standard3.yaml /home/wheeltec/projects/buff/big_buff.avi big "
      << "plan_big.csv plan_big_result.mp4\n";
    return 1;
  }

  const std::string config_path = argv[1];
  const std::string input_video = argv[2];
  const std::string mode_arg = argv[3];
  const std::string output_csv = argv[4];
  const std::string output_video = argc >= 6 ? argv[5] : "jlu_buff_plan_result.mp4";

  auto_buff::jlu::BuffMode mode;
  if (mode_arg == "big" || mode_arg == "BIG") {
    mode = auto_buff::jlu::BuffMode::BIG;
  } else if (mode_arg == "small" || mode_arg == "SMALL") {
    mode = auto_buff::jlu::BuffMode::SMALL;
  } else {
    std::cerr << "[ERROR] mode must be small or big, got: " << mode_arg << "\n";
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
    output_video, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));

  if (!writer.isOpened()) {
    std::cerr << "[ERROR] Failed to open output video: " << output_video << "\n";
    return 4;
  }

  std::ofstream csv(output_csv);
  if (!csv.is_open()) {
    std::cerr << "[ERROR] Failed to open output csv: " << output_csv << "\n";
    return 5;
  }

  csv << "frame_id,time_sec,mode,track_state,control,fire,"
      << "yaw,yaw_vel,yaw_acc,pitch,pitch_vel,pitch_acc,"
      << "blade_index,fly_time,total_ms\n";

  std::cout << "[INFO] config: " << config_path << "\n";
  std::cout << "[INFO] input:  " << input_video << "\n";
  std::cout << "[INFO] output video: " << output_video << "\n";
  std::cout << "[INFO] output csv:   " << output_csv << "\n";
  std::cout << "[INFO] size: " << width << "x" << height << " fps=" << fps << "\n";
  std::cout << "[INFO] mode: " << modeToString(mode) << "\n";

  auto_buff::jlu::JluBuffSystem system(config_path);

  if (!system.enabled()) {
    std::cerr << "[ERROR] jlu_buff.enable is false in config.\n";
    return 6;
  }

  // 离线测试用固定姿态。上车时这里由 gimbal.q(t) 提供。
  Eigen::Quaterniond gimbal_q = Eigen::Quaterniond::Identity();

  // 离线测试用伪造云台状态。重点是 bullet_speed，否则 trajectory 会用默认或判异常。
  io::GimbalState gimbal_state{};
  gimbal_state.yaw = 0.0F;
  gimbal_state.yaw_vel = 0.0F;
  gimbal_state.pitch = 0.0F;
  gimbal_state.pitch_vel = 0.0F;
  gimbal_state.bullet_speed = 22.0F;
  gimbal_state.bullet_count = 0;

  const auto start_time = std::chrono::steady_clock::now();

  cv::Mat frame;
  int frame_id = 0;

  int tracking_count = 0;
  int control_count = 0;
  int fire_count = 0;
  double total_ms_sum = 0.0;

  while (cap.read(frame)) {
    const double time_sec = static_cast<double>(frame_id) / fps;
    const auto timestamp =
      start_time + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                     std::chrono::duration<double>(time_sec));

    const auto t0 = std::chrono::steady_clock::now();
    auto plan = system.run(frame, mode, gimbal_q, gimbal_state, timestamp);
    const auto t1 = std::chrono::steady_clock::now();

    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_ms_sum += total_ms;

    if (plan.track_state == auto_buff::jlu::TrackState::TRACKING) tracking_count++;
    if (plan.control) control_count++;
    if (plan.fire) fire_count++;

    csv << frame_id << "," << std::fixed << std::setprecision(6) << time_sec << ","
        << modeToString(mode) << "," << auto_buff::jlu::to_string(plan.track_state) << ","
        << plan.control << "," << plan.fire << ","
        << plan.yaw << "," << plan.yaw_vel << "," << plan.yaw_acc << ","
        << plan.pitch << "," << plan.pitch_vel << "," << plan.pitch_acc << ","
        << plan.blade_index << "," << plan.fly_time << ","
        << total_ms << "\n";

    cv::Mat vis = frame.clone();
    drawPlan(vis, frame_id, time_sec, mode, plan);
    writer.write(vis);

    if (frame_id % 30 == 0) {
      std::cout << "[FRAME " << frame_id << "] "
                << "state=" << auto_buff::jlu::to_string(plan.track_state)
                << " control=" << plan.control
                << " fire=" << plan.fire
                << " yaw=" << plan.yaw
                << " pitch=" << plan.pitch
                << " blade=" << plan.blade_index
                << " fly=" << plan.fly_time
                << " total_ms=" << total_ms
                << "\n";
    }

    ++frame_id;
  }

  csv.close();
  writer.release();
  cap.release();

  const double avg_ms = frame_id > 0 ? total_ms_sum / static_cast<double>(frame_id) : 0.0;

  std::cout << "\n[DONE]\n";
  std::cout << "frames: " << frame_id << "\n";
  std::cout << "tracking_count: " << tracking_count << "\n";
  std::cout << "control_count: " << control_count << "\n";
  std::cout << "fire_count: " << fire_count << "\n";
  std::cout << "avg_total_ms: " << avg_ms << "\n";
  std::cout << "saved video: " << output_video << "\n";
  std::cout << "saved csv: " << output_csv << "\n";

  return 0;
}