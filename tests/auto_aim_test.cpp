#include <fmt/core.h>

#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

static constexpr const char * kVideoPath = "/home/wheeltec/projects/my_sp_main/assets/demo/demo.avi";

const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{config-path c  | configs/demo.yaml | yaml配置文件的路径}"
  "{start-index s  | 0                 | 视频起始帧下标    }"
  "{end-index e    | 0                 | 视频结束帧下标    }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>("config-path");
  auto start_index = cli.get<int>("start-index");
  auto end_index = cli.get<int>("end-index");

  tools::Plotter plotter;
  tools::Exiter exiter;

  cv::VideoCapture video(kVideoPath);
  if (!video.isOpened()) {
    tools::logger()->error("Failed to open video: {}", kVideoPath);
    return 1;
  }

  // 用视频 FPS 生成时间戳
  double fps = video.get(cv::CAP_PROP_FPS);
  if (fps <= 1e-3) {
    fps = 60.0;  // 兜底：拿不到fps就假设60
    tools::logger()->warn("Video FPS unavailable, fallback to {} fps", fps);
  }
  const double frame_dt_us = 1e6 / fps;

  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);

  cv::Mat img;
  auto t0 = std::chrono::steady_clock::now();

  auto_aim::Target last_target;
  io::Command last_command;
  double last_t = -1;  // 这里原本存 txt 的 t；现在存“视频时间秒”

  // 跳到起始帧
  video.set(cv::CAP_PROP_POS_FRAMES, start_index);

  for (int frame_count = start_index; !exiter.exit(); frame_count++) {
    if (end_index > 0 && frame_count > end_index) break;

    video.read(img);
    if (img.empty()) break;

    // 当前帧对应的时间（秒）与 timestamp
    double t_sec = (frame_count - start_index) / fps;
    auto timestamp = t0 + std::chrono::microseconds(
      (int64_t)((frame_count - start_index) * frame_dt_us));

    /// 自瞄核心逻辑
    // 不读txt：固定云台姿态为单位四元数
    solver.set_R_gimbal2world({1.0, 0.0, 0.0, 0.0});

    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);

    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, timestamp);

    auto aimer_start = std::chrono::steady_clock::now();
    auto command = aimer.aim(targets, timestamp, 27, false);

    // 保留你原来的简易发弹逻辑（基于上一帧command的yaw变化）
    if (
      !targets.empty() && aimer.debug_aim_point.valid &&
      std::abs(command.yaw - last_command.yaw) * 57.3 < 2)
      command.shoot = true;

    if (command.control) last_command = command;

    /// 调试输出
    auto finish = std::chrono::steady_clock::now();
    tools::logger()->info(
      "[{}] yolo: {:.1f}ms, tracker: {:.1f}ms, aimer: {:.1f}ms", frame_count,
      tools::delta_time(tracker_start, yolo_start) * 1e3,
      tools::delta_time(aimer_start, tracker_start) * 1e3,
      tools::delta_time(finish, aimer_start) * 1e3);

    tools::draw_text(
      img,
      fmt::format(
        "command is {},{:.2f},{:.2f},shoot:{}", command.control, command.yaw * 57.3,
        command.pitch * 57.3, command.shoot),
      {10, 60}, {154, 50, 205});

    // 固定姿态：云台角显示为0（预期）
    Eigen::Quaterniond gimbal_q(1.0, 0.0, 0.0, 0.0);
    tools::draw_text(
      img,
      fmt::format(
        "gimbal yaw{:.2f}", (tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3)[0]),
      {10, 90}, {255, 255, 255});

    tools::draw_text(
      img,
      fmt::format(
        "gimbal yaw{:.2f} pitch{:.2f}",
        (tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0, true) * 57.3)[0],
        (tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0, true) * 57.3)[1]),
      {10, 90}, {255, 255, 255});

    nlohmann::json data;

    // 装甲板原始观测数据
    data["armor_num"] = armors.size();
    if (!armors.empty()) {
      const auto & armor = armors.front();
      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
      data["armor_center_x"] = armor.center_norm.x;
      data["armor_center_y"] = armor.center_norm.y;
    }

    // 没有txt：gimbal yaw 用固定值
    data["gimbal_yaw"] = 0.0;
    data["cmd_yaw"] = command.yaw * 57.3;
    data["shoot"] = command.shoot;

    if (!targets.empty()) {
      auto target = targets.front();

      if (last_t == -1) {
        last_target = target;
        last_t = t_sec;
        // 保持你原来的行为：第一帧 target 有效时先 continue
        plotter.plot(data);
        cv::resize(img, img, {}, 0.5, 0.5);
        cv::imshow("reprojection", img);
        auto key = cv::waitKey(30);
        if (key == 'q') break;
        continue;
      }

      std::vector<Eigen::Vector4d> armor_xyza_list;

      // 当前帧 target 更新后：画所有装甲板（绿）
      armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      // aimer 瞄准位置（红）
      auto aim_point = aimer.debug_aim_point;
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (aim_point.valid) tools::draw_points(img, image_points, {0, 0, 255});

      // 观测器内部数据
      Eigen::VectorXd xk = target.ekf_x();
      data["x"] = xk[0];
      data["vx"] = xk[1];
      data["y"] = xk[2];
      data["vy"] = xk[3];
      data["z"] = xk[4];
      data["vz"] = xk[5];
      data["a"] = xk[6] * 57.3;
      data["w"] = xk[7];
      data["r"] = xk[8];
      data["l"] = xk[9];
      data["h"] = xk[10];
      data["last_id"] = target.last_id;

      // 卡方检验数据
      data["residual_yaw"] = target.ekf().data.at("residual_yaw");
      data["residual_pitch"] = target.ekf().data.at("residual_pitch");
      data["residual_distance"] = target.ekf().data.at("residual_distance");
      data["residual_angle"] = target.ekf().data.at("residual_angle");
      data["nis"] = target.ekf().data.at("nis");
      data["nees"] = target.ekf().data.at("nees");
      data["nis_fail"] = target.ekf().data.at("nis_fail");
      data["nees_fail"] = target.ekf().data.at("nees_fail");
      data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");
    }

    plotter.plot(data);

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("reprojection", img);
    auto key = cv::waitKey(30);
    if (key == 'q') break;
  }

  return 0;
}