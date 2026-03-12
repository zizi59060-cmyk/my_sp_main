// 从下位机读取弹速 + 同款发弹逻辑版本
// 基于 auto_aim_test_all.txt 改写：其余逻辑保持不变

#include <fmt/core.h>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{@config-path   | configs/demo.yaml | yaml配置文件路径}";

int main(int argc, char * argv[])
{
  // -------------------- 参数读取 --------------------
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  tools::Plotter plotter;
  tools::Exiter exiter;

  // -------------------- 模块初始化 --------------------
  io::Camera camera(config_path);  // 工业相机
  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);  // ✅ 同款 Shooter 发弹逻辑
  io::Gimbal gimbal(config_path);          // 串口通信对象（从下位机读弹速）

  cv::Mat img;
  double fps = 0.0;
  auto last_frame_time = std::chrono::steady_clock::now();
  int frame_count = 0;
  constexpr double fps_update_interval = 1.0;

  // -------------------- 主循环 --------------------
  while (!exiter.exit()) {
    auto frame_start = std::chrono::steady_clock::now();

    camera.read(img, frame_start);  // 读取图像和时间戳
    if (img.empty()) break;

    // -------------------- 云台状态 --------------------
    Eigen::Quaterniond gimbal_q = gimbal.q(frame_start);
    solver.set_R_gimbal2world(gimbal_q);

    // 读取下位机状态（含弹速）
    io::GimbalState gs = gimbal.state();

    // 当前云台欧拉角（弧度）：用于 Shooter 的 gimbal_pos 输入
    Eigen::Vector3d current_euler = tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0);
    Eigen::Vector3d gimbal_pos;
    gimbal_pos << current_euler[0], 0.0, current_euler[2];  // [0]=yaw, [2]=pitch（中间值此处不用）

    // -------------------- 自瞄核心 --------------------
    auto armors = yolo.detect(img, 0);
    auto targets = tracker.track(armors, frame_start);

    // ✅ 从下位机读取弹速（单位 m/s）
    // GimbalToVision/State 里就包含 bullet_speed 字段
    double bullet_speed = static_cast<double>(gs.bullet_speed);

    // 其余不变：交给 aimer 做预测与弹道解算6
    auto command = aimer.aim(targets, frame_start, bullet_speed, false);

    // ✅ 同款发弹控制逻辑：完全复用 Shooter::shoot 的判定
    // Shooter 内部会检查：command.control、targets 非空、auto_fire_、以及 yaw/aim_point 条件
    command.shoot = shooter.shoot(command, aimer, targets, gimbal_pos);

    // -------------------- 串口发送控制命令 --------------------
    gimbal.send(command.control, command.shoot, command.yaw, 0, 0, command.pitch, 0, 0);

    // -------------------- 帧率计算 --------------------
    auto frame_end = std::chrono::steady_clock::now();
    frame_count++;
    double elapsed_time = tools::delta_time(frame_end, last_frame_time);

    if (elapsed_time >= fps_update_interval) {
      fps = frame_count / elapsed_time;
      frame_count = 0;
      last_frame_time = frame_end;
      tools::logger()->info(
        "Current FPS: {:.1f}, Tracker State: {}, Bullet speed: {:.2f} m/s",
        fps, tracker.state(), bullet_speed);
    }

    // -------------------- 可视化 --------------------
    // 为了显示重新计算一遍角度（转为角度制）
    Eigen::Vector3d euler_deg = current_euler * 57.3;

    cv::Scalar state_color;
    std::string tracker_state = tracker.state();
    if (tracker_state == "tracking")
      state_color = {0, 255, 0};
    else if (tracker_state == "temp_lost")
      state_color = {0, 255, 255};
    else if (tracker_state == "detecting")
      state_color = {255, 255, 0};
    else
      state_color = {0, 0, 255};

    tools::draw_text(img, fmt::format("State: {}", tracker_state), {10, 30}, state_color);
    tools::draw_text(
      img, fmt::format("gimbal yaw:{:.2f}, pitch:{:.2f}", euler_deg[0], euler_deg[2]),
      {10, 60}, {255, 255, 255});
    tools::draw_text(
      img,
      fmt::format(
        "cmd yaw:{:.2f}, pitch:{:.2f}, shoot:{}, v:{:.2f}m/s",
        command.yaw * 57.3, command.pitch * 57.3, command.shoot, bullet_speed),
      {10, 90}, {154, 50, 205});

    if (!targets.empty() && aimer.debug_aim_point.valid) {
      auto & aim_point = aimer.debug_aim_point;
      Eigen::Vector4d aim_xyza;
      for (int i = 0; i < 4; ++i) aim_xyza[i] = static_cast<double>(aim_point.xyza[i]);

      auto & first_target = targets.front();
      auto image_points =
        solver.reproject_armor(aim_xyza.head<3>(), aim_xyza[3], first_target.armor_type, first_target.name);
      tools::draw_points(img, image_points, {0, 255, 0});
    }

    // -------------------- 绘图 --------------------
    nlohmann::json data;
    data["gimbal_yaw"] = euler_deg[0];
    data["cmd_yaw"] = command.yaw * 57.3;
    data["shoot"] = command.shoot;
    data["gimbal_pitch"] = euler_deg[2];
    data["cmd_pitch"] = command.pitch * 57.3;
    data["track_state"] = tracker_state;
    data["control"] = command.control;
    data["bullet_speed"] = bullet_speed;

    plotter.plot(data);

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("reprojection", img);
    if (cv::waitKey(1) == 'q') break;
  }

  return 0;
}

// [WARN] ekf_x() marker not found, debug snippet not auto-inserted.
