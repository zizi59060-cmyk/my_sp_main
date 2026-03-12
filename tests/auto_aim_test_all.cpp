#include <fmt/core.h>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"

#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明 }"
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

  tools::Exiter exiter;
  tools::Plotter plotter;

  // -------------------- 模块初始化 --------------------
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);

  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point frame_start;

  io::Command last_command{false, false, 0.f, 0.f};

  while (!exiter.exit()) {
    // 1) 读相机
    camera.read(img, frame_start);
    if (img.empty()) continue;

    // 2) 读云台姿态 + 云台状态（包含下位机弹速）
    Eigen::Quaterniond gimbal_q = gimbal.q(frame_start);
    solver.set_R_gimbal2world(gimbal_q);

    io::GimbalState gs = gimbal.state();  // ✅ 下位机回传状态
    double bullet_speed = static_cast<double>(gs.bullet_speed);  // ✅ 读取下位机弹速

    // 弹速兜底（防止串口刚启动时为0）
    if (!std::isfinite(bullet_speed) || bullet_speed < 1.0) bullet_speed = 27.0;

    // 3) YOLO检测
    auto armors = yolo.detect(img, 0);

    // 4) 跟踪
    auto targets = tracker.track(armors, frame_start);

    // 5) 瞄准（把弹速传进去）
    auto command = aimer.aim(targets, frame_start, bullet_speed, false);

    // 6) 一个简单的“接近上一帧”射击条件
    if (!targets.empty() && aimer.debug_aim_point.valid &&
        std::abs(command.yaw - last_command.yaw) * 57.3 < 1.5) {
      command.shoot = true;
    }


    if (command.control) last_command = command;

    // 7) 串口发送给下位机
    gimbal.send(
      command.control,
      command.shoot,
      static_cast<float>(command.yaw),
      0.f, 0.f,
      static_cast<float>(command.pitch),
      0.f, 0.f
    );

    // -------------------- 可视化/调试 --------------------
    Eigen::Vector3d euler_deg =
      tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3;

    tools::draw_text(
      img,
      fmt::format("bullet_speed(from gimbal): {:.2f}", bullet_speed),
      {10, 30}, {255, 255, 255});

    tools::draw_text(
      img,
      fmt::format("gimbal yaw:{:.2f} pitch:{:.2f}", euler_deg[0], euler_deg[1]),
      {10, 60}, {255, 255, 255});

    tools::draw_text(
      img,
      fmt::format("cmd yaw:{:.2f} pitch:{:.2f} shoot:{}",
                  command.yaw * 57.3, command.pitch * 57.3, command.shoot),
      {10, 90}, {154, 50, 205});

    nlohmann::json data;
    data["bullet_speed"] = bullet_speed;
    data["gimbal_yaw"] = euler_deg[0];
    data["gimbal_pitch"] = euler_deg[1];
    data["cmd_yaw"] = command.yaw * 57.3;
    data["cmd_pitch"] = command.pitch * 57.3;
    data["shoot"] = command.shoot;
    data["state"] = tracker.state();

    // =============== 新增 DEBUG 逻辑 (仿照 src/mt_auto_aim_debug.cpp) ===============
    if (!targets.empty()) {
      auto target = targets.front();

      // 1. 绘制当前跟踪目标的装甲板 (绿色)
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      // 2. 绘制 Aim Point (预测打击点)
      // 有效为红色 {0, 0, 255}, 无效为蓝色 {255, 0, 0}
      auto aim_point = aimer.debug_aim_point;
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      // 只有当计算出的点非零时才绘制，避免干扰
      if (aim_xyza != Eigen::Vector4d::Zero()) {
        auto image_points =
          solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
        if (aim_point.valid)
          tools::draw_points(img, image_points, {0, 0, 255});
        else
          tools::draw_points(img, image_points, {255, 0, 0});
      }

      // 3. 记录详细的 EKF 状态数据到 Plotter
      Eigen::VectorXd x = target.ekf_x();
      data["x"] = x[0];
      data["vx"] = x[1];
      data["y"] = x[2];
      data["vy"] = x[3];
      data["z"] = x[4];
      data["vz"] = x[5];
      data["a"] = x[6] * 57.3;
      data["w"] = x[7];
      data["r"] = x[8];
      data["l"] = x[9];
      data["h"] = x[10];
      data["last_id"] = target.last_id;
    }
    // ==============================================================================

    plotter.plot(data);

    cv::resize(img, img, {}, 0.5, 0.5); // 显示缩小，防止屏幕放不下
    cv::imshow("auto_aim_test_all", img);
    if (cv::waitKey(1) == 'q') break;
  }

  // 安全退出：停止控制/开火
  gimbal.send(false, false, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
  return 0;
}