#include <fmt/core.h>
#include <chrono>
#include <optional>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
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
    auto_aim::Planner planner(config_path);
    io::Gimbal gimbal(config_path);  // 串口通信对象

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
        auto gs = gimbal.state();

        // -------------------- 自瞄核心 --------------------
        auto armors = yolo.detect(img, 0);
        auto targets = tracker.track(armors, frame_start);

        std::optional<auto_aim::Target> target = std::nullopt;
        if (!targets.empty()) {
            target = targets.front();
        }

        // 使用下位机发送的弹速作为最终弹速，并启用 Planner(MPC) 输出
        auto plan = planner.plan(target, gs.bullet_speed);

        // -------------------- 串口发送控制命令 --------------------
        gimbal.send(
            plan.control,
            plan.fire,
            plan.yaw,
            plan.yaw_vel,
            plan.yaw_acc,
            plan.pitch,
            plan.pitch_vel,
            plan.pitch_acc);

        // -------------------- 帧率计算 --------------------
        auto frame_end = std::chrono::steady_clock::now();
        frame_count++;
        double elapsed_time = tools::delta_time(frame_end, last_frame_time);

        if (elapsed_time >= fps_update_interval) {
            fps = frame_count / elapsed_time;
            frame_count = 0;
            last_frame_time = frame_end;
            tools::logger()->info(
              "Current FPS: {:.1f}, Bullet Speed: {:.2f}, Tracker State: {}",
              fps,
              gs.bullet_speed,
              tracker.state());
        }

    //     // -------------------- 可视化 --------------------
    //     Eigen::Vector3d euler_deg = tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3;

    //     cv::Scalar state_color;
    //     std::string tracker_state = tracker.state();
    //     if (tracker_state == "tracking") state_color = {0, 255, 0};
    //     else if (tracker_state == "temp_lost") state_color = {0, 255, 255};
    //     else if (tracker_state == "detecting") state_color = {255, 255, 0};
    //     else state_color = {0, 0, 255};

    //     tools::draw_text(img, fmt::format("State: {}", tracker_state), {10, 30}, state_color);
    //     tools::draw_text(
    //       img,
    //       fmt::format("gimbal yaw:{:.2f}, pitch:{:.2f}", euler_deg[0], euler_deg[2]),
    //       {10, 60},
    //       {255, 255, 255});
    //     tools::draw_text(
    //       img,
    //       fmt::format("bullet speed:{:.2f} m/s", gs.bullet_speed),
    //       {10, 90},
    //       {255, 255, 255});
    //     tools::draw_text(
    //       img,
    //       fmt::format(
    //         "plan yaw:{:.2f}, pitch:{:.2f}, fire:{}, control:{}",
    //         plan.yaw * 57.3,
    //         plan.pitch * 57.3,
    //         plan.fire,
    //         plan.control),
    //       {10, 120},
    //       {154, 50, 205});
    //     tools::draw_text(
    //       img,
    //       fmt::format(
    //         "plan yaw_vel:{:.2f}, yaw_acc:{:.2f}, pitch_vel:{:.2f}, pitch_acc:{:.2f}",
    //         plan.yaw_vel,
    //         plan.yaw_acc,
    //         plan.pitch_vel,
    //         plan.pitch_acc),
    //       {10, 150},
    //       {154, 50, 205});

    //     // -------------------- 绘图 --------------------
    //     nlohmann::json data;
    //     data["gimbal_yaw"] = euler_deg[0];
    //     data["plan_yaw"] = plan.yaw * 57.3;
    //     data["plan_yaw_vel"] = plan.yaw_vel;
    //     data["plan_yaw_acc"] = plan.yaw_acc;
    //     data["fire"] = plan.fire;
    //     data["gimbal_pitch"] = euler_deg[2];
    //     data["plan_pitch"] = plan.pitch * 57.3;
    //     data["plan_pitch_vel"] = plan.pitch_vel;
    //     data["plan_pitch_acc"] = plan.pitch_acc;
    //     data["track_state"] = tracker_state;
    //     data["control"] = plan.control;
    //     data["bullet_speed"] = gs.bullet_speed;

    //     plotter.plot(data);

    //     cv::resize(img, img, {}, 0.5, 0.5);
    //     cv::imshow("reprojection", img);
    //     if (cv::waitKey(1) == 'q') break;
    }

    return 0;
}