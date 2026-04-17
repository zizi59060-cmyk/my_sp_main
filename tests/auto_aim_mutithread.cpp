#include <fmt/core.h>
#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"  // ✅ 改为多线程检测器
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
    io::Camera camera(config_path);
    auto_aim::multithread::MultiThreadDetector detector(config_path);
    auto_aim::Solver solver(config_path);
    auto_aim::Tracker tracker(config_path, solver);
    auto_aim::Aimer aimer(config_path);
    io::Gimbal gimbal(config_path);

    // -------------------- 检测线程 --------------------
    std::thread detect_thread([&]() {
        tools::logger()->info("[detect_thread] started.");
        cv::Mat img;
        while (!exiter.exit()) {
            auto t = std::chrono::steady_clock::now();
            camera.read(img, t);
            if (img.empty()) continue;
            detector.push(img, t);  // 推入检测队列
        }
        tools::logger()->info("[detect_thread] exited.");
    });

    // -------------------- 主线程逻辑 --------------------
    double fps = 0.0;
    int frame_count = 0;
    auto last_frame_time = std::chrono::steady_clock::now();
    constexpr double fps_update_interval = 1.0;

    while (!exiter.exit()) {
        // 从 detector 获取结果
        auto [img, armors, t] = detector.debug_pop();
        if (img.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // 云台姿态
        Eigen::Quaterniond gimbal_q = gimbal.q(t);
        solver.set_R_gimbal2world(gimbal_q);

        // -------------------- 自瞄主流程 --------------------
        auto targets = tracker.track(armors, t);
        auto command = aimer.aim(targets, t, 27, false);

        // 串口发送
        gimbal.send(command.control, command.shoot, command.yaw, 0, 0, command.pitch, 0, 0);

        // -------------------- FPS计算 --------------------
        frame_count++;
        double elapsed = tools::delta_time(std::chrono::steady_clock::now(), last_frame_time);
        if (elapsed >= fps_update_interval) {
            fps = frame_count / elapsed;
            frame_count = 0;
            last_frame_time = std::chrono::steady_clock::now();
            tools::logger()->info("FPS: {:.1f}, Tracker State: {}", fps, tracker.state());
        }

        // -------------------- 可视化 --------------------
        Eigen::Vector3d euler = tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3;

        cv::Scalar state_color;
        std::string tracker_state = tracker.state();
        if (tracker_state == "tracking") state_color = {0, 255, 0};
        else if (tracker_state == "temp_lost") state_color = {0, 255, 255};
        else if (tracker_state == "detecting") state_color = {255, 255, 0};
        else state_color = {0, 0, 255};

        tools::draw_text(img, fmt::format("State: {}", tracker_state), {10, 30}, state_color);
        tools::draw_text(img, fmt::format("FPS: {:.1f}", fps), {10, 60}, {255, 255, 255});
        tools::draw_text(img, fmt::format("Shoot: {}", command.shoot), {10, 90}, {154, 50, 205});

        cv::resize(img, img, {}, 0.5, 0.5);
        cv::imshow("debug_reprojection", img);
        if (cv::waitKey(1) == 'q') break;
    }

    // -------------------- 安全退出 --------------------
    if (detect_thread.joinable()) detect_thread.join();
    tools::logger()->info("Main exited safely.");
    return 0;
}
