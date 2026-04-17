#include <fmt/core.h>

#include <chrono>
#include <cmath>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>


#include "io/gimbal/gimbal.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/crc.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | /home/wheeltec/projects/my_sp_main/configs/demo.yaml | yaml配置文件路径 }";

int main(int argc, char * argv[])
{

  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");

  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;

  // -------------------- 初始化云台 --------------------
  io::Gimbal gimbal(config_path);
  tools::logger()->info("[Main] Gimbal test (no fire) started.");

  // -------------------- 参数定义 --------------------
  float yaw = 0.0f;
  float pitch = 0.0f;
  bool control = true;
  int step = 0;

  auto t0 = std::chrono::steady_clock::now();

  // -------------------- 主循环 --------------------
  while (!exiter.exit()) {
    // 构造指令：周期性摆动
    yaw =10.0f * std::sin(step * 0.1f);   // 左右摆动 ±10°
    pitch = 5.0f * std::cos(step * 0.1f);  // 上下摆动 ±5°
    step++;

    gimbal.send(control, false, yaw, 0, 0, pitch, 0, 0);

    auto t_now = std::chrono::steady_clock::now();
    auto state = gimbal.state();
    auto q = gimbal.q(t_now);
    auto euler = tools::eulers(q, 2, 1, 0) * 57.3;

    fmt::print("[Send] yaw={:.2f}, pitch={:.2f}\n", yaw, pitch);
    fmt::print(
      "[Recv] yaw={:.2f}, pitch={:.2f}, vyaw={:.2f}, vpitch={:.2f}\n", state.yaw, state.pitch,
      state.yaw_vel, state.pitch_vel);

    std::this_thread::sleep_for(100ms);
  }

  // -------------------- 程序退出：停止控制 --------------------
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
  tools::logger()->info("[Main] Gimbal test stopped.");

  return 0;
}
