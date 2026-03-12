#include <fmt/core.h>

#include <chrono>
#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

int main(int argc, char *argv[])
{
  // 初始化日志和退出控制器
  tools::Exiter exiter;
  
  // 配置文件路径 - 根据你的实际路径调整
  std::string config_path = "configs/demo.yaml";
  
  // 初始化各个模块
  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  
  // 打开摄像头
  cv::VideoCapture cap(0); // 0表示默认摄像头
  if (!cap.isOpened()) {
    tools::logger()->error("无法打开摄像头");
    return -1;
  }
  
  // 设置摄像头参数（根据你的摄像头调整）
  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  cap.set(cv::CAP_PROP_FPS, 30);
  
  cv::Mat frame;
  auto start_time = std::chrono::steady_clock::now();
  
  // 用于记录帧率和时间
  int frame_count = 0;
  double last_t = -1;
  
  // 用于存储命令
  io::Command last_command;
  
  // 固定四元数（因为没有IMU数据，使用默认值）
  Eigen::Quaterniond fixed_quaternion(1, 0, 0, 0);
  
  while (!exiter.exit()) {
    // 读取摄像头帧
    cap >> frame;
    if (frame.empty()) {
      tools::logger()->error("无法从摄像头读取帧");
      break;
    }
    
    // 计算时间戳（模拟）
    auto now = std::chrono::steady_clock::now();
    double timestamp = std::chrono::duration<double>(now - start_time).count();
    
    // 设置固定的云台姿态（因为没有IMU数据）
    solver.set_R_gimbal2world({fixed_quaternion.w(), fixed_quaternion.x(), 
                              fixed_quaternion.y(), fixed_quaternion.z()});
    
    // 自瞄核心逻辑
    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(frame, frame_count);
    
    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, now);
    
    auto aimer_start = std::chrono::steady_clock::now();
    auto command = aimer.aim(targets, now, 27, false);
    
    // 简单的射击判断逻辑
    if (!targets.empty() && aimer.debug_aim_point.valid &&
        std::abs(command.yaw - last_command.yaw) * 57.3 < 2) {
      command.shoot = true;
    }
    
    if (command.control) last_command = command;
    
    // 计算处理时间
    auto finish = std::chrono::steady_clock::now();
    double yolo_time = std::chrono::duration<double>(tracker_start - yolo_start).count() * 1000;
    double tracker_time = std::chrono::duration<double>(aimer_start - tracker_start).count() * 1000;
    double aimer_time = std::chrono::duration<double>(finish - aimer_start).count() * 1000;
    double total_time = std::chrono::duration<double>(finish - yolo_start).count() * 1000;
    
    // 计算帧率
    double fps = 1.0 / (timestamp - last_t);
    if (last_t < 0) fps = 0;
    last_t = timestamp;
    
    // 在图像上绘制信息
    tools::draw_text(
      frame,
      fmt::format(
        "FPS: {:.1f} | YOLO: {:.1f}ms | Tracker: {:.1f}ms | Aimer: {:.1f}ms | Total: {:.1f}ms",
        fps, yolo_time, tracker_time, aimer_time, total_time),
      {10, 30}, {0, 255, 0});
    
    tools::draw_text(
      frame,
      fmt::format(
        "Command: Yaw {:.2f}° | Pitch {:.2f}° | Shoot: {}",
        command.yaw * 57.3, command.pitch * 57.3, command.shoot ? "YES" : "NO"),
      {10, 60}, {0, 255, 255});
    
    // 绘制检测到的装甲板中心点
    for (const auto& armor : armors) {
      // 绘制装甲板中心点
      cv::circle(frame, cv::Point2f(
        armor.center_norm.x * frame.cols,
        armor.center_norm.y * frame.rows), 5, cv::Scalar(0, 0, 255), -1);
      
      // 显示装甲板信息
      std::string label = fmt::format("Conf: {:.2f}", armor.confidence);
      cv::putText(frame, label, 
                  cv::Point2f(armor.center_norm.x * frame.cols, 
                             armor.center_norm.y * frame.rows - 10), 
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
    }
    
    // 显示图像
    cv::imshow("Armor Detection", frame);
    
    // 按键处理
    int key = cv::waitKey(1);
    if (key == 'q' || key == 27) { // 'q' 或 ESC 键退出
      break;
    } else if (key == ' ') { // 空格键暂停
      cv::waitKey(0);
    }
    
    frame_count++;
  }
  
  // 释放资源
  cap.release();
  cv::destroyAllWindows();
  
  return 0;
}
