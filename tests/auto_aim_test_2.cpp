#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
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

namespace
{

struct ExpertVizConfig
{
  bool use_expert_visualization = true;
  bool show_detection_debug = true;
  bool show_selected_detected_armor = true;
  bool show_world_panel = true;
  bool show_plotter_expert_channels = true;

  int world_panel_size = 300;      // 面板边长
  double world_panel_scale = 180;  // 像素/米
};

ExpertVizConfig load_expert_viz_config(const std::string & config_path)
{
  ExpertVizConfig cfg;
  auto yaml = YAML::LoadFile(config_path);

  if (yaml["use_expert_visualization"])
    cfg.use_expert_visualization = yaml["use_expert_visualization"].as<bool>();
  if (yaml["show_detection_debug"])
    cfg.show_detection_debug = yaml["show_detection_debug"].as<bool>();
  if (yaml["show_selected_detected_armor"])
    cfg.show_selected_detected_armor = yaml["show_selected_detected_armor"].as<bool>();
  if (yaml["show_world_panel"])
    cfg.show_world_panel = yaml["show_world_panel"].as<bool>();
  if (yaml["show_plotter_expert_channels"])
    cfg.show_plotter_expert_channels = yaml["show_plotter_expert_channels"].as<bool>();
  if (yaml["world_panel_size"])
    cfg.world_panel_size = yaml["world_panel_size"].as<int>();
  if (yaml["world_panel_scale"])
    cfg.world_panel_scale = yaml["world_panel_scale"].as<double>();

  return cfg;
}

int choose_target_armor_idx(const auto_aim::Target & target)
{
  const auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) return -1;

  int best_idx = 0;
  double best_dist = std::numeric_limits<double>::infinity();

  for (int i = 0; i < static_cast<int>(armor_xyza_list.size()); ++i) {
    double d = armor_xyza_list[i].head<2>().norm();
    if (d < best_dist) {
      best_dist = d;
      best_idx = i;
    }
  }
  return best_idx;
}

std::optional<cv::Point2f> find_selected_detected_armor_center(
  const std::list<auto_aim::Armor> & armors,
  const std::optional<auto_aim::Target> & target_opt)
{
  if (!target_opt.has_value()) return std::nullopt;

  const auto & target = target_opt.value();
  const auto armor_xyza_list = target.armor_xyza_list();
  const int selected_idx = choose_target_armor_idx(target);
  if (selected_idx < 0) return std::nullopt;

  const auto selected_xyza = armor_xyza_list[selected_idx];
  const Eigen::Vector3d selected_xyz = selected_xyza.head<3>();
  const double selected_yaw = selected_xyza[3];

  bool found = false;
  cv::Point2f best_center;
  double best_score = std::numeric_limits<double>::infinity();

  for (const auto & armor : armors) {
    // 这里假设 armor 已经被 tracker 过程中 solve 过，至少目标相关装甲板是有世界坐标的
    // 如果你的 Armor 字段名不同，只需要把 xyz_in_world / ypr_in_world 改成你仓库里的实际字段名。
    if (armor.name != target.name || armor.type != target.armor_type) continue;

    const Eigen::Vector3d armor_xyz = armor.xyz_in_world;
    const double armor_yaw = armor.ypr_in_world[0];

    const double pos_err = (armor_xyz - selected_xyz).norm();
    const double yaw_err = std::abs(tools::limit_rad(armor_yaw - selected_yaw));
    const double score = pos_err + 0.2 * yaw_err;

    if (score < best_score) {
      best_score = score;
      best_center = armor.center;
      found = true;
    }
  }

  if (!found) return std::nullopt;
  return best_center;
}

void draw_detected_armors(
  cv::Mat & img,
  const std::list<auto_aim::Armor> & armors,
  const std::optional<auto_aim::Target> & target_opt,
  const ExpertVizConfig & cfg)
{
  if (!cfg.use_expert_visualization || !cfg.show_detection_debug) return;

  auto selected_center = find_selected_detected_armor_center(armors, target_opt);

  for (const auto & armor : armors) {
    cv::Scalar color(120, 180, 255);
    int radius = 6;
    int thickness = 2;

    if (selected_center.has_value() && cv::norm(armor.center - selected_center.value()) < 3.0) {
      color = cv::Scalar(0, 0, 255);
      radius = 10;
      thickness = 3;

      cv::circle(img, armor.center, 18, cv::Scalar(0, 255, 255), 2);
      cv::line(
        img,
        cv::Point2f(img.cols / 2.0f, img.rows / 2.0f),
        armor.center,
        cv::Scalar(0, 255, 255), 1);
      tools::draw_text(
        img,
        "SELECTED ARMOR",
        cv::Point2f(armor.center.x + 15, armor.center.y - 15),
        cv::Scalar(0, 255, 255));
    }

    cv::circle(img, armor.center, radius, color, thickness);
    tools::draw_text(
      img,
      fmt::format("P{}", static_cast<int>(armor.priority)),
      cv::Point2f(armor.center.x + 8, armor.center.y + 8),
      color);
  }
}

void overlay_panel(cv::Mat & img, const cv::Mat & panel, int x, int y)
{
  if (panel.empty()) return;
  if (x >= img.cols || y >= img.rows) return;

  const int w = std::min(panel.cols, img.cols - x);
  const int h = std::min(panel.rows, img.rows - y);
  if (w <= 0 || h <= 0) return;

  cv::Mat roi = img(cv::Rect(x, y, w, h));
  cv::Mat panel_crop = panel(cv::Rect(0, 0, w, h));

  cv::Mat blended;
  cv::addWeighted(roi, 0.25, panel_crop, 0.75, 0.0, blended);
  blended.copyTo(roi);
}

void draw_world_panel(
  cv::Mat & img,
  const std::optional<auto_aim::Target> & target_opt,
  const auto_aim::Plan & plan,
  const std::string & tracker_state,
  const ExpertVizConfig & cfg)
{
  if (!cfg.use_expert_visualization || !cfg.show_world_panel) return;
  if (!target_opt.has_value()) return;

  const auto & target = target_opt.value();
  const auto ekf_x = target.ekf_x();
  const auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) return;

  const int S = cfg.world_panel_size;
  cv::Mat panel(S, S, CV_8UC3, cv::Scalar(28, 28, 28));

  cv::rectangle(panel, cv::Rect(0, 0, S - 1, S - 1), cv::Scalar(180, 180, 180), 1);

  const cv::Point2f panel_center(S / 2.0f, S / 2.0f);
  const double scale = cfg.world_panel_scale;

  auto world_to_panel = [&](double x, double y) -> cv::Point2f {
    // 世界系 x 向右，y 向上（面板中上方是更远处）
    return cv::Point2f(
      static_cast<float>(panel_center.x + x * scale),
      static_cast<float>(panel_center.y - y * scale));
  };

  // 坐标轴
  cv::line(panel, cv::Point2f(0, panel_center.y), cv::Point2f(S, panel_center.y), cv::Scalar(60, 60, 60), 1);
  cv::line(panel, cv::Point2f(panel_center.x, 0), cv::Point2f(panel_center.x, S), cv::Scalar(60, 60, 60), 1);

  // 原点（枪口/云台近似位置）
  cv::circle(panel, panel_center, 4, cv::Scalar(255, 255, 255), -1);
  tools::draw_text(panel, "GIMBAL", cv::Point2f(panel_center.x + 6, panel_center.y - 6), cv::Scalar(255, 255, 255));

  // 目标中心
  const double cx = ekf_x[0];
  const double cy = ekf_x[2];
  const double vx = ekf_x[1];
  const double vy = ekf_x[3];
  const double angle = ekf_x[6];
  const double w = ekf_x[7];
  const double r = ekf_x[8];
  const double l = ekf_x[9];
  const double h = ekf_x[10];

  const cv::Point2f cpt = world_to_panel(cx, cy);
  cv::circle(panel, cpt, 6, cv::Scalar(0, 255, 0), -1);
  tools::draw_text(panel, "CENTER", cv::Point2f(cpt.x + 8, cpt.y - 8), cv::Scalar(0, 255, 0));

  // 中心速度箭头
  cv::arrowedLine(
    panel,
    cpt,
    world_to_panel(cx + vx * 0.2, cy + vy * 0.2),
    cv::Scalar(0, 255, 255),
    2,
    cv::LINE_AA,
    0,
    0.2);

  // 朝向箭头（整车相位）
  cv::arrowedLine(
    panel,
    cpt,
    world_to_panel(cx + 0.15 * std::cos(angle), cy + 0.15 * std::sin(angle)),
    cv::Scalar(255, 0, 255),
    2,
    cv::LINE_AA,
    0,
    0.2);

  const int selected_idx = choose_target_armor_idx(target);

  for (int i = 0; i < static_cast<int>(armor_xyza_list.size()); ++i) {
    const auto & xyza = armor_xyza_list[i];
    const cv::Point2f apt = world_to_panel(xyza[0], xyza[1]);

    cv::Scalar color = (i == selected_idx) ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 200, 0);
    int radius = (i == selected_idx) ? 8 : 5;
    int thickness = (i == selected_idx) ? -1 : 2;

    cv::circle(panel, apt, radius, color, thickness);
    cv::line(panel, cpt, apt, cv::Scalar(90, 90, 90), 1);
    tools::draw_text(panel, fmt::format("#{}", i), cv::Point2f(apt.x + 6, apt.y - 6), color);
  }

  // 状态文本
  tools::draw_text(panel, fmt::format("STATE: {}", tracker_state), cv::Point2f(10, 20), cv::Scalar(255, 255, 255));
  tools::draw_text(panel, fmt::format("FIRE: {}", plan.fire), cv::Point2f(10, 42), plan.fire ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255));
  tools::draw_text(panel, fmt::format("cx={:.2f} cy={:.2f}", cx, cy), cv::Point2f(10, 66), cv::Scalar(255, 255, 255));
  tools::draw_text(panel, fmt::format("vx={:.2f} vy={:.2f}", vx, vy), cv::Point2f(10, 88), cv::Scalar(255, 255, 255));
  tools::draw_text(panel, fmt::format("a={:.2f} w={:.2f}", angle, w), cv::Point2f(10, 110), cv::Scalar(255, 255, 255));
  tools::draw_text(panel, fmt::format("r={:.2f} l={:.2f} h={:.2f}", r, l, h), cv::Point2f(10, 132), cv::Scalar(255, 255, 255));
  tools::draw_text(panel, fmt::format("sel={}", selected_idx), cv::Point2f(10, 154), cv::Scalar(0, 255, 255));

  if (selected_idx >= 0) {
    const auto & sel = armor_xyza_list[selected_idx];
    tools::draw_text(
      panel,
      fmt::format("tx={:.2f} ty={:.2f} tz={:.2f}", sel[0], sel[1], sel[2]),
      cv::Point2f(10, 176),
      cv::Scalar(0, 255, 255));
    tools::draw_text(
      panel,
      fmt::format("ta={:.2f}", sel[3]),
      cv::Point2f(10, 198),
      cv::Scalar(0, 255, 255));
  }

  overlay_panel(img, panel, img.cols - S - 10, 10);
}

}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  ExpertVizConfig viz_cfg = load_expert_viz_config(config_path);

  tools::Plotter plotter;
  tools::Exiter exiter;

  io::Camera camera(config_path);
  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);
  io::Gimbal gimbal(config_path);

  cv::Mat img;
  double fps = 0.0;
  auto last_frame_time = std::chrono::steady_clock::now();
  int frame_count = 0;
  constexpr double fps_update_interval = 1.0;

  while (!exiter.exit()) {
    auto frame_start = std::chrono::steady_clock::now();

    camera.read(img, frame_start);
    if (img.empty()) break;

    Eigen::Quaterniond gimbal_q = gimbal.q(frame_start);
    solver.set_R_gimbal2world(gimbal_q);
    auto gs = gimbal.state();

    auto armors = yolo.detect(img, 0);
    auto targets = tracker.track(armors, frame_start);

    std::optional<auto_aim::Target> target = std::nullopt;
    if (!targets.empty()) {
      target = targets.front();
    }

    auto plan = planner.plan(target, gs.bullet_speed);

    gimbal.send(
      plan.control,
      plan.fire,
      plan.yaw,
      plan.yaw_vel,
      plan.yaw_acc,
      plan.pitch,
      plan.pitch_vel,
      plan.pitch_acc);

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

    // ==================== 文本信息 ====================
    Eigen::Vector3d euler_deg = tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3;

    cv::Scalar state_color;
    std::string tracker_state = tracker.state();
    if (tracker_state == "tracking") state_color = {0, 255, 0};
    else if (tracker_state == "temp_lost") state_color = {0, 255, 255};
    else if (tracker_state == "detecting") state_color = {255, 255, 0};
    else state_color = {0, 0, 255};

    tools::draw_text(img, fmt::format("State: {}", tracker_state), {10, 30}, state_color);
    tools::draw_text(
      img,
      fmt::format("gimbal yaw:{:.2f}, pitch:{:.2f}", euler_deg[0], euler_deg[2]),
      {10, 60},
      {255, 255, 255});
    tools::draw_text(
      img,
      fmt::format("bullet speed:{:.2f} m/s", gs.bullet_speed),
      {10, 90},
      {255, 255, 255});
    tools::draw_text(
      img,
      fmt::format(
        "plan yaw:{:.2f}, pitch:{:.2f}, fire:{}, control:{}",
        plan.yaw * 57.3,
        plan.pitch * 57.3,
        plan.fire,
        plan.control),
      {10, 120},
      {154, 50, 205});
    tools::draw_text(
      img,
      fmt::format(
        "plan yaw_vel:{:.2f}, yaw_acc:{:.2f}, pitch_vel:{:.2f}, pitch_acc:{:.2f}",
        plan.yaw_vel,
        plan.yaw_acc,
        plan.pitch_vel,
        plan.pitch_acc),
      {10, 150},
      {154, 50, 205});
    tools::draw_text(
      img,
      fmt::format("FPS:{:.1f}  Armors:{}", fps, armors.size()),
      {10, 180},
      {255, 255, 255});

    // ==================== 待击打装甲板 + 检测调试绘图 ====================
    draw_detected_armors(img, armors, target, viz_cfg);

    // ==================== 专家版世界面板 ====================
    draw_world_panel(img, target, plan, tracker_state, viz_cfg);

    // ==================== plotter 绘图 ====================
    nlohmann::json data;
    data["gimbal_yaw"] = euler_deg[0];
    data["plan_yaw"] = plan.yaw * 57.3;
    data["plan_yaw_vel"] = plan.yaw_vel;
    data["plan_yaw_acc"] = plan.yaw_acc;
    data["fire"] = plan.fire;
    data["gimbal_pitch"] = euler_deg[2];
    data["plan_pitch"] = plan.pitch * 57.3;
    data["plan_pitch_vel"] = plan.pitch_vel;
    data["plan_pitch_acc"] = plan.pitch_acc;
    data["track_state"] = tracker_state;
    data["control"] = plan.control;
    data["bullet_speed"] = gs.bullet_speed;

    if (viz_cfg.show_plotter_expert_channels && target.has_value()) {
      const auto ekf_x = target->ekf_x();
      data["target_center_x"] = ekf_x[0];
      data["target_center_vx"] = ekf_x[1];
      data["target_center_y"] = ekf_x[2];
      data["target_center_vy"] = ekf_x[3];
      data["target_center_z"] = ekf_x[4];
      data["target_center_vz"] = ekf_x[5];
      data["target_angle"] = ekf_x[6];
      data["target_w"] = ekf_x[7];
      data["target_r"] = ekf_x[8];
      data["target_l"] = ekf_x[9];
      data["target_h"] = ekf_x[10];

      const auto armor_xyza_list = target->armor_xyza_list();
      const int selected_idx = choose_target_armor_idx(*target);
      data["selected_idx"] = selected_idx;

      if (selected_idx >= 0 && selected_idx < static_cast<int>(armor_xyza_list.size())) {
        data["selected_armor_x"] = armor_xyza_list[selected_idx][0];
        data["selected_armor_y"] = armor_xyza_list[selected_idx][1];
        data["selected_armor_z"] = armor_xyza_list[selected_idx][2];
        data["selected_armor_yaw"] = armor_xyza_list[selected_idx][3];
      }
    }

    plotter.plot(data);

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("reprojection", img);
    if (cv::waitKey(1) == 'q') break;
  }

  return 0;
}