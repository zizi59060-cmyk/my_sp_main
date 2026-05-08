#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <Eigen/Geometry>
#include <fmt/core.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"

#include "tasks/auto_buff/jlu_buff/jlu_buff.hpp"

#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"

const std::string keys =
  "{help h usage ? | | print usage}"
  "{@config-path | configs/standard3.yaml | yaml config path}"
  "{send-control | false | actually send control command to gimbal}"
  "{allow-fire | false | allow fire flag to be sent; default false}"
  "{force-mode | auto | auto/small/big; auto means use gimbal mode}"
  "{show-scale | 0.5 | display resize scale}"
  "{plot-host | 127.0.0.1 | PlotJuggler UDP host}"
  "{plot-port | 9870 | PlotJuggler UDP port}";

namespace
{

std::string gimbalModeString(io::GimbalMode mode)
{
  switch (mode) {
    case io::GimbalMode::IDLE:
      return "IDLE";
    case io::GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case io::GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case io::GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "UNKNOWN";
  }
}

bool gimbalModeToBuffMode(io::GimbalMode gimbal_mode, auto_buff::jlu::BuffMode & out)
{
  if (gimbal_mode == io::GimbalMode::SMALL_BUFF) {
    out = auto_buff::jlu::BuffMode::SMALL;
    return true;
  }

  if (gimbal_mode == io::GimbalMode::BIG_BUFF) {
    out = auto_buff::jlu::BuffMode::BIG;
    return true;
  }

  return false;
}

void putText(
  cv::Mat & img,
  const std::string & text,
  int x,
  int y,
  const cv::Scalar & color = cv::Scalar(255, 255, 255),
  double scale = 0.65,
  int thickness = 2)
{
  cv::putText(img, text, {x, y}, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness);
}

cv::Scalar pointColor(int i)
{
  static const cv::Scalar colors[] = {
    {0, 255, 255},  // R
    {0, 0, 255},    // BR
    {0, 128, 255},  // TR
    {255, 0, 0},    // TL
    {255, 0, 255},  // BL
  };

  return colors[i % 5];
}

void drawBlade(cv::Mat & img, const auto_buff::jlu::BuffBlade & blade, int index)
{
  cv::rectangle(img, blade.rect, {0, 255, 0}, 2);

  std::ostringstream label;
  label << "#" << index
        << " conf=" << std::fixed << std::setprecision(2) << blade.confidence
        << " roll=" << std::setprecision(2) << blade.roll
        << " radius=" << std::setprecision(2)
        << (blade.position_world - blade.center_world).norm();

  putText(
    img,
    label.str(),
    static_cast<int>(blade.rect.x),
    std::max(20, static_cast<int>(blade.rect.y) - 8),
    {0, 255, 0},
    0.55,
    2);

  static const char * names[] = {"R", "BR", "TR", "TL", "BL"};

  for (int i = 0; i < auto_buff::jlu::kBuffBladePointCount; ++i) {
    const auto & p = blade.points.image[i];

    cv::circle(img, p, 5, pointColor(i), -1);

    putText(
      img,
      names[i],
      static_cast<int>(p.x + 6),
      static_cast<int>(p.y - 6),
      pointColor(i),
      0.45,
      1);
  }

  const auto & pts = blade.points.image;

  // JLU five-point order:
  // 0 R center, 1 bottom_right, 2 top_right, 3 top_left, 4 bottom_left
  cv::line(img, pts[1], pts[2], {255, 255, 0}, 2);
  cv::line(img, pts[2], pts[3], {255, 255, 0}, 2);
  cv::line(img, pts[3], pts[4], {255, 255, 0}, 2);
  cv::line(img, pts[4], pts[1], {255, 255, 0}, 2);
  cv::line(img, pts[0], (pts[1] + pts[4]) * 0.5F, {0, 255, 255}, 2);
}

void drawAimPoint(cv::Mat & img, const auto_buff::jlu::JluBuffDebug & dbg)
{
  if (!dbg.aim_point_image_valid) {
    return;
  }

  const cv::Point2f p = dbg.aim_point_image;

  if (p.x < 0 || p.y < 0 || p.x >= img.cols || p.y >= img.rows) {
    return;
  }

  const cv::Point center(static_cast<int>(p.x), static_cast<int>(p.y));

  // 红色大十字：trajectory 最终瞄准点
  cv::line(img, center + cv::Point(-18, 0), center + cv::Point(18, 0), {0, 0, 255}, 3);
  cv::line(img, center + cv::Point(0, -18), center + cv::Point(0, 18), {0, 0, 255}, 3);
  cv::circle(img, center, 8, {0, 0, 255}, 2);

  putText(
    img,
    "AIM",
    center.x + 12,
    center.y - 12,
    {0, 0, 255},
    0.65,
    2);
}

void drawHud(
  cv::Mat & img,
  const auto_buff::jlu::JluBuffDebug & dbg,
  io::GimbalMode gimbal_mode,
  const io::GimbalState & gs,
  const Eigen::Quaterniond & gimbal_q,
  bool send_control,
  bool allow_fire,
  bool active_buff_mode,
  bool tx_control,
  bool tx_fire)
{
  const auto & plan = dbg.plan;

  int y = 30;
  const int dy = 28;

  const cv::Scalar white(255, 255, 255);
  const cv::Scalar green(0, 255, 0);
  const cv::Scalar red(0, 0, 255);
  const cv::Scalar yellow(0, 255, 255);
  const cv::Scalar purple(255, 0, 255);

  const Eigen::Vector3d q_euler =
    tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0);

  const double q_yaw = q_euler[0];
  const double q_pitch = q_euler[1];
  const double q_roll = q_euler[2];

  auto row_color = [&](const std::string & s, const cv::Scalar & c) {
    putText(img, s, 12, y, c, 0.65, 2);
    y += dy;
  };

  auto row = [&](const std::string & s) {
    row_color(s, white);
  };

  row_color(
    fmt::format(
      "gimbal_mode={} active={} buff_mode={}",
      gimbalModeString(gimbal_mode),
      active_buff_mode,
      auto_buff::jlu::to_string(dbg.mode)),
    yellow);

  row_color(
    fmt::format(
      "send_control={} allow_fire={} tx_control={} tx_fire={}",
      send_control,
      allow_fire,
      tx_control,
      tx_fire),
    allow_fire ? red : green);

  row_color(
    fmt::format(
      "state={} plan_control={} plan_fire={}",
      auto_buff::jlu::to_string(plan.track_state),
      plan.control,
      plan.fire),
    plan.control ? green : red);

  row_color(
    fmt::format(
      "yaw={:.4f}rad/{:.2f}deg  pitch={:.4f}rad/{:.2f}deg",
      plan.yaw,
      plan.yaw * 57.2958F,
      plan.pitch,
      plan.pitch * 57.2958F),
    purple);

  row(
    fmt::format(
      "blade_index={} fly_time={:.4f} yaw_vel={:.4f} pitch_vel={:.4f}",
      plan.blade_index,
      plan.fly_time,
      plan.yaw_vel,
      plan.pitch_vel));

  row(
    fmt::format(
      "detected={} pnp={} detect={:.2f}ms pnp={:.2f}ms total={:.2f}ms",
      dbg.detected_blades.size(),
      dbg.pnp_blades.size(),
      dbg.detect_ms,
      dbg.pnp_ms,
      dbg.total_ms));

  row(
    fmt::format(
      "bullet_speed={:.2f} state_yaw={:.2f}deg state_pitch={:.2f}deg",
      gs.bullet_speed,
      gs.yaw * 57.2958F,
      gs.pitch * 57.2958F));

  row(
    fmt::format(
      "q_yaw={:.2f}deg q_pitch={:.2f}deg q_roll={:.2f}deg",
      q_yaw * 57.2958,
      q_pitch * 57.2958,
      q_roll * 57.2958));

  row(
    fmt::format(
      "aim_valid={} img_valid={} aim_img=({:.1f},{:.1f})",
      dbg.aim_point_valid,
      dbg.aim_point_image_valid,
      dbg.aim_point_image.x,
      dbg.aim_point_image.y));

  const cv::Point center(img.cols / 2, img.rows / 2);
  cv::line(img, center + cv::Point(-25, 0), center + cv::Point(25, 0), green, 1);
  cv::line(img, center + cv::Point(0, -25), center + cv::Point(0, 25), green, 1);
}

}  // namespace

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);

  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const std::string config_path = cli.get<std::string>(0);
  const bool send_control = cli.get<bool>("send-control");
  const bool allow_fire = cli.get<bool>("allow-fire");
  const std::string force_mode = cli.get<std::string>("force-mode");
  const double show_scale = cli.get<double>("show-scale");
  const std::string plot_host = cli.get<std::string>("plot-host");
  const int plot_port = cli.get<int>("plot-port");

  tools::logger()->info(
    "[JLU-Buff-Test] config={} send_control={} allow_fire={} force_mode={} plot={}:{}",
    config_path,
    send_control,
    allow_fire,
    force_mode,
    plot_host,
    plot_port);

  if (send_control && allow_fire) {
    tools::logger()->warn(
      "[JLU-Buff-Test] send-control=true and allow-fire=true. Make sure this is intentional.");
  }

  tools::Exiter exiter;
  tools::Plotter plotter(plot_host, static_cast<uint16_t>(plot_port));

  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);
  auto_buff::jlu::JluBuffSystem buff(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point frame_start;

  while (!exiter.exit()) {
    camera.read(img, frame_start);
    if (img.empty()) {
      continue;
    }

    const io::GimbalMode gm = gimbal.mode();
    io::GimbalState gs = gimbal.state();
    Eigen::Quaterniond gimbal_q = gimbal.q(frame_start);

    auto_buff::jlu::BuffMode buff_mode = auto_buff::jlu::BuffMode::SMALL;
    bool active_buff_mode = false;

    if (force_mode == "small" || force_mode == "SMALL") {
      buff_mode = auto_buff::jlu::BuffMode::SMALL;
      active_buff_mode = true;
    } else if (force_mode == "big" || force_mode == "BIG") {
      buff_mode = auto_buff::jlu::BuffMode::BIG;
      active_buff_mode = true;
    } else {
      active_buff_mode = gimbalModeToBuffMode(gm, buff_mode);
    }

    auto_buff::jlu::JluBuffPlan plan;

    if (active_buff_mode) {
      plan = buff.run(img, buff_mode, gimbal_q, gs, frame_start);
    } else {
      plan.control = false;
      plan.fire = false;
    }

    const auto & dbg = buff.debug();

    const bool tx_control = send_control && active_buff_mode && plan.control;
    const bool tx_fire = send_control && allow_fire && plan.fire;

    if (tx_control) {
      gimbal.send(
        true,
        tx_fire,
        plan.yaw,
        plan.yaw_vel,
        plan.yaw_acc,
        -plan.pitch,
        plan.pitch_vel,
        plan.pitch_acc);
    } else {
      gimbal.send(false, false, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);
    }

    cv::Mat vis = img.clone();

    for (int i = 0; i < static_cast<int>(dbg.pnp_blades.size()); ++i) {
      drawBlade(vis, dbg.pnp_blades[i], i);
    }

    // 红色 AIM：最终 trajectory 瞄准点投影
    drawAimPoint(vis, dbg);

    drawHud(
      vis,
      dbg,
      gm,
      gs,
      gimbal_q,
      send_control,
      allow_fire,
      active_buff_mode,
      tx_control,
      tx_fire);

    nlohmann::json data;

    data["buff/active"] = active_buff_mode;
    data["buff/mode"] = auto_buff::jlu::to_string(dbg.mode);
    data["buff/track_state"] = auto_buff::jlu::to_string(plan.track_state);

    data["buff/control"] = plan.control;
    data["buff/fire"] = plan.fire;
    data["buff/tx_control"] = tx_control;
    data["buff/tx_fire"] = tx_fire;

    data["buff/yaw"] = plan.yaw;
    data["buff/pitch"] = plan.pitch;
    data["buff/yaw_deg"] = plan.yaw * 57.2958F;
    data["buff/pitch_deg"] = plan.pitch * 57.2958F;
    data["buff/yaw_vel"] = plan.yaw_vel;
    data["buff/yaw_acc"] = plan.yaw_acc;
    data["buff/pitch_vel"] = plan.pitch_vel;
    data["buff/pitch_acc"] = plan.pitch_acc;

    data["buff/blade_index"] = plan.blade_index;
    data["buff/fly_time"] = plan.fly_time;

    data["aim/valid"] = dbg.aim_point_valid;
    data["aim/image_valid"] = dbg.aim_point_image_valid;
    data["aim/image_x"] = dbg.aim_point_image.x;
    data["aim/image_y"] = dbg.aim_point_image.y;
    data["aim/world_x"] = dbg.aim_point_world.x();
    data["aim/world_y"] = dbg.aim_point_world.y();
    data["aim/world_z"] = dbg.aim_point_world.z();

    data["buff/detected_blades"] = dbg.detected_blades.size();
    data["buff/pnp_blades"] = dbg.pnp_blades.size();

    data["time/detect_ms"] = dbg.detect_ms;
    data["time/pnp_ms"] = dbg.pnp_ms;
    data["time/total_ms"] = dbg.total_ms;

    const Eigen::Vector3d q_euler =
      tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0);

    data["gimbal/mode"] = gimbalModeString(gm);
    data["gimbal/bullet_speed"] = gs.bullet_speed;

    data["gimbal/state_yaw"] = gs.yaw;
    data["gimbal/state_yaw_deg"] = gs.yaw * 57.2958F;
    data["gimbal/state_yaw_vel"] = gs.yaw_vel;

    data["gimbal/state_pitch"] = gs.pitch;
    data["gimbal/state_pitch_deg"] = gs.pitch * 57.2958F;
    data["gimbal/state_pitch_vel"] = gs.pitch_vel;

    data["gimbal/q_yaw"] = q_euler[0];
    data["gimbal/q_pitch"] = q_euler[1];
    data["gimbal/q_roll"] = q_euler[2];

    data["gimbal/q_yaw_deg"] = q_euler[0] * 57.2958;
    data["gimbal/q_pitch_deg"] = q_euler[1] * 57.2958;
    data["gimbal/q_roll_deg"] = q_euler[2] * 57.2958;

    data["gimbal/bullet_count"] = gs.bullet_count;

    if (!dbg.pnp_blades.empty()) {
      const auto & b = dbg.pnp_blades.front();

      data["blade0/conf"] = b.confidence;
      data["blade0/roll"] = b.roll;
      data["blade0/pnp_ok"] = b.pnp_ok;
      data["blade0/radius"] = (b.position_world - b.center_world).norm();

      data["blade0/position_camera_x"] = b.position_camera.x();
      data["blade0/position_camera_y"] = b.position_camera.y();
      data["blade0/position_camera_z"] = b.position_camera.z();

      data["blade0/position_world_x"] = b.position_world.x();
      data["blade0/position_world_y"] = b.position_world.y();
      data["blade0/position_world_z"] = b.position_world.z();

      data["blade0/center_world_x"] = b.center_world.x();
      data["blade0/center_world_y"] = b.center_world.y();
      data["blade0/center_world_z"] = b.center_world.z();
    }

    if (dbg.pnp_blades.size() > 1) {
      const auto & b = dbg.pnp_blades[1];

      data["blade1/conf"] = b.confidence;
      data["blade1/roll"] = b.roll;
      data["blade1/pnp_ok"] = b.pnp_ok;
      data["blade1/radius"] = (b.position_world - b.center_world).norm();

      data["blade1/position_camera_x"] = b.position_camera.x();
      data["blade1/position_camera_y"] = b.position_camera.y();
      data["blade1/position_camera_z"] = b.position_camera.z();

      data["blade1/position_world_x"] = b.position_world.x();
      data["blade1/position_world_y"] = b.position_world.y();
      data["blade1/position_world_z"] = b.position_world.z();

      data["blade1/center_world_x"] = b.center_world.x();
      data["blade1/center_world_y"] = b.center_world.y();
      data["blade1/center_world_z"] = b.center_world.z();
    }

    plotter.plot(data);

    if (show_scale > 0.0 && std::abs(show_scale - 1.0) > 1e-6) {
      cv::resize(vis, vis, {}, show_scale, show_scale);
    }

    cv::imshow("jlu_buff_vehicle_test", vis);

    const int key = cv::waitKey(1);

    if (key == 'q' || key == 27) {
      break;
    }

    if (key == 's') {
      const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();

      const std::string name = fmt::format("jlu_buff_debug_{}.jpg", now_ms);
      cv::imwrite(name, vis);
      tools::logger()->info("[JLU-Buff-Test] saved {}", name);
    }
  }

  gimbal.send(false, false, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f);

  return 0;
}