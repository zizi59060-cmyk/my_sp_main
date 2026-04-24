#include "shooter.hpp"

#include <yaml-cpp/yaml.h>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
Shooter::Shooter(const std::string & config_path)
: last_command_{false, false, 0, 0},
  judge_distance_{0.0},
  first_tolerance_{0.0},
  second_tolerance_{0.0},
  auto_fire_{false},
  enable_switch_forbid_{false},
  switch_forbid_frames_{2},
  last_target_id_{-1},
  switch_block_frames_{0}
{
  auto yaml = YAML::LoadFile(config_path);

  first_tolerance_ = yaml["first_tolerance"].as<double>() / 57.3;    // degree to rad
  second_tolerance_ = yaml["second_tolerance"].as<double>() / 57.3;  // degree to rad
  judge_distance_ = yaml["judge_distance"].as<double>();
  auto_fire_ = yaml["auto_fire"].as<bool>();

  // 新增：切板禁射功能开关
  if (yaml["enable_switch_forbid"]) {
    enable_switch_forbid_ = yaml["enable_switch_forbid"].as<bool>();
  }

  // 新增：切板后禁射帧数
  if (yaml["switch_forbid_frames"]) {
    switch_forbid_frames_ = yaml["switch_forbid_frames"].as<int>();
  }

  if (switch_forbid_frames_ < 0) switch_forbid_frames_ = 0;

  tools::logger()->info(
    "[Shooter] auto_fire={}, enable_switch_forbid={}, switch_forbid_frames={}",
    auto_fire_, enable_switch_forbid_, switch_forbid_frames_);
}

bool Shooter::shoot(
  const io::Command & command, const auto_aim::Aimer & aimer,
  const std::list<auto_aim::Target> & targets, const Eigen::Vector3d & gimbal_pos)
{
  // 不控枪 / 没目标 / 不自动开火：直接不射，并清理切板状态
  if (!command.control || targets.empty() || !auto_fire_) {
    last_command_ = command;
    last_target_id_ = -1;
    switch_block_frames_ = 0;
    return false;
  }

  const auto & target = targets.front();

  auto target_x = target.ekf_x()[0];
  auto target_y = target.ekf_x()[2];
  auto tolerance = std::sqrt(tools::square(target_x) + tools::square(target_y)) > judge_distance_
                     ? second_tolerance_
                     : first_tolerance_;

  // ===== 切板禁射逻辑（可配置开关）=====
  if (enable_switch_forbid_) {
    // 检测 last_id 是否发生变化
    if (last_target_id_ != -1 && target.last_id != last_target_id_) {
      switch_block_frames_ = switch_forbid_frames_;
      tools::logger()->debug(
        "[Shooter] armor switch detected: {} -> {}, block {} frame(s)",
        last_target_id_, target.last_id, switch_forbid_frames_);
    }

    last_target_id_ = target.last_id;

    // 切板后的短暂禁射
    if (switch_block_frames_ > 0) {
      switch_block_frames_--;
      last_command_ = command;
      return false;
    }
  } else {
    // 功能关闭时，状态也保持干净
    last_target_id_ = target.last_id;
    switch_block_frames_ = 0;
  }

  // ===== 原有开火门控 =====
  // 1. command 不能突变过大
  // 2. 云台当前位置要接近上一帧 command
  // 3. aimer 的瞄点必须有效
  if (
    std::abs(last_command_.yaw - command.yaw) < tolerance * 2 &&
    std::abs(gimbal_pos[0] - last_command_.yaw) < tolerance &&
    aimer.debug_aim_point.valid) {
    last_command_ = command;
    return true;
  }

  last_command_ = command;
  return false;
}

}  // namespace auto_aim