#include "tasks/auto_buff/jlu_buff/trajectory.hpp"

#include <algorithm>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_buff::jlu
{

namespace
{

template <typename T>
T read_or(const YAML::Node & n, const std::string & key, const T & fallback)
{
  return n && n[key] ? n[key].as<T>() : fallback;
}

} // namespace

Trajectory::Trajectory(TrajectoryConfig config) : config_(config) {}

int Trajectory::selectBlade(const BuffState & state)
{
  int candidate = -1;

  for (int i = 0; i < 5; ++i) {
    if (state.blade_states[i] == BladeState::UNACTIVATED) {
      candidate = i;
      break;
    }
  }

  if (candidate < 0) {
    if (selected_blade_ < 0) return -1;

    if (change_count_ < config_.blade_select_change_count_thres) {
      ++change_count_;
      return selected_blade_;
    }

    selected_blade_ = -1;
    change_count_ = 0;
    return -1;
  }

  if (selected_blade_ < 0) {
    selected_blade_ = candidate;
    change_count_ = 0;
    return selected_blade_;
  }

  if (candidate != selected_blade_) {
    if (++change_count_ >= config_.blade_select_change_count_thres) {
      selected_blade_ = candidate;
      change_count_ = 0;
    }
  } else {
    change_count_ = 0;
  }

  return selected_blade_;
}

Eigen::Vector3d Trajectory::bladePoint(
  const BuffState & state, int idx, double predict_sec) const
{
  const int blade_idx = std::clamp(idx, 0, 4);

  // Original JLU trajectory:
  // buff_state.predict(fly_time).blades()[blade_index].getHitPosition()
  //
  // In this branch, JluBuffSystem::run() already gives solve() a state predicted by
  // predict_offset_sec. Therefore this function advances only the extra horizon,
  // normally the bullet fly time.
  const double blade_roll = tools::limit_rad(
    state.roll + blade_idx * kBuffBladeRollStep + state.vroll * predict_sec);

  return buff_blade_hit_position_from_center_roll(state.center_world, blade_roll, kBuffRadius);
}

TrajectorySolution Trajectory::solve(
  const BuffState & state, double bullet_speed, double predict_offset_sec)
{
  TrajectorySolution sol;

  if (state.track_state != TrackState::TRACKING) return sol;

  if (bullet_speed < config_.min_bullet_speed || bullet_speed > config_.max_bullet_speed) {
    tools::logger()->warn(
      "[JLU-Buff] Invalid bullet_speed={:.2f}; use default_bullet_speed={:.2f}",
      bullet_speed, config_.default_bullet_speed);

    bullet_speed = config_.default_bullet_speed;
  }

  if (bullet_speed <= 1e-3) return sol;

  const int blade = selectBlade(state);
  if (blade < 0 || blade >= 5) return sol;

  const int blade_idx = blade;

  double fly_time = 0.0;
  Eigen::Vector3d aim_point = bladePoint(state, blade_idx, 0.0);

  for (int i = 0; i < config_.max_aim_iterate_count; ++i) {
    aim_point = bladePoint(state, blade_idx, fly_time);

    const double new_fly_time = aim_point.norm() / bullet_speed;

    if (!config_.iterative_fly_time || std::abs(new_fly_time - fly_time) < 1e-4) {
      fly_time = new_fly_time;
      aim_point = bladePoint(state, blade_idx, fly_time);
      break;
    }

    fly_time = new_fly_time;
  }

  const double x = aim_point.x();
  const double y = aim_point.y();
  const double z = aim_point.z();

  const double horizontal = std::hypot(x, y);
  if (horizontal < 1e-6) return sol;

  sol.yaw = std::atan2(y, x) + config_.yaw_offset;

  double pitch = std::atan2(z, horizontal);

  for (int i = 0; i < config_.ballistic_max_iterate_count; ++i) {
    const double cos_pitch = std::max(1e-3, std::cos(pitch));

    const double effective_speed =
      config_.air_resistance_coefficient > 1e-9
        ? bullet_speed * std::exp(-config_.air_resistance_coefficient * std::max(0.0, fly_time))
        : bullet_speed;

    const double t = horizontal / std::max(1e-3, effective_speed * cos_pitch);

    const double compensated_z = z + 0.5 * config_.gravity * t * t;
    const double next_pitch = std::atan2(compensated_z, horizontal);

    if (std::abs(next_pitch - pitch) < 1e-5) {
      pitch = next_pitch;
      fly_time = t;
      break;
    }

    pitch = next_pitch;
    fly_time = t;
  }

  sol.pitch = pitch + config_.pitch_offset;
  sol.fly_time = fly_time;
  sol.selected_blade = blade_idx;
  sol.bullet_speed = bullet_speed;
  sol.aim_point_world = aim_point;
  sol.valid = std::isfinite(sol.yaw) && std::isfinite(sol.pitch) && fly_time > 0.0;
  sol.fire = sol.valid && state.blade_states[blade_idx] != BladeState::ACTIVATED;

  // tools::logger()->debug(
  //   "[JLU-Buff] trajectory bullet_speed={:.2f} offset={:.1f}ms yaw={:.3f} pitch={:.3f} "
  //   "fly={:.3f} blade={} fire={}",
  //   bullet_speed, predict_offset_sec * 1000.0, sol.yaw, sol.pitch, sol.fly_time,
  //   sol.selected_blade, sol.fire);

  return sol;
}

TrajectoryConfig load_trajectory_config(const YAML::Node & node)
{
  TrajectoryConfig c;

  c.max_aim_iterate_count =
    read_or<int>(node, "max_aim_iterate_count", c.max_aim_iterate_count);

  c.aim_ok_error_m = read_or<double>(node, "aim_ok_error_m", c.aim_ok_error_m);

  c.blade_select_change_count_thres =
    read_or<int>(node, "blade_select_change_count_thres", c.blade_select_change_count_thres);

  c.min_bullet_speed = read_or<double>(node, "min_bullet_speed", c.min_bullet_speed);
  c.max_bullet_speed = read_or<double>(node, "max_bullet_speed", c.max_bullet_speed);
  c.default_bullet_speed = read_or<double>(node, "default_bullet_speed", c.default_bullet_speed);

  c.iterative_fly_time = read_or<bool>(node, "iterative_fly_time", c.iterative_fly_time);

  c.yaw_offset = read_or<double>(node, "yaw_offset", c.yaw_offset) / 57.3;
  c.pitch_offset = read_or<double>(node, "pitch_offset", c.pitch_offset) / 57.3;

  c.gravity = read_or<double>(node, "gravity", c.gravity);
  c.air_resistance_coefficient =
    read_or<double>(node, "air_resistance_coefficient", c.air_resistance_coefficient);

  c.ballistic_max_iterate_count =
    read_or<int>(node, "ballistic_max_iterate_count", c.ballistic_max_iterate_count);

  return c;
}

} // namespace auto_buff::jlu