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
}  // namespace

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
  if (candidate < 0) candidate = 0;
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

Eigen::Vector3d Trajectory::bladePoint(const BuffState & state, int idx, double predict_sec) const
{
  Eigen::Vector3d radius_vec = state.blade_world[idx] - state.center_world;
  if (radius_vec.norm() < 1e-4) radius_vec = Eigen::Vector3d(0.0, 0.0, kBuffRadius);
  const double angle = state.roll + state.vroll * predict_sec;
  Eigen::AngleAxisd R(angle, Eigen::Vector3d::UnitX());
  return state.center_world + R * radius_vec;
}

TrajectorySolution Trajectory::solve(const BuffState & state, double bullet_speed, double predict_offset_sec)
{
  TrajectorySolution sol;
  if (state.track_state != TrackState::TRACKING) return sol;
  if (bullet_speed < config_.min_bullet_speed || bullet_speed > config_.max_bullet_speed) {
    tools::logger()->warn(
      "[JLU-Buff] Invalid bullet_speed={:.2f}; use default_bullet_speed={:.2f}", bullet_speed,
      config_.default_bullet_speed);
    bullet_speed = config_.default_bullet_speed;
  }
  if (bullet_speed <= 1e-3) return sol;

  const int blade = selectBlade(state);
  double fly_time = 0.0;
  Eigen::Vector3d aim_point = state.blade_world[std::max(0, blade)];
  for (int i = 0; i < config_.max_aim_iterate_count; ++i) {
    aim_point = bladePoint(state, blade, predict_offset_sec + fly_time);
    const double new_fly_time = aim_point.norm() / bullet_speed;
    if (!config_.iterative_fly_time || std::abs(new_fly_time - fly_time) < 1e-4) {
      fly_time = new_fly_time;
      break;
    }
    fly_time = new_fly_time;
  }

  constexpr double g = 9.80665;
  const double x = aim_point.x();
  const double y = aim_point.y();
  const double z = aim_point.z();
  const double horizontal = std::hypot(x, y);
  const double drop = 0.5 * g * fly_time * fly_time;
  sol.yaw = std::atan2(y, x) + config_.yaw_offset;
  sol.pitch = std::atan2(z + drop, horizontal) + config_.pitch_offset;
  sol.fly_time = fly_time;
  sol.selected_blade = blade;
  sol.bullet_speed = bullet_speed;
  sol.aim_point_world = aim_point;
  sol.valid = std::isfinite(sol.yaw) && std::isfinite(sol.pitch) && fly_time > 0.0;
  sol.fire = sol.valid && state.blade_states[blade] != BladeState::ACTIVATED;
  tools::logger()->debug(
    "[JLU-Buff] trajectory bullet_speed={:.2f} offset={:.1f}ms yaw={:.3f} pitch={:.3f} fly={:.3f} blade={} fire={}",
    bullet_speed, predict_offset_sec * 1000.0, sol.yaw, sol.pitch, sol.fly_time, sol.selected_blade,
    sol.fire);
  return sol;
}

TrajectoryConfig load_trajectory_config(const YAML::Node & node)
{
  TrajectoryConfig c;
  c.max_aim_iterate_count = read_or<int>(node, "max_aim_iterate_count", c.max_aim_iterate_count);
  c.aim_ok_error_m = read_or<double>(node, "aim_ok_error_m", c.aim_ok_error_m);
  c.blade_select_change_count_thres = read_or<int>(node, "blade_select_change_count_thres", c.blade_select_change_count_thres);
  c.min_bullet_speed = read_or<double>(node, "min_bullet_speed", c.min_bullet_speed);
  c.max_bullet_speed = read_or<double>(node, "max_bullet_speed", c.max_bullet_speed);
  c.default_bullet_speed = read_or<double>(node, "default_bullet_speed", c.default_bullet_speed);
  c.iterative_fly_time = read_or<bool>(node, "iterative_fly_time", c.iterative_fly_time);
  c.yaw_offset = read_or<double>(node, "yaw_offset", c.yaw_offset) / 57.3;
  c.pitch_offset = read_or<double>(node, "pitch_offset", c.pitch_offset) / 57.3;
  return c;
}
}  // namespace auto_buff::jlu
