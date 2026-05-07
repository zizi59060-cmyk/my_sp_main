#pragma once

#include <yaml-cpp/yaml.h>

#include "tasks/auto_buff/jlu_buff/types.hpp"

namespace auto_buff::jlu
{
struct TrajectoryConfig
{
  int max_aim_iterate_count = 20;
  double aim_ok_error_m = 0.005;
  int blade_select_change_count_thres = 20;
  double min_bullet_speed = 18.0;
  double max_bullet_speed = 25.0;
  double default_bullet_speed = 22.0;
  bool iterative_fly_time = true;
  double yaw_offset = 0.0;
  double pitch_offset = 0.0;
};

class Trajectory
{
public:
  explicit Trajectory(TrajectoryConfig config = {});
  TrajectorySolution solve(const BuffState & state, double bullet_speed, double predict_offset_sec);
private:
  int selectBlade(const BuffState & state);
  Eigen::Vector3d bladePoint(const BuffState & state, int idx, double predict_sec) const;

  TrajectoryConfig config_;
  int selected_blade_ = -1;
  int change_count_ = 0;
};

TrajectoryConfig load_trajectory_config(const YAML::Node & node);
}  // namespace auto_buff::jlu
