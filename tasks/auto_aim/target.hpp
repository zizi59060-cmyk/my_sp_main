#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{

struct MatchResult
{
  bool valid = false;
  int id = -1;
  double score = 1e9;
  bool phase_jump = false;
};

class Target
{
public:
  ArmorName name;
  ArmorType armor_type;
  ArmorPriority priority;
  bool jumped;
  int last_id;  // debug only

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);

  MatchResult match(const Armor & armor) const;
  void update(const Armor & armor);
  void update(const Armor & armor, const MatchResult & match);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool diverged() const;
  bool convergened();

  bool checkinit();

  void configure_phase_jump_guard(
    bool enable, double score_thresh, double guard_time_s, double dist_R_scale,
    double angle_R_scale);

  bool recent_phase_jump() const;
  double time_since_last_switch() const;

  bool isinit = false;

private:
  int armor_num_ = 4;
  int switch_count_ = 0;
  int update_count_ = 0;

  bool is_switch_ = false;
  bool is_converged_ = false;

  bool enable_phase_jump_guard_ = false;
  double phase_jump_score_thresh_ = 0.55;
  double phase_jump_guard_time_ = 0.10;
  double phase_jump_dist_R_scale_ = 3.0;
  double phase_jump_angle_R_scale_ = 4.0;
  bool recent_phase_jump_flag_ = false;

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_{};
  std::chrono::steady_clock::time_point last_switch_time_{};

  void update_ypda(const Armor & armor, int id, bool phase_jump);  // yaw pitch distance angle

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP