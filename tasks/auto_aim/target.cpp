#include "target.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <set>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_aim
{
Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  armor_num_(armor_num),
  t_(t),
  last_switch_time_(t)
{
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  auto center_x = xyz[0] + r * std::cos(ypr[0]);
  auto center_y = xyz[1] + r * std::sin(ypr[0]);
  auto center_z = xyz[2];

  // x vx y vy z vz a w r l h
  // a: angle
  // w: angular velocity
  // l: r2 - r1
  // h: z2 - z1
  Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0], 0, r, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
}

Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a + b;
    c[6] = tools::limit_rad(c[6]);
    return c;
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  t_ = std::chrono::steady_clock::now();
  last_switch_time_ = t_;
}

void Target::configure_phase_jump_guard(
  bool enable, double score_thresh, double guard_time_s, double dist_R_scale,
  double angle_R_scale)
{
  enable_phase_jump_guard_ = enable;
  phase_jump_score_thresh_ = score_thresh;
  phase_jump_guard_time_ = guard_time_s;
  phase_jump_dist_R_scale_ = dist_R_scale;
  phase_jump_angle_R_scale_ = angle_R_scale;
}

bool Target::recent_phase_jump() const
{
  if (!enable_phase_jump_guard_) return false;
  return time_since_last_switch() < phase_jump_guard_time_;
}

double Target::time_since_last_switch() const
{
  return tools::delta_time(t_, last_switch_time_);
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  Eigen::MatrixXd F{
    {1, dt, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 1, dt, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 1, dt, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 1, dt, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0},  {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}};

  double v1, v2;
  if (name == ArmorName::outpost) {
    v1 = 10;
    v2 = 0.1;
  } else {
    v1 = 100;
    v2 = 400;
  }

  auto a = dt * dt * dt * dt / 4;
  auto b = dt * dt * dt / 2;
  auto c = dt * dt;
  Eigen::MatrixXd Q{
    {a * v1, b * v1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {b * v1, c * v1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, a * v1, b * v1, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, b * v1, c * v1, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, a * v1, b * v1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, b * v1, c * v1, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, a * v2, b * v2, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, b * v2, c * v2, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};

  auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
    Eigen::VectorXd x_prior = F * x;
    x_prior[6] = tools::limit_rad(x_prior[6]);
    return x_prior;
  };

  if (this->convergened() && this->name == ArmorName::outpost && std::abs(this->ekf_.x[7]) > 2) {
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;
  }

  ekf_.predict(F, Q, f);
}

MatchResult Target::match(const Armor & armor) const
{
  MatchResult result;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();
  if (xyza_list.empty()) return result;

  std::vector<int> candidate_ids;
  candidate_ids.reserve(armor_num_);

  const auto push_unique = [&](int id) {
    if (id < 0 || id >= armor_num_) return;
    if (std::find(candidate_ids.begin(), candidate_ids.end(), id) == candidate_ids.end()) {
      candidate_ids.push_back(id);
    }
  };

  const double w = ekf_.x[7];
  if (
    !enable_phase_jump_guard_ || update_count_ < 2 || !is_converged_ ||
    std::abs(w) < 1.0) {
    for (int i = 0; i < armor_num_; ++i) push_unique(i);
  } else {
    push_unique(last_id);
    push_unique((last_id + 1) % armor_num_);
    push_unique((last_id - 1 + armor_num_) % armor_num_);
    if (candidate_ids.empty()) {
      for (int i = 0; i < armor_num_; ++i) push_unique(i);
    }
  }

  double best_score = std::numeric_limits<double>::infinity();
  int best_id = -1;

  for (int id : candidate_ids) {
    const auto & xyza = xyza_list[id];
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));

    double armor_yaw_err = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3]));
    double aim_yaw_err = std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));
    double pitch_err = std::abs(tools::limit_rad(armor.ypd_in_world[1] - ypd[1]));
    double dist_err = std::abs(armor.ypd_in_world[2] - ypd[2]);

    double score = armor_yaw_err + aim_yaw_err + 0.35 * pitch_err + 0.10 * dist_err;

    if (score < best_score) {
      best_score = score;
      best_id = id;
    }
  }

  if (best_id < 0) return result;

  result.valid = true;
  result.id = best_id;
  result.score = best_score;

  if (
    enable_phase_jump_guard_ && update_count_ >= 2 && best_id != last_id &&
    best_score < phase_jump_score_thresh_) {
    int delta = (best_id - last_id + armor_num_) % armor_num_;
    int signed_step = 0;
    if (delta == 1) signed_step = 1;
    else if (delta == armor_num_ - 1) signed_step = -1;

    bool is_neighbor_jump = (signed_step != 0);
    bool direction_ok =
      std::abs(w) < 1.0 ||
      (w > 0 ? (signed_step == 1) : (signed_step == -1));

    result.phase_jump = is_neighbor_jump && direction_ok;
  }

  return result;
}

void Target::update(const Armor & armor)
{
  auto match_result = match(armor);
  if (!match_result.valid) return;
  update(armor, match_result);
}

void Target::update(const Armor & armor, const MatchResult & match_result)
{
  if (!match_result.valid) return;

  int id = match_result.id;

  if (id != 0) jumped = true;

  bool switched = (id != last_id);
  is_switch_ = switched;

  if (switched) {
    switch_count_++;
    last_switch_time_ = t_;
  }

  recent_phase_jump_flag_ = match_result.phase_jump;
  last_id = id;
  update_count_++;

  update_ypda(armor, id, match_result.phase_jump);
}

void Target::update_ypda(const Armor & armor, int id, bool phase_jump)
{
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);

  auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);

  Eigen::VectorXd R_dig{
    {2e-2, 2e-2, log(std::abs(delta_angle) + 1) + 1.0,
     log(std::abs(armor.ypd_in_world[2]) + 1) / 100 + 0.2}};

  if (phase_jump) {
    R_dig[2] *= phase_jump_dist_R_scale_;
    R_dig[3] *= phase_jump_angle_R_scale_;
  }

  Eigen::MatrixXd R = R_dig.asDiagonal();

  auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
    Eigen::VectorXd xyz = h_armor_xyz(x, id);
    Eigen::VectorXd ypd = tools::xyz2ypd(xyz);
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    return {ypd[0], ypd[1], ypd[2], angle};
  };

  auto z_subtract = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
    Eigen::VectorXd c = a - b;
    c[0] = tools::limit_rad(c[0]);
    c[1] = tools::limit_rad(c[1]);
    c[3] = tools::limit_rad(c[3]);
    return c;
  };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z{{ypd[0], ypd[1], ypd[2], ypr[0]}};

  ekf_.update(z, H, R, h, z_subtract);
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], angle});
  }
  return _armor_xyza_list;
}

bool Target::diverged() const
{
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}

bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}

Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto armor_x = x[0] - r * std::cos(angle);
  auto armor_y = x[2] - r * std::sin(angle);
  auto armor_z = (use_l_h) ? x[4] + x[10] : x[4];

  return {armor_x, armor_y, armor_z};
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
  auto use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);

  auto r = (use_l_h) ? x[8] + x[9] : x[8];
  auto dx_da = r * std::sin(angle);
  auto dy_da = -r * std::cos(angle);

  auto dx_dr = -std::cos(angle);
  auto dy_dr = -std::sin(angle);
  auto dx_dl = (use_l_h) ? -std::cos(angle) : 0.0;
  auto dy_dl = (use_l_h) ? -std::sin(angle) : 0.0;

  auto dz_dh = (use_l_h) ? 1.0 : 0.0;

  Eigen::MatrixXd H_armor_xyza{
    {1, 0, 0, 0, 0, 0, dx_da, 0, dx_dr, dx_dl, 0},
    {0, 0, 1, 0, 0, 0, dy_da, 0, dy_dr, dy_dl, 0},
    {0, 0, 0, 0, 1, 0, 0, 0, 0, 0, dz_dh},
    {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0}};

  Eigen::VectorXd armor_xyz = h_armor_xyz(x, id);
  Eigen::MatrixXd H_armor_ypd = tools::xyz2ypd_jacobian(armor_xyz);
  Eigen::MatrixXd H_armor_ypda{
    {H_armor_ypd(0, 0), H_armor_ypd(0, 1), H_armor_ypd(0, 2), 0},
    {H_armor_ypd(1, 0), H_armor_ypd(1, 1), H_armor_ypd(1, 2), 0},
    {H_armor_ypd(2, 0), H_armor_ypd(2, 1), H_armor_ypd(2, 2), 0},
    {0, 0, 0, 1}};

  return H_armor_ypda * H_armor_xyza;
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim