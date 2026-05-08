#include "tasks/auto_buff/jlu_buff/targets.hpp"

#include <boost/make_shared.hpp>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/PriorFactor.h>

#include <algorithm>
#include <cmath>

#include "tasks/auto_buff/jlu_buff/factors.hpp"
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
BuffBlade choose_best(const std::vector<BuffBlade> & blades, const BuffState & last_state, bool has_last)
{
  if (!has_last) {
    return blades.front();
  }

  const auto best = std::min_element(blades.begin(), blades.end(), [&](const auto & a, const auto & b) {
    const double da = (a.center_world - last_state.center_world).norm() +
      0.15 * std::abs(tools::limit_rad(a.roll - last_state.roll));

    const double db = (b.center_world - last_state.center_world).norm() +
      0.15 * std::abs(tools::limit_rad(b.roll - last_state.roll));

    return da < db;
  });

  return *best;
}
// BuffBlade choose_best(const std::vector<BuffBlade> & blades, const BuffState & last_state, bool has_last)
// {
//   if (!has_last) {
//     return *std::max_element(
//       blades.begin(), blades.end(), [](const auto & a, const auto & b) { return a.confidence < b.confidence; });
//   }

//   const auto best = std::min_element(blades.begin(), blades.end(), [&](const auto & a, const auto & b) {
//     const double da = (a.center_world - last_state.center_world).norm() +
//                       0.15 * std::abs(tools::limit_rad(a.roll - last_state.roll));
//     const double db = (b.center_world - last_state.center_world).norm() +
//                       0.15 * std::abs(tools::limit_rad(b.roll - last_state.roll));
//     return da < db;
//   });
//   return *best;
// }

// void fill_blades_from_roll(BuffState & state)
// {
//   for (int i = 0; i < 5; ++i) {
//     const double roll = state.roll + i * 2.0 * CV_PI / 5.0;
//     state.blade_world[i] =
//       state.center_world + Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitZ()) * buff_blade_center_object_point(kBuffRadius);
//     state.blade_orientations[i] = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitZ()).toRotationMatrix();
//     if (state.blade_states[i] != BladeState::ACTIVATED) state.blade_states[i] = BladeState::UNACTIVATED;
//   }
// }
// void fill_blades_from_roll(BuffState & state)
// {
//   Eigen::Vector3d axis = state.rotation_axis_world;

//   if (!axis.allFinite() || axis.norm() < 1e-6) {
//     axis = Eigen::Vector3d::UnitZ();
//   }

//   axis.normalize();

//   Eigen::Vector3d radius_vec = state.radius_vector_world;

//   if (!radius_vec.allFinite() || radius_vec.norm() < 1e-4) {
//     radius_vec = Eigen::Vector3d(0.0, kBuffRadius, 0.0);
//   }

//   // 统一半径长度，避免 PnP 抖动导致圆半径忽大忽小
//   radius_vec = radius_vec.normalized() * kBuffRadius;

//   // 确保半径向量在旋转平面内：去掉沿旋转轴的分量
//   radius_vec = radius_vec - axis * radius_vec.dot(axis);

//   if (!radius_vec.allFinite() || radius_vec.norm() < 1e-4) {
//     radius_vec = Eigen::Vector3d(0.0, kBuffRadius, 0.0);
//   }

//   radius_vec = radius_vec.normalized() * kBuffRadius;

//   state.radius_vector_world = radius_vec;
//   state.rotation_axis_world = axis;

//   for (int i = 0; i < 5; ++i) {
//     const double delta = i * 2.0 * CV_PI / 5.0;
//     const Eigen::AngleAxisd R(delta, axis);

//     state.blade_world[i] = state.center_world + R * radius_vec;
//     state.blade_orientations[i] = R.toRotationMatrix();

//     if (state.blade_states[i] != BladeState::ACTIVATED) {
//       state.blade_states[i] = BladeState::UNACTIVATED;
//     }
//   }
// // }
void fill_blades_from_roll(BuffState & state)
{
  state.roll = tools::limit_rad(state.roll);

  state.rotation_axis_world = buff_rotation_axis_from_center(state.center_world);

  state.radius_vector_world =
    buff_blade_hit_position_from_center_roll(state.center_world, state.roll, kBuffRadius) -
    state.center_world;

  for (int i = 0; i < 5; ++i) {
    const double roll = tools::limit_rad(state.roll + i * kBuffBladeRollStep);

    state.blade_world[i] =
      buff_blade_hit_position_from_center_roll(state.center_world, roll, kBuffRadius);

    state.blade_orientations[i] =
      buff_blade_orientation_from_center_roll(state.center_world, roll);

    if (state.blade_states[i] != BladeState::ACTIVATED) {
      state.blade_states[i] = BladeState::UNACTIVATED;
    }
  }
}
gtsam::SharedNoiseModel diagonal(std::initializer_list<double> sigmas)
{
  gtsam::Vector v(static_cast<int>(sigmas.size()));
  int i = 0;
  for (double s : sigmas) v(i++) = s;
  return gtsam::noiseModel::Diagonal::Sigmas(v);
}
}  // namespace

SmallBuffTarget::SmallBuffTarget(TargetConfig config) : config_(config)
{
  gtsam::ISAM2Params params;
  params.relinearizeThreshold = 0.01;
  params.relinearizeSkip = 1;
  isam_ = gtsam::ISAM2(params);
}

BuffState SmallBuffTarget::update(const std::vector<BuffBlade> & blades, TimePoint timestamp)
{
  if (blades.empty()) {
    if (has_last_ && tools::delta_time(timestamp, last_timestamp_) < config_.lost_threshold_sec) {
      state_ = TrackState::TEMP_LOST;
      current_ = predict(tools::delta_time(timestamp, last_timestamp_));
      current_.track_state = state_;
      current_.timestamp = timestamp;
    } else {
      state_ = TrackState::LOST;
      current_.track_state = state_;
      current_.timestamp = timestamp;
    }
    //tools::logger()->debug("[JLU-Buff] TrackState={}", to_string(state_));
    return current_;
  }

  const auto best = choose_best(blades, current_, has_last_);
  const double dt = has_last_ ? std::max(1e-3, tools::delta_time(timestamp, last_timestamp_)) : 1e-3;
  const double measured_vroll = has_last_ ? tools::limit_rad(best.roll - last_roll_) / dt : 0.0;

  const size_t k = frame_index_++;
  const auto c_key = gtsam::Symbol('c', k);
  const auto r_key = gtsam::Symbol('r', k);
  const auto v_key = gtsam::Symbol('v', k);

  gtsam::NonlinearFactorGraph graph;
  gtsam::Values init;

  init.insert(c_key, gtsam::Point3(best.center_world.x(), best.center_world.y(), best.center_world.z()));
  init.insert(r_key, best.roll);
  init.insert(v_key, std::isfinite(measured_vroll) ? measured_vroll : last_vroll_);

  if (!has_last_) {
    graph.add(gtsam::PriorFactor<gtsam::Point3>(
      c_key, gtsam::Point3(best.center_world.x(), best.center_world.y(), best.center_world.z()),
      diagonal({0.10, 0.10, 0.10})));
    graph.add(gtsam::PriorFactor<double>(r_key, best.roll, diagonal({30.0 / 57.3})));
    graph.add(gtsam::PriorFactor<double>(v_key, 0.0, diagonal({6.0})));
  } else {
    const auto pc_key = gtsam::Symbol('c', k - 1);
    const auto pr_key = gtsam::Symbol('r', k - 1);
    const auto pv_key = gtsam::Symbol('v', k - 1);
    graph.add(boost::make_shared<ConstPositionFactor>(pc_key, c_key, diagonal({0.01, 0.01, 0.01})));
    graph.add(boost::make_shared<ConstVRollFactor>(pr_key, r_key, pv_key, dt, diagonal({0.10})));
    graph.add(gtsam::PriorFactor<double>(v_key, measured_vroll, diagonal({0.35})));
  }

  graph.add(boost::make_shared<RollFactor>(r_key, best.roll, diagonal({2.0 / 57.3})));

  // JLU tracker logic keeps all observed fan blades in the graph. The TensorRT detector
  // may produce one or more blades; map each observation to one of the five physical blades
  // by roll difference from the selected target blade, then add blade pose factors.
  std::array<BladeState, 5> measured_states{};
  measured_states.fill(BladeState::UNACTIVATED);
  for (const auto & blade : blades) {
    int blade_index = static_cast<int>(std::lround(tools::limit_rad(blade.roll - best.roll) / (2.0 * CV_PI / 5.0)));
    blade_index = (blade_index % 5 + 5) % 5;
    measured_states[blade_index] = blade.state;
    graph.add(boost::make_shared<BuffBladeFactor>(
      c_key, r_key, blade.position_world, blade.roll, blade_index,
      diagonal({0.03, 0.03, 0.03, 2.0 / 57.3})));
  }

  try {
    isam_.update(graph, init);
    isam_.update();
    const auto values = isam_.calculateEstimate();

    const auto center = values.at<gtsam::Point3>(c_key);
    current_.center_world = Eigen::Vector3d(center.x(), center.y(), center.z());
    current_.roll = values.at<double>(r_key);
    current_.vroll = values.at<double>(v_key);
  } catch (const std::exception & e) {
    tools::logger()->warn("[JLU-Buff] ISAM2 update failed, using raw measurement: {}", e.what());
    current_.center_world = best.center_world;
    current_.roll = best.roll;
    current_.vroll = std::isfinite(measured_vroll) ? measured_vroll : 0.0;
  }

  current_.track_state = TrackState::TRACKING;
  current_.timestamp = timestamp;
  current_.blade_states = measured_states;
  current_.blade_states[0] = best.state;
  fill_blades_from_roll(current_);

  state_ = TrackState::TRACKING;
  has_last_ = true;
  last_roll_ = current_.roll;
  last_vroll_ = current_.vroll;
  last_center_ = current_.center_world;
  last_timestamp_ = timestamp;
  // tools::logger()->debug(
  //   "[JLU-Buff] SmallBuff TrackState={} center=({:.3f},{:.3f},{:.3f}) roll={:.4f} vroll={:.4f}",
  //   to_string(state_), current_.center_world.x(), current_.center_world.y(), current_.center_world.z(),
  //   current_.roll, current_.vroll);
  return current_;
}

// BuffState SmallBuffTarget::predict(double predict_sec) const
// {
//   auto out = current_;
//   out.roll = predictedRoll(predict_sec);
//   fill_blades_from_roll(out);
//   return out;
// }
BuffState SmallBuffTarget::predict(double predict_sec) const
{
  auto out = current_;

  const double delta_roll = current_.vroll * predict_sec;

  Eigen::Vector3d axis = out.rotation_axis_world;
  if (!axis.allFinite() || axis.norm() < 1e-6) {
    axis = Eigen::Vector3d::UnitZ();
  }
  axis.normalize();

  out.roll = predictedRoll(predict_sec);
  out.radius_vector_world =
    Eigen::AngleAxisd(delta_roll, axis) * out.radius_vector_world;

  fill_blades_from_roll(out);
  return out;
}
void SmallBuffTarget::reset()
{
  state_ = TrackState::LOST;
  current_ = {};
  has_last_ = false;
  frame_index_ = 0;
  last_roll_ = 0.0;
  last_vroll_ = 0.0;
  last_center_ = Eigen::Vector3d::Zero();
  gtsam::ISAM2Params params;
  params.relinearizeThreshold = 0.01;
  params.relinearizeSkip = 1;
  isam_ = gtsam::ISAM2(params);
}

double SmallBuffTarget::predictedRoll(double dt) const { return tools::limit_rad(current_.roll + current_.vroll * dt); }

BigBuffTarget::BigBuffTarget(TargetConfig config, BuffFitterConfig fitter_config)
: SmallBuffTarget(config), fitter_(std::make_unique<BuffFitter>(fitter_config))
{
}

BuffState BigBuffTarget::update(const std::vector<BuffBlade> & blades, TimePoint timestamp)
{
  if (!has_start_) {
    start_time_ = timestamp;
    has_start_ = true;
  }
  auto state = SmallBuffTarget::update(blades, timestamp);
  if (state.track_state == TrackState::TRACKING && fitter_) {
    fitter_->push(tools::delta_time(timestamp, start_time_), state.roll);
  } else if (state.track_state == TrackState::LOST && fitter_) {
    fitter_->reset();
  }
  return state;
}

// BuffState BigBuffTarget::predict(double predict_sec) const
// {
//   auto out = current_;
//   if (!has_start_ || !fitter_) return SmallBuffTarget::predict(predict_sec);
//   const double t_now = tools::delta_time(current_.timestamp, start_time_);
//   const auto p0 = fitter_->getBuffCurvePoint(t_now);
//   const auto p1 = fitter_->getBuffCurvePoint(t_now + predict_sec);
//   out.roll = tools::limit_rad(current_.roll + p1.angle - p0.angle);
//   out.vroll = p1.velocity;
//   fill_blades_from_roll(out);
//   return out;
// }
BuffState BigBuffTarget::predict(double predict_sec) const
{
  auto out = current_;

  if (!has_start_ || !fitter_) {
    return SmallBuffTarget::predict(predict_sec);
  }

  const double t_now = tools::delta_time(current_.timestamp, start_time_);

  const auto p0 = fitter_->getBuffCurvePoint(t_now);
  const auto p1 = fitter_->getBuffCurvePoint(t_now + predict_sec);

  const double delta_roll = p1.angle - p0.angle;

  Eigen::Vector3d axis = out.rotation_axis_world;
  if (!axis.allFinite() || axis.norm() < 1e-6) {
    axis = Eigen::Vector3d::UnitZ();
  }
  axis.normalize();

  out.roll = tools::limit_rad(current_.roll + delta_roll);
  out.vroll = p1.velocity;
  out.radius_vector_world =
    Eigen::AngleAxisd(delta_roll, axis) * out.radius_vector_world;

  fill_blades_from_roll(out);
  return out;
}

void BigBuffTarget::reset()
{
  SmallBuffTarget::reset();
  if (fitter_) fitter_->reset();
  has_start_ = false;
}

double BigBuffTarget::predictedRoll(double dt) const { return predict(dt).roll; }

TargetConfig load_target_config(const YAML::Node & node)
{
  TargetConfig c;
  c.lost_threshold_sec = read_or<double>(node, "lost_threshold_sec", c.lost_threshold_sec);
  auto match = node ? node["match_conf"] : YAML::Node{};
  c.match_conf.max_match_distance_m = read_or<double>(match, "max_match_distance_m", c.match_conf.max_match_distance_m);
  c.match_conf.max_match_roll_diff_degree = read_or<double>(match, "max_match_roll_diff_degree", c.match_conf.max_match_roll_diff_degree);
  return c;
}
}  // namespace auto_buff::jlu
