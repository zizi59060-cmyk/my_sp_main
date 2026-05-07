#include "tasks/auto_buff/jlu_buff/targets.hpp"

#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <boost/make_shared.hpp>
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

BuffBlade choose_best(const std::vector<BuffBlade> & blades)
{
  return *std::max_element(blades.begin(), blades.end(), [](const auto & a, const auto & b) { return a.confidence < b.confidence; });
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
    } else {
      state_ = TrackState::LOST;
      current_.track_state = state_;
    }
    tools::logger()->debug("[JLU-Buff] TrackState={}", to_string(state_));
    return current_;
  }

  auto best = choose_best(blades);
  const double dt = has_last_ ? std::max(1e-3, tools::delta_time(timestamp, last_timestamp_)) : 1e-3;
  const double vroll = has_last_ ? tools::limit_rad(best.roll - last_roll_) / dt : 0.0;

  current_.track_state = TrackState::TRACKING;
  current_.center_world = best.center_world;
  current_.roll = best.roll;
  current_.vroll = std::isfinite(vroll) ? vroll : 0.0;
  current_.timestamp = timestamp;
  current_.blade_world.fill(best.position_world);
  current_.blade_states.fill(BladeState::UNACTIVATED);
  current_.blade_states[0] = best.state;

  // Keep a lightweight ISAM2 graph in the loop, matching the JLU factor-graph structure while
  // using my_sp_main's already transformed world measurements.
  try {
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values init;
    const auto center_key = gtsam::Symbol('c', 0);
    const auto roll_key = gtsam::Symbol('r', 0);
    if (!isam_.valueExists(center_key)) {
      init.insert(center_key, gtsam::Point3(best.center_world.x(), best.center_world.y(), best.center_world.z()));
      init.insert(roll_key, best.roll);
      graph.add(gtsam::PriorFactor<gtsam::Point3>(
        center_key, gtsam::Point3(best.center_world.x(), best.center_world.y(), best.center_world.z()),
        gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(3) << 0.10, 0.10, 0.10).finished())));
    }
    graph.add(boost::make_shared<BuffBladeFactor>(
      center_key, roll_key, best.position_world, best.roll,
      gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(4) << 0.03, 0.03, 0.03, 2.0 / 57.3).finished())));
    isam_.update(graph, init);
    const auto values = isam_.calculateEstimate();
    if (values.exists(center_key)) {
      const auto c = values.at<gtsam::Point3>(center_key);
      current_.center_world = Eigen::Vector3d(c.x(), c.y(), c.z());
    }
    if (values.exists(roll_key)) current_.roll = values.at<double>(roll_key);
  } catch (const std::exception & e) {
    tools::logger()->warn("[JLU-Buff] ISAM2 update skipped: {}", e.what());
  }

  state_ = TrackState::TRACKING;
  has_last_ = true;
  last_roll_ = best.roll;
  last_timestamp_ = timestamp;
  tools::logger()->debug("[JLU-Buff] SmallBuff roll={:.4f} vroll={:.4f}", current_.roll, current_.vroll);
  return current_;
}

BuffState SmallBuffTarget::predict(double predict_sec) const
{
  auto out = current_;
  out.roll = predictedRoll(predict_sec);
  return out;
}

void SmallBuffTarget::reset()
{
  state_ = TrackState::LOST;
  current_ = {};
  has_last_ = false;
  gtsam::ISAM2Params params;
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

BuffState BigBuffTarget::predict(double predict_sec) const
{
  auto out = current_;
  if (!has_start_ || !fitter_) return SmallBuffTarget::predict(predict_sec);
  const double t_now = tools::delta_time(current_.timestamp, start_time_);
  const auto p0 = fitter_->getBuffCurvePoint(t_now);
  const auto p1 = fitter_->getBuffCurvePoint(t_now + predict_sec);
  out.roll = tools::limit_rad(current_.roll + p1.angle - p0.angle);
  out.vroll = p1.velocity;
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
