#pragma once

#include <deque>
#include <memory>

#include <gtsam/nonlinear/ISAM2.h>
#include <yaml-cpp/yaml.h>

#include "tasks/auto_buff/jlu_buff/buff_fitter.hpp"
#include "tasks/auto_buff/jlu_buff/types.hpp"

namespace auto_buff::jlu
{
struct MatchConfig
{
  double max_match_distance_m = 1.0;
  double max_match_roll_diff_degree = 30.0;
};

struct TargetConfig
{
  double lost_threshold_sec = 0.8;
  MatchConfig match_conf;
};

class SmallBuffTarget
{
public:
  explicit SmallBuffTarget(TargetConfig config = {});
  virtual ~SmallBuffTarget() = default;

  virtual BuffState update(const std::vector<BuffBlade> & blades, TimePoint timestamp);
  virtual BuffState predict(double predict_sec) const;
  virtual void reset();
  TrackState state() const { return state_; }

protected:
  virtual double predictedRoll(double dt) const;
  TargetConfig config_;
  TrackState state_ = TrackState::LOST;
  BuffState current_;
  bool has_last_ = false;
  double last_roll_ = 0.0;
  TimePoint last_timestamp_{};
  gtsam::ISAM2 isam_;
};

class BigBuffTarget : public SmallBuffTarget
{
public:
  BigBuffTarget(TargetConfig config, BuffFitterConfig fitter_config);
  BuffState update(const std::vector<BuffBlade> & blades, TimePoint timestamp) override;
  BuffState predict(double predict_sec) const override;
  void reset() override;
private:
  double predictedRoll(double dt) const override;
  std::unique_ptr<BuffFitter> fitter_;
  TimePoint start_time_{};
  bool has_start_ = false;
};

TargetConfig load_target_config(const YAML::Node & node);
}  // namespace auto_buff::jlu
