#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include <yaml-cpp/yaml.h>

namespace auto_buff::jlu
{
struct BuffFitterConfig
{
  size_t queue_upper_limit = 200;
  size_t queue_lower_limit = 100;
  double param_lower_bound_scale = 1.0;
  double param_upper_bound_scale = 1.0;
  int curve_fitting_interval_time_ms = 20;
};

struct BuffCurveParams
{
  double a = 0.78;
  double omega = 1.884;
  double phi = 0.0;
  double b = 1.305;
};

struct BuffCurvePoint
{
  double angle = 0.0;
  double velocity = 0.0;
};

class BuffFitter
{
public:
  explicit BuffFitter(BuffFitterConfig config = {});
  ~BuffFitter();

  void push(double t_sec, double roll);
  void reset();
  BuffCurvePoint getBuffCurvePoint(double t_sec) const;
  BuffCurveParams params() const;

private:
  void loop();
  void fitOnce();

  BuffFitterConfig config_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::pair<double, double>> history_;
  BuffCurveParams params_;
  std::atomic<bool> quit_{false};
  std::thread thread_;
};

BuffFitterConfig load_fitter_config(const YAML::Node & node);
}  // namespace auto_buff::jlu
