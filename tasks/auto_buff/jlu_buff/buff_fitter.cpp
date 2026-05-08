#include "tasks/auto_buff/jlu_buff/buff_fitter.hpp"

#include <ceres/ceres.h>

#include <chrono>
#include <cmath>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_buff::jlu
{
namespace
{
struct CurveResidual
{
  CurveResidual(double t, double roll) : t_(t), roll_(roll) {}
  template <typename T>
  bool operator()(const T * const p, T * residual) const
  {
    const T pred = p[0] * sin(p[1] * T(t_) + p[2]) + p[3];
    // JLU big-buff fitting uses a wrapped angle residual. Keep the sin/cos double
    // residual so fitting remains continuous across +/-pi instead of directly
    // subtracting roll.
    residual[0] = sin(pred) - sin(T(roll_));
    residual[1] = cos(pred) - cos(T(roll_));
    return true;
  }
  double t_;
  double roll_;
};

template <typename T>
T read_or(const YAML::Node & n, const std::string & key, const T & fallback)
{
  return n && n[key] ? n[key].as<T>() : fallback;
}
}  // namespace

BuffFitter::BuffFitter(BuffFitterConfig config) : config_(config), thread_(&BuffFitter::loop, this) {}

BuffFitter::~BuffFitter()
{
  quit_ = true;
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void BuffFitter::push(double t_sec, double roll)
{
  {
    std::lock_guard lock(mutex_);
    history_.emplace_back(t_sec, roll);
    while (history_.size() > config_.queue_upper_limit) history_.pop_front();
  }
  cv_.notify_one();
}

void BuffFitter::reset()
{
  std::lock_guard lock(mutex_);
  history_.clear();
  params_ = {};
}

BuffCurveParams BuffFitter::params() const
{
  std::lock_guard lock(mutex_);
  return params_;
}

BuffCurvePoint BuffFitter::getBuffCurvePoint(double t_sec) const
{
  std::lock_guard lock(mutex_);
  BuffCurvePoint p;
  p.velocity = params_.a * std::sin(params_.omega * t_sec + params_.phi) + params_.b;
  p.angle = -(params_.a / params_.omega) * std::cos(params_.omega * t_sec + params_.phi) + params_.b * t_sec;
  return p;
}

void BuffFitter::loop()
{
  while (!quit_) {
    std::unique_lock lock(mutex_);
    cv_.wait_for(lock, std::chrono::milliseconds(config_.curve_fitting_interval_time_ms), [this] {
      return quit_.load() || history_.size() >= config_.queue_lower_limit;
    });
    lock.unlock();
    if (!quit_) fitOnce();
  }
}

void BuffFitter::fitOnce()
{
  std::deque<std::pair<double, double>> history;
  BuffCurveParams current;
  {
    std::lock_guard lock(mutex_);
    if (history_.size() < config_.queue_lower_limit) return;
    history = history_;
    current = params_;
  }

  double p[4] = {current.a, current.omega, current.phi, current.b};
  ceres::Problem problem;
  for (const auto & [t, roll] : history) {
    problem.AddResidualBlock(
      new ceres::AutoDiffCostFunction<CurveResidual, 2, 4>(new CurveResidual(t, roll)),
      new ceres::CauchyLoss(0.5), p);
  }
  problem.SetParameterLowerBound(p, 0, 0.6 * config_.param_lower_bound_scale);
  problem.SetParameterUpperBound(p, 0, 1.1 * config_.param_upper_bound_scale);
  problem.SetParameterLowerBound(p, 1, 1.6 * config_.param_lower_bound_scale);
  problem.SetParameterUpperBound(p, 1, 2.2 * config_.param_upper_bound_scale);
  problem.SetParameterLowerBound(p, 3, 0.8 * config_.param_lower_bound_scale);
  problem.SetParameterUpperBound(p, 3, 1.6 * config_.param_upper_bound_scale);

  ceres::Solver::Options options;
  options.max_num_iterations = 30;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  if (summary.IsSolutionUsable()) {
    std::lock_guard lock(mutex_);
    params_ = {p[0], p[1], tools::limit_rad(p[2]), p[3]};
    // tools::logger()->debug(
    //   "[JLU-Buff] BigBuff fitter params: a={:.4f} omega={:.4f} phi={:.4f} b={:.4f}",
    //   params_.a, params_.omega, params_.phi, params_.b);
  }
}

BuffFitterConfig load_fitter_config(const YAML::Node & node)
{
  BuffFitterConfig c;
  c.queue_upper_limit = read_or<size_t>(node, "queue_upper_limit", c.queue_upper_limit);
  c.queue_lower_limit = read_or<size_t>(node, "queue_lower_limit", c.queue_lower_limit);
  c.param_lower_bound_scale = read_or<double>(node, "param_lower_bound_scale", c.param_lower_bound_scale);
  c.param_upper_bound_scale = read_or<double>(node, "param_upper_bound_scale", c.param_upper_bound_scale);
  c.curve_fitting_interval_time_ms = read_or<int>(node, "curve_fitting_interval_time_ms", c.curve_fitting_interval_time_ms);
  return c;
}
}  // namespace auto_buff::jlu
