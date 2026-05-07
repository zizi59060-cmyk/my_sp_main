#pragma once

#include <memory>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <yaml-cpp/yaml.h>

#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_buff/jlu_buff/infer/trt_yolo_buff.hpp"
#include "tasks/auto_buff/jlu_buff/targets.hpp"
#include "tasks/auto_buff/jlu_buff/trajectory.hpp"

namespace auto_buff::jlu
{
class JluBuffSystem
{
public:
  explicit JluBuffSystem(const std::string & config_path);

  bool enabled() const { return enable_; }
  bool autoFireEnabled() const { return auto_fire_enable_; }
  JluBuffPlan run(
    const cv::Mat & image, BuffMode mode, const Eigen::Quaterniond & gimbal_q,
    const io::GimbalState & gimbal_state, TimePoint timestamp);
  void reset();

private:
  void set_R_gimbal2world(const Eigen::Quaterniond & q);
  void solvePnPAndTransform(std::vector<BuffBlade> & blades, TimePoint timestamp) const;
  bool enable_ = true;
  bool auto_fire_enable_ = true;
  double predict_offset_sec_ = 0.070;

  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
  Eigen::Matrix3d R_gimbal2imubody_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d R_camera2gimbal_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_camera2gimbal_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_gimbal2world_ = Eigen::Matrix3d::Identity();

  std::unique_ptr<TrtBuffDetector> detector_;
  std::unique_ptr<SmallBuffTarget> small_target_;
  std::unique_ptr<BigBuffTarget> big_target_;
  Trajectory trajectory_;
  BuffMode last_mode_ = BuffMode::SMALL;
};
}  // namespace auto_buff::jlu
