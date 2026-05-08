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

struct JluBuffDebug
{
  BuffMode mode = BuffMode::SMALL;

  // Detector 原始输出，未 PnP
  std::vector<BuffBlade> detected_blades;

  // solvePnPAndTransform 之后的 blades
  std::vector<BuffBlade> pnp_blades;

  // 最终输出计划
  JluBuffPlan plan;

  // Trajectory 最终瞄准点
  bool aim_point_valid = false;
  Eigen::Vector3d aim_point_world = Eigen::Vector3d::Zero();

  // aim_point_world 投影回图像后的像素点
  bool aim_point_image_valid = false;
  cv::Point2f aim_point_image{0.0F, 0.0F};

  double detect_ms = 0.0;
  double pnp_ms = 0.0;
  double total_ms = 0.0;

  float bullet_speed = 0.0F;
};

class JluBuffSystem
{
public:
  explicit JluBuffSystem(const std::string & config_path);

  bool enabled() const { return enable_; }
  bool autoFireEnabled() const { return auto_fire_enable_; }

  const JluBuffDebug & debug() const { return debug_; }

  JluBuffPlan run(
    const cv::Mat & image,
    BuffMode mode,
    const Eigen::Quaterniond & gimbal_q,
    const io::GimbalState & gimbal_state,
    TimePoint timestamp);

  void reset();

private:
  void set_R_gimbal2world(const Eigen::Quaterniond & q);

  void solvePnPAndTransform(std::vector<BuffBlade> & blades, TimePoint timestamp) const;

  bool projectWorldToImage(
    const Eigen::Vector3d & point_world,
    cv::Point2f & image_point) const;

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

  JluBuffDebug debug_;
};

}  // namespace auto_buff::jlu是