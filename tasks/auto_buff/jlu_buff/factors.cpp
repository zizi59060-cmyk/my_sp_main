#include "tasks/auto_buff/jlu_buff/factors.hpp"

#include <gtsam/base/Vector.h>

#include <cmath>
#include <utility>

#include <opencv2/calib3d.hpp>

#include "tools/math_tools.hpp"

namespace auto_buff::jlu
{

ConstPositionFactor::ConstPositionFactor(
  gtsam::Key previous_center_key, gtsam::Key current_center_key,
  const gtsam::SharedNoiseModel & model)
: NoiseModelFactor2(model, previous_center_key, current_center_key)
{
}

gtsam::Vector ConstPositionFactor::evaluateError(
  const gtsam::Point3 & previous, const gtsam::Point3 & current,
  boost::optional<gtsam::Matrix &> H1, boost::optional<gtsam::Matrix &> H2) const
{
  if (H1) {
    *H1 = -gtsam::Matrix::Identity(3, 3);
  }

  if (H2) {
    *H2 = gtsam::Matrix::Identity(3, 3);
  }

  gtsam::Vector e(3);
  e(0) = current.x() - previous.x();
  e(1) = current.y() - previous.y();
  e(2) = current.z() - previous.z();

  return e;
}

RollFactor::RollFactor(
  gtsam::Key roll_key, double measured_roll, const gtsam::SharedNoiseModel & model)
: NoiseModelFactor1(model, roll_key), measured_roll_(measured_roll)
{
}

gtsam::Vector RollFactor::evaluateError(
  const double & roll, boost::optional<gtsam::Matrix &> H) const
{
  if (H) {
    *H = gtsam::Matrix::Identity(1, 1);
  }

  gtsam::Vector e(1);
  e(0) = tools::limit_rad(roll - measured_roll_);

  return e;
}

ConstVRollFactor::ConstVRollFactor(
  gtsam::Key previous_roll_key, gtsam::Key current_roll_key, gtsam::Key previous_vroll_key,
  double dt, const gtsam::SharedNoiseModel & model)
: NoiseModelFactor3(model, previous_roll_key, current_roll_key, previous_vroll_key), dt_(dt)
{
}

gtsam::Vector ConstVRollFactor::evaluateError(
  const double & previous_roll, const double & current_roll, const double & previous_vroll,
  boost::optional<gtsam::Matrix &> H1, boost::optional<gtsam::Matrix &> H2,
  boost::optional<gtsam::Matrix &> H3) const
{
  if (H1) {
    *H1 = gtsam::Matrix::Zero(1, 1);
    (*H1)(0, 0) = -1.0;
  }

  if (H2) {
    *H2 = gtsam::Matrix::Zero(1, 1);
    (*H2)(0, 0) = 1.0;
  }

  if (H3) {
    *H3 = gtsam::Matrix::Zero(1, 1);
    (*H3)(0, 0) = -dt_;
  }

  gtsam::Vector e(1);
  e(0) = tools::limit_rad(current_roll - previous_roll - previous_vroll * dt_);

  return e;
}

BuffBladeReprojFactor::BuffBladeReprojFactor(
  gtsam::Key center_key, BuffBladePoints points, cv::Mat camera_matrix, cv::Mat distort_coeffs,
  const gtsam::SharedNoiseModel & model)
: NoiseModelFactor1(model, center_key),
  points_(points),
  camera_matrix_(std::move(camera_matrix)),
  distort_coeffs_(std::move(distort_coeffs))
{
}

gtsam::Vector BuffBladeReprojFactor::evaluateError(
  const gtsam::Point3 & center, boost::optional<gtsam::Matrix &> H) const
{
  if (H) {
    *H = gtsam::Matrix::Zero(10, 3);
  }

  gtsam::Vector error(10);

  const auto object_points = buff_blade_object_points();

  cv::Vec3d rvec(0.0, 0.0, 0.0);
  cv::Vec3d tvec(center.x(), center.y(), center.z());

  std::vector<cv::Point2f> projected_points;
  cv::projectPoints(
    object_points, rvec, tvec, camera_matrix_, distort_coeffs_, projected_points);

  for (int i = 0; i < kBuffBladePointCount; ++i) {
    error(2 * i) = projected_points[i].x - points_.image[i].x;
    error(2 * i + 1) = projected_points[i].y - points_.image[i].y;
  }

  return error;
}

BuffBladeFactor::BuffBladeFactor(
  gtsam::Key center_key, gtsam::Key roll_key, const Eigen::Vector3d & blade_position,
  double blade_roll, int blade_index, const gtsam::SharedNoiseModel & model)
: NoiseModelFactor2(model, center_key, roll_key),
  blade_position_(blade_position),
  blade_roll_(blade_roll),
  blade_index_(blade_index)
{
}

gtsam::Vector BuffBladeFactor::evaluateError(
  const gtsam::Point3 & center, const double & roll, boost::optional<gtsam::Matrix &> H1,
  boost::optional<gtsam::Matrix &> H2) const
{
  if (H1) {
    *H1 = gtsam::Matrix::Zero(4, 3);
    H1->block<3, 3>(0, 0) = gtsam::Matrix3::Identity();
  }

  if (H2) {
    *H2 = gtsam::Matrix::Zero(4, 1);
    (*H2)(3, 0) = 1.0;
  }

  const double predicted_blade_roll =
    tools::limit_rad(roll + blade_index_ * kBuffBladeRollStep);

  gtsam::Vector e(4);

  e(0) = center.x() - blade_position_.x();
  e(1) = center.y() - blade_position_.y();
  e(2) = center.z() - blade_position_.z();

  // 注意这里必须比较 predicted_blade_roll，而不是 roll。
  // 非 0 号扇叶需要加 blade_index * 2pi / 5。
  e(3) = tools::limit_rad(predicted_blade_roll - blade_roll_);

  return e;
}

} // namespace auto_buff::jlu