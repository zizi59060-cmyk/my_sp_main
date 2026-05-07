#include "tasks/auto_buff/jlu_buff/factors.hpp"

#include <gtsam/base/Vector.h>

#include <cmath>

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
  const gtsam::Point3 & previous, const gtsam::Point3 & current, boost::optional<gtsam::Matrix &> H1,
  boost::optional<gtsam::Matrix &> H2) const
{
  if (H1) *H1 = -gtsam::Matrix::Identity(3, 3);
  if (H2) *H2 = gtsam::Matrix::Identity(3, 3);
  return (current - previous).vector();
}

RollFactor::RollFactor(gtsam::Key roll_key, double measured_roll, const gtsam::SharedNoiseModel & model)
: NoiseModelFactor1(model, roll_key), measured_roll_(measured_roll)
{
}

gtsam::Vector RollFactor::evaluateError(const double & roll, boost::optional<gtsam::Matrix &> H) const
{
  if (H) *H = gtsam::Matrix::Identity(1, 1);
  gtsam::Vector1 e;
  e << tools::limit_rad(roll - measured_roll_);
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
  if (H1) (*H1) = (gtsam::Matrix(1, 1) << -1.0).finished();
  if (H2) (*H2) = (gtsam::Matrix(1, 1) << 1.0).finished();
  if (H3) (*H3) = (gtsam::Matrix(1, 1) << -dt_).finished();
  gtsam::Vector1 e;
  e << tools::limit_rad(current_roll - previous_roll - previous_vroll * dt_);
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
  if (H) *H = gtsam::Matrix::Zero(10, 3);
  gtsam::Vector error(10);
  const auto object = buff_blade_object_points();
  cv::Vec3d rvec(0, 0, 0), tvec(center.x(), center.y(), center.z());
  std::vector<cv::Point2f> proj;
  cv::projectPoints(object, rvec, tvec, camera_matrix_, distort_coeffs_, proj);
  for (int i = 0; i < kBuffBladePointCount; ++i) {
    error(2 * i) = proj[i].x - points_.image[i].x;
    error(2 * i + 1) = proj[i].y - points_.image[i].y;
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
  if (H1) *H1 = gtsam::Matrix::Zero(4, 3);
  if (H2) *H2 = gtsam::Matrix::Zero(4, 1);

  const double blade_roll = roll + blade_index_ * 2.0 * CV_PI / 5.0;
  const Eigen::Vector3d predicted_blade =
    center.vector() + Eigen::AngleAxisd(blade_roll, Eigen::Vector3d::UnitZ()) * buff_blade_center_object_point(kBuffRadius);

  gtsam::Vector e(4);
  e.segment<3>(0) = predicted_blade - blade_position_;
  e(3) = tools::limit_rad(roll - blade_roll_);
  return e;
}
}  // namespace auto_buff::jlu
