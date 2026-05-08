#pragma once

#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <opencv2/opencv.hpp>

#include "tasks/auto_buff/jlu_buff/types.hpp"

namespace auto_buff::jlu
{
// Keep the same factor names as the JLU buff_tracker. These factors are framework-free
// versions adapted to my_sp_main time/extrinsic types.
class ConstPositionFactor : public gtsam::NoiseModelFactor2<gtsam::Point3, gtsam::Point3>
{
public:
  ConstPositionFactor(
    gtsam::Key previous_center_key, gtsam::Key current_center_key,
    const gtsam::SharedNoiseModel & model);

  gtsam::Vector evaluateError(
    const gtsam::Point3 & previous, const gtsam::Point3 & current,
    boost::optional<gtsam::Matrix &> H1 = boost::none,
    boost::optional<gtsam::Matrix &> H2 = boost::none) const override;
};

class RollFactor : public gtsam::NoiseModelFactor1<double>
{
public:
  RollFactor(gtsam::Key roll_key, double measured_roll, const gtsam::SharedNoiseModel & model);
  gtsam::Vector evaluateError(
    const double & roll, boost::optional<gtsam::Matrix &> H = boost::none) const override;

private:
  double measured_roll_;
};

class ConstVRollFactor : public gtsam::NoiseModelFactor3<double, double, double>
{
public:
  ConstVRollFactor(
    gtsam::Key previous_roll_key, gtsam::Key current_roll_key, gtsam::Key previous_vroll_key,
    double dt, const gtsam::SharedNoiseModel & model);

  gtsam::Vector evaluateError(
    const double & previous_roll, const double & current_roll, const double & previous_vroll,
    boost::optional<gtsam::Matrix &> H1 = boost::none,
    boost::optional<gtsam::Matrix &> H2 = boost::none,
    boost::optional<gtsam::Matrix &> H3 = boost::none) const override;

private:
  double dt_;
};

class BuffBladeReprojFactor : public gtsam::NoiseModelFactor1<gtsam::Point3>
{
public:
  BuffBladeReprojFactor(
    gtsam::Key center_key, BuffBladePoints points, cv::Mat camera_matrix, cv::Mat distort_coeffs,
    const gtsam::SharedNoiseModel & model);
  gtsam::Vector evaluateError(
    const gtsam::Point3 & center, boost::optional<gtsam::Matrix &> H = boost::none) const override;

private:
  BuffBladePoints points_;
  cv::Mat camera_matrix_;
  cv::Mat distort_coeffs_;
};

class BuffBladeFactor : public gtsam::NoiseModelFactor2<gtsam::Point3, double>
{
public:
  BuffBladeFactor(
    gtsam::Key center_key, gtsam::Key roll_key, const Eigen::Vector3d & blade_position,
    double blade_roll, int blade_index, const gtsam::SharedNoiseModel & model);

  gtsam::Vector evaluateError(
    const gtsam::Point3 & center, const double & roll,
    boost::optional<gtsam::Matrix &> H1 = boost::none,
    boost::optional<gtsam::Matrix &> H2 = boost::none) const override;

private:
  Eigen::Vector3d blade_position_;
  double blade_roll_;
  int blade_index_;
};
}  // namespace auto_buff::jlu
