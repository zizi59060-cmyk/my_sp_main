#pragma once

#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <opencv2/opencv.hpp>

#include "tasks/auto_buff/jlu_buff/types.hpp"

namespace auto_buff::jlu
{
class ConstPositionFactor : public gtsam::NoiseModelFactor1<gtsam::Point3>
{
public:
  ConstPositionFactor(gtsam::Key key, const gtsam::Point3 & prior, const gtsam::SharedNoiseModel & model);
  gtsam::Vector evaluateError(const gtsam::Point3 & p, boost::optional<gtsam::Matrix &> H = boost::none) const override;
private:
  gtsam::Point3 prior_;
};

class RollFactor : public gtsam::NoiseModelFactor1<double>
{
public:
  RollFactor(gtsam::Key key, double measured_roll, const gtsam::SharedNoiseModel & model);
  gtsam::Vector evaluateError(const double & roll, boost::optional<gtsam::Matrix &> H = boost::none) const override;
private:
  double measured_roll_;
};

class ConstVRollFactor : public gtsam::NoiseModelFactor1<double>
{
public:
  ConstVRollFactor(gtsam::Key key, double prior, const gtsam::SharedNoiseModel & model);
  gtsam::Vector evaluateError(const double & vroll, boost::optional<gtsam::Matrix &> H = boost::none) const override;
private:
  double prior_;
};

class BuffBladeReprojFactor : public gtsam::NoiseModelFactor1<gtsam::Point3>
{
public:
  BuffBladeReprojFactor(
    gtsam::Key center_key, BuffBladePoints points, cv::Mat camera_matrix, cv::Mat distort_coeffs,
    const gtsam::SharedNoiseModel & model);
  gtsam::Vector evaluateError(const gtsam::Point3 & center, boost::optional<gtsam::Matrix &> H = boost::none) const override;
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
    double blade_roll, const gtsam::SharedNoiseModel & model);
  gtsam::Vector evaluateError(
    const gtsam::Point3 & center, const double & roll, boost::optional<gtsam::Matrix &> H1 = boost::none,
    boost::optional<gtsam::Matrix &> H2 = boost::none) const override;
private:
  Eigen::Vector3d blade_position_;
  double blade_roll_;
};
}  // namespace auto_buff::jlu
