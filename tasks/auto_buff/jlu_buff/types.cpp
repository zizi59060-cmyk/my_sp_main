#include "tasks/auto_buff/jlu_buff/types.hpp"

#include <algorithm>
#include <cmath>

namespace auto_buff::jlu
{

std::string to_string(BuffMode mode)
{
  return mode == BuffMode::SMALL ? "SMALL_BUFF" : "BIG_BUFF";
}

std::string to_string(TrackState state)
{
  switch (state) {
    case TrackState::LOST:
      return "LOST";
    case TrackState::TEMP_LOST:
      return "TEMP_LOST";
    case TrackState::TRACKING:
      return "TRACKING";
  }

  return "UNKNOWN";
}

std::string to_string(BladeState state)
{
  switch (state) {
    case BladeState::TARGET:
      return "TARGET";
    case BladeState::UNACTIVATED:
      return "UNACTIVATED";
    case BladeState::ACTIVATED:
      return "ACTIVATED";
  }

  return "UNKNOWN";
}

double normalize_buff_roll(double angle)
{
  while (angle > kPi) angle -= 2.0 * kPi;
  while (angle <= -kPi) angle += 2.0 * kPi;
  return angle;
}

std::vector<cv::Point3f> buff_blade_object_points(double radius)
{
  (void)radius;

  // Original JLU CAD points.
  // Order:
  // r_center, bottom_right, top_right, top_left, bottom_left.
  // Unit: metre.
  //
  // The blade lies in local X = 0 plane.
  return {
    {0.0F, 0.0F, 0.0F},
    {0.0F, -0.186F, 0.5415F},
    {0.0F, -0.160F, 0.8585F},
    {0.0F, 0.160F, 0.8585F},
    {0.0F, 0.186F, 0.5415F},
  };
}

Eigen::Vector3d buff_blade_center_object_point(double radius)
{
  // Kept for compatibility with old call sites.
  // In original JLU this is the hit point relative to r_center:
  // BUFF_BLADE_HIT_OBJ_POINT(0, 0, BUFF_RADIUS).
  return {0.0, 0.0, radius};
}

Eigen::Vector3d buff_rotation_axis_from_center(const Eigen::Vector3d & center_world)
{
  Eigen::Vector3d axis(center_world.x(), center_world.y(), 0.0);

  if (axis.squaredNorm() < 1e-12) {
    return Eigen::Vector3d::UnitX();
  }

  return axis.normalized();
}

Eigen::Vector3d buff_tangent_from_center(const Eigen::Vector3d & center_world)
{
  const Eigen::Vector3d axis = buff_rotation_axis_from_center(center_world);
  Eigen::Vector3d tangent = Eigen::Vector3d::UnitZ().cross(axis);

  if (tangent.squaredNorm() < 1e-12) {
    return Eigen::Vector3d::UnitY();
  }

  return tangent.normalized();
}

Eigen::Vector3d buff_blade_hit_position_from_center_roll(
  const Eigen::Vector3d & center_world, double roll, double radius)
{
  // Exactly the original BladePositionRoll::getHitPosition() geometry:
  //
  // z_hit = cos(roll) * BUFF_RADIUS + center.z()
  // horizontal_bias = -sin(roll) * BUFF_RADIUS
  // center_aim_yaw = atan2(center.y(), center.x())
  // x_hit = center.x() - horizontal_bias * sin(center_aim_yaw)
  // y_hit = center.y() + horizontal_bias * cos(center_aim_yaw)

  const double z_hit = std::cos(roll) * radius + center_world.z();
  const double horizontal_bias = -std::sin(roll) * radius;
  const double center_aim_yaw = std::atan2(center_world.y(), center_world.x());

  const double x_hit = center_world.x() - horizontal_bias * std::sin(center_aim_yaw);
  const double y_hit = center_world.y() + horizontal_bias * std::cos(center_aim_yaw);

  return {x_hit, y_hit, z_hit};
}

Eigen::Matrix3d buff_blade_orientation_from_center_roll(
  const Eigen::Vector3d & center_world, double roll)
{
  const Eigen::Vector3d axis = buff_rotation_axis_from_center(center_world);
  const Eigen::Vector3d tangent = buff_tangent_from_center(center_world);
  const Eigen::Vector3d up = Eigen::Vector3d::UnitZ();

  Eigen::Matrix3d R_plane = Eigen::Matrix3d::Identity();
  R_plane.col(0) = axis;
  R_plane.col(1) = tangent;
  R_plane.col(2) = up;

  return R_plane * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();
}

Eigen::Vector3d buff_rotation_matrix_to_rpy(const Eigen::Matrix3d & R)
{
  // R = Rz(yaw) * Ry(pitch) * Rx(roll).
  // This matches the original JLU usage of rotationMatrixToRPY():
  // rpy(0) is roll, rpy(1) is pitch, rpy(2) is yaw.
  const double pitch = std::asin(std::clamp(-R(2, 0), -1.0, 1.0));

  double roll = 0.0;
  double yaw = 0.0;

  const double cos_pitch = std::cos(pitch);

  if (std::abs(cos_pitch) > 1e-8) {
    roll = std::atan2(R(2, 1), R(2, 2));
    yaw = std::atan2(R(1, 0), R(0, 0));
  } else {
    roll = 0.0;
    yaw = std::atan2(-R(0, 1), R(1, 1));
  }

  return {
    normalize_buff_roll(roll),
    normalize_buff_roll(pitch),
    normalize_buff_roll(yaw),
  };
}

double blade_roll_from_points(const BuffBladePoints & points)
{
  // Fallback only.
  // The original JLU tracker gets roll from solvePnP pose, not from 2D keypoint direction.
  const cv::Point2f blade_mid{
    0.25F * (points.image[1].x + points.image[2].x + points.image[3].x + points.image[4].x),
    0.25F * (points.image[1].y + points.image[2].y + points.image[3].y + points.image[4].y),
  };

  const cv::Point2f v{
    blade_mid.x - points.image[0].x,
    blade_mid.y - points.image[0].y,
  };

  if (v.x * v.x + v.y * v.y < 1e-6F) {
    return 0.0;
  }

  const double angle_with_image_y_up =
    std::atan2(static_cast<double>(-v.y), static_cast<double>(v.x));

  return normalize_buff_roll(angle_with_image_y_up - kPi * 0.5);
}

} // namespace auto_buff::jlu