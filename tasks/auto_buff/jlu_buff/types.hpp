#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace auto_buff::jlu
{

using TimePoint = std::chrono::steady_clock::time_point;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kBuffRadius = 0.7;
constexpr double kBuffBladeRollStep = 2.0 * kPi / 5.0;
constexpr int kBuffBladePointCount = 5;

enum class BuffMode { SMALL, BIG };

enum class TrackState { LOST, TEMP_LOST, TRACKING };

enum class BladeState { TARGET, UNACTIVATED, ACTIVATED };

enum class BuffColor { UNKNOWN, RED, BLUE };

enum class BuffBladeType { UNKNOWN, TARGET, NORMAL, R_CENTER };

struct Header
{
  uint64_t seq = 0;
  TimePoint stamp{};
  std::string frame_id;
};

std::string to_string(BuffMode mode);

std::string to_string(TrackState state);

std::string to_string(BladeState state);

struct BuffBladePoints
{
  // JLU original five-point order:
  // 0 r_center, 1 bottom_right, 2 top_right, 3 top_left, 4 bottom_left.
  std::array<cv::Point2f, kBuffBladePointCount> image{};

  cv::Point2f r_center() const { return image[0]; }

  cv::Point2f bottom_right() const { return image[1]; }

  cv::Point2f top_right() const { return image[2]; }

  cv::Point2f top_left() const { return image[3]; }

  cv::Point2f bottom_left() const { return image[4]; }
};

struct BuffBlade
{
  Header header;

  BuffBladePoints points;

  BuffColor color = BuffColor::UNKNOWN;

  BuffBladeType type = BuffBladeType::UNKNOWN;

  BladeState state = BladeState::TARGET;

  int track_id = -1;

  float confidence = 0.0F;

  cv::Rect2f rect{};

  bool pnp_ok = false;

  cv::Vec3d rvec{};

  cv::Vec3d tvec{};

  // In the original JLU tracker, blade.position is the R-center position.
  Eigen::Vector3d position_camera = Eigen::Vector3d::Zero();

  Eigen::Vector3d position_world = Eigen::Vector3d::Zero();

  Eigen::Vector3d center_world = Eigen::Vector3d::Zero();

  Eigen::Matrix3d R_buff2world = Eigen::Matrix3d::Identity();

  // Original JLU convention:
  // roll = 0 means the hit point is vertically upward.
  double roll = 0.0;

  TimePoint timestamp{};
};

struct BuffObservation
{
  Header header;

  std::vector<BuffBlade> blades;

  TimePoint timestamp{};
};

struct BuffState
{
  TrackState track_state = TrackState::LOST;

  Eigen::Vector3d center_world = Eigen::Vector3d::Zero();

  // Hit positions of the five physical blades.
  std::array<Eigen::Vector3d, 5> blade_world{};

  std::array<Eigen::Matrix3d, 5> blade_orientations{};

  std::array<BladeState, 5> blade_states{};

  // Base roll of blade 0.
  double roll = 0.0;

  double vroll = 0.0;

  Eigen::Vector3d rotation_axis_world = Eigen::Vector3d::UnitX();

  Eigen::Vector3d radius_vector_world = Eigen::Vector3d(0.0, 0.0, kBuffRadius);

  TimePoint timestamp{};
};

struct TrajectorySolution
{
  bool valid = false;

  bool fire = false;

  int selected_blade = -1;

  double yaw = 0.0;

  double pitch = 0.0;

  double fly_time = 0.0;

  double bullet_speed = 0.0;

  Eigen::Vector3d aim_point_world = Eigen::Vector3d::Zero();
};

struct JluBuffPlan
{
  bool control = false;

  bool fire = false;

  float yaw = 0.0F;

  float yaw_vel = 0.0F;

  float yaw_acc = 0.0F;

  float pitch = 0.0F;

  float pitch_vel = 0.0F;

  float pitch_acc = 0.0F;

  TrackState track_state = TrackState::LOST;

  int blade_index = -1;

  double fly_time = 0.0;
};

std::vector<cv::Point3f> buff_blade_object_points(double radius = kBuffRadius);

Eigen::Vector3d buff_blade_center_object_point(double radius = kBuffRadius);

double normalize_buff_roll(double angle);

Eigen::Vector3d buff_rotation_axis_from_center(const Eigen::Vector3d & center_world);

Eigen::Vector3d buff_tangent_from_center(const Eigen::Vector3d & center_world);

Eigen::Vector3d buff_blade_hit_position_from_center_roll(
  const Eigen::Vector3d & center_world, double roll, double radius = kBuffRadius);

Eigen::Matrix3d buff_blade_orientation_from_center_roll(
  const Eigen::Vector3d & center_world, double roll);

Eigen::Vector3d buff_rotation_matrix_to_rpy(const Eigen::Matrix3d & R);

double blade_roll_from_points(const BuffBladePoints & points);

} // namespace auto_buff::jlu