#pragma once

#include <array>
#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

namespace auto_buff::jlu
{
using TimePoint = std::chrono::steady_clock::time_point;

constexpr double kBuffRadius = 0.7;
constexpr int kBuffBladePointCount = 5;

enum class BuffMode { SMALL, BIG };
enum class TrackState { LOST, TEMP_LOST, TRACKING };
enum class BladeState { TARGET, UNACTIVATED, ACTIVATED };

std::string to_string(BuffMode mode);
std::string to_string(TrackState state);
std::string to_string(BladeState state);

struct BuffBladePoints
{
  // JLU five-point order, preserved by every adapter in this directory:
  // 0 r_center, 1 bottom_right, 2 top_right, 3 top_left, 4 bottom_left.
  std::array<cv::Point2f, kBuffBladePointCount> image{};
};

struct BuffBlade
{
  BuffBladePoints points;
  BladeState state = BladeState::TARGET;
  float confidence = 0.0F;
  cv::Rect2f rect{};

  bool pnp_ok = false;
  cv::Vec3d rvec{};
  cv::Vec3d tvec{};
  Eigen::Vector3d position_camera = Eigen::Vector3d::Zero();
  Eigen::Vector3d position_world = Eigen::Vector3d::Zero();
  Eigen::Vector3d center_world = Eigen::Vector3d::Zero();
  double roll = 0.0;
  TimePoint timestamp{};
};

struct BuffObservation
{
  std::vector<BuffBlade> blades;
  TimePoint timestamp{};
};

struct BuffState
{
  TrackState track_state = TrackState::LOST;
  Eigen::Vector3d center_world = Eigen::Vector3d::Zero();
  std::array<Eigen::Vector3d, 5> blade_world{};
  std::array<BladeState, 5> blade_states{};
  double roll = 0.0;
  double vroll = 0.0;
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
double blade_roll_from_points(const BuffBladePoints & points);
}  // namespace auto_buff::jlu
