#include "tasks/auto_buff/jlu_buff/types.hpp"

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

std::vector<cv::Point3f> buff_blade_object_points(double radius)
{
  // JLU convention: r_center, bottom_right, top_right, top_left, bottom_left.
  // Keep the measured-layout style from JLU instead of a synthetic 3-D box: all five
  // points are on the fan plane, with the blade rectangle centered on BUFF_RADIUS.
  // Unit: metre. If the team's official CAD values change, only these constants need
  // to be updated; the point order and IPPE PnP path remain unchanged.
  constexpr float blade_width = 0.230F;
  constexpr float blade_height = 0.127F;
  const float half_width = blade_width * 0.5F;
  const float half_height = blade_height * 0.5F;
  const float r = static_cast<float>(radius);
  return {
    {0.0F, 0.0F, 0.0F},
    {half_width, r - half_height, 0.0F},
    {half_width, r + half_height, 0.0F},
    {-half_width, r + half_height, 0.0F},
    {-half_width, r - half_height, 0.0F},
  };
}

Eigen::Vector3d buff_blade_center_object_point(double radius)
{
  return {0.0, radius, 0.0};
}

double blade_roll_from_points(const BuffBladePoints & points)
{
  const auto v = points.image[0] - (points.image[1] + points.image[4]) * 0.5F;
  return std::atan2(static_cast<double>(v.y), static_cast<double>(v.x));
}
}  // namespace auto_buff::jlu
