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
  // The blade points are on a plane tangent to the fan radius. 0.7 m is JLU BUFF_RADIUS.
  constexpr float half_width = 0.115F;
  constexpr float half_height = 0.060F;
  const float r = static_cast<float>(radius);
  return {
    {0.0F, 0.0F, 0.0F},
    {half_width, -half_height, r},
    {half_width, half_height, r},
    {-half_width, half_height, r},
    {-half_width, -half_height, r},
  };
}

double blade_roll_from_points(const BuffBladePoints & points)
{
  const auto v = points.image[0] - (points.image[1] + points.image[4]) * 0.5F;
  return std::atan2(static_cast<double>(v.y), static_cast<double>(v.x));
}
}  // namespace auto_buff::jlu
