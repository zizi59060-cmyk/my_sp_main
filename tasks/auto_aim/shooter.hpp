#ifndef AUTO_AIM__SHOOTER_HPP
#define AUTO_AIM__SHOOTER_HPP

#include <string>

#include "io/command.hpp"
#include "tasks/auto_aim/aimer.hpp"

namespace auto_aim
{
class Shooter
{
public:
  Shooter(const std::string & config_path);

  bool shoot(
    const io::Command & command, const auto_aim::Aimer & aimer,
    const std::list<auto_aim::Target> & targets, const Eigen::Vector3d & gimbal_pos);

private:
  io::Command last_command_;
  double judge_distance_;
  double first_tolerance_;
  double second_tolerance_;
  bool auto_fire_;

  // 切板禁射功能开关与参数
  bool enable_switch_forbid_;
  int switch_forbid_frames_;

  // 运行时状态
  int last_target_id_;
  int switch_block_frames_;
};
}  // namespace auto_aim

#endif  // AUTO_AIM__SHOOTER_HPP