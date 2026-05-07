#include "tasks/auto_buff/jlu_buff/jlu_buff.hpp"

#include <algorithm>
#include <chrono>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace auto_buff::jlu
{
namespace
{
template <typename T>
T read_or(const YAML::Node & n, const std::string & key, const T & fallback)
{
  return n && n[key] ? n[key].as<T>() : fallback;
}
}  // namespace

JluBuffSystem::JluBuffSystem(const std::string & config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  auto root = yaml["jlu_buff"];
  enable_ = read_or<bool>(root, "enable", true);
  auto_fire_enable_ = read_or<bool>(root, "auto_fire_enable", true);
  predict_offset_sec_ = read_or<double>(root, "predict_offset_ms", 70.0) / 1000.0;

  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);

  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Map<Eigen::Vector3d>(t_camera2gimbal_data.data());

  if (enable_) detector_ = std::make_unique<TrtBuffDetector>(config_path);
  small_target_ = std::make_unique<SmallBuffTarget>(load_target_config(root ? root["small_buff_conf"] : YAML::Node{}));
  const auto big_conf = root ? root["big_buff_conf"] : YAML::Node{};
  big_target_ = std::make_unique<BigBuffTarget>(
    load_target_config(big_conf), load_fitter_config(big_conf ? big_conf["fitter_conf"] : YAML::Node{}));
  trajectory_ = Trajectory(load_trajectory_config(root ? root["trajectory_conf"] : YAML::Node{}));

  tools::logger()->info(
    "[JLU-Buff] enable={} auto_fire_enable={} predict_offset_ms={:.1f}", enable_,
    auto_fire_enable_, predict_offset_sec_ * 1000.0);
}

void JluBuffSystem::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  const Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

void JluBuffSystem::solvePnPAndTransform(std::vector<BuffBlade> & blades, TimePoint timestamp) const
{
  const auto object_points = buff_blade_object_points(kBuffRadius);
  for (auto & blade : blades) {
    std::vector<cv::Point2f> image_points(blade.points.image.begin(), blade.points.image.end());
    blade.timestamp = timestamp;

    // PnP uses the exact JLU point order: r_center, bottom_right, top_right, top_left, bottom_left.
    blade.pnp_ok = cv::solvePnP(
      object_points, image_points, camera_matrix_, distort_coeffs_, blade.rvec, blade.tvec, false,
      cv::SOLVEPNP_EPNP);
    if (!blade.pnp_ok) {
      tools::logger()->warn("[JLU-Buff] five-point PnP failed");
      continue;
    }

    cv::Mat rmat_cv;
    cv::Rodrigues(blade.rvec, rmat_cv);
    Eigen::Matrix3d R_buff2camera;
    cv::cv2eigen(rmat_cv, R_buff2camera);
    Eigen::Vector3d t_buff2camera;
    cv::cv2eigen(cv::Mat(blade.tvec), t_buff2camera);

    const Eigen::Vector3d blade_in_buff(0.0, 0.0, kBuffRadius);
    const Eigen::Vector3d center_in_camera = t_buff2camera;
    const Eigen::Vector3d blade_in_camera = R_buff2camera * blade_in_buff + t_buff2camera;
    const Eigen::Vector3d center_in_gimbal = R_camera2gimbal_ * center_in_camera + t_camera2gimbal_;
    const Eigen::Vector3d blade_in_gimbal = R_camera2gimbal_ * blade_in_camera + t_camera2gimbal_;
    blade.center_world = R_gimbal2world_ * center_in_gimbal;
    blade.position_camera = blade_in_camera;
    blade.position_world = R_gimbal2world_ * blade_in_gimbal;
    tools::logger()->debug(
      "[JLU-Buff] PnP ok conf={:.2f} points=[({:.1f},{:.1f}),({:.1f},{:.1f}),({:.1f},{:.1f}),({:.1f},{:.1f}),({:.1f},{:.1f})] center=({:.2f},{:.2f},{:.2f}) roll={:.3f}",
      blade.confidence, blade.points.image[0].x, blade.points.image[0].y, blade.points.image[1].x,
      blade.points.image[1].y, blade.points.image[2].x, blade.points.image[2].y,
      blade.points.image[3].x, blade.points.image[3].y, blade.points.image[4].x,
      blade.points.image[4].y, blade.center_world.x(), blade.center_world.y(),
      blade.center_world.z(), blade.roll);
  }
  blades.erase(std::remove_if(blades.begin(), blades.end(), [](const auto & b) { return !b.pnp_ok; }), blades.end());
}

JluBuffPlan JluBuffSystem::run(
  const cv::Mat & image, BuffMode mode, const Eigen::Quaterniond & gimbal_q,
  const io::GimbalState & gimbal_state, TimePoint timestamp)
{
  JluBuffPlan plan;
  if (!enable_) return plan;
  if (mode != last_mode_) {
    tools::logger()->info("[JLU-Buff] mode switch {} -> {}, reset tracker", to_string(last_mode_), to_string(mode));
    reset();
    last_mode_ = mode;
  }
  set_R_gimbal2world(gimbal_q);

  const auto t0 = std::chrono::steady_clock::now();
  auto blades = detector_ ? detector_->detect(image) : std::vector<BuffBlade>{};
  solvePnPAndTransform(blades, timestamp);
  const auto t1 = std::chrono::steady_clock::now();
  tools::logger()->debug(
    "[JLU-Buff] mode={} total_detect_ms={:.3f} pnp_blades={}", to_string(mode),
    tools::delta_time(t1, t0) * 1000.0, blades.size());

  BuffState state = mode == BuffMode::SMALL ? small_target_->update(blades, timestamp) : big_target_->update(blades, timestamp);
  plan.track_state = state.track_state;
  if (state.track_state != TrackState::TRACKING) {
    plan.control = false;
    plan.fire = false;
    return plan;
  }

  const auto predicted = mode == BuffMode::SMALL ? small_target_->predict(predict_offset_sec_) : big_target_->predict(predict_offset_sec_);
  auto sol = trajectory_.solve(predicted, gimbal_state.bullet_speed, predict_offset_sec_);
  if (!sol.valid) return plan;
  plan.control = true;
  plan.fire = auto_fire_enable_ && sol.fire;
  plan.yaw = static_cast<float>(sol.yaw);
  plan.pitch = static_cast<float>(sol.pitch);
  plan.blade_index = sol.selected_blade;
  plan.fly_time = sol.fly_time;
  tools::logger()->debug(
    "[JLU-Buff] TrackState={} bullet_speed={:.2f} yaw={:.4f} pitch={:.4f} fire={} auto_fire_enable={}",
    to_string(plan.track_state), gimbal_state.bullet_speed, plan.yaw, plan.pitch, plan.fire,
    auto_fire_enable_);
  return plan;
}

void JluBuffSystem::reset()
{
  if (small_target_) small_target_->reset();
  if (big_target_) big_target_->reset();
}
}  // namespace auto_buff::jlu
