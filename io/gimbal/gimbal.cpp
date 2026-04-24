#include "gimbal.hpp"

#include <iomanip>  // 添加这行以使用 std::setw
#include <sstream>  // 确保已经包含，用于 std::stringstream

#include "tools/crc.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/yaml.hpp"

namespace io
{
Gimbal::Gimbal(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto com_port = tools::read<std::string>(yaml, "com_port");

  try {
    serial_.setPort(com_port);
    serial_.open();
  } catch (const std::exception & e) {
    tools::logger()->error("[Gimbal] Failed to open serial: {}", e.what());
    exit(1);
  }

  thread_ = std::thread(&Gimbal::read_thread, this);

  queue_.pop();
  tools::logger()->info("[Gimbal] First q received.");
}

Gimbal::~Gimbal()
{
  quit_ = true;
  if (thread_.joinable()) thread_.join();
  serial_.close();
}

GimbalMode Gimbal::mode() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return mode_;
}

GimbalState Gimbal::state() const
{
  std::lock_guard<std::mutex> lock(mutex_);
  return state_;
}

std::string Gimbal::str(GimbalMode mode) const
{
  switch (mode) {
    case GimbalMode::IDLE:
      return "IDLE";
    case GimbalMode::AUTO_AIM:
      return "AUTO_AIM";
    case GimbalMode::SMALL_BUFF:
      return "SMALL_BUFF";
    case GimbalMode::BIG_BUFF:
      return "BIG_BUFF";
    default:
      return "INVALID";         
  }
}

Eigen::Quaterniond Gimbal::q(std::chrono::steady_clock::time_point t)
{
  while (true) {
    auto [q_a, t_a] = queue_.pop();
    auto [q_b, t_b] = queue_.front();
    auto t_ab = tools::delta_time(t_a, t_b);
    auto t_ac = tools::delta_time(t_a, t);
    auto k = t_ac / t_ab;
    Eigen::Quaterniond q_c = q_a.slerp(k, q_b).normalized();
    if (t < t_a) return q_c;
    if (!(t_a < t && t <= t_b)) continue;

    return q_c;
  }
}

void Gimbal::send(io::VisionToGimbal VisionToGimbal)
{
  tx_data_.mode = VisionToGimbal.mode;
  tx_data_.yaw = VisionToGimbal.yaw;
  tx_data_.yaw_vel = VisionToGimbal.yaw_vel;
  tx_data_.yaw_acc = VisionToGimbal.yaw_acc;
  tx_data_.pitch = VisionToGimbal.pitch;
  tx_data_.pitch_vel = VisionToGimbal.pitch_vel;
  tx_data_.pitch_acc = VisionToGimbal.pitch_acc;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
    tools::logger()->info("Send");
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

void Gimbal::send(
  bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
  float pitch_acc)
{
  tx_data_.mode = control ? (fire ? 2 : 1) : 0;
  tx_data_.yaw = yaw;
  tx_data_.yaw_vel = yaw_vel;
  tx_data_.yaw_acc = yaw_acc;
  tx_data_.pitch = pitch;
  tx_data_.pitch_vel = pitch_vel;
  tx_data_.pitch_acc = pitch_acc;
  tx_data_.crc16 = tools::get_crc16(
    reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_) - sizeof(tx_data_.crc16));

  try {
    serial_.write(reinterpret_cast<uint8_t *>(&tx_data_), sizeof(tx_data_));
    //tools::logger()->info("Write");
  } catch (const std::exception & e) {
    tools::logger()->warn("[Gimbal] Failed to write serial: {}", e.what());
  }
}

// bool Gimbal::read(uint8_t * buffer, size_t size)
// {
//   try {
//     return serial_.read(buffer, size) == size;

//   } catch (const std::exception & e) {
//     tools::logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
//     return false;
//   }
// }
bool Gimbal::read(uint8_t* buffer, size_t size)
{
  try {
    size_t total_read = 0;
    auto start = std::chrono::steady_clock::now();

    while (total_read < size) {
      size_t n = serial_.read(buffer + total_read, size - total_read);
      total_read += n;

      if (n == 0) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() > 100) {
          tools::logger()->warn("[Gimbal] read timeout, read {} / {} bytes", total_read, size);
          return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    return true;
  } catch (const std::exception& e) {
    tools::logger()->warn("[Gimbal] Failed to read serial: {}", e.what());
    return false;
  } catch (...) {
    tools::logger()->warn("[Gimbal] Failed to read serial: unknown exception");
    return false;
  }
}
// bool Gimbal::read(uint8_t * buffer, size_t size)
// {
//   size_t total_read = 0;
//   auto start = std::chrono::steady_clock::now();

//   while (total_read < size) {
//     if (quit_) return false;  // 防止程序退出时死锁

//     size_t n = serial_.read(buffer + total_read, size - total_read);

//     if (n > 0) {
//       total_read += n;
//       start = std::chrono::steady_clock::now();  // 只要读到数据，就重置超时计时器
//     } else {
//       // 超时保护
//       auto now = std::chrono::steady_clock::now();
//       if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() > 100) {
//         // 这里的警告日志可以注销，防止没连接时疯狂刷屏
//         // tools::logger()->warn("[Gimbal] read timeout, read {} / {} bytes", total_read, size);
//         return false;
//       }
//       std::this_thread::sleep_for(std::chrono::milliseconds(1));
//     }
//   }

//   return true;
// }

void Gimbal::read_thread()
{
  tools::logger()->info("[Gimbal] read_thread started.");
  int error_count = 0;

  while (!quit_) {
    // 【关键修改 1】：降低重连阈值。20次超时/错误（约2秒）无法恢复，直接强制重置串口
    if (error_count > 20) {
      error_count = 0;
      tools::logger()->warn("[Gimbal] Link lost or misaligned, attempting to reconnect...");
      reconnect();
      continue;
    }

    // 【关键修改 2】：逐字节寻找帧头 'S'，防止数据错位导致永远对不齐
    uint8_t header_byte = 0;
    if (!read(&header_byte, 1)) {
      error_count++;
      continue;
    }

    if (header_byte != 'S') {
      continue;  // 不是 'S'，继续吃掉底层缓冲区的垃圾数据
    }

    // 找到了 'S'，再读一个字节看是不是 'P'
    rx_data_.head[0] = 'S';
    if (!read(&header_byte, 1)) {
      error_count++;
      continue;
    }

    if (header_byte != 'P') {
      continue;  // 不是 'P'，说明前面的 'S' 只是碰巧，重新找
    }
    rx_data_.head[1] = 'P';

    // 成功对齐包头，记录时间并读取剩余包体
    auto t = std::chrono::steady_clock::now();

    if (!read(
          reinterpret_cast<uint8_t *>(&rx_data_) + sizeof(rx_data_.head),
          sizeof(rx_data_) - sizeof(rx_data_.head))) {
      error_count++;
      continue;
    }

    // CRC 校验
    if (!tools::check_crc16(reinterpret_cast<uint8_t *>(&rx_data_), sizeof(rx_data_))) {
      tools::logger()->debug("[Gimbal] CRC16 check failed.");
      error_count++;
      continue;
    }

    // --- 成功读到完整且正确的一帧，清零错误计数 ---
    error_count = 0;
    Eigen::Quaterniond q(rx_data_.q[0], rx_data_.q[1], rx_data_.q[2], rx_data_.q[3]);
    queue_.push({q, t});

    std::lock_guard<std::mutex> lock(mutex_);

    state_.yaw = rx_data_.yaw;
    state_.yaw_vel = rx_data_.yaw_vel;
    state_.pitch = rx_data_.pitch;
    state_.pitch_vel = rx_data_.pitch_vel;
    state_.bullet_speed = rx_data_.bullet_speed;
    state_.bullet_count = rx_data_.bullet_count;

    switch (rx_data_.mode) {
      case 0:
        mode_ = GimbalMode::IDLE;
        break;
      case 1:
        mode_ = GimbalMode::AUTO_AIM;
        break;
      case 2:
        mode_ = GimbalMode::SMALL_BUFF;
        break;
      case 3:
        mode_ = GimbalMode::BIG_BUFF;
        break;
      default:
        mode_ = GimbalMode::IDLE;
        tools::logger()->warn("[Gimbal] Invalid mode: {}", rx_data_.mode);
        break;
    }
  }

  tools::logger()->info("[Gimbal] read_thread stopped.");
}

void Gimbal::reconnect()
{
  int max_retry_count = 10;
  for (int i = 0; i < max_retry_count && !quit_; ++i) {
    // tools::logger()->warn("[Gimbal] Reconnecting serial, attempt {}/{}...", i + 1, max_retry_count);
    try {
      serial_.close();
      std::this_thread::sleep_for(std::chrono::milliseconds(500));  // 给硬件一点断开的时间
    } catch (...) {
    }

    try {
      serial_.open();  // 重新打开串口会清空底层操作系统和驱动的脏数据缓冲区
      queue_.clear();
      tools::logger()->info("[Gimbal] Reconnected serial successfully.");
      break;  // 成功则跳出重试循环
    } catch (const std::exception & e) {
      tools::logger()->warn("[Gimbal] Reconnect failed: {}", e.what());
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

}  // namespace io