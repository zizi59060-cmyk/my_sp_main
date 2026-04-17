#ifndef IO__DAHENG_HPP
#define IO__DAHENG_HPP

#include <atomic>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector> // [新增]

#include "include/GxIAPI.h"
#include "include/DxImageProc.h"

#include "io/camera.hpp"
#include "tools/thread_safe_queue.hpp"

namespace io
{

class DaHeng : public CameraBase
{
public:
  DaHeng(double exposure_ms, double gain, const std::string &vid_pid);
  ~DaHeng() override;

  void read(cv::Mat &img, std::chrono::steady_clock::time_point &timestamp) override;
  void onFrameCallback(GX_FRAME_CALLBACK_PARAM *pFrame);

private:
  struct CameraData
  {
    cv::Mat img;
    std::chrono::steady_clock::time_point timestamp;
  };

  double exposure_us_;
  double gain_;
  GX_DEV_HANDLE handle_;
  bool is_color_camera_;

  std::atomic<bool> capturing_;
  std::atomic<bool> capture_quit_;
  tools::ThreadSafeQueue<CameraData> queue_;

  int64_t pixel_color_filter_;
  int frame_count_;
  std::chrono::steady_clock::time_point last_fps_time_;

  // [新增] 存储 Gamma 查找表
  std::vector<VxUint8> lut_content_;
};

} // namespace io

#endif // IO__DAHENG_HPP