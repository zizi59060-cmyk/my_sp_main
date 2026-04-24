#pragma once

#include <list>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include "tasks/auto_aim/yolos/trt_logger.hpp"
#include "tasks/auto_aim/yolos/trt_utils.hpp"
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"

namespace auto_aim
{

struct GpuDecodedArmor
{
  float score = 0.0f;
  int color_id = 0;
  int num_id = 0;

  float x1 = 0.0f, y1 = 0.0f;
  float x2 = 0.0f, y2 = 0.0f;
  float x3 = 0.0f, y3 = 0.0f;
  float x4 = 0.0f, y4 = 0.0f;

  float min_x = 0.0f, min_y = 0.0f;
  float max_x = 0.0f, max_y = 0.0f;
};

class YOLOV5
{
public:
  static constexpr int kMaxDecodedArmors = 256;

  YOLOV5(const std::string & config_path, bool debug);
  ~YOLOV5();

  std::list<Armor> detect(const cv::Mat & raw_img, int frame_count);

private:
  std::list<Armor> buildArmorsFromDecoded(
    const std::vector<GpuDecodedArmor> & decoded,
    const cv::Mat & bgr_img,
    int frame_count);

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  void save(const Armor & armor) const;
  double sigmoid(double x);

  void allocateBindings();
  void freeBindings();

  void ensureGpuPreprocBuffers(int rows, int cols, int inW, int inH, int buffer_idx);

  void gpuPreprocessToDevice(
    const cv::Mat & src_bgr,
    void * device_input_ptr,
    int inW,
    int inH,
    nvinfer1::DataType dtype,
    cudaStream_t stream,
    int buffer_idx);

  static float iouRect(const cv::Rect & a, const cv::Rect & b);
  void nmsBoxesFast(
    const std::vector<cv::Rect> & boxes,
    const std::vector<float> & scores,
    float score_threshold,
    float nms_threshold,
    std::vector<int> & indices) const;

  const bool debug_;
  auto_aim::Detector detector_;
  std::string model_path_;

  double binary_threshold_ = 0.5;
  double min_confidence_ = 0.5;
  double score_threshold_ = 0.5;
  double nms_threshold_ = 0.5;

  bool use_roi_ = false;
  bool use_traditional_ = false;

  cv::Rect roi_;
  cv::Point2f offset_;

  std::string save_path_;
  cv::Mat tmp_img_;

  // 与 TRT 双缓冲对齐的上一帧图像缓存
  cv::Mat mFrameBuf_[2];

  // GPU preprocess cache，避免每帧临时分配 GpuMat
  cv::cuda::GpuMat mGpuSrc_[2];
  cv::cuda::GpuMat mGpuResized_[2];
  cv::cuda::GpuMat mGpuFloat_[2];

  // TRT runtime
  TrtLogger mLogger_;
  TrtUniquePtr<nvinfer1::IRuntime> mRuntime;
  TrtUniquePtr<nvinfer1::ICudaEngine> mEngine;

  TrtUniquePtr<nvinfer1::IExecutionContext> mContext[2];
  cudaStream_t mStream[2] = {nullptr, nullptr};
  cudaEvent_t mEvents[2] = {nullptr, nullptr};

  std::string mInputTensorName_;
  std::string mOutputTensorName_;
  nvinfer1::Dims mInputDims_{};
  nvinfer1::Dims mOutputDims_{};

  size_t mInputBytes_ = 0;
  size_t mOutputBytes_ = 0;

  nvinfer1::DataType mInputDtype_ = nvinfer1::DataType::kFLOAT;
  nvinfer1::DataType mOutputDtype_ = nvinfer1::DataType::kFLOAT;

  void * mGpuBuffers_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};

  // 小结果回传缓冲
  GpuDecodedArmor * mDecodedDev_[2] = {nullptr, nullptr};
  GpuDecodedArmor * mDecodedHost_[2] = {nullptr, nullptr};   // pinned host
  int * mDecodedCountDev_[2] = {nullptr, nullptr};
  int * mDecodedCountHost_[2] = {nullptr, nullptr};          // pinned host

  bool use_gpu_preproc_ = true;
  int mBufferIdx = 0;
  bool mFirstFrame = true;
};

}  // namespace auto_aim