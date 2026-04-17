#pragma once

#include <list>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <string>
#include <vector>
#include <memory>
#include <cstring>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include "tasks/auto_aim/yolos/trt_logger.hpp"
#include "tasks/auto_aim/yolos/trt_utils.hpp" 
#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"

namespace auto_aim
{
class YOLOV5
{
public:
  YOLOV5(const std::string & config_path, bool debug);
  ~YOLOV5();
  
  std::list<Armor> detect(const cv::Mat & raw_img, int frame_count);

private:
  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);
  std::list<Armor> postprocess(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);
  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  void save(const Armor & armor) const;
  double sigmoid(double x);

  void allocateBindings();
  void freeBindings();
  void initPinnedBuffers();
  void freePinnedBuffers();

  void gpuPreprocessToDevice(const cv::Mat &src_bgr, void* device_input_ptr, int inW, int inH, nvinfer1::DataType dtype, cudaStream_t stream);

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
  cv::Mat mPrevBgrImg; // 用于保存上一帧的图像 (为 parse 函数)
  
  // -----------------------------------------------------------------
  // ⬇️ ⬇️ [关键修复] 把 tmp_img_ 加回来 ⬇️ ⬇️
  // -----------------------------------------------------------------
  cv::Mat tmp_img_; // 用于 save() 函数

  // TRT runtime objects
  TrtLogger mLogger_;
  TrtUniquePtr<nvinfer1::IRuntime> mRuntime;
  TrtUniquePtr<nvinfer1::ICudaEngine> mEngine;
  
  TrtUniquePtr<nvinfer1::IExecutionContext> mContext[2];
  cudaStream_t mStream[2] = {nullptr, nullptr};
  cudaEvent_t mEvents[2] = {nullptr, nullptr};

  // (TRT 模型元数据)
  std::string mInputTensorName_;
  std::string mOutputTensorName_;
  nvinfer1::Dims mInputDims_{};
  nvinfer1::Dims mOutputDims_{};
  size_t mInputBytes_ = 0;
  size_t mOutputBytes_ = 0;
  nvinfer1::DataType mInputDtype_ = nvinfer1::DataType::kFLOAT;
  
  // (双缓冲管线资源)
  void* mGpuBuffers_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
  std::vector<float> mCpuOutputBuffer_[2];
  void* mPinnedOutputHost_[2] = {nullptr, nullptr};
  size_t mPinnedOutputBytes_ = 0;
  bool use_gpu_preproc_ = true; 
  int mBufferIdx = 0;
  bool mFirstFrame = true;
};

}  // namespace auto_aim