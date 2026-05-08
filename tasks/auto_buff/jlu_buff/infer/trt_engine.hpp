#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tasks/auto_buff/jlu_buff/infer/cuda_utils.hpp"
#include "tasks/auto_buff/jlu_buff/infer/trt_logger.hpp"

namespace auto_buff::jlu
{
struct TrtConfig
{
  std::string onnx_path;
  std::string engine_path;
  int input_width = 640;
  int input_height = 640;
  bool fp16 = true;
  int device_id = 0;
  std::string input_tensor_name;
  std::string output_tensor_name;
};

struct LetterboxInfo
{
  float scale = 1.0F;
  float pad_x = 0.0F;
  float pad_y = 0.0F;
  int original_width = 0;
  int original_height = 0;
};

class TrtEngine
{
public:
  explicit TrtEngine(TrtConfig config);
  ~TrtEngine();

  bool loadOrBuild();
  bool infer(const cv::Mat & image, std::vector<float> & output, LetterboxInfo * letterbox = nullptr);

  const nvinfer1::Dims & outputDims() const { return output_dims_; }
  const std::string & outputTensorName() const { return output_name_; }
  const std::string & inputTensorName() const { return input_name_; }
  bool ready() const { return engine_ && context_ && input_device_ && output_device_; }

private:
  bool loadSerializedEngine();
  bool buildFromOnnx();
  bool initializeContextAndBuffers();
  cv::Mat preprocess(const cv::Mat & image, LetterboxInfo & info) const;
  void releaseBuffers();

  TrtConfig config_;
  TrtLogger logger_;
  TrtUniquePtr<nvinfer1::IRuntime> runtime_;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
  TrtUniquePtr<nvinfer1::IExecutionContext> context_;
  cudaStream_t stream_ = nullptr;
  void * input_device_ = nullptr;
  void * output_device_ = nullptr;
  size_t input_bytes_ = 0;
  size_t output_bytes_ = 0;
  std::vector<float> input_host_;
  nvinfer1::Dims input_dims_{};
  nvinfer1::Dims output_dims_{};
  nvinfer1::DataType input_type_ = nvinfer1::DataType::kFLOAT;
  nvinfer1::DataType output_type_ = nvinfer1::DataType::kFLOAT;
  std::string input_name_;
  std::string output_name_;
};
}  // namespace auto_buff::jlu
