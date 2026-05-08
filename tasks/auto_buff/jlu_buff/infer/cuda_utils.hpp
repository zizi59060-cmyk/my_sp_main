#pragma once

#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace auto_buff::jlu
{
struct InferDeleter
{
  template <typename T>
  void operator()(T * obj) const noexcept
  {
    if (obj) delete obj;
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, InferDeleter>;

inline void check_cuda(cudaError_t status, const char * expr)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA runtime error in ") + expr + ": " + cudaGetErrorString(status));
  }
}

#define JLU_BUFF_CHECK_CUDA(expr) ::auto_buff::jlu::check_cuda((expr), #expr)

inline size_t element_size(nvinfer1::DataType type)
{
  using nvinfer1::DataType;
  switch (type) {
    case DataType::kFLOAT:
      return 4;
    case DataType::kHALF:
      return 2;
    case DataType::kINT8:
      return 1;
    case DataType::kINT32:
      return 4;
    case DataType::kBOOL:
      return 1;
#if NV_TENSORRT_MAJOR >= 10
    case DataType::kUINT8:
      return 1;
    case DataType::kFP8:
      return 1;
#endif
    default:
      throw std::runtime_error("unsupported TensorRT data type");
  }
}

inline size_t volume(const nvinfer1::Dims & dims)
{
  size_t v = 1;
  for (int i = 0; i < dims.nbDims; ++i) v *= static_cast<size_t>(dims.d[i] > 0 ? dims.d[i] : 1);
  return v;
}
}  // namespace auto_buff::jlu
