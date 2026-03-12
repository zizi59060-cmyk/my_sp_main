#pragma once

#include <memory>
#include <stdexcept>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>

namespace auto_aim
{

// -------------------------------
// 1. 智能指针 Deleter 定义
// -------------------------------
struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const noexcept
    {
        if (obj)
        {
#if NV_TENSORRT_MAJOR >= 10
            delete obj; // TRT 10+ 统一使用 delete
#else
            obj->destroy(); // 兼容旧版本
#endif
        }
    }
};

// -------------------------------
// 2. 通用的 TensorRT 智能指针模板
// -------------------------------
template <typename T>
using TrtUniquePtr = std::unique_ptr<T, InferDeleter>;

// -------------------------------
// 3. 获取 DataType 对应的元素大小
// -------------------------------
inline size_t getElementSize(nvinfer1::DataType t)
{
    using namespace nvinfer1;
    switch (t)
    {
        case DataType::kFLOAT: return 4;
        case DataType::kHALF:  return 2;
        case DataType::kINT8:  return 1;
        case DataType::kINT32: return 4;
        case DataType::kBOOL:  return 1;
#if NV_TENSORRT_MAJOR >= 10
        case DataType::kUINT8: return 1;
        case DataType::kFP8:   return 1;
#endif
        default:
            throw std::runtime_error("Unsupported TensorRT DataType");
    }
}

// -------------------------------
// 4. CUDA 错误检查辅助宏
// -------------------------------
#define CHECK_CUDA(call)                                                                 \
    do                                                                                   \
    {                                                                                    \
        cudaError_t status = (call);                                                     \
        if (status != cudaSuccess)                                                       \
        {                                                                                \
            throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(status)); \
        }                                                                                \
    } while (0)

// -------------------------------
// 5. 工具函数：计算 tensor 元素数
// -------------------------------
inline size_t getTensorVolume(const nvinfer1::Dims& dims)
{
    size_t vol = 1;
    for (int i = 0; i < dims.nbDims; ++i)
        vol *= static_cast<size_t>(dims.d[i] > 0 ? dims.d[i] : 1);
    return vol;
}

} // namespace auto_aim
