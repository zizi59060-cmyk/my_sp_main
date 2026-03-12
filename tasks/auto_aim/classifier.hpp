#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include "yolos/trt_logger.hpp"
#include "yolos/trt_utils.hpp" // <-- 包含新的工具头文件

#include "armor.hpp"

namespace auto_aim
{

class Classifier
{
public:
    explicit Classifier(const std::string & config_path);
    ~Classifier();

    // CPU-based OpenCV DNN 旧接口（保留）
    void classify(Armor & armor);

    // TensorRT 10.3 基于 I/O Tensor 的推理接口
    void ovclassify(Armor & armor);

    // 禁止拷贝
    Classifier(const Classifier&) = delete;
    Classifier& operator=(const Classifier&) = delete;

private:
    // OpenCV net (备用/兼容)
    cv::dnn::Net net_;

    // TRT objects
    TrtLogger mLogger_;
    TrtUniquePtr<nvinfer1::IRuntime> mRuntime{nullptr};
    TrtUniquePtr<nvinfer1::ICudaEngine> mEngine{nullptr};
    TrtUniquePtr<nvinfer1::IExecutionContext> mContext{nullptr};
    cudaStream_t mStream = nullptr;

    // GPU buffers: [0] input, [1] output
    void* mGpuBuffers_[2] = { nullptr, nullptr };
    std::vector<float> mCpuOutputBuffer_;

    // 保存张量名称
    std::string mInputTensorName_;
    std::string mOutputTensorName_;

    // 内部帮助函数
    void loadEngineFromFile(const std::string & engine_path);
    void allocateBindings();
    void freeBindings();
};

} // namespace auto_aim
