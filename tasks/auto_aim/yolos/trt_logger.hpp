#pragma once

#include <NvInferRuntime.h>
#include <iostream>

// TensorRT 日志记录器
class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // 仅打印 Error 和 Warning 信息
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
};