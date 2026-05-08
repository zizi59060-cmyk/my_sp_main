#pragma once

#include <NvInferRuntime.h>

#include "tools/logger.hpp"

namespace auto_buff::jlu
{
class TrtLogger final : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override
  {
    if (severity > Severity::kINFO) return;
    if (!msg) return;
    try {
      if (severity <= Severity::kERROR) tools::logger()->error("[JLU-Buff-TRT] {}", msg);
      else if (severity == Severity::kWARNING) tools::logger()->warn("[JLU-Buff-TRT] {}", msg);
      else tools::logger()->debug("[JLU-Buff-TRT] {}", msg);
    } catch (...) {
    }
  }
};
}  // namespace auto_buff::jlu
