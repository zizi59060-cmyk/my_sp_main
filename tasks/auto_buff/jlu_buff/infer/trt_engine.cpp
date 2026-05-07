#include "tasks/auto_buff/jlu_buff/infer/trt_engine.hpp"

#include <NvInferVersion.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>

#include "tools/logger.hpp"

namespace auto_buff::jlu
{
namespace
{
std::vector<char> read_binary(const std::string & path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) return {};
  const auto size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> data(static_cast<size_t>(size));
  if (!file.read(data.data(), size)) return {};
  return data;
}

bool write_binary(const std::string & path, const void * data, size_t size)
{
  if (path.empty() || !data || size == 0) return false;
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream file(path, std::ios::binary);
  if (!file.is_open()) return false;
  file.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
  return file.good();
}
}  // namespace

TrtEngine::TrtEngine(TrtConfig config) : config_(std::move(config)) {}

TrtEngine::~TrtEngine()
{
  releaseBuffers();
  if (stream_) cudaStreamDestroy(stream_);
}

bool TrtEngine::loadOrBuild()
{
  try {
    JLU_BUFF_CHECK_CUDA(cudaSetDevice(config_.device_id));
    tools::logger()->info(
      "[JLU-Buff-TRT] TensorRT version {}.{}.{}; engine='{}'; onnx='{}'; fp16={}",
      NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, config_.engine_path,
      config_.onnx_path, config_.fp16);

    const bool engine_loaded = loadSerializedEngine();
    if (!engine_loaded) {
      tools::logger()->warn(
        "[JLU-Buff-TRT] Serialized engine not available, try building from ONNX: {}",
        config_.onnx_path);
      if (!buildFromOnnx()) return false;
    }
    return initializeContextAndBuffers();
  } catch (const std::exception & e) {
    tools::logger()->error("[JLU-Buff-TRT] load/build failed: {}", e.what());
    return false;
  }
}

bool TrtEngine::loadSerializedEngine()
{
  if (config_.engine_path.empty() || !std::filesystem::exists(config_.engine_path)) return false;
  const auto engine_data = read_binary(config_.engine_path);
  if (engine_data.empty()) return false;

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) return false;
  engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!engine_) {
    tools::logger()->error("[JLU-Buff-TRT] deserializeCudaEngine failed: {}", config_.engine_path);
    return false;
  }
  tools::logger()->info("[JLU-Buff-TRT] Loaded serialized engine: {}", config_.engine_path);
  return true;
}

bool TrtEngine::buildFromOnnx()
{
  if (config_.onnx_path.empty() || !std::filesystem::exists(config_.onnx_path)) {
    tools::logger()->error(
      "[JLU-Buff-TRT] ONNX model not found. Put model.onnx at '{}' or provide engine '{}'.",
      config_.onnx_path, config_.engine_path);
    return false;
  }

  TrtUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger_));
  if (!builder) return false;
  const auto flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  TrtUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flags));
  if (!network) return false;
  TrtUniquePtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger_));
  if (!parser) return false;
  if (!parser->parseFromFile(config_.onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    tools::logger()->error("[JLU-Buff-TRT] ONNX parse failed: {}", config_.onnx_path);
    for (int i = 0; i < parser->getNbErrors(); ++i) {
      tools::logger()->error("[JLU-Buff-TRT] parser error {}: {}", i, parser->getError(i)->desc());
    }
    return false;
  }

  TrtUniquePtr<nvinfer1::IBuilderConfig> builder_config(builder->createBuilderConfig());
  if (!builder_config) return false;
  builder_config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
  if (config_.fp16 && builder->platformHasFastFp16()) {
    builder_config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  auto * input = network->getInput(0);
  if (!input) return false;
  auto dims = input->getDimensions();
  if (dims.nbDims == 4 && (dims.d[2] < 0 || dims.d[3] < 0 || dims.d[0] < 0)) {
    TrtUniquePtr<nvinfer1::IOptimizationProfile> profile(builder->createOptimizationProfile());
    nvinfer1::Dims fixed = dims;
    fixed.d[0] = 1;
    fixed.d[2] = config_.input_height;
    fixed.d[3] = config_.input_width;
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, fixed);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, fixed);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, fixed);
    builder_config->addOptimizationProfile(profile.release());
  }

  tools::logger()->info("[JLU-Buff-TRT] Building TensorRT engine from ONNX, this may take a while...");
  TrtUniquePtr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *builder_config));
  if (!plan) {
    tools::logger()->error("[JLU-Buff-TRT] buildSerializedNetwork failed");
    return false;
  }
  if (!config_.engine_path.empty() && write_binary(config_.engine_path, plan->data(), plan->size())) {
    tools::logger()->info("[JLU-Buff-TRT] Saved engine: {}", config_.engine_path);
  }

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  engine_.reset(runtime_->deserializeCudaEngine(plan->data(), plan->size()));
  return static_cast<bool>(engine_);
}

bool TrtEngine::initializeContextAndBuffers()
{
  if (!engine_) return false;
  context_.reset(engine_->createExecutionContext());
  if (!context_) return false;
  JLU_BUFF_CHECK_CUDA(cudaStreamCreate(&stream_));

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char * name = engine_->getIOTensorName(i);
    if (!name) continue;
    const auto mode = engine_->getTensorIOMode(name);
    auto dims = engine_->getTensorShape(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      input_name_ = config_.input_tensor_name.empty() ? name : config_.input_tensor_name;
      input_type_ = engine_->getTensorDataType(name);
      if (dims.nbDims == 4) {
        dims.d[0] = 1;
        dims.d[2] = config_.input_height;
        dims.d[3] = config_.input_width;
        context_->setInputShape(name, dims);
      }
      input_dims_ = dims;
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      output_name_ = config_.output_tensor_name.empty() ? name : config_.output_tensor_name;
      output_type_ = engine_->getTensorDataType(name);
    }
  }
  if (input_name_.empty() || output_name_.empty()) return false;

  output_dims_ = context_->getTensorShape(output_name_.c_str());
  if (output_dims_.nbDims <= 0) output_dims_ = engine_->getTensorShape(output_name_.c_str());

  input_bytes_ = volume(input_dims_) * element_size(input_type_);
  output_bytes_ = volume(output_dims_) * element_size(output_type_);
  if (output_type_ != nvinfer1::DataType::kFLOAT) {
    tools::logger()->warn("[JLU-Buff-TRT] Output is not FP32; current postprocess expects FP32.");
  }
  JLU_BUFF_CHECK_CUDA(cudaMalloc(&input_device_, input_bytes_));
  JLU_BUFF_CHECK_CUDA(cudaMalloc(&output_device_, output_bytes_));
  context_->setTensorAddress(input_name_.c_str(), input_device_);
  context_->setTensorAddress(output_name_.c_str(), output_device_);
  tools::logger()->info(
    "[JLU-Buff-TRT] Engine ready. input='{}' bytes={} output='{}' bytes={}", input_name_,
    input_bytes_, output_name_, output_bytes_);
  return true;
}

cv::Mat TrtEngine::preprocess(const cv::Mat & image, LetterboxInfo & info) const
{
  info.original_width = image.cols;
  info.original_height = image.rows;
  info.scale = std::min(
    static_cast<float>(config_.input_width) / static_cast<float>(image.cols),
    static_cast<float>(config_.input_height) / static_cast<float>(image.rows));
  const int resized_w = static_cast<int>(std::round(image.cols * info.scale));
  const int resized_h = static_cast<int>(std::round(image.rows * info.scale));
  info.pad_x = (config_.input_width - resized_w) * 0.5F;
  info.pad_y = (config_.input_height - resized_h) * 0.5F;

  cv::Mat resized;
  cv::resize(image, resized, {resized_w, resized_h});
  cv::Mat canvas(config_.input_height, config_.input_width, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(canvas(cv::Rect(static_cast<int>(info.pad_x), static_cast<int>(info.pad_y), resized_w, resized_h)));
  cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
  canvas.convertTo(canvas, CV_32F, 1.0 / 255.0);
  return canvas;
}

bool TrtEngine::infer(const cv::Mat & image, std::vector<float> & output, LetterboxInfo * letterbox)
{
  if (!ready() || image.empty()) return false;
  try {
    LetterboxInfo info;
    cv::Mat chw_src = preprocess(image, info);
    if (letterbox) *letterbox = info;

    const int hw = config_.input_width * config_.input_height;
    input_host_.assign(static_cast<size_t>(3 * hw), 0.0F);
    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; ++c) channels[c] = cv::Mat(config_.input_height, config_.input_width, CV_32F, input_host_.data() + c * hw);
    cv::split(chw_src, channels);

    JLU_BUFF_CHECK_CUDA(cudaMemcpyAsync(input_device_, input_host_.data(), input_bytes_, cudaMemcpyHostToDevice, stream_));
    if (!context_->enqueueV3(stream_)) return false;
    output.resize(output_bytes_ / sizeof(float));
    JLU_BUFF_CHECK_CUDA(cudaMemcpyAsync(output.data(), output_device_, output_bytes_, cudaMemcpyDeviceToHost, stream_));
    JLU_BUFF_CHECK_CUDA(cudaStreamSynchronize(stream_));
    return true;
  } catch (const std::exception & e) {
    tools::logger()->error("[JLU-Buff-TRT] inference failed: {}", e.what());
    return false;
  }
}

void TrtEngine::releaseBuffers()
{
  if (input_device_) cudaFree(input_device_);
  if (output_device_) cudaFree(output_device_);
  input_device_ = nullptr;
  output_device_ = nullptr;
}
}  // namespace auto_buff::jlu
