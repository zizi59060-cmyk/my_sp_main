#include "yolov5.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <cstring>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include <opencv2/dnn.hpp>
#include <cuda_fp16.h> // __half

#include <opencv2/cudawarping.hpp> 
#include <opencv2/core/cuda_stream_accessor.hpp> 


// [不变] CUDA Kernel
// extern "C" __global__
static __global__
void bgr_to_rgb_nchw_kernel_float(const float* __restrict__ src, size_t src_step_bytes, void* dst_void,
                                  int H, int W, bool dst_fp16)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;

  const char* row_ptr_char = reinterpret_cast<const char*>(src) + y * src_step_bytes;
  const float* row_ptr = reinterpret_cast<const float*>(row_ptr_char);
  int idx = x * 3;
  float b = row_ptr[idx + 0];
  float g = row_ptr[idx + 1];
  float r = row_ptr[idx + 2];

  int hw = H * W;
  int pix = y * W + x;

  if (!dst_fp16) {
    float* dst = reinterpret_cast<float*>(dst_void);
    dst[0 * hw + pix] = r; 
    dst[1 * hw + pix] = g;
    dst[2 * hw + pix] = b;
  } else {
    __half* dst = reinterpret_cast<__half*>(dst_void);
    dst[0 * hw + pix] = __float2half_rn(r);
    dst[1 * hw + pix] = __float2half_rn(g);
    dst[2 * hw + pix] = __float2half_rn(b);
  }
}

namespace auto_aim
{

// (构造函数, 析构函数, allocateBindings, freeBindings... 和上个回答一样，保持不变)
YOLOV5::YOLOV5(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false), mBufferIdx(0), mFirstFrame(true)
{
  try {
    auto yaml = YAML::LoadFile(config_path);
    model_path_ = yaml["yolov5_model_path"].as<std::string>();
    if (yaml["threshold"]) binary_threshold_ = yaml["threshold"].as<double>();
    if (yaml["min_confidence"]) min_confidence_ = yaml["min_confidence"].as<double>();
    if (yaml["score_threshold"]) score_threshold_ = yaml["score_threshold"].as<double>();
    if (yaml["nms_threshold"]) nms_threshold_ = yaml["nms_threshold"].as<double>();
    int x = 0, y = 0, width = -1, height = -1;
    if (yaml["roi"]) {
      x = yaml["roi"]["x"].as<int>();
      y = yaml["roi"]["y"].as<int>();
      width = yaml["roi"]["width"].as<int>();
      height = yaml["roi"]["height"].as<int>();
      use_roi_ = yaml["use_roi"].as<bool>();
    }
    if (yaml["use_traditional"]) use_traditional_ = yaml["use_traditional"].as<bool>();
    roi_ = cv::Rect(x, y, width, height);
    offset_ = cv::Point2f(x, y);
    if (yaml["use_gpu_preproc"]) use_gpu_preproc_ = yaml["use_gpu_preproc"].as<bool>();
    save_path_ = "imgs";
    std::filesystem::create_directory(save_path_);
    std::ifstream engineFile(model_path_, std::ios::binary);
    if (!engineFile.is_open()) { /*...*/ return; }
    engineFile.seekg(0, std::ios::end);
    std::streamsize fsize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    std::vector<char> engineData(static_cast<size_t>(fsize));
    if (!engineFile.read(engineData.data(), fsize)) { /*...*/ return; }
    engineFile.close();
    nvinfer1::IRuntime* runtime_ptr = nvinfer1::createInferRuntime(mLogger_);
    if (!runtime_ptr) { /*...*/ return; }
    mRuntime.reset(runtime_ptr);
    nvinfer1::ICudaEngine* engine_ptr = mRuntime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine_ptr) { /*...*/ return; }
    mEngine.reset(engine_ptr);
    
    mContext[0].reset(mEngine->createExecutionContext());
    mContext[1].reset(mEngine->createExecutionContext());
    if (!mContext[0] || !mContext[1]) {
      tools::logger()->error("YOLOV5: createExecutionContext failed");
      return;
    }

    CHECK_CUDA(cudaStreamCreate(&mStream[0]));
    CHECK_CUDA(cudaStreamCreate(&mStream[1]));
    CHECK_CUDA(cudaEventCreate(&mEvents[0]));
    CHECK_CUDA(cudaEventCreate(&mEvents[1]));

    allocateBindings();
    tools::logger()->info("YOLOV5: TensorRT engine loaded. Input: {} Output: {}",
                          mInputTensorName_, mOutputTensorName_);
  }
  catch (const std::exception & e) {
    tools::logger()->error("YOLOV5: Exception in ctor: {}", e.what());
    throw std::runtime_error("Failed during YOLOV5 ctor: " + std::string(e.what()));
  }
}
YOLOV5::~YOLOV5()
{
  try {
    tools::logger()->info("YOLOV5: Releasing resources...");
    freeBindings(); 
    freePinnedBuffers(); 

    if (mStream[0]) CHECK_CUDA(cudaStreamDestroy(mStream[0]));
    if (mStream[1]) CHECK_CUDA(cudaStreamDestroy(mStream[1]));
    if (mEvents[0]) CHECK_CUDA(cudaEventDestroy(mEvents[0]));
    if (mEvents[1]) CHECK_CUDA(cudaEventDestroy(mEvents[1]));

    tools::logger()->info("YOLOV5: All resources released.");
  } catch (...) {}
}
void YOLOV5::allocateBindings()
{
  if (!mEngine) return;
  int nbIOTensors = mEngine->getNbIOTensors();
  if (nbIOTensors <= 0) { /*...*/ return; }
  
  size_t elems_in = 0;
  size_t elems_out = 0;

  for (int i = 0; i < nbIOTensors; ++i) {
    const char* tname = mEngine->getIOTensorName(i);
    std::string name(tname);
    nvinfer1::DataType dtype = mEngine->getTensorDataType(name.c_str());
    nvinfer1::TensorIOMode mode = mEngine->getTensorIOMode(name.c_str());
    nvinfer1::Dims dims = mEngine->getTensorShape(name.c_str());
    size_t elems = getTensorVolume(dims);
    size_t bytes = elems * getElementSize(dtype);

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      mInputTensorName_ = name;
      mInputDims_ = dims;
      mInputBytes_ = bytes; 
      mInputDtype_ = dtype;
      elems_in = elems;

      CHECK_CUDA(cudaMalloc(&mGpuBuffers_[0][0], bytes));
      CHECK_CUDA(cudaMalloc(&mGpuBuffers_[1][0], bytes));
      tools::logger()->info("YOLOV5: Alloc 2 sets of TRT input buffers ({} bytes each)", bytes);
    
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      mOutputTensorName_ = name;
      mOutputDims_ = dims;
      mOutputBytes_ = bytes; 
      elems_out = elems;

      CHECK_CUDA(cudaMalloc(&mGpuBuffers_[0][1], bytes));
      CHECK_CUDA(cudaMalloc(&mGpuBuffers_[1][1], bytes));
      mCpuOutputBuffer_[0].resize(elems_out);
      mCpuOutputBuffer_[1].resize(elems_out);
      tools::logger()->info("YOLOV5: Alloc 2 sets of TRT output buffers ({} bytes each)", bytes);
    }
  }
  initPinnedBuffers();
}
void YOLOV5::freeBindings()
{
  for (int i = 0; i < 2; ++i) {
    if (mGpuBuffers_[i][0]) CHECK_CUDA(cudaFree(mGpuBuffers_[i][0]));
    if (mGpuBuffers_[i][1]) CHECK_CUDA(cudaFree(mGpuBuffers_[i][1]));
    mGpuBuffers_[i][0] = nullptr;
    mGpuBuffers_[i][1] = nullptr;
    mCpuOutputBuffer_[i].clear();
  }
}
void YOLOV5::initPinnedBuffers()
{
  if (mPinnedOutputHost_[0] || mPinnedOutputHost_[1]) return;
  if (mOutputBytes_ == 0) { /*...*/ return; }
  mPinnedOutputBytes_ = mOutputBytes_;
  
  CHECK_CUDA(cudaHostAlloc(&mPinnedOutputHost_[0], mPinnedOutputBytes_, cudaHostAllocDefault));
  CHECK_CUDA(cudaHostAlloc(&mPinnedOutputHost_[1], mPinnedOutputBytes_, cudaHostAllocDefault));
  
  if (debug_) tools::logger()->info("YOLOV5: 2 pinned output buffers allocated {} bytes each", mPinnedOutputBytes_);
  
  size_t elems = mPinnedOutputBytes_ / sizeof(float);
  mCpuOutputBuffer_[0].resize(elems);
  mCpuOutputBuffer_[1].resize(elems);
}
void YOLOV5::freePinnedBuffers()
{
  if (mPinnedOutputHost_[0]) { cudaFreeHost(mPinnedOutputHost_[0]); mPinnedOutputHost_[0] = nullptr; }
  if (mPinnedOutputHost_[1]) { cudaFreeHost(mPinnedOutputHost_[1]); mPinnedOutputHost_[1] = nullptr; }
}
void YOLOV5::gpuPreprocessToDevice(const cv::Mat &src_bgr, void* device_input_ptr, int inW, int inH, nvinfer1::DataType dtype, cudaStream_t stream)
{
  auto cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);

  cv::cuda::GpuMat d_src;
  d_src.upload(src_bgr, cv_stream);

  cv::cuda::GpuMat d_resized;
  cv::cuda::resize(d_src, d_resized, cv::Size(inW, inH), 0, 0, cv::INTER_LINEAR, cv_stream);

  cv::cuda::GpuMat d_float;
  d_resized.convertTo(d_float, CV_32F, 1.0 / 255.0, 0.0, cv_stream);

  dim3 block(16, 16);
  dim3 grid((inW + block.x - 1) / block.x, (inH + block.y - 1) / block.y);

  bool dst_fp16 = (dtype == nvinfer1::DataType::kHALF);
  float* d_float_ptr = reinterpret_cast<float*>(d_float.data);
  size_t src_step_bytes = d_float.step;
  
  bgr_to_rgb_nchw_kernel_float<<<grid, block, 0, stream>>>(d_float_ptr, src_step_bytes, device_input_ptr, inH, inW, dst_fp16);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    tools::logger()->error("YOLOV5: gpu preprocess kernel error: {}", cudaGetErrorString(err));
  }
}
std::list<Armor> YOLOV5::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    if (debug_) tools::logger()->warn("YOLOV5::detect empty image");
    return {};
  }
  cv::Mat bgr_img = use_roi_ ? raw_img(roi_) : raw_img;
  if (mInputDims_.nbDims < 4) {
    tools::logger()->error("YOLOV5: invalid input dims");
    return {};
  }
  
  const int current_idx = mBufferIdx;
  const int prev_idx = 1 - current_idx;

  cudaStream_t current_stream = mStream[current_idx];
  auto current_context = mContext[current_idx].get();
  void* current_input_buffer = mGpuBuffers_[current_idx][0];
  void* current_output_buffer = mGpuBuffers_[current_idx][1];
  void* current_pinned_output = mPinnedOutputHost_[current_idx];

  int inN = mInputDims_.d[0] > 0 ? mInputDims_.d[0] : 1;
  int inC = mInputDims_.d[1];
  int inH = mInputDims_.d[2];
  int inW = mInputDims_.d[3];

  if (use_gpu_preproc_ && current_input_buffer) {
    gpuPreprocessToDevice(bgr_img, current_input_buffer, inW, inH, mInputDtype_, current_stream);
  } else {
    cv::Mat resized;
    cv::resize(bgr_img, resized, cv::Size(inW, inH));
    cv::Mat blob;
    cv::dnn::blobFromImage(resized, blob, 1.0 / 255.0, cv::Size(inW, inH), cv::Scalar(), true, false, CV_32F);
    cv::Mat input_contig = blob.isContinuous() ? blob : blob.clone();
    CHECK_CUDA(cudaMemcpyAsync(current_input_buffer, input_contig.data, std::min(input_contig.total() * input_contig.elemSize(), mInputBytes_), cudaMemcpyHostToDevice, current_stream));
  }

  current_context->setTensorAddress(mInputTensorName_.c_str(), current_input_buffer);
  current_context->setTensorAddress(mOutputTensorName_.c_str(), current_output_buffer);
  if (!current_context->enqueueV3(current_stream)) {
    tools::logger()->error("YOLOV5: enqueueV3 failed");
    return {};
  }

  size_t to_copy_back = mPinnedOutputBytes_ ? std::min(mPinnedOutputBytes_, mOutputBytes_) : mOutputBytes_;
  CHECK_CUDA(cudaMemcpyAsync(current_pinned_output, current_output_buffer, to_copy_back, cudaMemcpyDeviceToHost, current_stream));
  
  CHECK_CUDA(cudaEventRecord(mEvents[current_idx], current_stream));

  
  std::list<Armor> armors; 

  if (mFirstFrame) {
    mFirstFrame = false;
  } else {
    CHECK_CUDA(cudaEventSynchronize(mEvents[prev_idx]));

    size_t floats = to_copy_back / sizeof(float);
    if (mCpuOutputBuffer_[prev_idx].size() < floats) mCpuOutputBuffer_[prev_idx].resize(floats);
    std::memcpy(mCpuOutputBuffer_[prev_idx].data(), mPinnedOutputHost_[prev_idx], floats * sizeof(float));

    int out_rows = 1;
    int out_cols = static_cast<int>(mCpuOutputBuffer_[prev_idx].size());
    if (mOutputDims_.nbDims >= 3) {
        out_rows = mOutputDims_.d[mOutputDims_.nbDims - 2];
        out_cols = mOutputDims_.d[mOutputDims_.nbDims - 1];
    } else if (mOutputDims_.nbDims == 2) {
        out_rows = mOutputDims_.d[0];
        out_cols = mOutputDims_.d[1];
    }
    if (static_cast<size_t>(out_rows * out_cols) != mCpuOutputBuffer_[prev_idx].size() && mCpuOutputBuffer_[prev_idx].size() > 0) {
      if (mOutputDims_.nbDims >= 2 && mOutputDims_.d[mOutputDims_.nbDims - 1] > 0) {
          out_cols = mOutputDims_.d[mOutputDims_.nbDims - 1];
          out_rows = mCpuOutputBuffer_[prev_idx].size() / out_cols;
      }
      if (static_cast<size_t>(out_rows * out_cols) != mCpuOutputBuffer_[prev_idx].size()) {
         out_rows = 1;
         out_cols = static_cast<int>(mCpuOutputBuffer_[prev_idx].size());
      }
    }
    
    cv::Mat output(out_rows, out_cols, CV_32F, mCpuOutputBuffer_[prev_idx].data());
    
    armors = parse(1.0, output, mPrevBgrImg, frame_count - 1);
    
    if (debug_) {
        draw_detections(mPrevBgrImg, armors, frame_count - 1);
    }
  }

  bgr_img.copyTo(mPrevBgrImg);
  mBufferIdx = prev_idx;

  return armors;
}
std::list<Armor> YOLOV5::parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  double orig_h = (double)bgr_img.rows;
  double orig_w = (double)bgr_img.cols;
  double resized_h = (mInputDims_.nbDims >= 3) ? (double)mInputDims_.d[2] : orig_h;
  double resized_w = (mInputDims_.nbDims >= 4) ? (double)mInputDims_.d[3] : orig_w;
  if (resized_h == 0) resized_h = orig_h;
  if (resized_w == 0) resized_w = orig_w;
  double scale_w = orig_w / resized_w;
  double scale_h = orig_h / resized_h;

  std::vector<int> color_ids, num_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;

  if (output.empty() || output.type() != CV_32F) {
      tools::logger()->error("YOLOV5::parse: Invalid output mat.");
      return {};
  }
  
  for (int r = 0; r < output.rows; ++r) {
    if (output.cols < 22) {
        if(r == 0 && debug_) tools::logger()->warn("YOLOV5::parse: Output cols ({}) < 22. May truncate.", output.cols);
    }
        
    double score = sigmoid(output.at<float>(r, 8));
    if (score < score_threshold_) continue;

    cv::Mat color_scores = output.row(r).colRange(9, std::min(13, output.cols));
    cv::Mat class_scores = output.row(r).colRange(13, std::min(22, output.cols));
    cv::Point class_id, color_id;
    double score_class = 0.0, score_color = 0.0;
    if (class_scores.cols > 0) cv::minMaxLoc(class_scores, NULL, &score_class, NULL, &class_id);
    if (color_scores.cols > 0) cv::minMaxLoc(color_scores, NULL, &score_color, NULL, &color_id);

    std::vector<cv::Point2f> armor_key_points;
    auto val_at = [&](int rr, int cc)->float {
      return (cc < output.cols) ? output.at<float>(rr, cc) : 0.f;
    };
    
    armor_key_points.push_back({ val_at(r,0) * (float)scale_w, val_at(r,1) * (float)scale_h });
    armor_key_points.push_back({ val_at(r,6) * (float)scale_w, val_at(r,7) * (float)scale_h });
    armor_key_points.push_back({ val_at(r,4) * (float)scale_w, val_at(r,5) * (float)scale_h });
    armor_key_points.push_back({ val_at(r,2) * (float)scale_w, val_at(r,3) * (float)scale_h });

    float min_x = armor_key_points[0].x, max_x = armor_key_points[0].x;
    float min_y = armor_key_points[0].y, max_y = armor_key_points[0].y;
    for (size_t i = 1; i < armor_key_points.size(); ++i) {
      min_x = std::min(min_x, armor_key_points[i].x);
      max_x = std::max(max_x, armor_key_points[i].x);
      min_y = std::min(min_y, armor_key_points[i].y);
      max_y = std::max(max_y, armor_key_points[i].y);
    }

    boxes.emplace_back(static_cast<int>(min_x), static_cast<int>(min_y),
                       static_cast<int>(std::max(0.f, max_x - min_x)),
                       static_cast<int>(std::max(0.f, max_y - min_y)));
    color_ids.emplace_back(color_id.x);
    num_ids.emplace_back(class_id.x);
    confidences.emplace_back(static_cast<float>(score));
    armors_key_points.emplace_back(armor_key_points);
  }

  std::vector<int> indices;
  if (!boxes.empty()) cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (int idx : indices) {
    if (use_roi_) {
      armors.emplace_back(color_ids[idx], num_ids[idx], confidences[idx], boxes[idx], armors_key_points[idx], offset_);
    } else {
      armors.emplace_back(color_ids[idx], num_ids[idx], confidences[idx], boxes[idx], armors_key_points[idx]);
    }
  }
  
  // -----------------------------------------------------------------
  // ⬇️ ⬇️ [关键修复] 在 parse 中保存 tmp_img_ ⬇️ ⬇️
  // -----------------------------------------------------------------
  tmp_img_ = bgr_img; 
  
  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it) || !check_type(*it)) {
      it = armors.erase(it);
      continue;
    }
    if (use_traditional_) detector_.detect(*it, bgr_img); 
    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }
  
  return armors;
}
std::list<Armor> YOLOV5::postprocess(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}
bool YOLOV5::check_name(const Armor & armor) const
{
  return armor.name != ArmorName::not_armor && armor.confidence > min_confidence_;
}
bool YOLOV5::check_type(const Armor & armor) const
{
  if (armor.type == ArmorType::small)
    return armor.name != ArmorName::one && armor.name != ArmorName::base;
  else
    // -----------------------------------------------------------------
    // ⬇️ ⬇️ [关键修复] 修复 ArmorNameD 拼写错误 ⬇️ ⬇️
    // -----------------------------------------------------------------
    return armor.name != ArmorName::two && armor.name != ArmorName::sentry && armor.name != ArmorName::outpost;
}
cv::Point2f YOLOV5::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  return { center.x / static_cast<float>(bgr_img.cols), center.y / static_cast<float>(bgr_img.rows) };
}
void YOLOV5::draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  if (debug_) {
    tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  }
  for (const auto & armor : armors) {
    auto info = fmt::format("{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    if (debug_) tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }
  if (use_roi_) {
    cv::rectangle(detection, roi_, cv::Scalar(0,255,0), 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection", detection);
}
void YOLOV5::save(const Armor & armor) const
{
  // -----------------------------------------------------------------
  // ⬇️ ⬇️ [关键修复] 使用 tmp_img_ ⬇️ ⬇️
  // -----------------------------------------------------------------
  if (tmp_img_.empty()) return;
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}
double YOLOV5::sigmoid(double x)
{
  if (x > 0) return 1.0 / (1.0 + exp(-x));
  return exp(x) / (1.0 + exp(x));
}

} // namespace auto_aim