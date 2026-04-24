#include "yolov5.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <cuda_fp16.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudawarping.hpp>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace
{

static __global__
void bgr_to_rgb_nchw_kernel_float(
  const float * __restrict__ src,
  size_t src_step_bytes,
  void * dst_void,
  int H,
  int W,
  bool dst_fp16)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;

  const char * row_ptr_char =
    reinterpret_cast<const char *>(src) + y * src_step_bytes;
  const float * row_ptr = reinterpret_cast<const float *>(row_ptr_char);

  int idx = x * 3;
  float b = row_ptr[idx + 0];
  float g = row_ptr[idx + 1];
  float r = row_ptr[idx + 2];

  int hw = H * W;
  int pix = y * W + x;

  if (!dst_fp16) {
    float * dst = reinterpret_cast<float *>(dst_void);
    dst[0 * hw + pix] = r;
    dst[1 * hw + pix] = g;
    dst[2 * hw + pix] = b;
  } else {
    __half * dst = reinterpret_cast<__half *>(dst_void);
    dst[0 * hw + pix] = __float2half_rn(r);
    dst[1 * hw + pix] = __float2half_rn(g);
    dst[2 * hw + pix] = __float2half_rn(b);
  }
}

template<typename T>
__device__ inline float read_as_float(T v);

template<>
__device__ inline float read_as_float<float>(float v)
{
  return v;
}

template<>
__device__ inline float read_as_float<__half>(__half v)
{
  return __half2float(v);
}

__device__ inline float sigmoid_device(float x)
{
  return 1.0f / (1.0f + expf(-x));
}

template<typename T>
__device__ inline int argmax_device(const T * p, int n)
{
  if (n <= 0) return 0;
  int best_i = 0;
  float best_v = read_as_float<T>(p[0]);
  for (int i = 1; i < n; ++i) {
    float v = read_as_float<T>(p[i]);
    if (v > best_v) {
      best_v = v;
      best_i = i;
    }
  }
  return best_i;
}

// 每行输出格式假定与当前 CPU parse 一致：
// [0..7] 四点, [8] obj score, [9..12] color, [13..21] class
template<typename T>
__global__ void decode_yolov5_armor_kernel(
  const T * output,
  int rows,
  int cols,
  float score_threshold,
  float scale_w,
  float scale_h,
  auto_aim::GpuDecodedArmor * out_items,
  int * out_count,
  int max_items)
{
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  if (cols < 22) return;

  const T * row = output + r * cols;

  float score = sigmoid_device(read_as_float<T>(row[8]));
  if (score < score_threshold) return;

  int color_id = argmax_device(row + 9, 4);
  int num_id = argmax_device(row + 13, 9);

  int idx = atomicAdd(out_count, 1);
  if (idx >= max_items) return;

  float x1 = read_as_float<T>(row[0]) * scale_w;
  float y1 = read_as_float<T>(row[1]) * scale_h;
  float x4 = read_as_float<T>(row[2]) * scale_w;
  float y4 = read_as_float<T>(row[3]) * scale_h;
  float x3 = read_as_float<T>(row[4]) * scale_w;
  float y3 = read_as_float<T>(row[5]) * scale_h;
  float x2 = read_as_float<T>(row[6]) * scale_w;
  float y2 = read_as_float<T>(row[7]) * scale_h;

  float min_x = fminf(fminf(x1, x2), fminf(x3, x4));
  float min_y = fminf(fminf(y1, y2), fminf(y3, y4));
  float max_x = fmaxf(fmaxf(x1, x2), fmaxf(x3, x4));
  float max_y = fmaxf(fmaxf(y1, y2), fmaxf(y3, y4));

  auto_aim::GpuDecodedArmor item;
  item.score = score;
  item.color_id = color_id;
  item.num_id = num_id;

  item.x1 = x1; item.y1 = y1;
  item.x2 = x2; item.y2 = y2;
  item.x3 = x3; item.y3 = y3;
  item.x4 = x4; item.y4 = y4;

  item.min_x = min_x;
  item.min_y = min_y;
  item.max_x = max_x;
  item.max_y = max_y;

  out_items[idx] = item;
}

}  // namespace

namespace auto_aim
{

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
      if (yaml["use_roi"]) {
        use_roi_ = yaml["use_roi"].as<bool>();
      }
    }

    if (yaml["use_traditional"]) {
      use_traditional_ = yaml["use_traditional"].as<bool>();
    }
    if (yaml["use_gpu_preproc"]) {
      use_gpu_preproc_ = yaml["use_gpu_preproc"].as<bool>();
    }

    roi_ = cv::Rect(x, y, width, height);
    offset_ = cv::Point2f(x, y);

    save_path_ = "imgs";
    std::filesystem::create_directory(save_path_);

    std::ifstream engine_file(model_path_, std::ios::binary);
    if (!engine_file.is_open()) {
      throw std::runtime_error("YOLOV5: cannot open engine file: " + model_path_);
    }

    engine_file.seekg(0, std::ios::end);
    std::streamsize fsize = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);

    std::vector<char> engine_data(static_cast<size_t>(fsize));
    if (!engine_file.read(engine_data.data(), fsize)) {
      throw std::runtime_error("YOLOV5: failed to read engine file: " + model_path_);
    }
    engine_file.close();

    nvinfer1::IRuntime * runtime_ptr = nvinfer1::createInferRuntime(mLogger_);
    if (!runtime_ptr) {
      throw std::runtime_error("YOLOV5: createInferRuntime failed");
    }
    mRuntime.reset(runtime_ptr);

    nvinfer1::ICudaEngine * engine_ptr =
      mRuntime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (!engine_ptr) {
      throw std::runtime_error("YOLOV5: deserializeCudaEngine failed");
    }
    mEngine.reset(engine_ptr);

    mContext[0].reset(mEngine->createExecutionContext());
    mContext[1].reset(mEngine->createExecutionContext());
    if (!mContext[0] || !mContext[1]) {
      throw std::runtime_error("YOLOV5: createExecutionContext failed");
    }

    CHECK_CUDA(cudaStreamCreate(&mStream[0]));
    CHECK_CUDA(cudaStreamCreate(&mStream[1]));
    CHECK_CUDA(cudaEventCreate(&mEvents[0]));
    CHECK_CUDA(cudaEventCreate(&mEvents[1]));

    allocateBindings();

    for (int i = 0; i < 2; ++i) {
      CHECK_CUDA(cudaMalloc(&mDecodedDev_[i], sizeof(GpuDecodedArmor) * kMaxDecodedArmors));
      CHECK_CUDA(cudaHostAlloc(&mDecodedHost_[i],
                               sizeof(GpuDecodedArmor) * kMaxDecodedArmors,
                               cudaHostAllocDefault));

      CHECK_CUDA(cudaMalloc(&mDecodedCountDev_[i], sizeof(int)));
      CHECK_CUDA(cudaHostAlloc(&mDecodedCountHost_[i], sizeof(int), cudaHostAllocDefault));

      *mDecodedCountHost_[i] = 0;
    }

    tools::logger()->info(
      "YOLOV5: TensorRT engine loaded. Input: {} Output: {}",
      mInputTensorName_, mOutputTensorName_);
  } catch (const std::exception & e) {
    tools::logger()->error("YOLOV5: Exception in ctor: {}", e.what());
    throw;
  }
}

YOLOV5::~YOLOV5()
{
  try {
    for (int i = 0; i < 2; ++i) {
      if (mDecodedDev_[i]) CHECK_CUDA(cudaFree(mDecodedDev_[i]));
      if (mDecodedHost_[i]) CHECK_CUDA(cudaFreeHost(mDecodedHost_[i]));
      if (mDecodedCountDev_[i]) CHECK_CUDA(cudaFree(mDecodedCountDev_[i]));
      if (mDecodedCountHost_[i]) CHECK_CUDA(cudaFreeHost(mDecodedCountHost_[i]));

      mDecodedDev_[i] = nullptr;
      mDecodedHost_[i] = nullptr;
      mDecodedCountDev_[i] = nullptr;
      mDecodedCountHost_[i] = nullptr;
    }

    freeBindings();

    if (mStream[0]) CHECK_CUDA(cudaStreamDestroy(mStream[0]));
    if (mStream[1]) CHECK_CUDA(cudaStreamDestroy(mStream[1]));
    if (mEvents[0]) CHECK_CUDA(cudaEventDestroy(mEvents[0]));
    if (mEvents[1]) CHECK_CUDA(cudaEventDestroy(mEvents[1]));
  } catch (...) {
  }
}

void YOLOV5::allocateBindings()
{
  if (!mEngine) return;

  int nb_io_tensors = mEngine->getNbIOTensors();
  if (nb_io_tensors <= 0) {
    throw std::runtime_error("YOLOV5: no IO tensors found");
  }

  for (int i = 0; i < nb_io_tensors; ++i) {
    const char * tname = mEngine->getIOTensorName(i);
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

      CHECK_CUDA(cudaMalloc(&mGpuBuffers_[0][0], bytes));
      CHECK_CUDA(cudaMalloc(&mGpuBuffers_[1][0], bytes));
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      mOutputTensorName_ = name;
      mOutputDims_ = dims;
      mOutputBytes_ = bytes;
      mOutputDtype_ = dtype;

      CHECK_CUDA(cudaMalloc(&mGpuBuffers_[0][1], bytes));
      CHECK_CUDA(cudaMalloc(&mGpuBuffers_[1][1], bytes));
    }
  }
}

void YOLOV5::freeBindings()
{
  for (int i = 0; i < 2; ++i) {
    if (mGpuBuffers_[i][0]) CHECK_CUDA(cudaFree(mGpuBuffers_[i][0]));
    if (mGpuBuffers_[i][1]) CHECK_CUDA(cudaFree(mGpuBuffers_[i][1]));
    mGpuBuffers_[i][0] = nullptr;
    mGpuBuffers_[i][1] = nullptr;
  }
}

void YOLOV5::ensureGpuPreprocBuffers(int rows, int cols, int inW, int inH, int buffer_idx)
{
  if (mGpuSrc_[buffer_idx].rows != rows ||
      mGpuSrc_[buffer_idx].cols != cols ||
      mGpuSrc_[buffer_idx].type() != CV_8UC3) {
    mGpuSrc_[buffer_idx].create(rows, cols, CV_8UC3);
  }

  if (mGpuResized_[buffer_idx].rows != inH ||
      mGpuResized_[buffer_idx].cols != inW ||
      mGpuResized_[buffer_idx].type() != CV_8UC3) {
    mGpuResized_[buffer_idx].create(inH, inW, CV_8UC3);
  }

  if (mGpuFloat_[buffer_idx].rows != inH ||
      mGpuFloat_[buffer_idx].cols != inW ||
      mGpuFloat_[buffer_idx].type() != CV_32FC3) {
    mGpuFloat_[buffer_idx].create(inH, inW, CV_32FC3);
  }
}

void YOLOV5::gpuPreprocessToDevice(
  const cv::Mat & src_bgr,
  void * device_input_ptr,
  int inW,
  int inH,
  nvinfer1::DataType dtype,
  cudaStream_t stream,
  int buffer_idx)
{
  auto cv_stream = cv::cuda::StreamAccessor::wrapStream(stream);

  ensureGpuPreprocBuffers(src_bgr.rows, src_bgr.cols, inW, inH, buffer_idx);

  cv::cuda::GpuMat & d_src = mGpuSrc_[buffer_idx];
  cv::cuda::GpuMat & d_resized = mGpuResized_[buffer_idx];
  cv::cuda::GpuMat & d_float = mGpuFloat_[buffer_idx];

  d_src.upload(src_bgr, cv_stream);
  cv::cuda::resize(d_src, d_resized, cv::Size(inW, inH), 0, 0, cv::INTER_LINEAR, cv_stream);
  d_resized.convertTo(d_float, CV_32F, 1.0 / 255.0, 0.0, cv_stream);

  dim3 block(16, 16);
  dim3 grid((inW + block.x - 1) / block.x, (inH + block.y - 1) / block.y);

  bool dst_fp16 = (dtype == nvinfer1::DataType::kHALF);
  float * d_float_ptr = reinterpret_cast<float *>(d_float.data);
  size_t src_step_bytes = d_float.step;

  bgr_to_rgb_nchw_kernel_float<<<grid, block, 0, stream>>>(
    d_float_ptr, src_step_bytes, device_input_ptr, inH, inW, dst_fp16);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    tools::logger()->error(
      "YOLOV5: gpu preprocess kernel error: {}",
      cudaGetErrorString(err));
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
  void * current_input_buffer = mGpuBuffers_[current_idx][0];
  void * current_output_buffer = mGpuBuffers_[current_idx][1];

  int inH = mInputDims_.d[2];
  int inW = mInputDims_.d[3];

  if (use_gpu_preproc_ && current_input_buffer) {
    gpuPreprocessToDevice(
      bgr_img, current_input_buffer, inW, inH, mInputDtype_, current_stream, current_idx);
  } else {
    cv::Mat resized;
    cv::resize(bgr_img, resized, cv::Size(inW, inH));

    cv::Mat blob;
    cv::dnn::blobFromImage(
      resized, blob, 1.0 / 255.0, cv::Size(inW, inH),
      cv::Scalar(), true, false, CV_32F);

    cv::Mat input_contig = blob.isContinuous() ? blob : blob.clone();
    CHECK_CUDA(cudaMemcpyAsync(
      current_input_buffer,
      input_contig.data,
      std::min(input_contig.total() * input_contig.elemSize(), mInputBytes_),
      cudaMemcpyHostToDevice,
      current_stream));
  }

  current_context->setTensorAddress(mInputTensorName_.c_str(), current_input_buffer);
  current_context->setTensorAddress(mOutputTensorName_.c_str(), current_output_buffer);

  if (!current_context->enqueueV3(current_stream)) {
    tools::logger()->error("YOLOV5: enqueueV3 failed");
    return {};
  }

  // 当前帧保存到对应 buffer，上一帧结果出来时再用 prev_idx 图像构造 Armor
  bgr_img.copyTo(mFrameBuf_[current_idx]);

  int zero = 0;
  CHECK_CUDA(cudaMemcpyAsync(
    mDecodedCountDev_[current_idx],
    &zero,
    sizeof(int),
    cudaMemcpyHostToDevice,
    current_stream));

  int out_rows = 1;
  int out_cols = 1;
  if (mOutputDims_.nbDims >= 3) {
    out_rows = mOutputDims_.d[mOutputDims_.nbDims - 2];
    out_cols = mOutputDims_.d[mOutputDims_.nbDims - 1];
  } else if (mOutputDims_.nbDims == 2) {
    out_rows = mOutputDims_.d[0];
    out_cols = mOutputDims_.d[1];
  }

  float scale_w = static_cast<float>(bgr_img.cols) / static_cast<float>(inW);
  float scale_h = static_cast<float>(bgr_img.rows) / static_cast<float>(inH);

  dim3 block(256);
  dim3 grid((out_rows + block.x - 1) / block.x);

  if (mOutputDtype_ == nvinfer1::DataType::kFLOAT) {
    decode_yolov5_armor_kernel<float><<<grid, block, 0, current_stream>>>(
      reinterpret_cast<const float *>(current_output_buffer),
      out_rows,
      out_cols,
      static_cast<float>(score_threshold_),
      scale_w,
      scale_h,
      mDecodedDev_[current_idx],
      mDecodedCountDev_[current_idx],
      kMaxDecodedArmors);
  } else if (mOutputDtype_ == nvinfer1::DataType::kHALF) {
    decode_yolov5_armor_kernel<__half><<<grid, block, 0, current_stream>>>(
      reinterpret_cast<const __half *>(current_output_buffer),
      out_rows,
      out_cols,
      static_cast<float>(score_threshold_),
      scale_w,
      scale_h,
      mDecodedDev_[current_idx],
      mDecodedCountDev_[current_idx],
      kMaxDecodedArmors);
  } else {
    tools::logger()->error("YOLOV5: unsupported output dtype for gpu decode");
    return {};
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    tools::logger()->error("YOLOV5: decode kernel error: {}", cudaGetErrorString(err));
    return {};
  }

  CHECK_CUDA(cudaMemcpyAsync(
    mDecodedCountHost_[current_idx],
    mDecodedCountDev_[current_idx],
    sizeof(int),
    cudaMemcpyDeviceToHost,
    current_stream));

  // 固定上限的小结果回传，比整块 raw output 回传轻得多
  CHECK_CUDA(cudaMemcpyAsync(
    mDecodedHost_[current_idx],
    mDecodedDev_[current_idx],
    sizeof(GpuDecodedArmor) * kMaxDecodedArmors,
    cudaMemcpyDeviceToHost,
    current_stream));

  CHECK_CUDA(cudaEventRecord(mEvents[current_idx], current_stream));

  std::list<Armor> armors;

  if (mFirstFrame) {
    mFirstFrame = false;
  } else {
    CHECK_CUDA(cudaEventSynchronize(mEvents[prev_idx]));

    int num = *mDecodedCountHost_[prev_idx];
    if (num < 0) num = 0;
    if (num > kMaxDecodedArmors) num = kMaxDecodedArmors;

    std::vector<GpuDecodedArmor> decoded(static_cast<size_t>(num));
    if (num > 0) {
      std::memcpy(
        decoded.data(),
        mDecodedHost_[prev_idx],
        sizeof(GpuDecodedArmor) * static_cast<size_t>(num));
    }

    armors = buildArmorsFromDecoded(decoded, mFrameBuf_[prev_idx], frame_count - 1);

    if (debug_) {
      draw_detections(mFrameBuf_[prev_idx], armors, frame_count - 1);
    }
  }

  mBufferIdx = prev_idx;
  return armors;
}

float YOLOV5::iouRect(const cv::Rect & a, const cv::Rect & b)
{
  int inter_x1 = std::max(a.x, b.x);
  int inter_y1 = std::max(a.y, b.y);
  int inter_x2 = std::min(a.x + a.width,  b.x + b.width);
  int inter_y2 = std::min(a.y + a.height, b.y + b.height);

  int inter_w = std::max(0, inter_x2 - inter_x1);
  int inter_h = std::max(0, inter_y2 - inter_y1);

  float inter_area = static_cast<float>(inter_w * inter_h);
  float union_area = static_cast<float>(a.area() + b.area()) - inter_area;

  if (union_area <= 1e-6f) return 0.0f;
  return inter_area / union_area;
}

void YOLOV5::nmsBoxesFast(
  const std::vector<cv::Rect> & boxes,
  const std::vector<float> & scores,
  float score_threshold,
  float nms_threshold,
  std::vector<int> & indices) const
{
  indices.clear();
  if (boxes.empty() || boxes.size() != scores.size()) return;

  std::vector<int> order;
  order.reserve(boxes.size());

  for (int i = 0; i < static_cast<int>(boxes.size()); ++i) {
    if (scores[i] >= score_threshold) order.push_back(i);
  }

  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return scores[a] > scores[b];
  });

  std::vector<char> suppressed(order.size(), 0);

  for (size_t i = 0; i < order.size(); ++i) {
    if (suppressed[i]) continue;

    int keep = order[i];
    indices.push_back(keep);

    for (size_t j = i + 1; j < order.size(); ++j) {
      if (suppressed[j]) continue;
      if (iouRect(boxes[keep], boxes[order[j]]) > nms_threshold) {
        suppressed[j] = 1;
      }
    }
  }
}

std::list<Armor> YOLOV5::buildArmorsFromDecoded(
  const std::vector<GpuDecodedArmor> & decoded,
  const cv::Mat & bgr_img,
  int frame_count)
{
  (void)frame_count;

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  boxes.reserve(decoded.size());
  scores.reserve(decoded.size());

  for (const auto & d : decoded) {
    boxes.emplace_back(
      static_cast<int>(d.min_x),
      static_cast<int>(d.min_y),
      static_cast<int>(std::max(0.0f, d.max_x - d.min_x)),
      static_cast<int>(std::max(0.0f, d.max_y - d.min_y)));
    scores.emplace_back(d.score);
  }

  std::vector<int> keep;
  nmsBoxesFast(
    boxes,
    scores,
    static_cast<float>(score_threshold_),
    static_cast<float>(nms_threshold_),
    keep);

  std::list<Armor> armors;
  for (int idx : keep) {
    const auto & d = decoded[idx];

    std::vector<cv::Point2f> pts;
    pts.reserve(4);
    pts.push_back({d.x1, d.y1});
    pts.push_back({d.x2, d.y2});
    pts.push_back({d.x3, d.y3});
    pts.push_back({d.x4, d.y4});

    if (use_roi_) {
      armors.emplace_back(
        d.color_id, d.num_id, d.score,
        boxes[idx], pts, offset_);
    } else {
      armors.emplace_back(
        d.color_id, d.num_id, d.score,
        boxes[idx], pts);
    }
  }

  tmp_img_ = bgr_img;

  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it) || !check_type(*it)) {
      it = armors.erase(it);
      continue;
    }

    if (use_traditional_) {
      detector_.detect(*it, bgr_img);
    }

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  return armors;
}

bool YOLOV5::check_name(const Armor & armor) const
{
  return armor.name != ArmorName::not_armor && armor.confidence > min_confidence_;
}

bool YOLOV5::check_type(const Armor & armor) const
{
  if (armor.type == ArmorType::small) {
    return armor.name != ArmorName::one && armor.name != ArmorName::base;
  } else {
    return armor.name != ArmorName::two &&
           armor.name != ArmorName::sentry &&
           armor.name != ArmorName::outpost;
  }
}

cv::Point2f YOLOV5::get_center_norm(
  const cv::Mat & bgr_img,
  const cv::Point2f & center) const
{
  return {
    center.x / static_cast<float>(bgr_img.cols),
    center.y / static_cast<float>(bgr_img.rows)
  };
}

void YOLOV5::draw_detections(
  const cv::Mat & img,
  const std::list<Armor> & armors,
  int frame_count) const
{
  auto detection = img.clone();

  if (debug_) {
    tools::draw_text(
      detection,
      fmt::format("[{}]", frame_count),
      {10, 30},
      {255, 255, 255});
  }

  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}",
      armor.confidence,
      COLORS[armor.color],
      ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);

    tools::draw_points(detection, armor.points, {0, 255, 0});
    if (debug_) {
      tools::draw_text(detection, info, armor.center, {0, 255, 0});
    }
  }

  if (use_roi_) {
    cv::rectangle(detection, roi_, cv::Scalar(0, 255, 0), 2);
  }

  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection", detection);
}

void YOLOV5::save(const Armor & armor) const
{
  if (tmp_img_.empty()) return;

  auto file_name = fmt::format(
    "{:%Y-%m-%d_%H-%M-%S}",
    std::chrono::system_clock::now());

  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

double YOLOV5::sigmoid(double x)
{
  if (x > 0) return 1.0 / (1.0 + std::exp(-x));
  return std::exp(x) / (1.0 + std::exp(x));
}

}  // namespace auto_aim