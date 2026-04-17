#include "classifier.hpp"

#include <yaml-cpp/yaml.h>
#include <fstream>
#include <iterator>

#include "tools/logger.hpp" // 你的项目 logger（用于记录高层信息）

namespace auto_aim
{

// -----------------------------
// Helper: read whole file into vector<char>
// -----------------------------
static std::vector<char> readFile(const std::string & filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(static_cast<size_t>(size));
    if (!file.read(buffer.data(), size)) return {};
    return buffer;
}

// -----------------------------
// Constructor
// -----------------------------
Classifier::Classifier(const std::string & config_path)
{
    try
    {
        auto yaml = YAML::LoadFile(config_path);
        auto model_path = yaml["classify_model"].as<std::string>(); // e.g., "assets/tiny_resnet.onnx"

        // 保留 OpenCV ONNX 方案（作为 fallback / for net_.forward）
        try {
            net_ = cv::dnn::readNetFromONNX(model_path);
        } catch (const std::exception & e) {
            tools::logger()->warn("Classifier: readNetFromONNX failed (may be OK): {}", e.what());
        }

        // engine 路径 (.engine)
        std::string engine_path = model_path;
        size_t dot_pos = engine_path.rfind('.');
        if (dot_pos != std::string::npos) {
            engine_path.replace(dot_pos, engine_path.length() - dot_pos, ".engine");
        } else {
            engine_path += ".engine";
        }

        tools::logger()->info("Classifier: Loading TensorRT engine: {}", engine_path);

        // 读取 engine 文件
        std::vector<char> engineData = readFile(engine_path);
        if (engineData.empty()) {
            tools::logger()->error("Classifier: Failed to read engine file: {}", engine_path);
            return;
        }

        // 创建 Runtime — 注意：createInferRuntime 需要 ILogger&（不是指针）
        nvinfer1::IRuntime* runtime_ptr = nvinfer1::createInferRuntime(mLogger_);
        if (!runtime_ptr) {
            tools::logger()->error("Classifier: createInferRuntime returned nullptr");
            return;
        }
        mRuntime.reset(runtime_ptr);

        // 反序列化引擎
        nvinfer1::ICudaEngine* engine_ptr =
            mRuntime->deserializeCudaEngine(engineData.data(), engineData.size());
        if (!engine_ptr) {
            tools::logger()->error("Classifier: deserializeCudaEngine failed");
            return;
        }
        mEngine.reset(engine_ptr);

        // 创建 execution context
        nvinfer1::IExecutionContext* ctx_ptr = mEngine->createExecutionContext();
        if (!ctx_ptr) {
            tools::logger()->error("Classifier: createExecutionContext failed");
            return;
        }
        mContext.reset(ctx_ptr);

        // 创建 CUDA stream（显式检查返回值）
        {
            cudaError_t err = cudaStreamCreate(&mStream);
            if (err != cudaSuccess) {
                tools::logger()->error("Classifier: cudaStreamCreate failed: {}", cudaGetErrorString(err));
                mStream = nullptr;
                return;
            }
        }

        // 分配绑定缓冲区（输入/输出）
        allocateBindings();

        tools::logger()->info("Classifier: TensorRT Engine loaded successfully.");
    }
    catch (const std::exception & e)
    {
        tools::logger()->error("Classifier: Exception during init: {}", e.what());
    }
}

// -----------------------------
// Destructor
// -----------------------------
Classifier::~Classifier()
{
    try {
        freeBindings();

        if (mStream) {
            cudaError_t err = cudaStreamDestroy(mStream);
            if (err != cudaSuccess) {
                tools::logger()->warn("Classifier: cudaStreamDestroy failed: {}", cudaGetErrorString(err));
            }
            mStream = nullptr;
        }

        tools::logger()->info("Classifier: TensorRT resources released.");
    } catch (...) {
        // 不抛异常于析构函数
    }
}

// -----------------------------
// allocateBindings: 获取 I/O Tensor 信息并分配 GPU/CPU 缓冲区
// -----------------------------
void Classifier::allocateBindings()
{
    if (!mEngine) return;

    // iterate I/O tensors
    int nbIOTensors = mEngine->getNbIOTensors();
    if (nbIOTensors <= 0) {
        tools::logger()->error("Classifier: engine has no I/O tensors");
        return;
    }

    // We expect usually 2 (input + output). If engine 不同，按 mode 匹配。
    for (int i = 0; i < nbIOTensors; ++i)
    {
        const char* name = mEngine->getIOTensorName(i);
        if (!name) continue;
        std::string sname(name);

        // get tensor data type & shape
        nvinfer1::DataType dtype = mEngine->getTensorDataType(name);
        nvinfer1::TensorIOMode ioMode = mEngine->getTensorIOMode(name);
        nvinfer1::Dims shape = mEngine->getTensorShape(name);

        // 使用 trt_utils 提供的辅助函数（注意命名空间）
        size_t elements = auto_aim::getTensorVolume(shape);
        size_t bytes = elements * auto_aim::getElementSize(dtype);

        if (ioMode == nvinfer1::TensorIOMode::kINPUT) {
            mInputTensorName_ = sname;
            cudaError_t err = cudaMalloc(&mGpuBuffers_[0], bytes);
            if (err != cudaSuccess) {
                tools::logger()->error("Classifier: cudaMalloc input '{}' failed: {}", sname, cudaGetErrorString(err));
                mGpuBuffers_[0] = nullptr;
                return;
            }
            tools::logger()->info("Classifier: Allocated input tensor '{}' elements={} bytes={}", sname, elements, bytes);
        }
        else if (ioMode == nvinfer1::TensorIOMode::kOUTPUT) {
            mOutputTensorName_ = sname;
            mCpuOutputBuffer_.resize(elements); // 假设输出为 float（如非 float 需处理）
            cudaError_t err = cudaMalloc(&mGpuBuffers_[1], bytes);
            if (err != cudaSuccess) {
                tools::logger()->error("Classifier: cudaMalloc output '{}' failed: {}", sname, cudaGetErrorString(err));
                mGpuBuffers_[1] = nullptr;
                return;
            }
            tools::logger()->info("Classifier: Allocated output tensor '{}' elements={} bytes={}", sname, elements, bytes);
        }
        else {
            tools::logger()->warn("Classifier: Unknown I/O Tensor mode for '{}'", sname);
        }
    }

    // 如果某些 buffer 未分配，记日志
    for (int i = 0; i < 2; ++i) {
        if (!mGpuBuffers_[i]) {
            tools::logger()->warn("Classifier: mGpuBuffers_[{}] not allocated", i);
        }
    }
}

// -----------------------------
// freeBindings
// -----------------------------
void Classifier::freeBindings()
{
    for (int i = 0; i < 2; ++i) {
        if (mGpuBuffers_[i]) {
            cudaError_t err = cudaFree(mGpuBuffers_[i]);
            if (err != cudaSuccess) {
                tools::logger()->warn("Classifier: cudaFree mGpuBuffers_[{}] failed: {}", i, cudaGetErrorString(err));
            }
            mGpuBuffers_[i] = nullptr;
        }
    }
    mCpuOutputBuffer_.clear();
}

// -----------------------------
// classify: 旧的 OpenCV DNN CPU 逻辑（保留）
// -----------------------------
void Classifier::classify(Armor & armor)
{
    if (armor.pattern.empty()) {
        armor.name = ArmorName::not_armor;
        return;
    }

    cv::Mat gray;
    cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

    auto input = cv::Mat(32, 32, CV_8UC1, cv::Scalar(0));
    auto x_scale = static_cast<double>(32) / gray.cols;
    auto y_scale = static_cast<double>(32) / gray.rows;
    auto scale = std::min(x_scale, y_scale);
    auto h = static_cast<int>(gray.rows * scale);
    auto w = static_cast<int>(gray.cols * scale);

    if (h == 0 || w == 0) {
        armor.name = ArmorName::not_armor;
        return;
    }
    auto roi = cv::Rect(0, 0, w, h);
    cv::resize(gray, input(roi), {w, h});

    auto blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar());

    net_.setInput(blob);
    cv::Mat outputs = net_.forward();

    // softmax
    float max = *std::max_element(outputs.begin<float>(), outputs.end<float>());
    cv::exp(outputs - max, outputs);
    float sum = cv::sum(outputs)[0];
    outputs /= sum;

    double confidence;
    cv::Point label_point;
    cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
    int label_id = label_point.x;

    armor.confidence = confidence;
    armor.name = static_cast<ArmorName>(label_id);
}

// -----------------------------
// ovclassify: TensorRT 10.3 (I/O Tensors API) 推理实现
// -----------------------------
void Classifier::ovclassify(Armor & armor)
{
    if (!mEngine || !mContext) {
        // fallback to CPU classify if TRT not ready
        classify(armor);
        return;
    }

    if (armor.pattern.empty()) {
        armor.name = ArmorName::not_armor;
        return;
    }

    // preprocess: gray -> resized 32x32 float normalized
    cv::Mat gray;
    cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

    cv::Mat input_mat(32, 32, CV_32F, cv::Scalar(0));
    auto x_scale = static_cast<double>(32) / gray.cols;
    auto y_scale = static_cast<double>(32) / gray.rows;
    auto scale = std::min(x_scale, y_scale);
    int h = static_cast<int>(gray.rows * scale);
    int w = static_cast<int>(gray.cols * scale);

    if (h == 0 || w == 0) {
        armor.name = ArmorName::not_armor;
        return;
    }

    cv::Rect roi(0, 0, w, h);
    cv::resize(gray, input_mat(roi), {w, h});
    input_mat.convertTo(input_mat, CV_32F, 1.0 / 255.0);

    // Ensure contiguity
    cv::Mat input_contig = input_mat.isContinuous() ? input_mat : input_mat.clone();

    // Copy H->D
    // We assume input dtype is float32. If engine input dtype not float, conversion needed.
    size_t input_bytes = input_contig.total() * input_contig.elemSize(); // bytes
    if (!mGpuBuffers_[0]) {
        tools::logger()->error("Classifier: input buffer not allocated");
        classify(armor); // fallback
        return;
    }

    {
        cudaError_t err = cudaMemcpyAsync(mGpuBuffers_[0], input_contig.data, input_bytes, cudaMemcpyHostToDevice, mStream);
        if (err != cudaSuccess) {
            tools::logger()->error("Classifier: cudaMemcpyAsync H2D failed: {}", cudaGetErrorString(err));
            classify(armor);
            return;
        }
    }

    // set addresses for I/O tensors
    mContext->setTensorAddress(mInputTensorName_.c_str(), mGpuBuffers_[0]);
    mContext->setTensorAddress(mOutputTensorName_.c_str(), mGpuBuffers_[1]);

    // enqueue inference
    if (!mContext->enqueueV3(mStream)) {
        tools::logger()->error("Classifier: enqueueV3 failed");
        classify(armor);
        return;
    }

    // copy D->H
    if (!mGpuBuffers_[1]) {
        tools::logger()->error("Classifier: output buffer not allocated");
        classify(armor);
        return;
    }

    {
        cudaError_t err = cudaMemcpyAsync(mCpuOutputBuffer_.data(), mGpuBuffers_[1], mCpuOutputBuffer_.size() * sizeof(float), cudaMemcpyDeviceToHost, mStream);
        if (err != cudaSuccess) {
            tools::logger()->error("Classifier: cudaMemcpyAsync D2H failed: {}", cudaGetErrorString(err));
            classify(armor);
            return;
        }
    }

    // sync stream
    {
        cudaError_t err = cudaStreamSynchronize(mStream);
        if (err != cudaSuccess) {
            tools::logger()->error("Classifier: cudaStreamSynchronize failed: {}", cudaGetErrorString(err));
            classify(armor);
            return;
        }
    }

    // form cv::Mat from CPU buffer
    cv::Mat outputs(1, static_cast<int>(mCpuOutputBuffer_.size()), CV_32F, mCpuOutputBuffer_.data());

    // softmax
    float max = *std::max_element(outputs.begin<float>(), outputs.end<float>());
    cv::exp(outputs - max, outputs);
    float sum = cv::sum(outputs)[0];
    outputs /= sum;

    double confidence;
    cv::Point label_point;
    cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
    int label_id = label_point.x;

    armor.confidence = confidence;
    armor.name = static_cast<ArmorName>(label_id);
}

} // namespace auto_aim

