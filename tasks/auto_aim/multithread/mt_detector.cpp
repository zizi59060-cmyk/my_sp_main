#include "tasks/auto_aim/multithread/mt_detector.hpp"

// 引入 YAML 解析器
#include <yaml-cpp/yaml.h> 
// 引入你修改后的 TensorRT YOLOV5
#include "tasks/auto_aim/yolos/yolov5.hpp"
// 引入日志工具
#include "tools/logger.hpp"

namespace auto_aim {
namespace multithread {

// -----------------------------------------------------------------
// ⬇️ ⬇️ [关键修复 1] 解决构造函数错误 ⬇️ ⬇️
// -----------------------------------------------------------------
MultiThreadDetector::MultiThreadDetector(const std::string &config_path)
    : config_path_(config_path),
      // 错误日志显示：ThreadSafeQueue 构造函数需要2个参数
      // 我们在成员初始化列表中显式调用它
      // 参数 1: 队列大小 (我们用 2 作为"双缓冲"大小)
      // 参数 2: 队列满时的回调函数
      input_queue_(2, [] { tools::logger()->warn("mt_detector: input_queue is full!"); }),
      output_queue_(2, [] { tools::logger()->warn("mt_detector: output_queue is full!"); }) {
    
    // 尝试从 YAML 文件中读取 debug 标志
    try {
        YAML::Node config = YAML::LoadFile(config_path);
        debug_ = config["debug"].as<bool>(false);
        tools::logger()->info("mt_detector: Read debug flag = {}", debug_);
    } catch (const std::exception &e) {
        tools::logger()->error("mt_detector: Failed to load debug flag from config: {}. Defaulting to false.", e.what());
        debug_ = false;
    }

    // 启动工作线程
    worker_thread_ = std::thread(&MultiThreadDetector::run, this);
}

MultiThreadDetector::~MultiThreadDetector() {
    // 发送退出信号
    exit_.store(true);
    // 等待工作线程安全退出
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    tools::logger()->info("mt_detector: Worker thread joined.");
}

void MultiThreadDetector::push(const cv::Mat &img, std::chrono::steady_clock::time_point t) {
    if (!exit_) {
        // 注意：这里需要 clone 图像，因为原始的 img 缓冲区可能会在
        // push 线程中被立即重用（例如相机 SDK 的环形缓冲区）。
        // clone 确保了工作线程有自己的数据副本。
        input_queue_.push(std::make_tuple(img.clone(), t));
    }
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::debug_pop() {
    std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> data;
    
    // 使用 try_pop，主线程不会被阻塞
    if (output_queue_.try_pop(data)) {
        return data;
    }
    
    // 如果队列为空，返回一个空的 tuple
    return {}; 
}

void MultiThreadDetector::run() {
    tools::logger()->info("mt_detector: Worker thread started.");

    // --- 关键：在此线程中实例化 YOLOV5 ---
    // 这将确保 TensorRT 引擎、CUDA 上下文等资源
    // 在这个专用的工作线程上创建和绑定。
    std::unique_ptr<auto_aim::YOLOV5> yolo;
    try {
        yolo = std::make_unique<auto_aim::YOLOV5>(config_path_, debug_);
    } catch (const std::exception &e) {
        // 如果 TRT 初始化失败，线程无法工作
        tools::logger()->critical("mt_detector: Failed to initialize YOLOV5 in worker thread: {}", e.what());
        return; // 线程退出
    }
    
    tools::logger()->info("mt_detector: YOLOV5 model initialized successfully in worker thread.");

    std::tuple<cv::Mat, std::chrono::steady_clock::time_point> input_data;
    int frame_count = 0;

    // 线程主循环
    while (!exit_) {
        // -----------------------------------------------------------------
        // ⬇️ ⬇️ [关键修复 2] 解决 try_pop 错误 ⬇️ ⬇️
        // -----------------------------------------------------------------
        // 错误日志显示：try_pop 只接受 1 个参数
        if (input_queue_.try_pop(input_data)) {
            frame_count++;
            
            // 从 tuple 中解包数据
            cv::Mat img = std::get<0>(input_data);
            auto t = std::get<1>(input_data);

            // --- 执行 TensorRT 推理 ---
            std::list<Armor> armors;
            try {
                armors = yolo->detect(img, frame_count);
            } catch (const std::exception &e) {
                tools::logger()->error("mt_detector: Error during yolo->detect(): {}", e.what());
                // 即使检测出错，也继续处理下一帧
            }
            // ---------------------------

            // 将结果（即使是空的）推送到输出队列
            // 我们使用 std::move 来避免不必要的拷贝
            output_queue_.push(std::make_tuple(std::move(img), std::move(armors), t));

        } else {
            // try_pop 失败，说明输入队列为空
            // sleep 1ms 来防止此循环 100% 占用 CPU (忙等)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    tools::logger()->info("mt_detector: Worker thread exiting.");
}

} // namespace multithread
} // namespace auto_aim