#pragma once

#include <atomic>
#include <chrono>
#include <list>
#include <thread>
#include <tuple>
#include <string>
#include <opencv2/opencv.hpp>
#include "tasks/auto_aim/armor.hpp"
#include "tools/thread_safe_queue.hpp" // 关键依赖：线程安全队列

namespace auto_aim {
namespace multithread {

/**
 * @brief 多线程探测器
 * * 该类启动一个独立的工作线程，专门用于运行 YOLOV5 (TensorRT) 模型。
 * 它使用两个线程安全队列在 "push" 线程、工作线程 和 "pop" 线程之间
 * 解耦数据流，实现流水线并行处理。
 */
class MultiThreadDetector {
public:
    /**
     * @brief 构造函数
     * @param config_path 配置文件的路径，将传递给 YOLOV5
     */
    explicit MultiThreadDetector(const std::string &config_path);

    /**
     * @brief 析构函数
     * 负责停止并 join 工作线程
     */
    ~MultiThreadDetector();

    // 删除拷贝和赋值构造函数
    MultiThreadDetector(const MultiThreadDetector&) = delete;
    MultiThreadDetector& operator=(const MultiThreadDetector&) = delete;

    /**
     * @brief [线程安全] 生产者调用（例如：相机线程）
     * 向输入队列推送原始图像和时间戳。
     * @param img 原始图像
     * @param t 图像时间戳
     */
    void push(const cv::Mat &img, std::chrono::steady_clock::time_point t);

    /**
     * @brief [线程安全] 消费者调用（例如：主线程）
     * 从输出队列中尝试弹出一个已处理完毕的结果。
     * @return 包含 (图像, 装甲板列表, 时间戳) 的 tuple。如果队列为空，则返回空的 tuple。
     */
    std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> debug_pop();

private:
    /**
     * @brief 内部工作线程的运行函数
     */
    void run();

    // 配置文件路径，传递给 YOLOV5
    std::string config_path_;
    // Debug 标志，从 config_path_ 中读取，传递给 YOLOV5
    bool debug_ = false;

    // 输入队列（[相机线程] -> [YOLO线程]）
    tools::ThreadSafeQueue<std::tuple<cv::Mat, std::chrono::steady_clock::time_point>> input_queue_;
    
    // 输出队列（[YOLO线程] -> [主线程]）
    tools::ThreadSafeQueue<std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>> output_queue_;

    // 线程退出标志
    std::atomic<bool> exit_{false};
    // 内部工作线程（YOLO 推理线程）
    std::thread worker_thread_;
};

} // namespace multithread
} // namespace auto_aim