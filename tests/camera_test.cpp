#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <chrono>

#include "io/camera.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"

namespace fs = std::filesystem;

int main(int argc, char** argv)
{
  const std::string keys =
      "{help h usage ? |                      | 打印帮助信息}"
      "{@config        | configs/camera.yaml  | 相机配置文件 }"
      "{out o          | assets/img_with_q    | 输出文件夹 }"
      "{start s        | 1                    | 起始编号 }"
      "{q quality      | 95                   | JPEG质量(0-100) }"
      "{two-digit      |                      | 同时保存 01.jpg 这种两位编号 }";

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help") || !cli.has("@config")) {
    cli.printMessage();
    return 0;
  }

  std::string config_path = cli.get<std::string>("@config");
  std::string out_dir = cli.get<std::string>("out");
  int index = cli.get<int>("start");
  int jpeg_q = cli.get<int>("quality");
  bool two_digit = cli.has("two-digit");

  if (jpeg_q < 0) jpeg_q = 0;
  if (jpeg_q > 100) jpeg_q = 100;

  fs::create_directories(out_dir);

  tools::Exiter exiter;
  io::Camera camera(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  tools::logger()->info("工业相机拍照程序启动");
  tools::logger()->info("SPACE拍照保存，q/ESC退出");

  const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, jpeg_q};

  while (!exiter.exit()) {
    camera.read(img, timestamp);

    if (img.empty()) {
      // 有些工业相机偶尔会读到空帧，直接跳过
      continue;
    }

    cv::imshow("Industrial Camera Capture (SPACE=save, q/ESC=quit)", img);

    int key = cv::waitKey(1);

    if (key == 27 || key == 'q' || key == 'Q') {  // ESC or q
      break;
    }

    if (key == 32) { // SPACE
      // 1.jpg
      std::string path1 = out_dir + "/" + std::to_string(index) + ".jpg";
      bool ok1 = cv::imwrite(path1, img, params);

      // 01.jpg（可选，兼容你之前 loader 同时尝试 1.jpg 和 01.jpg 的逻辑）
      bool ok2 = true;
      std::string path2;
      if (two_digit) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%02d.jpg", index);
        path2 = out_dir + "/" + std::string(buf);
        ok2 = cv::imwrite(path2, img, params);
      }

      if (ok1 && ok2) {
        tools::logger()->info("Saved {}", path1);
        if (two_digit) tools::logger()->info("Saved {}", path2);
        index++;
      } else {
        tools::logger()->error("保存失败（检查输出目录权限/磁盘空间）");
      }
    }
  }

  tools::logger()->info("程序退出");
  cv::destroyAllWindows();
  return 0;
}
