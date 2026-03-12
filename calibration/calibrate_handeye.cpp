// #include <fmt/core.h>
// #include <yaml-cpp/yaml.h>

// #include <Eigen/Dense>  // 必须在opencv2/core/eigen.hpp上面
// #include <fstream>
// #include <opencv2/core/eigen.hpp>
// #include <opencv2/opencv.hpp>

// #include "tools/img_tools.hpp"
// #include "tools/math_tools.hpp"

// const std::string keys =
//   "{help h usage ? |                          | 输出命令行参数说明}"
//   "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
//   "{@input-folder  | assets/img_with_q        | 输入文件夹路径   }";

// std::vector<cv::Point3f> centers_3d(const cv::Size & pattern_size, const float center_distance)
// {
//   std::vector<cv::Point3f> centers_3d;

//   for (int i = 0; i < pattern_size.height; i++)
//     for (int j = 0; j < pattern_size.width; j++)
//       centers_3d.push_back({j * center_distance, i * center_distance, 0});

//   return centers_3d;
// }

// Eigen::Quaterniond read_q(const std::string & q_path)
// {
//   std::ifstream q_file(q_path);
//   double w, x, y, z;
//   q_file >> w >> x >> y >> z;
//   return {w, x, y, z};
// }

// void load(
//   const std::string & input_folder, const std::string & config_path,
//   std::vector<double> & R_gimbal2imubody_data, std::vector<cv::Mat> & R_gimbal2world_list,
//   std::vector<cv::Mat> & t_gimbal2world_list, std::vector<cv::Mat> & rvecs,
//   std::vector<cv::Mat> & tvecs)
// {
//   // 读取yaml参数
//   auto yaml = YAML::LoadFile(config_path);
//   auto pattern_cols = yaml["pattern_cols"].as<int>();
//   auto pattern_rows = yaml["pattern_rows"].as<int>();
//   auto center_distance_mm = yaml["center_distance_mm"].as<double>();
//   R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
//   auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
//   auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();

//   cv::Size pattern_size(pattern_cols, pattern_rows);
//   Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_gimbal2imubody(R_gimbal2imubody_data.data());
//   cv::Matx33d camera_matrix(camera_matrix_data.data());
//   cv::Mat distort_coeffs(distort_coeffs_data);

//   for (int i = 1; true; i++) {
//     // 读取图片和对应四元数
//     auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
//     auto q_path = fmt::format("{}/{}.txt", input_folder, i);
//     auto img = cv::imread(img_path);
//     Eigen::Quaterniond q = read_q(q_path);
//     if (img.empty()) break;

//     // 计算云台的欧拉角
//     Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
//     Eigen::Matrix3d R_gimbal2world =
//       R_gimbal2imubody.transpose() * R_imubody2imuabs * R_gimbal2imubody;
//     Eigen::Vector3d ypr = tools::eulers(R_gimbal2world, 2, 1, 0) * 57.3;  // degree

//     // 在图片上显示云台的欧拉角，用来检验R_gimbal2imubody是否正确
//     auto drawing = img.clone();
//     tools::draw_text(drawing, fmt::format("yaw   {:.2f}", ypr[0]), {40, 40}, {0, 0, 255});
//     tools::draw_text(drawing, fmt::format("pitch {:.2f}", ypr[1]), {40, 80}, {0, 0, 255});
//     tools::draw_text(drawing, fmt::format("roll  {:.2f}", ypr[2]), {40, 120}, {0, 0, 255});

//     // 识别标定板
//     std::vector<cv::Point2f> centers_2d;
//     auto success = cv::findCirclesGrid(img, pattern_size, centers_2d);  // 默认是对称圆点图案

//     // 显示识别结果
//     cv::drawChessboardCorners(drawing, pattern_size, centers_2d, success);
//     cv::resize(drawing, drawing, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
//     cv::imshow("Press any to continue", drawing);
//     cv::waitKey(0);

//     // 输出识别结果
//     fmt::print("[{}] {}\n", success ? "success" : "failure", img_path);
//     if (!success) continue;

//     // 计算所需的数据
//     cv::Mat t_gimbal2world = (cv::Mat_<double>(3, 1) << 0, 0, 0);
//     cv::Mat R_gimbal2world_cv;
//     cv::eigen2cv(R_gimbal2world, R_gimbal2world_cv);
//     cv::Mat rvec, tvec;
//     auto centers_3d_ = centers_3d(pattern_size, center_distance_mm);
//     cv::solvePnP(
//       centers_3d_, centers_2d, camera_matrix, distort_coeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE);

//     // 记录所需的数据
//     R_gimbal2world_list.emplace_back(R_gimbal2world_cv);
//     t_gimbal2world_list.emplace_back(t_gimbal2world);
//     rvecs.emplace_back(rvec);
//     tvecs.emplace_back(tvec);
//   }
// }

// void print_yaml(
//   const std::vector<double> & R_gimbal2imubody_data, const cv::Mat & R_camera2gimbal,
//   const cv::Mat & t_camera2gimbal, const Eigen::Vector3d & ypr)
// {
//   YAML::Emitter result;
//   std::vector<double> R_camera2gimbal_data(
//     R_camera2gimbal.begin<double>(), R_camera2gimbal.end<double>());
//   std::vector<double> t_camera2gimbal_data(
//     t_camera2gimbal.begin<double>(), t_camera2gimbal.end<double>());

//   result << YAML::BeginMap;
//   result << YAML::Key << "R_gimbal2imubody";
//   result << YAML::Value << YAML::Flow << R_gimbal2imubody_data;
//   result << YAML::Newline;
//   result << YAML::Newline;
//   result << YAML::Comment(fmt::format(
//     "相机同理想情况的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree", ypr[0], ypr[1], ypr[2]));
//   result << YAML::Key << "R_camera2gimbal";
//   result << YAML::Value << YAML::Flow << R_camera2gimbal_data;
//   result << YAML::Key << "t_camera2gimbal";
//   result << YAML::Value << YAML::Flow << t_camera2gimbal_data;
//   result << YAML::Newline;
//   result << YAML::EndMap;

//   fmt::print("\n{}\n", result.c_str());
// }

// int main(int argc, char * argv[])
// {
//   // 读取命令行参数
//   cv::CommandLineParser cli(argc, argv, keys);
//   if (cli.has("help")) {
//     cli.printMessage();
//     return 0;
//   }
//   auto input_folder = cli.get<std::string>(0);
//   auto config_path = cli.get<std::string>("config-path");

//   // 从输入文件夹中加载标定所需的数据
//   std::vector<double> R_gimbal2imubody_data;
//   std::vector<cv::Mat> R_gimbal2world_list, t_gimbal2world_list;
//   std::vector<cv::Mat> rvecs, tvecs;
//   load(
//     input_folder, config_path, R_gimbal2imubody_data, R_gimbal2world_list, t_gimbal2world_list,
//     rvecs, tvecs);

//   // 手眼标定
//   cv::Mat R_camera2gimbal, t_camera2gimbal;
//   cv::calibrateHandEye(
//     R_gimbal2world_list, t_gimbal2world_list, rvecs, tvecs, R_camera2gimbal, t_camera2gimbal);
//   t_camera2gimbal /= 1e3;  // mm to m

//   // 计算相机同理想情况的偏角
//   Eigen::Matrix3d R_camera2gimbal_eigen;
//   cv::cv2eigen(R_camera2gimbal, R_camera2gimbal_eigen);
//   Eigen::Matrix3d R_gimbal2ideal{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}};
//   Eigen::Matrix3d R_camera2ideal = R_gimbal2ideal * R_camera2gimbal_eigen;
//   Eigen::Vector3d ypr = tools::eulers(R_camera2ideal, 1, 0, 2) * 57.3;  // degree

//   // 输出yaml
//   print_yaml(R_gimbal2imubody_data, R_camera2gimbal, t_camera2gimbal, ypr);
// }
// #include <fmt/core.h>
// #include <yaml-cpp/yaml.h>

// #include <Eigen/Dense>
// #include <fstream>
// #include <opencv2/core/eigen.hpp>
// #include <opencv2/opencv.hpp>

// #include "tools/img_tools.hpp"
// #include "tools/math_tools.hpp"

// const std::string keys =
//   "{help h usage ? |                          | 输出命令行参数说明}"
//   "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
//   "{@input-folder  | assets/img_with_q        | 输入文件夹路径   }";

// // 生成棋盘格角点的3D坐标
// std::vector<cv::Point3f> chessboard_corners_3d(const cv::Size & pattern_size, const float square_size)
// {
//   std::vector<cv::Point3f> corners_3d;

//   for (int i = 0; i < pattern_size.height; i++) {
//     for (int j = 0; j < pattern_size.width; j++) {
//       corners_3d.push_back(cv::Point3f(j * square_size, i * square_size, 0));
//     }
//   }
//   return corners_3d;
// }

// Eigen::Quaterniond read_q(const std::string & q_path)
// {
//   std::ifstream q_file(q_path);
//   double w, x, y, z;
//   q_file >> w >> x >> y >> z;
//   return Eigen::Quaterniond(w, x, y, z);
// }

// // 解析ROS格式的相机参数
// void parse_ros_camera_params(const std::string & config_path, 
//                             cv::Matx33d & camera_matrix,
//                             cv::Mat & distort_coeffs)
// {
//   fmt::print("Loading ROS camera parameters from: {}\n", config_path);
  
//   YAML::Node yaml;
//   try {
//     yaml = YAML::LoadFile(config_path);
//   } catch (const YAML::ParserException& e) {
//     throw std::runtime_error(fmt::format("YAML parsing error: {}", e.what()));
//   }
  
//   // 检查必需的字段
//   if (!yaml["camera_matrix"]) {
//     throw std::runtime_error("Missing 'camera_matrix' in YAML config");
//   }
//   if (!yaml["distortion_coefficients"]) {
//     throw std::runtime_error("Missing 'distortion_coefficients' in YAML config");
//   }
  
//   // 解析相机内参矩阵
//   auto camera_matrix_node = yaml["camera_matrix"];
//   auto camera_matrix_data = camera_matrix_node["data"].as<std::vector<double>>();
  
//   if (camera_matrix_data.size() != 9) {
//     throw std::runtime_error("Camera matrix should have 9 elements");
//   }
  
//   camera_matrix = cv::Matx33d(
//     camera_matrix_data[0], camera_matrix_data[1], camera_matrix_data[2],
//     camera_matrix_data[3], camera_matrix_data[4], camera_matrix_data[5],
//     camera_matrix_data[6], camera_matrix_data[7], camera_matrix_data[8]
//   );
  
//   // 解析畸变系数
//   auto distort_coeffs_node = yaml["distortion_coefficients"];
//   auto distort_coeffs_data = distort_coeffs_node["data"].as<std::vector<double>>();
  
//   distort_coeffs = cv::Mat(distort_coeffs_data).reshape(1, 1);
  
//   fmt::print("Camera matrix loaded:\n");
//   fmt::print("[{:.5f}, {:.5f}, {:.5f}]\n", camera_matrix(0,0), camera_matrix(0,1), camera_matrix(0,2));
//   fmt::print("[{:.5f}, {:.5f}, {:.5f}]\n", camera_matrix(1,0), camera_matrix(1,1), camera_matrix(1,2));
//   fmt::print("[{:.5f}, {:.5f}, {:.5f}]\n", camera_matrix(2,0), camera_matrix(2,1), camera_matrix(2,2));
//   fmt::print("Distortion coefficients: ");
//   for (int i = 0; i < distort_coeffs.cols; i++) {
//     fmt::print("{:.6f} ", distort_coeffs.at<double>(0, i));
//   }
//   fmt::print("\n");
// }

// void load(
//   const std::string & input_folder, const std::string & config_path,
//   std::vector<double> & R_gimbal2imubody_data, std::vector<cv::Mat> & R_gimbal2world_list,
//   std::vector<cv::Mat> & t_gimbal2world_list, std::vector<cv::Mat> & rvecs,
//   std::vector<cv::Mat> & tvecs)
// {
//   fmt::print("Loading YAML config from: {}\n", config_path);
  
//   // 读取yaml参数
//   YAML::Node yaml;
//   try {
//     yaml = YAML::LoadFile(config_path);
//     fmt::print("YAML config loaded successfully\n");
//   } catch (const YAML::ParserException& e) {
//     throw std::runtime_error(fmt::format("YAML parsing error: {}", e.what()));
//   } catch (const YAML::BadFile& e) {
//     throw std::runtime_error(fmt::format("YAML file not found: {}", e.what()));
//   }
  
//   // 添加错误检查
//   if (!yaml["pattern_cols"]) {
//     throw std::runtime_error("Missing 'pattern_cols' in YAML config");
//   }
//   if (!yaml["pattern_rows"]) {
//     throw std::runtime_error("Missing 'pattern_rows' in YAML config");
//   }
//   if (!yaml["square_size_mm"]) {
//     throw std::runtime_error("Missing 'square_size_mm' in YAML config");
//   }
//   if (!yaml["R_gimbal2imubody"]) {
//     throw std::runtime_error("Missing 'R_gimbal2imubody' in YAML config");
//   }
//   if (!yaml["camera_params_path"]) {
//     throw std::runtime_error("Missing 'camera_params_path' in YAML config");
//   }

//   auto pattern_cols = yaml["pattern_cols"].as<int>();
//   auto pattern_rows = yaml["pattern_rows"].as<int>();
//   auto square_size_mm = yaml["square_size_mm"].as<double>();
//   R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
//   auto camera_params_path = yaml["camera_params_path"].as<std::string>();
  
//   fmt::print("Pattern size: {}x{}\n", pattern_cols, pattern_rows);
//   fmt::print("Square size: {}mm\n", square_size_mm);
//   fmt::print("R_gimbal2imubody size: {}\n", R_gimbal2imubody_data.size());
//   fmt::print("Camera params path: {}\n", camera_params_path);

//   // 解析ROS格式的相机参数
//   cv::Matx33d camera_matrix;
//   cv::Mat distort_coeffs;
//   parse_ros_camera_params(camera_params_path, camera_matrix, distort_coeffs);

//   // 棋盘格参数 (6×8 30mm)
//   const cv::Size pattern_size(pattern_cols - 1, pattern_rows - 1); // 内部角点数 (列数-1, 行数-1)
//   const float square_size = square_size_mm;
//   const cv::Scalar borderColor(0, 255, 0); // 绿色边框 (BGR格式)
//   const int borderThickness = 2;

//   // 检查矩阵数据大小
//   if (R_gimbal2imubody_data.size() != 9) {
//     throw std::runtime_error("R_gimbal2imubody should have 9 elements (3x3 matrix)");
//   }

//   Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_gimbal2imubody(R_gimbal2imubody_data.data());

//   fmt::print("Processing images in: {}\n", input_folder);
  
//   for (int i = 1; true; i++) {
//     // 读取图片和对应四元数
//     auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
//     auto q_path = fmt::format("{}/{}.txt", input_folder, i);
    
//     fmt::print("Reading image: {}\n", img_path);
//     auto img = cv::imread(img_path);
    
//     // 检查文件是否存在
//     if (img.empty()) {
//       fmt::print("No more images found. Processed {} images.\n", i-1);
//       break;
//     }
    
//     fmt::print("Reading quaternion: {}\n", q_path);
//     std::ifstream q_file(q_path);
//     if (!q_file.is_open()) {
//       fmt::print("Warning: Quaternion file not found: {}\n", q_path);
//       continue;
//     }
    
//     Eigen::Quaterniond q = read_q(q_path);
//     fmt::print("Quaternion: w={}, x={}, y={}, z={}\n", q.w(), q.x(), q.y(), q.z());

//     // 计算云台的欧拉角
//     Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
//     Eigen::Matrix3d R_gimbal2world =
//       R_gimbal2imubody.transpose() * R_imubody2imuabs * R_gimbal2imubody;
//     Eigen::Vector3d ypr = tools::eulers(R_gimbal2world, 2, 1, 0) * 57.3;  // degree
//     fmt::print("Gimbal Euler angles: yaw={:.2f}, pitch={:.2f}, roll={:.2f}\n", ypr[0], ypr[1], ypr[2]);

//     // 在图片上显示云台的欧拉角
//     auto drawing = img.clone();
//     tools::draw_text(drawing, fmt::format("yaw   {:.2f}", ypr[0]), {40, 40}, {0, 0, 255});
//     tools::draw_text(drawing, fmt::format("pitch {:.2f}", ypr[1]), {40, 80}, {0, 0, 255});
//     tools::draw_text(drawing, fmt::format("roll  {:.2f}", ypr[2]), {40, 120}, {0, 0, 255});

//     // 识别棋盘格
//     std::vector<cv::Point2f> corners_2d;
//     cv::Mat gray;
//     cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    
//     // 检测棋盘格角点
//     fmt::print("Detecting chessboard corners...\n");
//     bool success = cv::findChessboardCorners(
//         gray, pattern_size, corners_2d,
//         cv::CALIB_CB_ADAPTIVE_THRESH | 
//         cv::CALIB_CB_NORMALIZE_IMAGE |
//         cv::CALIB_CB_FAST_CHECK);
    
//     fmt::print("Chessboard detection: {}\n", success ? "success" : "failure");
    
//     if (success) {
//         // 亚像素优化
//         fmt::print("Performing sub-pixel optimization...\n");
//         cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
//         cv::cornerSubPix(gray, corners_2d, cv::Size(11, 11), cv::Size(-1, -1), criteria);
        
//         // 绘制角点连线
//         cv::drawChessboardCorners(drawing, pattern_size, corners_2d, success);
        
//         // 获取四个角点：左上、右上、右下、左下
//         cv::Point2f topLeft = corners_2d[0];
//         cv::Point2f topRight = corners_2d[pattern_size.width - 1];
//         cv::Point2f bottomRight = corners_2d.back();
//         cv::Point2f bottomLeft = corners_2d[corners_2d.size() - pattern_size.width];
        
//         // 绘制绿色边框
//         std::vector<cv::Point> borderPoints = {
//             topLeft, topRight, bottomRight, bottomLeft
//         };
//         cv::polylines(drawing, borderPoints, true, borderColor, borderThickness);
        
//         // 在边框上显示尺寸信息
//         std::string sizeInfo = fmt::format("{}x{} {}mm", 
//             pattern_size.width + 1, pattern_size.height + 1, 
//             static_cast<int>(square_size));
//         cv::putText(drawing, sizeInfo, 
//                     topLeft + cv::Point2f(0, -10), 
//                     cv::FONT_HERSHEY_SIMPLEX, 0.5, borderColor, 1);
//     }

//     // 显示识别结果
//     cv::resize(drawing, drawing, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
//     fmt::print("Showing image...\n");
//     cv::imshow("Press any to continue", drawing);
//     fmt::print("Waiting for key press...\n");
//     int key = cv::waitKey(0);
//     fmt::print("Key pressed: {}\n", key);
//     if (key == 'q') {
//       fmt::print("User requested exit.\n");
//       break;  // 按q键提前退出
//     }

//     if (!success) {
//       fmt::print("Skipping image due to failed detection\n");
//       continue;
//     }

//     // 计算所需的数据
//     fmt::print("Solving PnP...\n");
//     cv::Mat t_gimbal2world = (cv::Mat_<double>(3, 1) << 0, 0, 0);
//     cv::Mat R_gimbal2world_cv;
//     cv::eigen2cv(R_gimbal2world, R_gimbal2world_cv);
//     cv::Mat rvec, tvec;
//     auto corners_3d = chessboard_corners_3d(pattern_size, square_size);
//     cv::solvePnP(
//       corners_3d, corners_2d, camera_matrix, distort_coeffs, rvec, tvec, false, cv::SOLVEPNP_IPPE);

//     // 记录所需的数据
//     R_gimbal2world_list.push_back(R_gimbal2world_cv);
//     t_gimbal2world_list.push_back(t_gimbal2world);
//     rvecs.push_back(rvec);
//     tvecs.push_back(tvec);
    
//     fmt::print("Data recorded for image {}\n", i);
//   }
// }

// void print_yaml(
//   const std::vector<double> & R_gimbal2imubody_data, const cv::Mat & R_camera2gimbal,
//   const cv::Mat & t_camera2gimbal, const Eigen::Vector3d & ypr)
// {
//   YAML::Emitter result;
//   std::vector<double> R_camera2gimbal_data(
//     R_camera2gimbal.begin<double>(), R_camera2gimbal.end<double>());
//   std::vector<double> t_camera2gimbal_data(
//     t_camera2gimbal.begin<double>(), t_camera2gimbal.end<double>());

//   result << YAML::BeginMap;
//   result << YAML::Key << "R_gimbal2imubody";
//   result << YAML::Value << YAML::Flow << R_gimbal2imubody_data;
//   result << YAML::Newline;
//   result << YAML::Newline;
//   result << YAML::Comment(fmt::format(
//     "相机同理想情况的偏角: yaw{:.2f} pitch{:.2f} roll{:.2f} degree", ypr[0], ypr[1], ypr[2]));
//   result << YAML::Key << "R_camera2gimbal";
//   result << YAML::Value << YAML::Flow << R_camera2gimbal_data;
//   result << YAML::Key << "t_camera2gimbal";
//   result << YAML::Value << YAML::Flow << t_camera2gimbal_data;
//   result << YAML::Newline;
//   result << YAML::EndMap;

//   fmt::print("\n{}\n", result.c_str());
// }

// int main(int argc, char * argv[])
// {
//   try {
//     // 读取命令行参数
//     cv::CommandLineParser cli(argc, argv, keys);
//     if (cli.has("help")) {
//       cli.printMessage();
//       return 0;
//     }
//     auto input_folder = cli.get<std::string>(0);
//     auto config_path = cli.get<std::string>("config-path");

//     fmt::print("Loading configuration from: {}\n", config_path);
//     fmt::print("Input folder: {}\n", input_folder);

//     // 从输入文件夹中加载标定所需的数据
//     std::vector<double> R_gimbal2imubody_data;
//     std::vector<cv::Mat> R_gimbal2world_list, t_gimbal2world_list;
//     std::vector<cv::Mat> rvecs, tvecs;
//     load(
//       input_folder, config_path, R_gimbal2imubody_data, R_gimbal2world_list, t_gimbal2world_list,
//       rvecs, tvecs);

//     // 检查是否有足够的数据进行标定
//     if (R_gimbal2world_list.size() < 5) {
//       fmt::print(stderr, "Error: Not enough valid samples (minimum 5 required, got {})\n", 
//                 R_gimbal2world_list.size());
//       return 1;
//     }

//     fmt::print("Performing hand-eye calibration with {} samples...\n", R_gimbal2world_list.size());

//     // 手眼标定
//     cv::Mat R_camera2gimbal, t_camera2gimbal;
//     cv::calibrateHandEye(
//       R_gimbal2world_list, t_gimbal2world_list, rvecs, tvecs, R_camera2gimbal, t_camera2gimbal,
//       cv::CALIB_HAND_EYE_TSAI);
//     t_camera2gimbal /= 1e3;  // mm to m

//     // 计算相机同理想情况的偏角
//     Eigen::Matrix3d R_camera2gimbal_eigen;
//     cv::cv2eigen(R_camera2gimbal, R_camera2gimbal_eigen);
//     Eigen::Matrix3d R_gimbal2ideal{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}};
//     Eigen::Matrix3d R_camera2ideal = R_gimbal2ideal * R_camera2gimbal_eigen;
//     Eigen::Vector3d ypr = tools::eulers(R_camera2ideal, 1, 0, 2) * 57.3;  // degree

//     // 输出yaml
//     print_yaml(R_gimbal2imubody_data, R_camera2gimbal, t_camera2gimbal, ypr);
    
//     return 0;
//   } catch (const std::exception& e) {
//     fmt::print(stderr, "Error: {}\n", e.what());
//     return 1;
//   }
// }

// 

#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "tools/img_tools.hpp"
#include "tools/math_tools.hpp"

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{@input-folder  | assets/img_with_q        | 输入文件夹路径   }";

// 生成棋盘格角点的3D坐标
std::vector<cv::Point3f> chessboard_corners_3d(const cv::Size & pattern_size, float square_size)
{
  std::vector<cv::Point3f> corners;
  for (int i = 0; i < pattern_size.height; i++)
    for (int j = 0; j < pattern_size.width; j++)
      corners.emplace_back(j * square_size, i * square_size, 0);
  return corners;
}

Eigen::Quaterniond read_q(const std::string & q_path)
{
  std::ifstream q_file(q_path);
  double w, x, y, z;
  q_file >> w >> x >> y >> z;
  return Eigen::Quaterniond(w, x, y, z);
}

void parse_camera_params(const YAML::Node& yaml,
                         cv::Matx33d & camera_matrix, cv::Mat & distort_coeffs)
{
  try {
    // 直接从当前配置文件读取相机参数
    auto cm = yaml["camera_matrix"].as<std::vector<double>>();
    auto dc = yaml["distort_coeffs"].as<std::vector<double>>();

    // 验证数据
    if (cm.size() != 9) {
      throw std::runtime_error("相机内参矩阵必须包含9个元素");
    }
    if (dc.size() != 5) {
      throw std::runtime_error(fmt::format("畸变系数必须包含5个元素，实际有{}个", dc.size()));
    }

    camera_matrix = cv::Matx33d(cm.data());
    distort_coeffs = cv::Mat(dc).reshape(1, 1);
    
    // 详细输出用于调试
    fmt::print("=== 从配置文件加载相机参数 ===\n");
    fmt::print("内参矩阵:\n");
    for (int i = 0; i < 3; i++) {
        fmt::print("[ {:.5f}, {:.5f}, {:.5f} ]\n", 
                   camera_matrix(i, 0), camera_matrix(i, 1), camera_matrix(i, 2));
    }
    fmt::print("畸变系数: [ ");
    for (size_t i = 0; i < dc.size(); i++) {
        fmt::print("{:.5f} ", dc[i]);
    }
    fmt::print("]\n");
    fmt::print("棋盘格尺寸: {}x{}\n", yaml["pattern_cols"].as<int>()-1, yaml["pattern_rows"].as<int>()-1);
    fmt::print("方格尺寸: {:.2f} mm\n", yaml["square_size_mm"].as<double>());
    fmt::print("===================================\n");
  } catch (const YAML::Exception& e) {
    fmt::print("❌ 解析相机参数失败: {}\n", e.what());
    throw;
  } catch (const std::exception& e) {
    fmt::print("❌ 加载相机参数失败: {}\n", e.what());
    throw;
  }
}

// 检查矩阵是否包含NaN或Inf
bool contains_invalid(const cv::Mat& m) {
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            double val = m.at<double>(i, j);
            if (std::isnan(val) || std::isinf(val)) {
                return true;
            }
        }
    }
    return false;
}
void verify_coordinate_system(const cv::Mat& tvec, const Eigen::Vector3d& gimbal_ypr, int image_index) {
    fmt::print("\n=== 坐标系验证 (图像 {}) ===\n", image_index);
    
    // 1. 检查OpenCV检测结果
    fmt::print("OpenCV检测结果:\n");
    fmt::print("  tvec: [{:.2f}, {:.2f}, {:.2f}] mm\n", 
               tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    
    // 2. 检查云台姿态
    fmt::print("云台姿态 (FLU坐标系):\n");
    fmt::print("  yaw={:.2f}°, pitch={:.2f}°, roll={:.2f}°\n", 
               gimbal_ypr[0], gimbal_ypr[1], gimbal_ypr[2]);
    
    // 3. 物理一致性检查
    bool physical_consistency = true;
    std::vector<std::string> inconsistencies;
    
    // 检查标定板位置
    if (tvec.at<double>(2) <= 0) {
        physical_consistency = false;
        inconsistencies.push_back("标定板在相机后方 (tvec.z <= 0)");
    }
    
    // 检查云台姿态与标定板位置的物理关系
    // 如果云台朝前且标定板在相机前方，应该是合理的情况
    if (std::abs(gimbal_ypr[0]) > 90 || std::abs(gimbal_ypr[1]) > 90) {
        physical_consistency = false;
        inconsistencies.push_back("云台姿态异常 (角度过大)");
    }
    
    // 检查重力方向验证
    // 这里可以添加基于四元数的重力方向验证
    
    // 输出验证结果
    if (physical_consistency) {
        fmt::print("✅ 物理姿态一致\n");
    } else {
        fmt::print("❌ 物理姿态不一致！可能存在坐标系混淆\n");
        for (const auto& issue : inconsistencies) {
            fmt::print("   - {}\n", issue);
        }
    }
}
void load(const std::string & input_folder, const std::string & config_path,
          std::vector<double> & R_gimbal2imubody_data,
          std::vector<cv::Mat> & R_gimbal2world_list,
          std::vector<cv::Mat> & t_gimbal2world_list,
          std::vector<cv::Mat> & rvecs, std::vector<cv::Mat> & tvecs)
{
  YAML::Node yaml = YAML::LoadFile(config_path);
  auto pattern_cols = yaml["pattern_cols"].as<int>();
  auto pattern_rows = yaml["pattern_rows"].as<int>();
  auto square_size_mm = yaml["square_size_mm"].as<double>();
  R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();

  cv::Matx33d camera_matrix;
  cv::Mat distort_coeffs;
  
  // 直接从当前配置文件读取相机参数，不再从外部文件读取
  parse_camera_params(yaml, camera_matrix, distort_coeffs);

  const cv::Size pattern_size(pattern_cols - 1, pattern_rows - 1);
  const float square_size = square_size_mm;

  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_gimbal2imubody(R_gimbal2imubody_data.data());

  int success_count = 0;
  int total_count = 0;
  int invalid_pose = 0;

  fmt::print("开始加载数据...\n");

  for (int i = 1;; i++) {
    auto img_path = fmt::format("{}/{}.jpg", input_folder, i);
    auto q_path = fmt::format("{}/{}.txt", input_folder, i);

    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        // 尝试不同命名格式
        img_path = fmt::format("{}/{:02d}.jpg", input_folder, i);
        img = cv::imread(img_path);
        if (img.empty()) {
            fmt::print("无法读取图像: {}, 停止加载\n", img_path);
            break;
        }
    }

    total_count++;
    Eigen::Quaterniond q = read_q(q_path);
    Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
    Eigen::Matrix3d R_gimbal2world =
      R_gimbal2imubody.transpose() * R_imubody2imuabs * R_gimbal2imubody;

    Eigen::Vector3d ypr = tools::eulers(R_gimbal2world, 2, 1, 0) * 57.3;
    if (std::isnan(ypr[0]) || std::isnan(ypr[1])) {
      fmt::print("⚠️ 第 {} 张：IMU 姿态异常（含 NaN）\n", i);
      invalid_pose++;
      continue;
    }

    fmt::print("第 {:02d} 张 -> yaw:{:6.2f}°, pitch:{:6.2f}°, roll:{:6.2f}°\n",
               i, ypr[0], ypr[1], ypr[2]);

    std::vector<cv::Point2f> corners_2d;
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    
    // 图像预处理
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.5);
    cv::equalizeHist(gray, gray);

    bool found = cv::findChessboardCorners(
      gray, pattern_size, corners_2d,
      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);

    if (!found) {
      fmt::print("    ❌ 棋盘格未检测到\n");
      continue;
    }

    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
    cv::cornerSubPix(gray, corners_2d, cv::Size(11, 11), cv::Size(-1, -1), criteria);

    // 可视化检测结果
    cv::Mat img_display = img.clone();
    cv::drawChessboardCorners(img_display, pattern_size, corners_2d, found);
    cv::imshow("Chessboard Corners", img_display);
    cv::waitKey(100);  // 短暂显示

    std::vector<cv::Point3f> corners_3d = chessboard_corners_3d(pattern_size, square_size);

    cv::Mat rvec, tvec;
    bool pnp_ok = cv::solvePnP(
      corners_3d, corners_2d, camera_matrix, distort_coeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

    // 检查结果是否有效
    if (!pnp_ok || contains_invalid(rvec) || contains_invalid(tvec)) {
      fmt::print("    ⚠️ solvePnP 结果异常\n");
      if (pnp_ok) {
        fmt::print("        rvec: [{:.2f}, {:.2f}, {:.2f}]\n", 
                   rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
        fmt::print("        tvec: [{:.2f}, {:.2f}, {:.2f}]\n", 
                   tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
      }
      continue;
    }

    fmt::print("    ✅ 棋盘格检测成功 | tvec = [{:.2f}, {:.2f}, {:.2f}] mm\n",
               tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    verify_coordinate_system(tvec, ypr, i);
    cv::Mat R_cv;
    cv::eigen2cv(R_gimbal2world, R_cv);
    cv::Mat t_gimbal2world = (cv::Mat_<double>(3, 1) << 0, 0, 0);

    R_gimbal2world_list.push_back(R_cv);
    t_gimbal2world_list.push_back(t_gimbal2world);
    rvecs.push_back(rvec);
    tvecs.push_back(tvec);
    success_count++;
  }

  cv::destroyAllWindows();  // 关闭所有OpenCV窗口

  fmt::print("\n========== 数据加载完成 ==========\n");
  fmt::print("总图像数：{}\n", total_count);
  fmt::print("棋盘格识别成功：{}\n", success_count);
  fmt::print("IMU 姿态异常：{}\n", invalid_pose);
  fmt::print("有效样本（用于标定）：{}\n\n", success_count);

  if (success_count < 5)
    fmt::print("⚠️ 样本太少，建议至少拍摄 8-10 张不同角度图像\n");
}

void print_yaml(const std::vector<double> & R_gimbal2imubody_data,
                const cv::Mat & R_camera2gimbal, const cv::Mat & t_camera2gimbal,
                const Eigen::Vector3d & ypr)
{
  YAML::Emitter out;
  std::vector<double> R_data(R_camera2gimbal.begin<double>(), R_camera2gimbal.end<double>());
  std::vector<double> t_data(t_camera2gimbal.begin<double>(), t_camera2gimbal.end<double>());

  out << YAML::BeginMap;
  out << YAML::Key << "R_gimbal2imubody" << YAML::Value << YAML::Flow << R_gimbal2imubody_data;
  out << YAML::Comment(fmt::format(" 相机偏角: yaw {:.2f}°, pitch {:.2f}°, roll {:.2f}° ", ypr[0], ypr[1], ypr[2]));
  out << YAML::Key << "R_camera2gimbal" << YAML::Value << YAML::Flow << R_data;
  out << YAML::Key << "t_camera2gimbal" << YAML::Value << YAML::Flow << t_data;
  out << YAML::EndMap;

  fmt::print("\n============== 标定结果 ==============\n{}\n", out.c_str());
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto input_folder = cli.get<std::string>(0);
  auto config_path = cli.get<std::string>("config-path");

  std::vector<double> R_gimbal2imubody_data;
  std::vector<cv::Mat> R_gimbal2world_list, t_gimbal2world_list;
  std::vector<cv::Mat> rvecs, tvecs;

  load(input_folder, config_path, R_gimbal2imubody_data,
       R_gimbal2world_list, t_gimbal2world_list, rvecs, tvecs);

  if (R_gimbal2world_list.size() < 5) {
    fmt::print("❌ 有效数据不足，无法进行手眼标定。\n");
    return 0;
  }

  // 检查tvec变化情况
  double tvec_change = 0;
  for (size_t i = 1; i < tvecs.size(); i++) {
    double diff = cv::norm(tvecs[i] - tvecs[i-1]);
    tvec_change += diff;
  }
  double avg_tvec_change = tvec_change / (tvecs.size() - 1);
  fmt::print("平均tvec变化: {:.2f} mm\n", avg_tvec_change);
  
  if (avg_tvec_change < 10) {
    fmt::print("⚠️ 平移变化过小，使用纯旋转标定方法\n");
  }

  cv::Mat R_camera2gimbal, t_camera2gimbal;
  
  // 使用CALIB_HAND_EYE_PARK方法 - 针对纯旋转场景优化
  cv::calibrateHandEye(R_gimbal2world_list, t_gimbal2world_list,
                       rvecs, tvecs, R_camera2gimbal, t_camera2gimbal,
                       cv::CALIB_HAND_EYE_HORAUD);
  
  t_camera2gimbal /= 1e3;  // 毫米转米

  // 计算标定结果的欧拉角
  Eigen::Matrix3d R_cam2gimbal_eigen;
  cv::cv2eigen(R_camera2gimbal, R_cam2gimbal_eigen);
  
  // 理想坐标系定义：X轴向前，Y轴向左，Z轴向上
  Eigen::Matrix3d R_gimbal2ideal{{0, -1, 0}, {0, 0, -1}, {1, 0, 0}};
  Eigen::Matrix3d R_camera2ideal = R_gimbal2ideal * R_cam2gimbal_eigen;
  
  // 计算欧拉角（YPR顺序）
  Eigen::Vector3d ypr = tools::eulers(R_camera2ideal, 1, 0, 2) * 57.3;

  print_yaml(R_gimbal2imubody_data, R_camera2gimbal, t_camera2gimbal, ypr);
  
  // 添加标定质量评估
  fmt::print("\n========== 标定质量评估 ==========\n");
  fmt::print("旋转矩阵行列式: {:.6f} (应接近1)\n", cv::determinant(R_camera2gimbal));
  return 0;
}