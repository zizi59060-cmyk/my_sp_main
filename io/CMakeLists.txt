cmake_minimum_required(VERSION 3.16)

find_package(yaml-cpp REQUIRED)

add_subdirectory(serial)

# 确保头文件路径被正确包含
include_directories(daheng/include)

################## 检查 ROS 环境 ##################
find_package(ament_cmake QUIET)
find_package(rclcpp QUIET)
find_package(std_msgs QUIET)
find_package(rosidl_typesupport_cpp QUIET)
find_package(sp_msgs QUIET)

# 设置 ROS 库变量，如果找到的话
set(ROS_LIBS "")
if(ament_cmake_FOUND AND rclcpp_FOUND AND std_msgs_FOUND AND rosidl_typesupport_cpp_FOUND AND sp_msgs_FOUND)
  message(STATUS "ROS2 environment found, preparing ROS2 libraries for linking.")
    # 设置要链接的 ROS 库
  set(ROS_LIBS rclcpp std_msgs sp_msgs)
else()
  message(WARNING "ROS2 not found, skipping ROS2 specific code.")
endif()
##################################################

# 创建目标 io
add_library(io STATIC 
  hikrobot/hikrobot.cpp
  mindvision/mindvision.cpp 
  daheng/daheng.cpp
  usbcamera/usbcamera.cpp 
  camera.cpp
  cboard.cpp
  dm_imu/dm_imu.cpp
  gimbal/gimbal.cpp
)

# 如果找到 ROS，则添加 ROS 相关的源文件
if(ROS_LIBS)
  target_sources(io PRIVATE ros2/publish2nav.cpp ros2/ros2.cpp ros2/subscribe2nav.cpp)
    # 确保 ROS 的头文件也被包含
  ament_target_dependencies(io rclcpp std_msgs sp_msgs)
endif()

# hikrobot
target_include_directories(io PUBLIC hikrobot/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC hikrobot/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}! Only 'aarch64' is supported by this configuration for linking HikRobot.")
endif()

# mindvision
target_include_directories(io PUBLIC mindvision/include)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC mindvision/lib/arm64)
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}! Only 'aarch64' is supported by this configuration for linking MindVision.")
endif()

# daheng
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  target_link_directories(io PUBLIC daheng/lib/armv8) # 使用您提供的 armv8 目录
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}! Only 'aarch64' is supported by this configuration for linking DaHeng.")
endif()


# 【关键修复点：将所有非 ROS 依赖项合并到一个 PUBLIC 链接命令中】
# 这会解决 "undefined reference" 错误
target_link_libraries(io PUBLIC
    MvCameraControl 
    MVSDK 
    gxiapi 
    log4cplus_gx 
    usb-1.0 
    yaml-cpp 
    serial
)

# 【关键修复点：将 ROS 依赖项保留在 ament_target_dependencies 中】
# 我们将非 ROS 库设为 PUBLIC，将 ROS 库设为 plain (通过 ament 宏)
# 这在现代 CMake 中是允许的，只要 PUBLIC/PRIVATE 调用在 "plain" 调用之前。
if(ROS_LIBS)
    # 注意：ament_target_dependencies 必须在 target_link_libraries 之后
    # 以确保 "keyword" 签名在 "plain" 签名之前被评估
    ament_target_dependencies(io rclcpp std_msgs sp_msgs)
endif()