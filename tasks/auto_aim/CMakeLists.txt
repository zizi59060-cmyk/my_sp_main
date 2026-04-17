cmake_minimum_required(VERSION 3.16)

add_subdirectory(planner/tinympc)

enable_language(CUDA)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
set_source_files_properties(tasks/auto_aim/multithread/mt_detector.cu PROPERTIES LANGUAGE CUDA)
add_library(auto_aim OBJECT 
    armor.cpp
    classifier.cpp 
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolov5.cu
    # yolos/yolov8.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
)

# TensorRT 库路径
set(TENSORRT_LIB_DIR "/usr/lib/aarch64-linux-gnu")
set(TENSORRT_INCLUDE_DIR "/usr/include/aarch64-linux-gnu")

# 包含路径
include_directories(${TENSORRT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})

# 链接 TensorRT + CUDA 库
target_link_libraries(auto_aim PUBLIC 
    io 
    nvinfer
    cudart
    opencv_core
    opencv_cudaimgproc
    opencv_cudawarping
    tinympcstatic
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIB_DIR}/libnvinfer.so
    ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so
    ${TENSORRT_LIB_DIR}/libnvonnxparser.so
)
