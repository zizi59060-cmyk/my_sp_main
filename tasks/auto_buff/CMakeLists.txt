cmake_minimum_required(VERSION 3.16)

find_package(Ceres REQUIRED)
find_package(GTSAM REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)

add_library(auto_buff OBJECT
    buff_type.cpp
    jlu_buff/types.cpp
    jlu_buff/factors.cpp
    jlu_buff/buff_fitter.cpp
    jlu_buff/targets.cpp
    jlu_buff/trajectory.cpp
    jlu_buff/jlu_buff.cpp
    jlu_buff/infer/trt_engine.cpp
    jlu_buff/infer/trt_yolo_buff.cpp
)

target_compile_features(auto_buff PUBLIC cxx_std_20)
target_include_directories(auto_buff PUBLIC
    ${PROJECT_SOURCE_DIR}
    ${TensorRT_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(auto_buff PUBLIC
    auto_aim
    ${CERES_LIBRARIES}
    gtsam
    ${TensorRT_LIBRARIES}
    CUDA::cudart
)
