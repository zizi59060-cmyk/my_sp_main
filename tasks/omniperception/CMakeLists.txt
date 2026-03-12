cmake_minimum_required(VERSION 3.16)

# set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")

# find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# add_library(omniperception OBJECT 
#     decider.cpp
#     perceptron.cpp
# )
# target_link_libraries(omniperception openvino::runtime)
# 修改后 (替换为这些行)
# target_link_libraries(omniperception PUBLIC
#     ${TensorRT_LIBRARIES}
#     ${CUDA_LIBRARIES}
# )