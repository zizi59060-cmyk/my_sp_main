find_path(TensorRT_INCLUDE_DIR NvInfer.h
  HINTS $ENV{TensorRT_ROOT} ${TensorRT_ROOT} /usr /usr/local/TensorRT /usr/local/cuda
  PATH_SUFFIXES include targets/x86_64-linux/include targets/aarch64-linux/include)

find_library(TensorRT_NVINFER_LIBRARY nvinfer
  HINTS $ENV{TensorRT_ROOT} ${TensorRT_ROOT} /usr /usr/local/TensorRT /usr/local/cuda
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu targets/x86_64-linux/lib targets/aarch64-linux/lib)
find_library(TensorRT_NVONNXPARSER_LIBRARY nvonnxparser
  HINTS $ENV{TensorRT_ROOT} ${TensorRT_ROOT} /usr /usr/local/TensorRT /usr/local/cuda
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu targets/x86_64-linux/lib targets/aarch64-linux/lib)
find_library(TensorRT_NVINFER_PLUGIN_LIBRARY nvinfer_plugin
  HINTS $ENV{TensorRT_ROOT} ${TensorRT_ROOT} /usr /usr/local/TensorRT /usr/local/cuda
  PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu lib/aarch64-linux-gnu targets/x86_64-linux/lib targets/aarch64-linux/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT DEFAULT_MSG
  TensorRT_INCLUDE_DIR TensorRT_NVINFER_LIBRARY TensorRT_NVONNXPARSER_LIBRARY)

if(TensorRT_FOUND)
  set(TensorRT_LIBRARIES ${TensorRT_NVINFER_LIBRARY} ${TensorRT_NVONNXPARSER_LIBRARY})
  if(TensorRT_NVINFER_PLUGIN_LIBRARY)
    list(APPEND TensorRT_LIBRARIES ${TensorRT_NVINFER_PLUGIN_LIBRARY})
  endif()
  set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
endif()
