# my_sp_main / jiu_sp_main

## 1. 项目说明

本分支 `jiu_sp_main` 在 `my_sp_main` 的主程序、相机、云台、串口、自瞄 `auto_aim`、配置和日志风格基础上，新增并接入一套 JLU 风格的能量机关打符后端：

- 打符入口仍在 `standard_mpc` 主循环中，由下位机 `mode` 选择 `AUTO_AIM`、`SMALL_BUFF`、`BIG_BUFF`。
- 自瞄分支保持原 `auto_aim::YOLO -> Tracker -> Planner -> gimbal.send` 流程。
- 小符/大符分支进入 `tasks/auto_buff/jlu_buff`。
- 检测推理不再使用 JLU 原 OpenVINO 链路，而是使用 TensorRT 10.3 C++ API + CUDA Runtime。
- 打符跟踪后端包含 JLU 五点关键点、PnP、GTSAM ISAM2 风格因子图、BigBuff Ceres 曲线拟合和弹道预测。
- 参考项目：`https://github.com/Fskaaaaaaaa/jlu_vision_26.git`。当前仓库没有引入 iceoryx、fast_tf、quill、rfl、ConfigManager 或 JLU 节点系统。

## 2. 分支说明

```bash
git fetch origin
git checkout main
git pull origin main
git checkout -b jiu_sp_main
git branch --show-current
```

本次修改应只提交在 `jiu_sp_main`。`main` 不应直接修改或提交。

## 3. 系统环境

推荐环境：

- Ubuntu 22.04 LTS
- NVIDIA Driver（与 CUDA/TensorRT 匹配）
- CUDA Toolkit（建议 12.x，按 TensorRT 10.3 包要求选择）
- cuDNN（如果导出的 ONNX/插件需要）
- TensorRT 10.3（必须，含 `libnvinfer.so` 与 `libnvonnxparser.so`）
- OpenCV 4（含 core/imgproc/highgui/dnn/cuda 模块）
- Ceres Solver
- GTSAM
- yaml-cpp
- Eigen3
- fmt / spdlog / nlohmann-json
- OpenVINO：新 JLU 打符链路不依赖 OpenVINO；旧文件如需单独启用才需要 OpenVINO。

## 4. 从零配置环境命令

### 4.1 基础工具

```bash
sudo apt update
sudo apt install -y build-essential cmake git pkg-config curl wget unzip tar \
  libopencv-dev libeigen3-dev libyaml-cpp-dev libfmt-dev libspdlog-dev \
  nlohmann-json3-dev libgoogle-glog-dev libgflags-dev libatlas-base-dev \
  libsuitesparse-dev libboost-all-dev
```

### 4.2 检查 NVIDIA 驱动和 CUDA

```bash
nvidia-smi
nvcc --version || true
ls /usr/local/cuda/include/cuda_runtime_api.h
```

如果没有 CUDA，请按 NVIDIA 官方 Ubuntu 22.04 CUDA repo 安装；示例（请按实际 CUDA 版本替换 pin/deb URL）：

```bash
cd /tmp
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-5
nvcc --version
```

### 4.3 安装 TensorRT 10.3

方式 A：deb local repo（推荐，需从 NVIDIA TensorRT 10.3 下载页手动下载 Ubuntu 22.04 x86_64 或 aarch64 local repo deb）。

```bash
mkdir -p ~/Downloads/tensorrt-10.3
# 将 NVIDIA 官网下载的 nv-tensorrt-local-repo-ubuntu2204-10.3.*.deb 放入该目录
cd ~/Downloads/tensorrt-10.3
sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-10.3.*.deb
sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.3.*/*keyring.gpg /usr/share/keyrings/
sudo apt update
sudo apt install -y tensorrt libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev
```

方式 B：tar 包安装（需从 NVIDIA TensorRT 10.3 下载页手动下载 Linux tar 包）。

```bash
sudo mkdir -p /opt/tensorrt
cd /opt/tensorrt
sudo tar -xf ~/Downloads/TensorRT-10.3.*.Linux.*.tar.gz
export TensorRT_ROOT=/opt/tensorrt/TensorRT-10.3.*
export LD_LIBRARY_PATH=${TensorRT_ROOT}/lib:${LD_LIBRARY_PATH}
echo 'export TensorRT_ROOT=/opt/tensorrt/TensorRT-10.3.*' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=${TensorRT_ROOT}/lib:${LD_LIBRARY_PATH}' >> ~/.bashrc
```

验证：

```bash
ldconfig -p | grep libnvinfer
ldconfig -p | grep nvonnxparser
python3 - <<'PY'
import ctypes
for lib in ['libnvinfer.so', 'libnvonnxparser.so']:
    ctypes.CDLL(lib)
    print('OK', lib)
PY
```

### 4.4 安装 Ceres

Ubuntu 包方式：

```bash
sudo apt install -y libceres-dev
```

源码方式（包版本不满足时）：

```bash
cd /tmp
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
make -j$(nproc)
sudo make install
sudo ldconfig
```

### 4.5 安装 GTSAM

```bash
cd /tmp
git clone https://github.com/borglab/gtsam.git
cd gtsam
git checkout 4.2.0 || true
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGTSAM_BUILD_TESTS=OFF -DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF
make -j$(nproc)
sudo make install
sudo ldconfig
```

### 4.6 获取、切换、编译项目

```bash
git clone https://github.com/zizi59060-cmyk/my_sp_main.git
cd my_sp_main
git checkout jiu_sp_main || git checkout -b jiu_sp_main origin/jiu_sp_main
git submodule update --init --recursive
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## 5. TensorRT 10.3 配置

确认版本和库：

```bash
grep NV_TENSORRT_MAJOR /usr/include/NvInferVersion.h || grep NV_TENSORRT_MAJOR ${TensorRT_ROOT}/include/NvInferVersion.h
ldconfig -p | grep libnvinfer.so
ldconfig -p | grep libnvonnxparser.so
```

如果使用 tar 包：

```bash
export TensorRT_ROOT=/opt/tensorrt/TensorRT-10.3.x.x
export LD_LIBRARY_PATH=${TensorRT_ROOT}/lib:${LD_LIBRARY_PATH}
cmake .. -DTensorRT_ROOT=${TensorRT_ROOT}
```

ONNX 转 engine 有两种方式：

```bash
# 方式 1：程序首次运行自动 build engine（推荐）
mkdir -p models/buff
cp /path/to/buff.onnx models/buff/buff.onnx
./build/standard_mpc configs/demo.yaml

# 方式 2：trtexec 手工生成
/usr/src/tensorrt/bin/trtexec \
  --onnx=models/buff/buff.onnx \
  --saveEngine=models/buff/buff_trt10_3.engine \
  --fp16 \
  --explicitBatch
```

FP16 开关在配置中修改：

```yaml
jlu_buff:
  detector:
    fp16: true   # 关闭时改为 false
```

## 6. 模型文件准备

默认配置：

```yaml
jlu_buff:
  detector:
    onnx_path: "models/buff/buff.onnx"
    engine_path: "models/buff/buff_trt10_3.engine"
```

程序优先加载 `engine_path`；不存在时从 `onnx_path` 自动构建并保存 engine。两者都不存在时，日志会明确报错，检测返回空结果，不会误开火。

当前后处理假定输出候选格式为 bbox + confidence + 5 个关键点，关键点顺序为：

1. `r_center`
2. `bottom_right`
3. `top_right`
4. `top_left`
5. `bottom_left`

如果你的 ONNX 输出不是该格式，需要修改 `tasks/auto_buff/jlu_buff/infer/trt_yolo_buff.cpp` 中的 adapter。

## 7. 编译

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

如果 TensorRT 是 tar 包安装：

```bash
mkdir -p build
cd build
cmake .. -DTensorRT_ROOT=${TensorRT_ROOT}
make -j$(nproc)
```

## 8. 运行

```bash
./build/standard_mpc configs/demo.yaml
```

模式由下位机串口包 `GimbalToVision.mode` 控制：

- `0`：IDLE
- `1`：AUTO_AIM（原自瞄）
- `2`：SMALL_BUFF（JLU 小符）
- `3`：BIG_BUFF（JLU 大符）

配置文件路径默认是 `configs/demo.yaml`。实车时需要确认相机标定、`R_camera2gimbal`、`t_camera2gimbal`、`R_gimbal2imubody`、CAN/串口设备名和模型路径。

## 9. 调试 jlu_vision_26 方法

```bash
cd /tmp
git clone https://github.com/Fskaaaaaaaa/jlu_vision_26.git
cd jlu_vision_26
find src/auto_buff -maxdepth 3 -type f | sort
sed -n '1,220p' configs/auto_buff/buff_detector.yaml
sed -n '1,260p' configs/auto_buff/buff_tracker.yaml
```

阅读顺序建议：

1. `src/auto_buff/buff_detector`：确认模型输入、输出和关键点语义。
2. `src/auto_buff/buff_tracker/include/types.hpp` 与 `src/.../types.cpp`：确认 `BuffBlade`、状态枚举、五点顺序。
3. `factors.hpp/cpp`：理解位置、roll、vroll、reprojection、blade 因子。
4. `targets.hpp/cpp`：理解 `SmallBuffTarget` 的 ISAM2 更新和丢失状态。
5. `buff_fitter.hpp/cpp`：理解大符 Ceres 变速曲线拟合。
6. `trajectory.hpp/cpp`：理解选叶、弹速检查、飞行时间迭代、yaw/pitch 输出。
7. 回到本仓库对照 `tasks/auto_buff/jlu_buff`，确认没有引入 iceoryx/fast_tf/quill/rfl/ConfigManager。

调试检查项：

```bash
# 确认点顺序注释和 adapter
sed -n '1,220p' tasks/auto_buff/jlu_buff/types.hpp
sed -n '1,260p' tasks/auto_buff/jlu_buff/infer/trt_yolo_buff.cpp

# 确认 PnP 使用五点
sed -n '1,260p' tasks/auto_buff/jlu_buff/jlu_buff.cpp

# 查看 GTSAM/Ceres/Trajectory 移植代码
sed -n '1,260p' tasks/auto_buff/jlu_buff/targets.cpp
sed -n '1,260p' tasks/auto_buff/jlu_buff/buff_fitter.cpp
sed -n '1,220p' tasks/auto_buff/jlu_buff/trajectory.cpp
```

日志中重点看：TensorRT engine 加载/构建、infer 耗时、检测数量、五点坐标、PnP 是否成功、`TrackState`、SmallBuff roll/vroll、BigBuff fitter params、bullet speed、predict offset、yaw/pitch/fire。

## 10. 常见问题

- 找不到 TensorRT：设置 `TensorRT_ROOT`，确认 `NvInfer.h`、`libnvinfer.so` 存在，重新 `cmake .. -DTensorRT_ROOT=...`。
- 找不到 `nvonnxparser`：安装 `libnvonnxparsers-dev` 或检查 tar 包 `lib` 路径是否在 `LD_LIBRARY_PATH`。
- CUDA runtime error：检查 `nvidia-smi`、`nvcc --version`、driver/CUDA/TensorRT ABI 是否匹配。
- engine 反序列化失败：engine 与当前 GPU/TensorRT 版本不兼容，删除 engine 后让程序从 ONNX 重建。
- ONNX build engine 失败：用 `trtexec --onnx=... --verbose` 看 parser 错误，必要时固定输入 shape 或重导出 ONNX。
- GTSAM 找不到：源码安装后执行 `sudo ldconfig`，并确认 `/usr/local/lib/cmake/GTSAM` 可见。
- Ceres 找不到：安装 `libceres-dev` 或源码安装后确认 `CeresConfig.cmake` 可见。
- yaw/pitch 方向反了：检查 `R_camera2gimbal`、`R_gimbal2imubody`、下位机 yaw/pitch 正方向和 `jlu_buff.trajectory_conf.yaw_offset/pitch_offset`。
- 点顺序错误：按 JLU 顺序重排 adapter，只改 `trt_yolo_buff.cpp`，不要改内部 `BuffBladePoints` 顺序。
- 大符拟合不收敛：增大历史队列、检查 roll 连续性、降低异常检测阈值、确认进入 BIG_BUFF 模式后没有频繁 reset。
- 自动开火太激进：设置 `jlu_buff.auto_fire_enable: false` 或收紧 trajectory/fire gate。
- 弹速异常：配置 `min_bullet_speed`、`max_bullet_speed`、`default_bullet_speed`；异常弹速会使用默认弹速。

## 11. 关闭自动开火

```yaml
jlu_buff:
  auto_fire_enable: false
```

关闭后小符/大符仍会输出 yaw/pitch 控制，但不会自动 fire。
