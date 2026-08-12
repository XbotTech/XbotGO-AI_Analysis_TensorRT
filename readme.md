# XbotGO AI 分析 — TensorRT 加速

基于 TensorRT 的 YOLOv11 目标检测与篮球跟踪细化，C++ 实现。

## 依赖

- CUDA Toolkit
- TensorRT 10.14.1.48
- OpenCV 4.6.0
- CMake ≥ 3.12

## 快速开始

1. 修改 `build_and_run.sh` 顶部两个路径变量：

```bash
TENSORRT_DIR="$ROOT/TensorRT-10.14.1.48"             # TensorRT 安装目录
OpenCV_DIR="/home/xiaodai/miniconda3/envs/opencv_cpp/lib/cmake/opencv4"  # OpenCV cmake 目录
```

2. 一键构建 + 运行：

```bash
./build_and_run.sh
```

## 产出

| 可执行文件            | 用途                    |
| --------------------- | ----------------------- |
| `yolov11-tensorrt`  | ONNX → TensorRT engine |
| `exec_detect_video` | 全帧目标检测，输出 JSON |
| `exec_yolo_refine`  | 跟踪引导的篮球检测细化  |

## 可选：Python 调用

```python
from CppExecCall import CppExecCall

cpp = CppExecCall()
cpp.extract_info("<engine>", "<video>")   # → detect.json
cpp.refine_ball("<engine>", "<video>")    # → ball boxes JSON
```

---

## Docker 部署（推荐用于 T4 / GPU 服务器）

### 前提

- T4 服务器已安装 NVIDIA 驱动（≥ 525）、nvidia-container-toolkit、Docker
- 本机已登录 NGC Registry：

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <NGC API Key>（在 https://ngc.nvidia.com/setup/api-key 生成）
```

### 文件说明

| 文件                     | 用途                                                                               |
| ------------------------ | ---------------------------------------------------------------------------------- |
| `Dockerfile`           | 两阶段构建，基于`nvcr.io/nvidia/tensorrt:24.12-py3`（自带 CUDA 12.6 + TensorRT） |
| `.dockerignore`        | 排除 build 产物、测试视频等不必要文件                                              |
| `docker-entrypoint.sh` | 统一入口，封装`build-engine` / `detect` / `refine` 子命令                    |
| `docker-compose.yml`   | 一键启动交互式容器                                                                 |
| `deploy.sh`            | 一键：构建 → 导出 → scp 上传 → T4 上导入                                        |

### 方式一：deploy.sh 一键部署

```bash
# 构建镜像 → 导出 tar.gz → scp 上传到 T4 → 自动导入
./deploy.sh <T4服务器IP> [用户名]

# 示例
./deploy.sh 192.168.1.100 ubuntu
```

### 方式二：手动部署

**本地构建镜像：**

```bash
docker build -t xbotgo-tensorrt:latest .
```

**导出并上传到 T4 服务器：**

```bash
docker save xbotgo-tensorrt:latest | gzip > xbotgo-tensorrt.tar.gz
scp xbotgo-tensorrt.tar.gz user@t4-server:~/

# 在 T4 上导入
ssh user@t4-server "docker load < ~/xbotgo-tensorrt.tar.gz"
```

### T4 上运行

**1. 生成 TensorRT engine（仅第一次）：**

```bash
docker run --gpus all -it --rm \
    -v $(pwd)/model:/workspace/model \
    xbotgo-tensorrt:latest \
    build-engine
```

**2. 全帧目标检测：**

```bash
docker run --gpus all -it --rm \
    -v $(pwd)/model:/workspace/model \
    -v $(pwd)/data:/data \
    xbotgo-tensorrt:latest \
    detect \
    /workspace/model/<engine文件> \
    /data/<视频文件> \
    /data/detect.json
```

**3. 篮球跟踪细化：**

```bash
docker run --gpus all -it --rm \
    -v $(pwd)/model:/workspace/model \
    -v $(pwd)/data:/data \
    xbotgo-tensorrt:latest \
    refine \
    /workspace/model/<engine文件> \
    /data/<视频文件>
```

**4. 交互式进入容器：**

```bash
docker run --gpus all -it --rm \
    -v $(pwd)/model:/workspace/model \
    -v $(pwd)/data:/data \
    xbotgo-tensorrt:latest

# 进入后可直接执行:
# build-engine
# detect <engine> <video> <output.json>
# refine <engine> <video>
```

**5. 使用 docker-compose：**

```bash
docker compose up -d
docker compose exec xbotgo-tensorrt bash
```

### 镜像内已编译可执行文件

| 路径                                        | 用途                         |
| ------------------------------------------- | ---------------------------- |
| `/workspace/build/yolov11-tensorrt`       | ONNX → TensorRT engine 转换 |
| `/workspace/build/exec_detect_video`      | 全帧目标检测                 |
| `/workspace/build/exec_yolo_refine`       | 篮球跟踪细化                 |
| `/workspace/build/libyolov11_tensorrt.so` | 动态库（供 Python 调用）     |
