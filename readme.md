# XbotGO AI 分析 — TensorRT 加速

基于 TensorRT 的 YOLOv11 目标检测与篮球跟踪细化，C++ 实现。

## 依赖
使用docker无需关注，只作为记录

- CUDA Toolkit
- TensorRT 10.14.1.48
- OpenCV 4.6.0
- CMake ≥ 3.12

## 快速开始（Docker）

```bash
# 1. 构建镜像
docker build -t xbotgo-tensorrt:latest .

# 2. 生成 engine（T4 上首次运行）
docker run --gpus all -it --rm \
    -v $(pwd)/model:/workspace/model \
    xbotgo-tensorrt:latest \
    build-engine

# 3. 检测
docker run --gpus all -it --rm \
    -v $(pwd)/model:/workspace/model \
    -v $(pwd)/data:/data \
    xbotgo-tensorrt:latest \
    detect
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

### 文件说明

| 文件                     | 用途                                                                               |
| ------------------------ | ---------------------------------------------------------------------------------- |
| `Dockerfile`           | 两阶段构建，基于`nvcr.io/nvidia/tensorrt:24.12-py3`（自带 CUDA 12.6 + TensorRT） |
| `.dockerignore`        | 排除 build 产物、测试视频等不必要文件                                              |
| `docker-entrypoint.sh` | 统一入口，封装`build-engine` / `detect` / `refine` 子命令                    |
| `docker-compose.yml`   | 一键启动交互式容器                                                                 |

### 部署步骤（在 T4 上直接构建）

**1. 登录 T4 服务器：**

```bash
ssh user@t4-server
```

**2. 获取源码（两种方式任选）：**

```bash
# 方式 a: git clone
git clone <你的仓库地址>
cd XbotGO-AI_Analysis_TensorRT-main

# 方式 b: scp 上传本地源码
scp -r . user@t4-server:~/XbotGO-AI_Analysis_TensorRT-main
```

**3. 在 T4 上构建镜像：**

```bash
docker build -t xbotgo-tensorrt:latest .
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
    detect
```

<!-- **3. 篮球跟踪细化：**

```bash
docker run --gpus all -it --rm \
    -v $(pwd)/model:/workspace/model \
    -v $(pwd)/data:/data \
    xbotgo-tensorrt:latest \
    refine \
    /workspace/model/<engine文件> \
    /data/<视频文件>
``` -->

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
