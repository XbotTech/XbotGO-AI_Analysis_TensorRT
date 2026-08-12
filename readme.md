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

| 可执行文件 | 用途 |
|---|---|
| `yolov11-tensorrt` | ONNX → TensorRT engine |
| `exec_detect_video` | 全帧目标检测，输出 JSON |
| `exec_yolo_refine` | 跟踪引导的篮球检测细化 |

## 可选：Python 调用

```python
from CppExecCall import CppExecCall

cpp = CppExecCall()
cpp.extract_info("<engine>", "<video>")   # → detect.json
cpp.refine_ball("<engine>", "<video>")    # → ball boxes JSON
```