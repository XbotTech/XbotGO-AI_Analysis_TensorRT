# ============================================================
# Stage 1: Build stage
#   基于 NVIDIA 官方 TensorRT 镜像（已包含 CUDA + TensorRT）
#   可用 tag 查看: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt
# ============================================================
FROM nvcr.io/nvidia/tensorrt:24.12-py3 AS builder
# 该镜像自带: CUDA 12.6 + TensorRT 10.x + cuDNN + Python3

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# -------------------- Install OpenCV & build tools --------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# -------------------- Copy source code --------------------
WORKDIR /workspace
COPY CMakeLists.txt .
COPY main.cpp .
COPY CppExecCall.py .
COPY src/ src/
COPY model/ model/

# -------------------- Build --------------------
# T4 GPU = sm_75 (compute capability 7.5)
ENV TENSORRT_DIR=/usr
ENV OpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4

RUN mkdir -p build && cd build && \
    cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="75" \
    -DTENSORRT_DIR="$TENSORRT_DIR" \
    -DOpenCV_DIR="$OpenCV_DIR" && \
    make -j$(nproc)

# ============================================================
# Stage 2: Runtime stage (smaller image)
# ============================================================
FROM nvcr.io/nvidia/tensorrt:24.12-py3

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# -------------------- Runtime deps --------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# -------------------- Copy built artifacts from builder --------------------
WORKDIR /workspace
COPY --from=builder /workspace/build/yolov11-tensorrt         /workspace/build/yolov11-tensorrt
COPY --from=builder /workspace/build/exec_detect_video        /workspace/build/exec_detect_video
COPY --from=builder /workspace/build/exec_yolo_refine         /workspace/build/exec_yolo_refine
COPY --from=builder /workspace/build/libyolov11_tensorrt.so   /workspace/build/libyolov11_tensorrt.so
COPY --from=builder /workspace/model/                          /workspace/model/
COPY --from=builder /workspace/CppExecCall.py                  /workspace/

# -------------------- Environment --------------------
ENV LD_LIBRARY_PATH=/workspace/build:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV TENSORRT_DIR=/usr

# 构建阶段直接验证下游镜像依赖的运行环境和 C++ 产物。
RUN test -x /bin/sh && \
    test -x /bin/bash && \
    command -v python >/dev/null && \
    test -x /workspace/build/exec_detect_video && \
    test -x /workspace/build/exec_yolo_refine && \
    ! ldd /workspace/build/exec_detect_video | grep -q "not found"

# -------------------- Entrypoint --------------------
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
