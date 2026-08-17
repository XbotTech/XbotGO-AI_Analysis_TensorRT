ARG BUILD_IMAGE=nvcr.io/nvidia/tensorrt:24.12-py3
ARG RUNTIME_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu24.04

# ============================================================
# Stage 1: compile the two inference executables and the engine generator.
# The large TensorRT image and all development packages stay in this stage.
# ============================================================
FROM ${BUILD_IMAGE} AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
# -------------------- Build dependencies --------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# -------------------- Copy source code --------------------
WORKDIR /workspace
COPY CMakeLists.txt .
COPY main.cpp .
COPY src/ src/

# -------------------- Build --------------------
# T4 GPU = sm_75 (compute capability 7.5)
ENV TENSORRT_DIR=/usr
ENV OpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4

RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=75 \
        -DTENSORRT_DIR="$TENSORRT_DIR" \
        -DOpenCV_DIR="$OpenCV_DIR" \
    && cmake --build build \
        --target yolov11-tensorrt exec_detect_video exec_yolo_refine \
        --parallel "$(nproc)" \
    && strip --strip-unneeded \
        build/yolov11-tensorrt \
        build/exec_detect_video \
        build/exec_yolo_refine

# ============================================================
# Stage 2: inference-only runtime.
# It intentionally cannot build an engine from ONNX. Mount pre-built engines
# when the container starts.
# ============================================================
FROM ${RUNTIME_IMAGE} AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ARG TENSORRT_VERSION=10.7.0.23-1+cuda12.6

# TensorRT 24.12 uses TensorRT 10.7 and CUDA 12.6. Pin the Debian
# packages so a future repository update cannot silently change the ABI.
# libnvinfer_builder_resource is only needed by createInferBuilder(); both
# retained executables only deserialize an existing engine, so omit it from
# the committed layer to save several GB.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnvinfer10="${TENSORRT_VERSION}" \
        libnvinfer-plugin10="${TENSORRT_VERSION}" \
        libnvonnxparsers10="${TENSORRT_VERSION}" \
        libopencv-core406t64 \
        libopencv-imgproc406t64 \
        libopencv-imgcodecs406t64 \
        libopencv-videoio406t64 \
    && rm -f /usr/lib/x86_64-linux-gnu/libnvinfer_builder_resource.so* \
    && rm -rf /var/lib/apt/lists/*

# -------------------- Runtime artifacts --------------------
WORKDIR /workspace
COPY --from=builder /workspace/build/exec_detect_video /usr/local/bin/exec_detect_video
COPY --from=builder /workspace/build/exec_yolo_refine  /usr/local/bin/exec_yolo_refine

# Catch missing link-time dependencies without requiring a GPU during build.
RUN test -x /usr/local/bin/exec_detect_video \
    && test -x /usr/local/bin/exec_yolo_refine \
    && ! ldd /usr/local/bin/exec_detect_video | grep -q "not found" \
    && ! ldd /usr/local/bin/exec_yolo_refine | grep -q "not found"

# -------------------- Entrypoint --------------------
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["help"]
