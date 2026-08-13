#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ============================================
# 可配置路径（迁移到其他机器时修改这里）
# ============================================
TENSORRT_DIR="$ROOT/TensorRT-10.14.1.48"
OpenCV_DIR="/home/xiaodai/miniconda3/envs/opencv_cpp/lib/cmake/opencv4"

# 将 TensorRT lib 加入库搜索路径，解决 dlopen 加载 builder_resource 的问题
export LD_LIBRARY_PATH="$TENSORRT_DIR/lib:$LD_LIBRARY_PATH"

echo "============================================"
echo " Step 1/4: Build project"
echo "============================================"
mkdir -p build
# 如果 build 目录下有内容，则清空
if [ "$(ls -A build 2>/dev/null)" ]; then
    echo "[CLEAN] Removing existing build contents..."
    rm -rf build/*
fi
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORRT_DIR="$TENSORRT_DIR" \
    -DOpenCV_DIR="$OpenCV_DIR"
make -j$(nproc)
cd ..

echo ""
echo "============================================"
echo " Step 2/4: Export ONNX → TensorRT engine"
echo "============================================"
ONNX="model/yolo11s_person_basketball_backboard_hoop_1920_1088_AIAnalysi1_without_nms_batch4.onnx"
ENGINE="model/yolo11s_person_basketball_backboard_hoop_1920_1088_AIAnalysi1_without_nms_batch4.engine"

if [ -s "$ENGINE" ]; then
    echo "[SKIP] Engine already exists: $ENGINE"
else
    echo "[RUN] ./build/yolov11-tensorrt $ONNX"
    ./build/yolov11-tensorrt "$ONNX"
fi

echo ""
echo "============================================"
echo " Step 3/4: Run full-frame detection"
echo "============================================"
VIDEO="data/test.mp4"
OUTPUT_JSON="testcpp.json"

echo "[RUN] ./build/exec_detect_video $ENGINE $VIDEO $OUTPUT_JSON"
./build/exec_detect_video "$ENGINE" "$VIDEO" "$OUTPUT_JSON"

echo ""
echo "============================================"
echo " Step 4/4: Run CppExecCall.py"
echo "============================================"
if [ -f "CppExecCall.py" ]; then
    echo "[RUN] python3 CppExecCall.py"
    python3 CppExecCall.py
else
    echo "[SKIP] CppExecCall.py not found"
fi

echo ""
echo "============================================"
echo " All steps completed successfully!"
echo "============================================"