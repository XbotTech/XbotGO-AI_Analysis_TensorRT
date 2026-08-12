#!/bin/bash
set -e

# ============================================================
# Docker entrypoint — handles ONNX → engine conversion + run
# ============================================================

ONNX_DIR="/workspace/model"
ENGINE_DIR="/workspace/model"

# If no command given, drop into bash
if [ $# -eq 0 ]; then
    exec bash
fi

case "$1" in
    # --------------------------------------------------
    # build-engine: ONNX → TensorRT engine
    #   docker run --gpus all <image> build-engine [model.onnx]
    # --------------------------------------------------
    build-engine)
        ONNX="${2:-$ONNX_DIR/yolo11s_person_basketball_backboard_hoop_1920_1088_AIAnalysi1_without_nms_batch4.onnx}"
        echo "[INFO] Building TensorRT engine from: $ONNX"
        /workspace/build/yolov11-tensorrt "$ONNX"
        echo "[INFO] Done. Engine saved to: $ONNX_DIR"
        ;;

    # --------------------------------------------------
    # detect: full-frame detection on video
    #   docker run --gpus all -v /host/data:/data <image> detect <engine> <video> [output.json]
    # --------------------------------------------------
    detect)
        ENGINE="${2:-$ONNX_DIR/yolo11s_person_basketball_backboard_hoop_1920_1088_AIAnalysi1_without_nms_batch4.engine}"
        VIDEO="${3:-/data/test.mp4}"
        OUTPUT="${4:-/data/detect.json}"
        echo "[INFO] Running detection: engine=$ENGINE video=$VIDEO output=$OUTPUT"
        /workspace/build/exec_detect_video "$ENGINE" "$VIDEO" "$OUTPUT"
        echo "[INFO] Detection complete → $OUTPUT"
        ;;

    # --------------------------------------------------
    # refine: tracking-guided ball refinement
    #   docker run --gpus all -v /host/data:/data <image> refine <engine> <video>
    # --------------------------------------------------
    refine)
        ENGINE="${2:-$ONNX_DIR/yolo11s_person_basketball_backboard_hoop_1920_1088_AIAnalysi1_without_nms_batch4.engine}"
        VIDEO="${3:-/data/test.mp4}"
        echo "[INFO] Running ball refinement: engine=$ENGINE video=$VIDEO"
        /workspace/build/exec_yolo_refine "$ENGINE" "$VIDEO"
        echo "[INFO] Refinement complete."
        ;;

    # --------------------------------------------------
    # shell / custom command
    # --------------------------------------------------
    *)
        exec "$@"
        ;;
esac
