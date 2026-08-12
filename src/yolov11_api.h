/**
 * @file yolov11_api.h
 * @brief Public C API for YOLOv11 TensorRT inference library.
 *
 * This header defines the stable, C-compatible interface for:
 *   1. Exporting an ONNX model to a TensorRT engine.
 *   2. Running object detection on a video file with the engine.
 *
 * All functions are thread-safe to call from different threads as long as
 * each call operates on a distinct invocation context.  Concurrent calls
 * that share the same engine file are safe (the engine is deserialized
 * per invocation).
 */

#ifndef YOLOV11_API_H
#define YOLOV11_API_H

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Error codes                                                       */
/* ------------------------------------------------------------------ */
typedef enum {
    YOLOV11_OK                 =  0,  /**< Operation succeeded. */
    YOLOV11_ERROR_PARAM        = -1,  /**< Invalid parameter (e.g. NULL path). */
    YOLOV11_ERROR_FILE_OPEN    = -2,  /**< Cannot open input file. */
    YOLOV11_ERROR_FILE_WRITE   = -3,  /**< Cannot write output file. */
    YOLOV11_ERROR_CUDA         = -4,  /**< CUDA runtime error. */
    YOLOV11_ERROR_TENSORRT     = -5,  /**< TensorRT build or runtime error. */
    YOLOV11_ERROR_OPENCV       = -6,  /**< OpenCV error (e.g. video codec). */
    YOLOV11_ERROR_NO_FRAME     = -7,  /**< No frame could be read from video. */
    YOLOV11_ERROR_UNKNOWN      = -99  /**< Unknown internal error. */
} yolov11_error_t;

/* ------------------------------------------------------------------ */
/*  Progress callback                                                 */
/* ------------------------------------------------------------------ */

/**
 * @brief Callback for per-frame progress reporting.
 * @param current_frame  0-based index of the frame just processed.
 * @param total_frames   Total number of frames in the video (best-effort;
 *                       may be 0 if the container does not report it).
 * @param user_data      Opaque pointer passed through from the API call.
 */
typedef void (*yolov11_progress_callback)(int current_frame,
                                          int total_frames,
                                          void* user_data);

/* ------------------------------------------------------------------ */
/*  API functions                                                     */
/* ------------------------------------------------------------------ */

/**
 * @brief Build a TensorRT engine from an ONNX model.
 *
 * The generated engine file is written alongside the ONNX file with the
 * same base name and the `.engine` extension.
 *
 * Example: `yolov11_export_engine("model.onnx")` produces `model.engine`.
 *
 * @param onnx_path  Path to the input ONNX model file (must exist).
 * @return YOLOV11_OK on success, or a negative error code.
 */
yolov11_error_t yolov11_export_engine(const char* onnx_path);

/**
 * @brief Run object detection on every frame of a video.
 *
 * @param engine_path       Path to a pre-built `.engine` file (or `.onnx`
 *                          to auto-build the engine first).
 * @param video_path        Path to the input video (mp4 / avi / mov / …).
 * @param json_output_path  Path for the per-frame detection JSON file.
 *                          Pass NULL or "" to skip JSON generation.
 * @param output_video_path Path for the output annotated video.
 *                          Pass NULL or "" to skip video generation.
 * @param conf_threshold    Minimum confidence threshold (0.0 ~ 1.0).
 *                          Detections below this value are discarded.
 *                          Pass a value <= 0 to use the default (0.3).
 * @param progress_cb       Optional callback invoked after each frame is
 *                          processed.  Pass NULL if not needed.
 * @param user_data         Opaque pointer forwarded to `progress_cb`.
 * @return YOLOV11_OK on success, or a negative error code.
 */
yolov11_error_t yolov11_detect_video(
    const char* engine_path,
    const char* video_path,
    const char* json_output_path,
    const char* output_video_path,
    float conf_threshold,
    yolov11_progress_callback progress_cb,
    void* user_data);

/**
 * @brief Run tracking-guided basketball ball refinement.
 *
 * Uses a secondary 640x640 ball-detection engine for crop-based re-detection
 * of balls missed by the primary YOLO model.  Reads per-event YOLO detections
 * from ``JsonForLLM_with_objects.json``, processes each event segment of the
 * video, and outputs refined ball boxes.
 *
 * @param events_json_path  Path to JsonForLLM_with_objects.json.
 * @param video_path        Path to the input video.
 * @param ball_engine_path  Path to the 640x640 ball-detection .engine file.
 * @param output_json_path  Path for the aggregate output JSON.
 * @param detect_stride     Process every N-th frame (>= 1).
 * @param det_conf          Confidence threshold for crop re-detection (0.0~1.0).
 * @param ball_cls          Ball class ID in the crop detection model.
 * @param json_ball_cls     Ball class ID in the input JSON's "Objects" field.
 * @param max_lookback      Max lookback frames for tracking (default 5).
 * @param proximity_threshold  Max center distance (pixels) to consider a
 *                          detection as "covering" a previously tracked ball.
 * @param progress_cb       Optional callback.  Pass NULL if not needed.
 * @param user_data         Opaque pointer forwarded to `progress_cb`.
 * @return YOLOV11_OK on success, or a negative error code.
 */
yolov11_error_t yolov11_track_refine(
    const char* events_json_path,
    const char* video_path,
    const char* ball_engine_path,
    const char* output_json_path,
    int detect_stride,
    float det_conf,
    int ball_cls,
    int json_ball_cls,
    int max_lookback,
    float proximity_threshold,
    yolov11_progress_callback progress_cb,
    void* user_data);

#ifdef __cplusplus
}
#endif

#endif /* YOLOV11_API_H */
