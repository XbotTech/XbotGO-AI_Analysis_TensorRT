#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include <cstdint>

using namespace nvinfer1;
using namespace std;
using namespace cv;

struct Detection
{
    float conf;
    int class_id;
    Rect2f bbox;
};

struct DetailedTimingStats
{
    double preprocess_gpu_ms = 0.0;   // H2D + warpaffine, accumulated per batch
    double inference_gpu_ms = 0.0;    // TensorRT execution, accumulated per batch
    double post_decode_gpu_ms = 0.0;  // Decode/filter kernel + filtered-count D2H
    double post_copy_gpu_ms = 0.0;    // Detection-result D2H
    double post_wait_cpu_ms = 0.0;    // Time blocked in existing stream synchronizations
    double post_cpu_ms = 0.0;         // Result parsing/vector construction/NMS
    uint64_t batches = 0;
    uint64_t frames = 0;
};

class YOLOv11
{
public:
    static constexpr int NUM_STREAMS = 2;

    YOLOv11(string model_path, nvinfer1::ILogger& logger);
    ~YOLOv11();

    // ---- Pipeline API: slot = 0 or 1, batch_idx = 0..batch_size-1 ----
    void preprocess(Mat& image, int slot, int batch_idx = 0);
    void infer(int slot);
    void postprocess(vector<Detection>& output, int slot, int batch_idx = 0);
    void syncSlot(int slot);
    vector<Detection> mapDetectionsToOriginal(
        const Size& image_size,
        const vector<Detection>& output) const;

    int getBatchSize() const { return batch_size; }    //!< Engine batch size
    int getStreamCount() const { return active_stream_count; }
    bool detailedTimingEnabled() const { return detailed_timing_enabled; }
    const DetailedTimingStats& getDetailedTimingStats() const { return detailed_timing_stats; }

    void setConfThreshold(float threshold) { conf_threshold = threshold; }
    float getConfThreshold() const { return conf_threshold; }

    // ---- Legacy single-stream API (backward compat) ----
    void preprocess(Mat& image) { preprocess(image, 0); }
    void infer()                 { infer(0); }
    void postprocess(vector<Detection>& output) { postprocess(output, 0); }

    void draw(Mat& image, const vector<Detection>& output);

private:
    void init(std::string engine_path, nvinfer1::ILogger& logger);

    // Double-buffered GPU resources (2 slots for pipelining)
    float* gpu_buffers[NUM_STREAMS][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    float* gpu_filtered_boxes[NUM_STREAMS] = {nullptr, nullptr};
    int*   gpu_filtered_count[NUM_STREAMS] = {nullptr, nullptr};
    float* cpu_filtered_boxes[NUM_STREAMS] = {nullptr, nullptr};

    cudaStream_t streams[NUM_STREAMS] = {nullptr, nullptr};
    IExecutionContext* contexts[NUM_STREAMS] = {nullptr, nullptr};

    // Optional, non-blocking detailed timing. Enabled with YOLO_DETAILED_TIMING=1.
    enum TimingEvent {
        PREPROCESS_START,
        PREPROCESS_END,
        INFERENCE_START,
        INFERENCE_END,
        POST_DECODE_START,
        POST_DECODE_END,
        POST_COPY_START,
        POST_COPY_END,
        TIMING_EVENT_COUNT
    };
    cudaEvent_t timing_events[NUM_STREAMS][TIMING_EVENT_COUNT] = {};
    bool timing_batch_pending[NUM_STREAMS] = {false, false};
    bool detailed_timing_enabled = false;
    DetailedTimingStats detailed_timing_stats;
    int active_stream_count = NUM_STREAMS;

    bool inference_initialized = false;

    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;

    // Model parameters
    int input_w = 0, input_h = 0;
    int batch_size = 1;           // Read from engine; 4 for batch export
    int num_detections = 0;       // 8400 or 300
    int detection_attribute_size = 0; // 84 or 6
    bool has_nms = false;        // true if engine has built-in NMS (det_attr <= 6)
    int num_classes = 80;
    const int MAX_IMAGE_SIZE = 4096 * 4096;
    const int MAX_OUTPUT_DETECTIONS = 1000;
    float conf_threshold = 0.3f;
    float nms_threshold = 0.4f;

    vector<Scalar> colors;

    void collectBatchGpuTiming(int slot);
    float elapsedGpuMs(int slot, TimingEvent start, TimingEvent end) const;
    void build(std::string onnxPath, nvinfer1::ILogger& logger);
};
