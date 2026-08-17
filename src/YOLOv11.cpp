#include "YOLOv11.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"
#include "preprocess.h"
#include "postprocess.h"
#include <NvOnnxParser.h>
#include "common.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>


static Logger logger;
#define isFP16 true
#define warmup true


YOLOv11::YOLOv11(string model_path, nvinfer1::ILogger& logger)
{
    if (model_path.find(".onnx") == std::string::npos)
    {
        // Load pre-built engine
        init(model_path, logger);
    }
    else
    {
        // Build engine from ONNX
        build(model_path, logger);

        // Re-load the engine we just built to set up inference resources
        string engine_path = model_path.substr(0, model_path.find_last_of(".")) + ".engine";
        init(engine_path, logger);
    }
}


void YOLOv11::init(std::string engine_path, nvinfer1::ILogger& logger)
{
    const char* timing_env = std::getenv("YOLO_DETAILED_TIMING");
    detailed_timing_enabled = timing_env != nullptr &&
                              std::strcmp(timing_env, "0") != 0 &&
                              std::strlen(timing_env) != 0;

    const char* streams_env = std::getenv("YOLO_NUM_STREAMS");
    if (streams_env != nullptr && std::strlen(streams_env) != 0) {
        const int requested_streams = std::atoi(streams_env);
        if (requested_streams == 1 || requested_streams == 2) {
            active_stream_count = requested_streams;
        } else {
            fprintf(stderr,
                    "Invalid YOLO_NUM_STREAMS=%s; expected 1 or 2. Using 2 streams.\n",
                    streams_env);
            active_stream_count = NUM_STREAMS;
        }
    }

    // Read the engine file
    ifstream engineStream(engine_path, ios::binary);
    engineStream.seekg(0, ios::end);
    const size_t modelSize = engineStream.tellg();
    engineStream.seekg(0, ios::beg);
    unique_ptr<char[]> engineData(new char[modelSize]);
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();

    // Deserialize the tensorrt engine (shared across all contexts)
    runtime = createInferRuntime(logger);
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);

    // Get input and output sizes of the model (same for all slots)
#if NV_TENSORRT_MAJOR < 10
    input_h = engine->getBindingDimensions(0).d[2];
    input_w = engine->getBindingDimensions(0).d[3];
    detection_attribute_size = engine->getBindingDimensions(1).d[1];
    num_detections = engine->getBindingDimensions(1).d[2];
#else
    auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
    batch_size = input_dims.d[0];
    input_h = input_dims.d[2];
    input_w = input_dims.d[3];
    auto output_dims = engine->getTensorShape(engine->getIOTensorName(1));
    // NMS models output [batch, num_dets, det_attr]; non-NMS output [batch, det_attr, num_dets]
    // Heuristic: if d[2] <= 6, it's NMS format — swap d[1] and d[2]
    bool is_nms_format = (output_dims.d[2] <= 6);
    if (is_nms_format) {
        num_detections = output_dims.d[1];
        detection_attribute_size = output_dims.d[2];
    } else {
        detection_attribute_size = output_dims.d[1];
        num_detections = output_dims.d[2];
    }
#endif
    // Auto-detect NMS model: use the format check, NOT det_attr size
    // (1-class non-NMS models have det_attr=5 which would falsely trigger has_nms)
    has_nms = is_nms_format;
    // num_classes only meaningful for non-NMS; NMS output has class_id directly
    num_classes = has_nms ? 0 : detection_attribute_size - 4;
    printf("Model: batch=%d, input=%dx%d, det_attr=%d, num_dets=%d, classes=%d, NMS=%s, streams=%d\n",
           batch_size, input_w, input_h, detection_attribute_size, num_detections, num_classes,
           has_nms ? "built-in" : "CPU", active_stream_count);

    // ---- Per-slot initialization (2 slots for pipelining) ----
    for (int s = 0; s < active_stream_count; s++) {
        // Context per slot
        contexts[s] = engine->createExecutionContext();

        // GPU buffers: input [batch * 3 * H * W] + output [batch * det_attr * num_dets]
        CUDA_CHECK(cudaMalloc(&gpu_buffers[s][0], batch_size * 3 * input_w * input_h * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&gpu_buffers[s][1], batch_size * detection_attribute_size * num_detections * sizeof(float)));

#if NV_TENSORRT_MAJOR >= 10
        contexts[s]->setInputTensorAddress(engine->getIOTensorName(0), gpu_buffers[s][0]);
        contexts[s]->setOutputTensorAddress(engine->getIOTensorName(1), gpu_buffers[s][1]);
#endif

        // Postprocess GPU buffers (only needed for non-NMS models)
        if (!has_nms) {
            CUDA_CHECK(cudaMalloc(&gpu_filtered_boxes[s], MAX_OUTPUT_DETECTIONS * 6 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&gpu_filtered_count[s], sizeof(int)));
            cpu_filtered_boxes[s] = new float[MAX_OUTPUT_DETECTIONS * 6];
        }

        // Stream per slot
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
        if (detailed_timing_enabled) {
            for (int e = 0; e < TIMING_EVENT_COUNT; e++) {
                CUDA_CHECK(cudaEventCreate(&timing_events[s][e]));
            }
        }
    }

    cuda_preprocess_init(MAX_IMAGE_SIZE, active_stream_count, batch_size);

    // Warmup on slot 0
    if (warmup) {
        for (int i = 0; i < 2; i++) {
            this->infer(0);
        }
        printf("model warmup 2 times\n");
    }

    if (detailed_timing_enabled) {
        printf("Detailed timing enabled (CUDA events, no additional stream synchronization)\n");
    }

    inference_initialized = true;
}

YOLOv11::~YOLOv11()
{
    if (inference_initialized) {
        for (int s = 0; s < active_stream_count; s++) {
            CUDA_CHECK(cudaStreamSynchronize(streams[s]));
            CUDA_CHECK(cudaStreamDestroy(streams[s]));
            if (detailed_timing_enabled) {
                for (int e = 0; e < TIMING_EVENT_COUNT; e++) {
                    CUDA_CHECK(cudaEventDestroy(timing_events[s][e]));
                }
            }
            CUDA_CHECK(cudaFree(gpu_buffers[s][0]));
            CUDA_CHECK(cudaFree(gpu_buffers[s][1]));
            if (!has_nms) {
                CUDA_CHECK(cudaFree(gpu_filtered_boxes[s]));
                CUDA_CHECK(cudaFree(gpu_filtered_count[s]));
                delete[] cpu_filtered_boxes[s];
            }
            delete contexts[s];
        }
        cuda_preprocess_destroy();
    }

    // Engine and runtime are shared, always created
    delete engine;
    delete runtime;
}

void YOLOv11::preprocess(Mat& image, int slot, int batch_idx) {
    // Launch GPU preprocess on stream[slot] — writes to batch slot
    float* dst = gpu_buffers[slot][0] + batch_idx * 3 * input_w * input_h;
    if (detailed_timing_enabled && inference_initialized && batch_idx == 0) {
        CUDA_CHECK(cudaEventRecord(timing_events[slot][PREPROCESS_START], streams[slot]));
    }
    cuda_preprocess(image.ptr(), image.cols, image.rows,
                    dst, input_w, input_h, streams[slot], slot, batch_idx);
}

void YOLOv11::infer(int slot)
{
    const bool capture_timing = detailed_timing_enabled && inference_initialized;
    if (capture_timing) {
        // Both events are placed after all preprocess calls for this batch.
        CUDA_CHECK(cudaEventRecord(timing_events[slot][PREPROCESS_END], streams[slot]));
        CUDA_CHECK(cudaEventRecord(timing_events[slot][INFERENCE_START], streams[slot]));
    }
#if NV_TENSORRT_MAJOR < 10
    contexts[slot]->enqueueV2((void**)gpu_buffers[slot], streams[slot], nullptr);
#else
    contexts[slot]->enqueueV3(streams[slot]);
#endif
    if (capture_timing) {
        CUDA_CHECK(cudaEventRecord(timing_events[slot][INFERENCE_END], streams[slot]));
        timing_batch_pending[slot] = true;
    }
}

float YOLOv11::elapsedGpuMs(int slot, TimingEvent start, TimingEvent end) const
{
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms,
                                    timing_events[slot][start],
                                    timing_events[slot][end]));
    return elapsed_ms;
}

void YOLOv11::collectBatchGpuTiming(int slot)
{
    if (!detailed_timing_enabled || !timing_batch_pending[slot]) return;

    // Called only after postprocess's existing stream synchronization, so these
    // event queries do not introduce a new wait or serialize the pipeline.
    detailed_timing_stats.preprocess_gpu_ms +=
        elapsedGpuMs(slot, PREPROCESS_START, PREPROCESS_END);
    detailed_timing_stats.inference_gpu_ms +=
        elapsedGpuMs(slot, INFERENCE_START, INFERENCE_END);
    detailed_timing_stats.batches++;
    timing_batch_pending[slot] = false;
}

void YOLOv11::postprocess(vector<Detection>& output, int slot, int batch_idx)
{
    using TimingClock = std::chrono::steady_clock;
    const bool capture_timing = detailed_timing_enabled && inference_initialized;

    // Raw output for this batch element: offset in the full batched buffer
    int per_image_size = detection_attribute_size * num_detections;
    float* raw_output = gpu_buffers[slot][1] + batch_idx * per_image_size;

    if (has_nms) {
        // ---- NMS mode: engine already outputs [num_dets, 6] NMS'd results ----
        int output_size = num_detections * detection_attribute_size;  // 300 * 6
        float* cpu_output = new float[output_size];
        if (capture_timing) {
            CUDA_CHECK(cudaEventRecord(timing_events[slot][POST_COPY_START], streams[slot]));
        }
        CUDA_CHECK(cudaMemcpyAsync(cpu_output, raw_output, output_size * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[slot]));
        if (capture_timing) {
            CUDA_CHECK(cudaEventRecord(timing_events[slot][POST_COPY_END], streams[slot]));
        }
        TimingClock::time_point wait_start;
        if (capture_timing) wait_start = TimingClock::now();
        CUDA_CHECK(cudaStreamSynchronize(streams[slot]));
        if (capture_timing) {
            detailed_timing_stats.post_wait_cpu_ms +=
                std::chrono::duration<double, std::milli>(TimingClock::now() - wait_start).count();
            collectBatchGpuTiming(slot);
            detailed_timing_stats.post_copy_gpu_ms +=
                elapsedGpuMs(slot, POST_COPY_START, POST_COPY_END);
        }

        TimingClock::time_point cpu_start;
        if (capture_timing) cpu_start = TimingClock::now();
        for (int i = 0; i < num_detections; i++) {
            float x1   = cpu_output[i * 6 + 0];
            float y1   = cpu_output[i * 6 + 1];
            float x2   = cpu_output[i * 6 + 2];
            float y2   = cpu_output[i * 6 + 3];
            float conf = cpu_output[i * 6 + 4];
            int   cls  = (int)cpu_output[i * 6 + 5];
            if (conf <= 0.0f) continue;
            Detection det;
            det.conf = conf;
            det.class_id = cls;
            det.bbox = Rect2f(x1, y1, x2 - x1, y2 - y1);
            output.push_back(det);
        }
        delete[] cpu_output;
        if (capture_timing) {
            detailed_timing_stats.post_cpu_ms +=
                std::chrono::duration<double, std::milli>(TimingClock::now() - cpu_start).count();
            detailed_timing_stats.frames++;
        }
        return;
    }

    // ---- Non-NMS mode: GPU decode + CPU NMS (original path) ----
    if (capture_timing) {
        CUDA_CHECK(cudaEventRecord(timing_events[slot][POST_DECODE_START], streams[slot]));
    }
    CUDA_CHECK(cudaMemsetAsync(gpu_filtered_count[slot], 0, sizeof(int), streams[slot]));

    // ----- Step 2: GPU kernel: decode boxes + filter by confidence (async) -----
    cuda_postprocess_decode(
        raw_output,
        gpu_filtered_boxes[slot],
        gpu_filtered_count[slot],
        num_detections, num_classes, detection_attribute_size,
        conf_threshold, MAX_OUTPUT_DETECTIONS,
        streams[slot]
    );

    // ----- Step 3: Copy filtered count back to CPU (async) -----
    int filtered_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&filtered_count, gpu_filtered_count[slot], sizeof(int),
                               cudaMemcpyDeviceToHost, streams[slot]));
    if (capture_timing) {
        CUDA_CHECK(cudaEventRecord(timing_events[slot][POST_DECODE_END], streams[slot]));
    }

    // ----- Step 4: Sync: wait for all GPU work on this slot -----
    TimingClock::time_point first_wait_start;
    if (capture_timing) first_wait_start = TimingClock::now();
    CUDA_CHECK(cudaStreamSynchronize(streams[slot]));
    if (capture_timing) {
        detailed_timing_stats.post_wait_cpu_ms +=
            std::chrono::duration<double, std::milli>(TimingClock::now() - first_wait_start).count();
        collectBatchGpuTiming(slot);
        detailed_timing_stats.post_decode_gpu_ms +=
            elapsedGpuMs(slot, POST_DECODE_START, POST_DECODE_END);
    }

    if (filtered_count == 0) {
        if (capture_timing) detailed_timing_stats.frames++;
        return;
    }
    if (filtered_count > MAX_OUTPUT_DETECTIONS) filtered_count = MAX_OUTPUT_DETECTIONS;

    // ----- Step 5: Copy filtered detections (GPU is done, safe to use sync memcpy) -----
    if (capture_timing) {
        CUDA_CHECK(cudaEventRecord(timing_events[slot][POST_COPY_START], streams[slot]));
    }
    CUDA_CHECK(cudaMemcpyAsync(cpu_filtered_boxes[slot], gpu_filtered_boxes[slot],
                               filtered_count * 6 * sizeof(float),
                               cudaMemcpyDeviceToHost, streams[slot]));
    if (capture_timing) {
        CUDA_CHECK(cudaEventRecord(timing_events[slot][POST_COPY_END], streams[slot]));
    }
    TimingClock::time_point second_wait_start;
    if (capture_timing) second_wait_start = TimingClock::now();
    CUDA_CHECK(cudaStreamSynchronize(streams[slot]));
    if (capture_timing) {
        detailed_timing_stats.post_wait_cpu_ms +=
            std::chrono::duration<double, std::milli>(TimingClock::now() - second_wait_start).count();
        detailed_timing_stats.post_copy_gpu_ms +=
            elapsedGpuMs(slot, POST_COPY_START, POST_COPY_END);
    }

    // ----- Step 6: Build detection lists + CPU NMS -----
    TimingClock::time_point cpu_start;
    if (capture_timing) cpu_start = TimingClock::now();
    vector<Rect> boxes;
    vector<Rect2f> precise_boxes;
    vector<int> class_ids;
    vector<float> confidences;
    boxes.reserve(filtered_count);
    precise_boxes.reserve(filtered_count);
    class_ids.reserve(filtered_count);
    confidences.reserve(filtered_count);

    for (int i = 0; i < filtered_count; i++) {
        float x    = cpu_filtered_boxes[slot][i * 6 + 0];
        float y    = cpu_filtered_boxes[slot][i * 6 + 1];
        float w    = cpu_filtered_boxes[slot][i * 6 + 2];
        float h    = cpu_filtered_boxes[slot][i * 6 + 3];
        float conf = cpu_filtered_boxes[slot][i * 6 + 4];
        int   cls  = (int)cpu_filtered_boxes[slot][i * 6 + 5];

        Rect box;
        box.x = static_cast<int>(x);
        box.y = static_cast<int>(y);
        box.width = static_cast<int>(w);
        box.height = static_cast<int>(h);

        boxes.push_back(box);
        precise_boxes.emplace_back(x, y, w, h);
        class_ids.push_back(cls);
        confidences.push_back(conf);
    }

    vector<int> nms_result;
    dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection result;
        int idx = nms_result[i];
        result.class_id = class_ids[idx];
        result.conf = confidences[idx];
        result.bbox = precise_boxes[idx];
        output.push_back(result);
    }
    if (capture_timing) {
        detailed_timing_stats.post_cpu_ms +=
            std::chrono::duration<double, std::milli>(TimingClock::now() - cpu_start).count();
        detailed_timing_stats.frames++;
    }
}

void YOLOv11::syncSlot(int slot)
{
    CUDA_CHECK(cudaStreamSynchronize(streams[slot]));
}

vector<Detection> YOLOv11::mapDetectionsToOriginal(
    const Size& image_size,
    const vector<Detection>& output) const
{
    vector<Detection> mapped = output;
    const float ratio_h = input_h / static_cast<float>(image_size.height);
    const float ratio_w = input_w / static_cast<float>(image_size.width);

    for (Detection& detection : mapped) {
        Rect2f& box = detection.bbox;
        if (ratio_h > ratio_w) {
            box.x /= ratio_w;
            box.y = (box.y - (input_h - ratio_w * image_size.height) / 2) / ratio_w;
            box.width /= ratio_w;
            box.height /= ratio_w;
        }
        else {
            box.x = (box.x - (input_w - ratio_h * image_size.width) / 2) / ratio_h;
            box.y /= ratio_h;
            box.width /= ratio_h;
            box.height /= ratio_h;
        }
    }

    return mapped;
}

void YOLOv11::build(std::string onnxPath, nvinfer1::ILogger& logger)
{
    auto builder = createInferBuilder(logger);
#if NV_TENSORRT_MAJOR < 10
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
#else
    INetworkDefinition* network = builder->createNetworkV2(0U);
#endif
    IBuilderConfig* config = builder->createBuilderConfig();
    if (isFP16)
    {
#if NV_TENSORRT_MAJOR < 10
        config->setFlag(BuilderFlag::kFP16);
#else
        config->setFlag(BuilderFlag::kFP16);
#endif
    }
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));

    // Set optimization profile only if input has dynamic dimensions
    auto input_dims = network->getInput(0)->getDimensions();
    bool has_dynamic = false;
    for (int i = 0; i < input_dims.nbDims; i++) {
        if (input_dims.d[i] < 0) { has_dynamic = true; break; }
    }
    if (has_dynamic) {
        auto profile = builder->createOptimizationProfile();
        profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMIN,
                               Dims4{1, input_dims.d[1], input_dims.d[2], input_dims.d[3]});
        profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kOPT,
                               Dims4{4, input_dims.d[1], input_dims.d[2], input_dims.d[3]});
        profile->setDimensions(network->getInput(0)->getName(), OptProfileSelector::kMAX,
                               Dims4{4, input_dims.d[1], input_dims.d[2], input_dims.d[3]});
        config->addOptimizationProfile(profile);
        printf("Optimization profile: MIN=[1,%lld,%lld,%lld] OPT=[4,%lld,%lld,%lld] MAX=[4,%lld,%lld,%lld]\n",
               static_cast<long long>(input_dims.d[1]),
               static_cast<long long>(input_dims.d[2]),
               static_cast<long long>(input_dims.d[3]),
               static_cast<long long>(input_dims.d[1]),
               static_cast<long long>(input_dims.d[2]),
               static_cast<long long>(input_dims.d[3]),
               static_cast<long long>(input_dims.d[1]),
               static_cast<long long>(input_dims.d[2]),
               static_cast<long long>(input_dims.d[3]));
    }

    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };

    // Write the serialized engine to file (engine will be loaded later by init())
    string engine_path = onnxPath.substr(0, onnxPath.find_last_of(".")) + ".engine";
    std::ofstream file(engine_path, std::ios::binary | std::ios::out);
    file.write((const char*)plan->data(), plan->size());
    file.close();

    delete plan;
    delete parser;
    delete config;
    delete network;
    delete builder;
}

void YOLOv11::draw(Mat& image, const vector<Detection>& output)
{
    const vector<Detection> mapped = mapDetectionsToOriginal(image.size(), output);

    for (int i = 0; i < mapped.size(); i++)
    {
        auto detection = mapped[i];
        auto box = detection.bbox;
        auto class_id = detection.class_id;
        auto conf = detection.conf;
        cv::Scalar color = cv::Scalar(COLORS[class_id][0], COLORS[class_id][1], COLORS[class_id][2]);

        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

        // Detection box text
        string class_string = CLASS_NAMES[class_id] + ' ' + to_string(conf).substr(0, 4);
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        rectangle(image, text_rect, color, FILLED);
        putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}
