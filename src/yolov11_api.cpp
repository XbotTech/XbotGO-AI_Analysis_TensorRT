/**
 * @file yolov11_api.cpp
 * @brief Implementation of the public YOLOv11 TensorRT C API.
 */

#include "yolov11_api.h"

#include "YOLOv11.h"
#include "YoloTrackRefine.h"
#include "logging.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

/* =================================================================== */
/*  Internal helpers                                                   */
/* =================================================================== */

namespace {

// ---------------------------------------------------------------------------
// Logger – shared across all API invocations.  TensorRT logging is
//          thread-safe per the official documentation.
// ---------------------------------------------------------------------------
class ApiLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << std::endl;
    }
};

// A single static logger instance is sufficient.
static ApiLogger g_api_logger;

// ---------------------------------------------------------------------------
// Tiny filesystem helpers
// ---------------------------------------------------------------------------
bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

bool ensure_parent_dir(const std::string& path) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path parent = fs::path(path).parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent, ec);
        return !ec;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Thread-safe frame queue (same design as in main.cpp)
// ---------------------------------------------------------------------------
class FrameQueue {
public:
    explicit FrameQueue(size_t max_size = 128)
        : max_size_(max_size) {}

    void push(cv::Mat frame) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_not_full_.wait(lock, [this] { return q_.size() < max_size_ || done_; });
        if (done_) return;
        q_.push(std::move(frame));
        cv_not_empty_.notify_one();
    }

    bool pop(cv::Mat& frame) {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_not_empty_.wait(lock, [this] { return !q_.empty() || done_; });
        if (q_.empty() && done_) return false;
        frame = std::move(q_.front());
        q_.pop();
        cv_not_full_.notify_one();
        return true;
    }

    void setDone() {
        std::unique_lock<std::mutex> lock(mtx_);
        done_ = true;
        cv_not_empty_.notify_all();
    }

private:
    std::queue<cv::Mat> q_;
    std::mutex mtx_;
    std::condition_variable cv_not_empty_;
    std::condition_variable cv_not_full_;
    size_t max_size_;
    bool done_ = false;
};

// ---------------------------------------------------------------------------
// Convert exception / error to yolov11_error_t
// ---------------------------------------------------------------------------
yolov11_error_t to_error_code(const std::exception& /*e*/) {
    // Could inspect e.what() for finer granularity.
    return YOLOV11_ERROR_UNKNOWN;
}

}  // anonymous namespace

/* =================================================================== */
/*  API: Export ONNX → Engine                                          */
/* =================================================================== */

yolov11_error_t yolov11_export_engine(const char* onnx_path) {
    if (onnx_path == nullptr || std::strlen(onnx_path) == 0) {
        return YOLOV11_ERROR_PARAM;
    }

    if (!file_exists(onnx_path)) {
        std::fprintf(stderr, "yolov11_export_engine: ONNX file not found: %s\n",
                     onnx_path);
        return YOLOV11_ERROR_FILE_OPEN;
    }

    try {
        // The YOLOv11 constructor auto-detects .onnx and builds + saves the
        // engine.  We only need to construct (and immediately destroy) it.
        YOLOv11 model(onnx_path, g_api_logger);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "yolov11_export_engine: build failed: %s\n",
                     e.what());
        return to_error_code(e);
    } catch (...) {
        std::fprintf(stderr, "yolov11_export_engine: unknown build error\n");
        return YOLOV11_ERROR_UNKNOWN;
    }

    // Verify the engine was written
    std::string engine_path =
        std::string(onnx_path).substr(0, std::string(onnx_path).find_last_of('.'))
        + ".engine";
    if (!file_exists(engine_path)) {
        std::fprintf(stderr,
                     "yolov11_export_engine: engine file was not created: %s\n",
                     engine_path.c_str());
        return YOLOV11_ERROR_TENSORRT;
    }

    std::printf("yolov11_export_engine: engine saved to %s\n",
                engine_path.c_str());
    return YOLOV11_OK;
}

/* =================================================================== */
/*  API: Detect video                                                  */
/* =================================================================== */

yolov11_error_t yolov11_detect_video(
    const char* engine_path,
    const char* video_path,
    const char* json_output_path,
    const char* output_video_path,
    float conf_threshold,
    yolov11_progress_callback progress_cb,
    void* user_data)
{
    /* ---------- parameter validation ---------- */
    if (engine_path == nullptr || std::strlen(engine_path) == 0 ||
        video_path  == nullptr || std::strlen(video_path)  == 0) {
        return YOLOV11_ERROR_PARAM;
    }

    const bool save_json  = (json_output_path  != nullptr &&
                             std::strlen(json_output_path)  > 0);
    const bool save_video = (output_video_path != nullptr &&
                             std::strlen(output_video_path) > 0);

    // Guard against overwriting inputs
    if (save_json &&
        (std::strcmp(json_output_path, engine_path) == 0 ||
         std::strcmp(json_output_path, video_path)  == 0 ||
         (save_video && std::strcmp(json_output_path, output_video_path) == 0))) {
        std::fprintf(stderr, "yolov11_detect_video: JSON output must not "
                     "overwrite an input file.\n");
        return YOLOV11_ERROR_PARAM;
    }

    if (!file_exists(engine_path)) {
        std::fprintf(stderr, "yolov11_detect_video: engine not found: %s\n",
                     engine_path);
        return YOLOV11_ERROR_FILE_OPEN;
    }

    try {
        /* ---------- open video ---------- */
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::fprintf(stderr, "yolov11_detect_video: cannot open video: %s\n",
                         video_path);
            return YOLOV11_ERROR_FILE_OPEN;
        }

        const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        std::printf("Video: %d frames\n", total_frames);

        auto t_start = std::chrono::steady_clock::now();

        /* ---------- create output directories ---------- */
        if (save_video && !ensure_parent_dir(output_video_path)) {
            cap.release();
            return YOLOV11_ERROR_FILE_WRITE;
        }
        if (save_json && !ensure_parent_dir(json_output_path)) {
            cap.release();
            return YOLOV11_ERROR_FILE_WRITE;
        }

        /* ---------- model ---------- */
        YOLOv11 model(engine_path, g_api_logger);
        if (conf_threshold > 0.0f) {
            model.setConfThreshold(conf_threshold);
            std::printf("Confidence threshold set to: %.2f\n", conf_threshold);
        }
        const int B = model.getBatchSize();
        const int stream_count = model.getStreamCount();

        /* ---------- video writer ---------- */
        cv::VideoWriter video_writer;
        if (save_video) {
            const int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
            const double fps = cap.get(cv::CAP_PROP_FPS);
            const int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            const int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            video_writer.open(output_video_path, codec, fps, cv::Size(w, h));
            if (!video_writer.isOpened()) {
                std::fprintf(stderr,
                    "yolov11_detect_video: cannot open output video: %s\n",
                    output_video_path);
                cap.release();
                return YOLOV11_ERROR_FILE_WRITE;
            }
            std::printf("Saving output to: %s  (%dx%d @ %.1f FPS)\n",
                        output_video_path, w, h, fps);
        }

        /* ---------- JSON output ---------- */
        std::ofstream json_out;
        bool first_json_frame = true;
        // 大缓冲区 (4 MB)：对于 3 小时视频 (~300 MB JSON) 仅触发约 75 次 write，
        // 每次约 10-15 ms，总计不到 1 秒，对推理管线几乎无影响。
        // 同时内存占用固定为 4 MB，不会随视频时长增长。
        std::vector<char> json_buf(4 * 1024 * 1024);  // 4 MB
        if (save_json) {
            json_out.open(json_output_path, std::ios::out | std::ios::trunc);
            if (!json_out.is_open()) {
                std::fprintf(stderr,
                    "yolov11_detect_video: cannot open JSON output: %s\n",
                    json_output_path);
                cap.release();
                video_writer.release();
                return YOLOV11_ERROR_FILE_WRITE;
            }
            json_out.rdbuf()->pubsetbuf(json_buf.data(), json_buf.size());
            json_out << "[\n"
                     << std::setprecision(std::numeric_limits<float>::max_digits10);
        }

        /* ---------- async decode pipeline ---------- */
        FrameQueue frame_queue(128);

        std::thread producer([&]() {
            cv::Mat frame;
            while (cap.read(frame)) {
                frame_queue.push(std::move(frame));
            }
            frame_queue.setDone();
        });

        /* ---------- batched inference ---------- */
        int frame_count = 0;
        int batch_count = 0;

        std::vector<cv::Mat> images_buf[2] = {
            std::vector<cv::Mat>(B),
            std::vector<cv::Mat>(B)
        };

        auto load_batch = [&](int slot) {
            int actual = 0;
            for (int b = 0; b < B; b++) {
                if (!frame_queue.pop(images_buf[slot][b])) break;
                actual++;
            }
            return actual;
        };

        auto launch_batch = [&](int slot, int actual) {
            for (int b = 0; b < actual; b++)
                model.preprocess(images_buf[slot][b], slot, b);
            model.infer(slot);
            batch_count++;
        };

        auto finish_batch = [&](int slot, int actual) {
            for (int b = 0; b < actual; b++) {
                std::vector<Detection> objects;
                model.postprocess(objects, slot, b);

                /* ---- JSON ---- */
                if (json_out.is_open()) {
                    const std::vector<Detection> detections =
                        model.mapDetectionsToOriginal(
                            images_buf[slot][b].size(), objects);

                    if (!first_json_frame) json_out << ",\n";
                    first_json_frame = false;

                    json_out << "  {\"frame_id\": " << frame_count
                             << ", \"Detect4in1\": [";
                    for (size_t i = 0; i < detections.size(); ++i) {
                        if (i != 0) json_out << ", ";
                        const Detection& d = detections[i];
                        json_out << "{\"cls\": " << d.class_id
                                 << ", \"conf\": " << d.conf
                                 << ", \"bbox\": ["
                                 << d.bbox.x << ", " << d.bbox.y << ", "
                                 << (d.bbox.x + d.bbox.width) << ", "
                                 << (d.bbox.y + d.bbox.height) << "]}";
                    }
                    json_out << "]}";
                }

                /* ---- annotated video ---- */
                if (video_writer.isOpened()) {
                    model.draw(images_buf[slot][b], objects);
                    video_writer.write(images_buf[slot][b]);
                }

                frame_count++;

                /* ---- progress ---- */
                if (progress_cb) {
                    progress_cb(frame_count, total_frames, user_data);
                }
            }
        };

        // Prime the pipeline
        int cur_slot = 0;
        int cur_actual = load_batch(cur_slot);
        if (cur_actual == 0) {
            producer.join();
            cap.release();
            video_writer.release();
            if (json_out.is_open()) { json_out << "]\n"; json_out.close(); }
            return YOLOV11_ERROR_NO_FRAME;
        }
        launch_batch(cur_slot, cur_actual);

        // Main loop
        if (stream_count == 1) {
            while (true) {
                finish_batch(cur_slot, cur_actual);
                cur_actual = load_batch(cur_slot);
                if (cur_actual == 0) break;
                launch_batch(cur_slot, cur_actual);
            }
        } else {
            while (true) {
                const int nxt_slot = 1 - cur_slot;
                const int nxt_actual = load_batch(nxt_slot);
                if (nxt_actual == 0) break;
                launch_batch(nxt_slot, nxt_actual);
                finish_batch(cur_slot, cur_actual);
                cur_slot   = nxt_slot;
                cur_actual = nxt_actual;
            }
            finish_batch(cur_slot, cur_actual);
        }

        /* ---------- cleanup ---------- */
        producer.join();
        cap.release();
        video_writer.release();
        if (json_out.is_open()) {
            json_out << "\n]\n";
            json_out.close();
            if (!json_out) {
                std::fprintf(stderr,
                    "yolov11_detect_video: error writing JSON output: %s\n",
                    json_output_path);
                return YOLOV11_ERROR_FILE_WRITE;
            }
            std::printf("Detection JSON saved to: %s (%d frames)\n",
                        json_output_path, frame_count);
        }
        if (save_video) {
            std::printf("Output video saved to: %s\n", output_video_path);
        }

        auto t_end = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double fps = frame_count > 0 ? 1000.0 * frame_count / elapsed_ms : 0.0;

        std::printf("yolov11_detect_video: done, %d frames, %.1f s, %.1f FPS\n",
                    frame_count, elapsed_ms / 1000.0, fps);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "yolov11_detect_video: exception: %s\n", e.what());
        return to_error_code(e);
    } catch (...) {
        std::fprintf(stderr, "yolov11_detect_video: unknown exception\n");
        return YOLOV11_ERROR_UNKNOWN;
    }

    return YOLOV11_OK;
}

/* =================================================================== */
/*  API: YoloTrackRefine – tracking-guided ball refinement             */
/* =================================================================== */

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
    void* user_data)
{
    (void)progress_cb;
    (void)user_data;

    if (events_json_path == nullptr || std::strlen(events_json_path) == 0 ||
        video_path       == nullptr || std::strlen(video_path)       == 0 ||
        ball_engine_path == nullptr || std::strlen(ball_engine_path) == 0 ||
        output_json_path == nullptr || std::strlen(output_json_path) == 0) {
        return YOLOV11_ERROR_PARAM;
    }

    if (detect_stride < 1) return YOLOV11_ERROR_PARAM;

    YoloTrackRefineConfig config;
    config.detectStride       = detect_stride;
    config.detConf            = det_conf;
    config.cropSize           = 640;
    config.maxLookback        = max_lookback;
    config.proximityThreshold = proximity_threshold;
    config.yoloConfThresh     = 0.0f;
    config.ballCls            = ball_cls;
    config.jsonBallCls        = json_ball_cls;

    try {
        int ret = YoloTrackRefine::run(
            events_json_path, video_path, ball_engine_path,
            output_json_path, config);
        return (ret == 0) ? YOLOV11_OK : YOLOV11_ERROR_UNKNOWN;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "yolov11_track_refine: exception: %s\n", e.what());
        return YOLOV11_ERROR_UNKNOWN;
    } catch (...) {
        std::fprintf(stderr, "yolov11_track_refine: unknown exception\n");
        return YOLOV11_ERROR_UNKNOWN;
    }
}
