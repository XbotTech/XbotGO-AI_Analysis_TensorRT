#include <cstdio>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <utility>
#include <vector>
#include "YOLOv11.h"


struct FrameQueueStats {
    double pop_wait_ms = 0.0;
    double pop_wait_p95_ms = 0.0;
    double pop_wait_p99_ms = 0.0;
    double pop_wait_max_ms = 0.0;
    double push_wait_ms = 0.0;
    double push_wait_max_ms = 0.0;
    double average_depth = 0.0;
    uint64_t pop_wait_events = 0;
    uint64_t push_wait_events = 0;
    uint64_t pushed_frames = 0;
    uint64_t popped_frames = 0;
    size_t max_depth = 0;
    size_t capacity = 0;
};


struct VideoProducerStats {
    double read_decode_ms = 0.0;
    double read_decode_max_ms = 0.0;
    double wall_ms = 0.0;
    uint64_t decoded_frames = 0;
};


/**
 * @brief Thread-safe frame queue with back-pressure.
 *
 * Producer thread reads frames from video and pushes them into the queue.
 * Consumer (main thread) pops frames for inference.
 * This decouples I/O from GPU computation: while GPU is busy, the producer
 * pre-reads frames and reduces consumer stalls when decoding can keep up.
 */
class FrameQueue {
public:
    explicit FrameQueue(size_t max_size = 64)
        : max_size_(max_size),
          stats_start_(StatsClock::now()),
          last_depth_update_(stats_start_) {}

    /**
     * @brief Push a frame into the queue (called by producer thread).
     *        Blocks if the queue is full (back-pressure to prevent memory blowup).
     */
    void push(cv::Mat frame) {
        std::unique_lock<std::mutex> lock(mtx_);

        if (q_.size() >= max_size_ && !done_) {
            const auto wait_start = StatsClock::now();
            push_wait_events_++;
            cv_not_full_.wait(lock, [this] { return q_.size() < max_size_ || done_; });
            const double wait_ms = elapsedMs(wait_start, StatsClock::now());
            push_wait_ms_ += wait_ms;
            push_wait_max_ms_ = std::max(push_wait_max_ms_, wait_ms);
        }

        if (done_) return;  // Shouldn't happen, but safety

        updateDepthAreaLocked(StatsClock::now());
        q_.push(std::move(frame));
        pushed_frames_++;
        max_depth_ = std::max(max_depth_, q_.size());
        cv_not_empty_.notify_one();  // Wake up consumer
    }

    /**
     * @brief Pop a frame from the queue (called by consumer/main thread).
     * @return true if a frame was popped, false if producer is done and queue is empty.
     */
    bool pop(cv::Mat& frame) {
        std::unique_lock<std::mutex> lock(mtx_);

        double wait_ms = 0.0;
        if (q_.empty() && !done_) {
            const auto wait_start = StatsClock::now();
            pop_wait_events_++;
            cv_not_empty_.wait(lock, [this] { return !q_.empty() || done_; });
            wait_ms = elapsedMs(wait_start, StatsClock::now());
            pop_wait_ms_ += wait_ms;
            pop_wait_max_ms_ = std::max(pop_wait_max_ms_, wait_ms);
        }

        if (q_.empty() && done_) return false;  // All frames consumed

        updateDepthAreaLocked(StatsClock::now());
        frame = std::move(q_.front());
        q_.pop();
        popped_frames_++;
        pop_wait_samples_ms_.push_back(wait_ms);
        cv_not_full_.notify_one();  // Wake up producer (room available)
        return true;
    }

    /**
     * @brief Signal that no more frames will be pushed.
     *        Wakes up consumer so it can drain remaining frames and exit.
     */
    void setDone() {
        std::unique_lock<std::mutex> lock(mtx_);
        updateDepthAreaLocked(StatsClock::now());
        done_ = true;
        cv_not_empty_.notify_all();
    }

    FrameQueueStats stats() {
        std::unique_lock<std::mutex> lock(mtx_);
        const auto now = StatsClock::now();
        updateDepthAreaLocked(now);

        FrameQueueStats result;
        result.pop_wait_ms = pop_wait_ms_;
        result.pop_wait_max_ms = pop_wait_max_ms_;
        result.push_wait_ms = push_wait_ms_;
        result.push_wait_max_ms = push_wait_max_ms_;
        result.pop_wait_events = pop_wait_events_;
        result.push_wait_events = push_wait_events_;
        result.pushed_frames = pushed_frames_;
        result.popped_frames = popped_frames_;
        result.max_depth = max_depth_;
        result.capacity = max_size_;

        const double elapsed_ms = elapsedMs(stats_start_, now);
        result.average_depth = elapsed_ms > 0.0 ? depth_time_area_ / elapsed_ms : 0.0;

        if (!pop_wait_samples_ms_.empty()) {
            std::vector<double> sorted_samples = pop_wait_samples_ms_;
            std::sort(sorted_samples.begin(), sorted_samples.end());
            result.pop_wait_p95_ms = percentile(sorted_samples, 0.95);
            result.pop_wait_p99_ms = percentile(sorted_samples, 0.99);
        }
        return result;
    }

private:
    using StatsClock = std::chrono::steady_clock;

    static double elapsedMs(StatsClock::time_point start, StatsClock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    static double percentile(const std::vector<double>& sorted_samples, double quantile) {
        const size_t index = static_cast<size_t>(
            std::ceil(quantile * static_cast<double>(sorted_samples.size()))) - 1;
        return sorted_samples[std::min(index, sorted_samples.size() - 1)];
    }

    void updateDepthAreaLocked(StatsClock::time_point now) {
        const double elapsed_ms = elapsedMs(last_depth_update_, now);
        depth_time_area_ += elapsed_ms * static_cast<double>(q_.size());
        last_depth_update_ = now;
    }

    std::queue<cv::Mat> q_;
    std::mutex mtx_;
    std::condition_variable cv_not_empty_;  // Consumer waits on this
    std::condition_variable cv_not_full_;   // Producer waits on this (back-pressure)
    size_t max_size_;
    bool done_ = false;

    StatsClock::time_point stats_start_;
    StatsClock::time_point last_depth_update_;
    double depth_time_area_ = 0.0;  // Integral of queue depth over milliseconds.
    double pop_wait_ms_ = 0.0;
    double pop_wait_max_ms_ = 0.0;
    double push_wait_ms_ = 0.0;
    double push_wait_max_ms_ = 0.0;
    uint64_t pop_wait_events_ = 0;
    uint64_t push_wait_events_ = 0;
    uint64_t pushed_frames_ = 0;
    uint64_t popped_frames_ = 0;
    size_t max_depth_ = 0;
    std::vector<double> pop_wait_samples_ms_;
};


bool IsPathExist(const string& path) {
#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    return (access(path.c_str(), F_OK) == 0);
#endif
}
bool IsFile(const string& path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

#ifdef _WIN32
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}

/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

static void printDetailedTiming(const YOLOv11& model)
{
    if (!model.detailedTimingEnabled()) return;

    const DetailedTimingStats& timing = model.getDetailedTimingStats();
    if (timing.frames == 0 || timing.batches == 0) {
        printf("Detailed timing: no completed frames\n");
        return;
    }

    const double frames = static_cast<double>(timing.frames);
    const double batches = static_cast<double>(timing.batches);
    printf("--- Detailed timing (CUDA events; no extra synchronization) ---\n");
    printf("  GPU preprocess:       %.3f ms/batch  (%.3f ms/frame)\n",
           timing.preprocess_gpu_ms / batches, timing.preprocess_gpu_ms / frames);
    printf("  GPU TensorRT infer:   %.3f ms/batch  (%.3f ms/frame)\n",
           timing.inference_gpu_ms / batches, timing.inference_gpu_ms / frames);
    printf("  GPU post decode:      %.3f ms/frame\n",
           timing.post_decode_gpu_ms / frames);
    printf("  GPU result D2H:       %.3f ms/frame\n",
           timing.post_copy_gpu_ms / frames);
    printf("  CPU stream wait:      %.3f ms/frame\n",
           timing.post_wait_cpu_ms / frames);
    printf("  CPU result parse/NMS: %.3f ms/frame\n",
           timing.post_cpu_ms / frames);
    printf("  Timed work:           %llu batches, %llu frames\n",
           static_cast<unsigned long long>(timing.batches),
           static_cast<unsigned long long>(timing.frames));
}

static void printUsage(const char* executable)
{
    printf("Usage:\n");
    printf("  Build engine:    %s <model.onnx>\n", executable);
    printf("  Run inference:   %s <model.engine|model.onnx> <video_or_image> [output.mp4]\n",
           executable);
    printf("                   [--json <detect.json>]\n");
    printf("  JSON schema:     XbotGo pipeline step-1 detect.json\n");
    printf("  Stream mode:     YOLO_NUM_STREAMS=1|2 (default: 2)\n");
    printf("  Detailed timing: YOLO_DETAILED_TIMING=1\n");
}

int main(int argc, char** argv)
{
    // Usage check
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    // 统计最最开始的时间
    auto program_start_time = std::chrono::system_clock::now();
    const string engine_file_path{ argv[1] };

    // Optional inference outputs/options. The first positional argument after
    // the input remains the legacy output-video path.
    string output_path;
    string json_output_path;

    // ---- Mode 1: Build engine only (1 argument, .onnx file) ----
    bool is_onnx = engine_file_path.find(".onnx") != std::string::npos;
    if (argc == 2) {
        if (!is_onnx) {
            printf("Error: single argument must be an .onnx file to build engine.\n");
            return 1;
        }
        printf("Building TensorRT engine from %s ...\n", engine_file_path.c_str());
        YOLOv11 model(engine_file_path, logger);
        printf("Engine built successfully.\n");
        printf("Program runtime: %.0f ms\n",
            (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now() - program_start_time).count());
        return 0;
    }

    for (int i = 3; i < argc; ++i) {
        const string argument(argv[i]);
        if (argument == "--json") {
            if (i + 1 >= argc || json_output_path.size() != 0) {
                fprintf(stderr, "Error: --json requires exactly one output path.\n");
                printUsage(argv[0]);
                return 1;
            }
            json_output_path = argv[++i];
        }
        else if (argument.rfind("--", 0) == 0) {
            fprintf(stderr, "Error: unknown option '%s'.\n", argument.c_str());
            printUsage(argv[0]);
            return 1;
        }
        else if (output_path.empty()) {
            output_path = argument;
        }
        else {
            fprintf(stderr, "Error: unexpected positional argument '%s'.\n",
                    argument.c_str());
            printUsage(argv[0]);
            return 1;
        }
    }
    const bool save_output = !output_path.empty();

    // ---- Mode 2: Inference (2 arguments) ----
    const string path{ argv[2] };
    vector<string> imagePathList;
    bool isVideo{ false };

    if (IsFile(path))
    {
        string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png")
        {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov" || suffix == "mkv" || suffix == "webm")
        {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            abort();
        }
    }
    else if (IsPathExist(path))
    {
        glob(path + "/*.jpg", imagePathList);
    }

    if (!isVideo && imagePathList.empty()) {
        fprintf(stderr, "Error: no readable video or image input found at %s\n", path.c_str());
        return 1;
    }

    if (!json_output_path.empty() && !isVideo) {
        fprintf(stderr, "Error: --json is only supported for video input.\n");
        return 1;
    }
    if (!json_output_path.empty() &&
        (json_output_path == engine_file_path ||
         json_output_path == path ||
         json_output_path == output_path)) {
        fprintf(stderr, "Error: JSON output must not overwrite an input or output file.\n");
        return 1;
    }

    YOLOv11 model(engine_file_path, logger);

    if (isVideo) {
        cv::VideoCapture cap(path);
        if (!cap.isOpened()) {
            printf("Error: Cannot open video: %s\n", path.c_str());
            return 1;
        }
        int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
        printf("Video: %d frames\n", total_frames);

        // Setup video writer if saving output
        cv::VideoWriter video_writer;
        if (save_output) {
            int codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');  // H.264
            double out_fps = cap.get(cv::CAP_PROP_FPS);
            int out_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
            int out_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            video_writer.open(output_path, codec, out_fps, cv::Size(out_w, out_h));
            if (!video_writer.isOpened()) {
                printf("Error: Cannot open output video: %s\n", output_path.c_str());
                return 1;
            }
            printf("Saving output to: %s  (%dx%d @ %.1f FPS)\n",
                   output_path.c_str(), out_w, out_h, out_fps);
        }

        std::ofstream json_output;
        bool first_json_frame = true;
        if (!json_output_path.empty()) {
            const std::filesystem::path parent =
                std::filesystem::path(json_output_path).parent_path();
            if (!parent.empty()) {
                std::error_code error;
                std::filesystem::create_directories(parent, error);
                if (error) {
                    fprintf(stderr, "Error: cannot create JSON output directory: %s\n",
                            error.message().c_str());
                    return 1;
                }
            }

            json_output.open(json_output_path, std::ios::out | std::ios::trunc);
            if (!json_output.is_open()) {
                fprintf(stderr, "Error: cannot open JSON output: %s\n",
                        json_output_path.c_str());
                return 1;
            }
            json_output << "[\n"
                        << std::setprecision(std::numeric_limits<float>::max_digits10);
        }

        // ================================================================
        //  Async frame decoding: producer thread reads frames ahead of time
        //  to reduce GPU pipeline waits when video decoding can keep up.
        // ================================================================
        FrameQueue frame_queue(128);  // Buffer up to 128 frames (~2GB for 1920x1920)
        VideoProducerStats producer_stats;

        // Producer thread: continuously read frames from video
        std::thread producer([&]() {
            const auto producer_start = std::chrono::steady_clock::now();
            cv::Mat frame;
            while (true) {
                const auto read_start = std::chrono::steady_clock::now();
                const bool read_ok = cap.read(frame);
                const auto read_end = std::chrono::steady_clock::now();
                if (!read_ok) break;

                const double read_ms =
                    std::chrono::duration<double, std::milli>(read_end - read_start).count();
                producer_stats.read_decode_ms += read_ms;
                producer_stats.read_decode_max_ms =
                    std::max(producer_stats.read_decode_max_ms, read_ms);
                producer_stats.decoded_frames++;
                frame_queue.push(std::move(frame));
            }
            frame_queue.setDone();  // Signal consumer: no more frames
            producer_stats.wall_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - producer_start).count();
        });

        // ---- Configurable single/dual-stream batched inference ----
        double total_pre_ms = 0, total_inf_ms = 0, total_post_ms = 0;
        double total_draw_ms = 0, total_write_ms = 0;
        int frame_count = 0;
        int batch_count = 0;
        const int B = model.getBatchSize();
        const int stream_count = model.getStreamCount();

        // Wall-clock timer: measures real end-to-end throughput including I/O
        auto inference_wall_start = std::chrono::system_clock::now();
        // Two slots are allocated; single-stream mode only uses slot 0.
        vector<Mat> images_buf[2] = {vector<Mat>(B), vector<Mat>(B)};

        auto load_batch = [&](int slot) {
            int actual = 0;
            for (int b = 0; b < B; b++) {
                if (!frame_queue.pop(images_buf[slot][b])) break;
                actual++;
            }
            return actual;
        };

        auto launch_batch = [&](int slot, int actual) {
            auto pre_start = std::chrono::system_clock::now();
            for (int b = 0; b < actual; b++)
                model.preprocess(images_buf[slot][b], slot, b);
            auto pre_end = std::chrono::system_clock::now();
            model.infer(slot);
            auto infer_end = std::chrono::system_clock::now();
            batch_count++;
            total_pre_ms += (double)std::chrono::duration_cast<std::chrono::microseconds>(pre_end - pre_start).count() / 1000.;
            total_inf_ms += (double)std::chrono::duration_cast<std::chrono::microseconds>(infer_end - pre_end).count() / 1000.;
        };

        auto finish_batch = [&](int slot, int actual) {
            auto tp0 = std::chrono::system_clock::now();
            const double draw_ms_before = total_draw_ms;
            const double write_ms_before = total_write_ms;
            for (int b = 0; b < actual; b++) {
                vector<Detection> objects;
                model.postprocess(objects, slot, b);

                if (json_output.is_open()) {
                    const vector<Detection> detections =
                        model.mapDetectionsToOriginal(images_buf[slot][b].size(), objects);

                    if (!first_json_frame) {
                        json_output << ",\n";
                    }
                    first_json_frame = false;

                    json_output << "  {\"frame_id\": " << frame_count + b
                                << ", \"Detect4in1\": [";
                    for (size_t i = 0; i < detections.size(); ++i) {
                        if (i != 0) {
                            json_output << ", ";
                        }
                        const Detection& detection = detections[i];
                        const float x2 = detection.bbox.x + detection.bbox.width;
                        const float y2 = detection.bbox.y + detection.bbox.height;

                        json_output << "{\"cls\": " << detection.class_id
                                    << ", \"conf\": " << detection.conf
                                    << ", \"bbox\": ["
                                    << detection.bbox.x << ", "
                                    << detection.bbox.y << ", "
                                    << x2 << ", " << y2 << "]}";
                    }
                    json_output << "]}";
                }

                if (save_output) {
                    auto draw_start = std::chrono::steady_clock::now();
                    model.draw(images_buf[slot][b], objects);
                    auto draw_end = std::chrono::steady_clock::now();
                    video_writer.write(images_buf[slot][b]);
                    auto write_end = std::chrono::steady_clock::now();
                    total_draw_ms += std::chrono::duration<double, std::milli>(draw_end - draw_start).count();
                    total_write_ms += std::chrono::duration<double, std::milli>(write_end - draw_end).count();
                }
            }
            auto tp1 = std::chrono::system_clock::now();
            total_post_ms += (double)std::chrono::duration_cast<std::chrono::microseconds>(tp1 - tp0).count() / 1000.
                           - (total_draw_ms - draw_ms_before)
                           - (total_write_ms - write_ms_before);
            frame_count += actual;
        };

        int cur_slot = 0;
        int cur_actual = load_batch(cur_slot);
        if (cur_actual == 0) {
            producer.join();
            cap.release();
            if (json_output.is_open()) {
                json_output << "]\n";
                json_output.close();
            }
            return 0;
        }
        launch_batch(cur_slot, cur_actual);

        if (stream_count == 1) {
            // A slot cannot be reused until its output has been consumed.
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

                // Let the other stream run while the current batch is consumed.
                launch_batch(nxt_slot, nxt_actual);
                finish_batch(cur_slot, cur_actual);
                cur_slot = nxt_slot;
                cur_actual = nxt_actual;
            }
            finish_batch(cur_slot, cur_actual);
        }

        // Cleanup: wait for producer to finish, release resources
        producer.join();
        cap.release();
        if (save_output) {
            video_writer.release();
        }
        if (json_output.is_open()) {
            json_output << "\n]\n";
            json_output.close();
            if (!json_output) {
                fprintf(stderr, "Error: failed to write JSON output: %s\n",
                        json_output_path.c_str());
                return 1;
            }
            printf("Detection JSON saved to: %s (%d frames)\n",
                   json_output_path.c_str(), frame_count);
        }

        // Stop end-to-end timing before sorting/reporting diagnostic samples.
        auto inference_wall_end = std::chrono::system_clock::now();
        double wall_sec = std::chrono::duration<double>(inference_wall_end - inference_wall_start).count();
        const FrameQueueStats queue_stats = frame_queue.stats();

        if (save_output) {
            printf("Output saved to: %s\n", output_path.c_str());
        }

        printf("--- %s-stream + Batch%d + AsyncDecode (%d frames) ---\n",
               stream_count == 1 ? "Single" : "Dual", B, frame_count);
        printf("  preprocess host submit: %.2f ms/frame\n", total_pre_ms / frame_count);
        printf("  inference enqueue:      %.2f ms/batch  (%.2f ms/frame)\n",
               total_inf_ms / batch_count, total_inf_ms / frame_count);
        printf("  postprocess wall:       %.2f ms/frame\n", total_post_ms / frame_count);
        if (save_output) {
            printf("  draw:                   %.2f ms/frame\n", total_draw_ms / frame_count);
            printf("  video write:            %.2f ms/frame\n", total_write_ms / frame_count);
        }
        printf("  stage wall sum:         %.2f ms/frame  (%.1f FPS equivalent)\n",
               (total_pre_ms + total_inf_ms + total_post_ms) / frame_count,
               1000.0 * frame_count / (total_pre_ms + total_inf_ms + total_post_ms));
        printf("--- Queue / producer diagnostics ---\n");
        printf("  queue_pop_wait_ms/frame:  %.3f  (%.1f%% wall; %llu empty waits, p95 %.3f, p99 %.3f, max %.3f ms)\n",
               queue_stats.pop_wait_ms / frame_count,
               100.0 * queue_stats.pop_wait_ms / (wall_sec * 1000.0),
               static_cast<unsigned long long>(queue_stats.pop_wait_events),
               queue_stats.pop_wait_p95_ms,
               queue_stats.pop_wait_p99_ms,
               queue_stats.pop_wait_max_ms);
        printf("  queue_push_wait_ms/frame: %.3f  (%.1f%% producer wall; %llu full waits, max %.3f ms)\n",
               producer_stats.decoded_frames > 0
                   ? queue_stats.push_wait_ms / producer_stats.decoded_frames : 0.0,
               producer_stats.wall_ms > 0.0
                   ? 100.0 * queue_stats.push_wait_ms / producer_stats.wall_ms : 0.0,
               static_cast<unsigned long long>(queue_stats.push_wait_events),
               queue_stats.push_wait_max_ms);
        printf("  queue depth average/max:  %.2f / %zu frames  (capacity %zu, pushed/popped %llu/%llu)\n",
               queue_stats.average_depth,
               queue_stats.max_depth,
               queue_stats.capacity,
               static_cast<unsigned long long>(queue_stats.pushed_frames),
               static_cast<unsigned long long>(queue_stats.popped_frames));

        const double decoded_frames = static_cast<double>(producer_stats.decoded_frames);
        const double read_ms_per_frame = decoded_frames > 0.0
            ? producer_stats.read_decode_ms / decoded_frames : 0.0;
        const double producer_ms_per_frame = decoded_frames > 0.0
            ? producer_stats.wall_ms / decoded_frames : 0.0;
        printf("  video read/decode:         %.3f ms/frame  (%.1f FPS, max %.3f ms)\n",
               read_ms_per_frame,
               producer_stats.read_decode_ms > 0.0
                   ? 1000.0 * decoded_frames / producer_stats.read_decode_ms : 0.0,
               producer_stats.read_decode_max_ms);
        printf("  producer effective:        %.3f ms/frame  (%.1f FPS incl. queue back-pressure)\n",
               producer_ms_per_frame,
               producer_stats.wall_ms > 0.0
                   ? 1000.0 * decoded_frames / producer_stats.wall_ms : 0.0);
        printf("  batch fill average:        %.3f/%d frames  (%.1f%%)\n",
               static_cast<double>(frame_count) / batch_count,
               B,
               100.0 * frame_count / (static_cast<double>(batch_count) * B));
        printDetailedTiming(model);

        // Real end-to-end throughput: wall clock from first frame to last frame
        printf("  real:        %.2f ms/frame  (%.1f end-to-end FPS, %.1fs wall clock)\n",
               1000.0 * wall_sec / frame_count, frame_count / wall_sec, wall_sec);
    }
    else if (!imagePathList.empty()) {
        // ============================================================
        //  Image mode: process a list of images (from folder or single)
        //  Uses same dual-stream + batch pipeline; images preloaded into queue
        // ============================================================
        FrameQueue frame_queue(128);

        // Producer thread: load images from disk
        std::thread producer([&]() {
            for (const auto& imgPath : imagePathList) {
                cv::Mat img = cv::imread(imgPath);
                if (!img.empty()) frame_queue.push(std::move(img));
            }
            frame_queue.setDone();
        });

        double total_pre_ms = 0, total_inf_ms = 0, total_post_ms = 0;
        int frame_count = 0;
        int batch_count = 0;
        const int B = model.getBatchSize();
        const int stream_count = model.getStreamCount();
        auto inference_wall_start = std::chrono::system_clock::now();
        vector<Mat> images_buf[2] = {vector<Mat>(B), vector<Mat>(B)};

        auto load_batch = [&](int slot) {
            int actual = 0;
            for (int b = 0; b < B; b++) {
                if (!frame_queue.pop(images_buf[slot][b])) break;
                actual++;
            }
            return actual;
        };

        auto launch_batch = [&](int slot, int actual) {
            auto pre_start = std::chrono::system_clock::now();
            for (int b = 0; b < actual; b++)
                model.preprocess(images_buf[slot][b], slot, b);
            auto pre_end = std::chrono::system_clock::now();
            model.infer(slot);
            auto infer_end = std::chrono::system_clock::now();
            batch_count++;
            total_pre_ms += (double)std::chrono::duration_cast<std::chrono::microseconds>(pre_end - pre_start).count() / 1000.;
            total_inf_ms += (double)std::chrono::duration_cast<std::chrono::microseconds>(infer_end - pre_end).count() / 1000.;
        };

        auto finish_batch = [&](int slot, int actual) {
            auto tp0 = std::chrono::system_clock::now();
            for (int b = 0; b < actual; b++) {
                vector<Detection> objects;
                model.postprocess(objects, slot, b);
            }
            auto tp1 = std::chrono::system_clock::now();
            total_post_ms += (double)std::chrono::duration_cast<std::chrono::microseconds>(tp1 - tp0).count() / 1000.;
            frame_count += actual;
        };

        int cur_slot = 0;
        int cur_actual = load_batch(cur_slot);
        if (cur_actual == 0) { producer.join(); return 0; }
        launch_batch(cur_slot, cur_actual);

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
                cur_slot = nxt_slot;
                cur_actual = nxt_actual;
            }
            finish_batch(cur_slot, cur_actual);
        }

        producer.join();

        auto inference_wall_end = std::chrono::system_clock::now();
        double wall_sec = std::chrono::duration<double>(inference_wall_end - inference_wall_start).count();
        const FrameQueueStats queue_stats = frame_queue.stats();

        printf("--- %s-stream + Batch%d + Images (%d frames) ---\n",
               stream_count == 1 ? "Single" : "Dual", B, frame_count);
        printf("  preprocess host submit: %.2f ms/frame\n", total_pre_ms / frame_count);
        printf("  inference enqueue:      %.2f ms/batch  (%.2f ms/frame)\n",
               total_inf_ms / batch_count, total_inf_ms / frame_count);
        printf("  postprocess wall:       %.2f ms/frame\n", total_post_ms / frame_count);
        printf("  stage wall sum:         %.2f ms/frame  (%.1f FPS equivalent)\n",
               (total_pre_ms + total_inf_ms + total_post_ms) / frame_count,
               1000.0 * frame_count / (total_pre_ms + total_inf_ms + total_post_ms));
        printf("--- Queue diagnostics ---\n");
        printf("  queue_pop_wait_ms/frame:  %.3f  (%.1f%% wall; %llu empty waits, p95 %.3f, p99 %.3f, max %.3f ms)\n",
               queue_stats.pop_wait_ms / frame_count,
               100.0 * queue_stats.pop_wait_ms / (wall_sec * 1000.0),
               static_cast<unsigned long long>(queue_stats.pop_wait_events),
               queue_stats.pop_wait_p95_ms,
               queue_stats.pop_wait_p99_ms,
               queue_stats.pop_wait_max_ms);
        printf("  queue_push_wait_ms/frame: %.3f  (%llu full waits, max %.3f ms)\n",
               queue_stats.push_wait_ms / frame_count,
               static_cast<unsigned long long>(queue_stats.push_wait_events),
               queue_stats.push_wait_max_ms);
        printf("  queue depth average/max:  %.2f / %zu frames  (capacity %zu, pushed/popped %llu/%llu)\n",
               queue_stats.average_depth,
               queue_stats.max_depth,
               queue_stats.capacity,
               static_cast<unsigned long long>(queue_stats.pushed_frames),
               static_cast<unsigned long long>(queue_stats.popped_frames));
        printf("  batch fill average:        %.3f/%d frames  (%.1f%%)\n",
               static_cast<double>(frame_count) / batch_count,
               B,
               100.0 * frame_count / (static_cast<double>(batch_count) * B));
        printDetailedTiming(model);

        printf("  real:        %.2f ms/frame  (%.1f end-to-end FPS, %.1fs wall clock)\n",
               1000.0 * wall_sec / frame_count, frame_count / wall_sec, wall_sec);
    }
    else{
        printf("not video\n");
        return 0;
    }

    auto program_end_time = std::chrono::system_clock::now();
    auto program_duration = std::chrono::duration_cast<std::chrono::milliseconds>(program_end_time - program_start_time).count();
    printf("Program runtime: %ld ms\n", program_duration);
    return 0;
}
