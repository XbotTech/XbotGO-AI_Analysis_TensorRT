/**
 * @file exec_detect_video.cpp
 * @brief 全帧视频目标检测 CLI — 调用 yolov11_detect_video() C API。
 *
 * Build: 根目录 cmake .. && make exec_detect_video
 * Run:   ./build/exec_detect_video <engine> <video> [output.json] [output.mp4] [conf]
 */

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include "yolov11_api.h"

static void on_progress(int cur, int total, void* user) {
    (void)user;
    // if (total > 0) {
    //     std::printf("\r  Progress: %d / %d  (%.1f%%)", cur, total, 100.0 * cur / total);
    //     std::fflush(stdout);
    // } else {
    //     std::printf("\r  Progress: %d frames", cur);
    //     std::fflush(stdout);
    // }
    return;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Usage: %s <engine> <video> [output.json] [output.mp4] [conf]\n",
                     argv[0]);
        std::fprintf(stderr, "  Use \"None\" to skip an output file.\n");
        std::fprintf(stderr, "  conf: confidence threshold (0.0~1.0), default 0.3\n");
        return 1;
    }

    const char* engine   = argv[1];
    const char* video    = argv[2];
    const char* out_json = (argc > 3) ? argv[3] : nullptr;
    const char* out_mp4  = (argc > 4) ? argv[4] : nullptr;
    float conf_threshold = (argc > 5) ? std::atof(argv[5]) : 0.0f;

    // 如果参数是 "None"，则表示不保存对应的输出文件。
    if (out_json && std::strcmp(out_json, "None") == 0) {
        out_json = nullptr;
    }
    if (out_mp4 && std::strcmp(out_mp4, "None") == 0) {
        out_mp4 = nullptr;
    }

    // 未指定 conf 时使用 API 内部默认值 (0.3)
    if (conf_threshold <= 0.0f) {
        conf_threshold = 0.3f;
    }

    std::cout << "=== YOLOv11 TensorRT Demo ===" << std::endl;
    std::cout << "  engine:   " << engine << std::endl;
    std::cout << "  video:    " << video << std::endl;
    std::cout << "  out_json: " << (out_json ? out_json : "(none)") << std::endl;
    std::cout << "  out_mp4:  " << (out_mp4  ? out_mp4  : "(none)") << std::endl;
    std::cout << "  conf:     " << conf_threshold << std::endl;
    std::cout << "==============================" << std::endl;

    auto t_start = std::chrono::steady_clock::now();

    yolov11_error_t ret = yolov11_detect_video(
        engine, video, out_json, out_mp4, conf_threshold, on_progress, nullptr);

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    std::cout << std::endl;

    if (ret == YOLOV11_OK) {
        std::cout << "Done." << std::endl;
    } else {
        std::cerr << "Error: " << ret << std::endl;
    }

    std::cout << "Total time: " << elapsed_ms / 1000.0 << " s  ("
              << elapsed_ms << " ms)" << std::endl;
    return (ret == YOLOV11_OK) ? 0 : 1;
}
