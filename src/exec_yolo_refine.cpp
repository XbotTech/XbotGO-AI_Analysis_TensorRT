/**
 * @file exec_yolo_refine.cpp
 * @brief 跟踪引导篮球检测细化 CLI — 直接调用 YoloTrackRefine::run()。
 *
 * Build: 根目录 cmake .. && make exec_yolo_refine
 * Run:   ./build/exec_yolo_refine <json> <video> <engine> [output.json] [options]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include "YoloTrackRefine.h"

static void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s <json> <video> <engine> [output.json] [options]\n"
        "\n"
        "  <json>         Path to JsonForLLM_with_objects.json\n"
        "  <video>        Path to input video (mp4)\n"
        "  <engine>       Path to 640x640 ball detection .engine file\n"
        "  [output.json]  Output path (default: <json_dir>/YoloRefineBallboxes_cpp.json)\n"
        "\n"
        "Options:\n"
        "  --stride N     Frame stride (default: 2)\n"
        "  --conf C       Re-detect confidence threshold (default: 0.25)\n"
        "  --lookback N   Max lookback frames (default: 5)\n"
        "  --proximity P  Proximity threshold in pixels (default: 150)\n"
        "  --ball-cls N   Class index for ball in crop model (default: 1)\n"
        "  --json-ball-cls N  Class index for ball in input JSON (default: 1)\n",
        prog);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    std::string jsonInput  = argv[1];
    std::string videoPath  = argv[2];
    std::string enginePath = argv[3];

    // Parse optional positional output json
    std::string outputJson;
    int posIdx = 4;
    if (argc > posIdx && argv[posIdx][0] != '-') {
        outputJson = argv[posIdx++];
    } else {
        std::filesystem::path inPath(jsonInput);
        outputJson = (inPath.parent_path() / "YoloRefineBallboxes.json").string();
    }

    // Defaults
    int   detectStride       = 1;
    float detConf            = 0.25f;
    int   maxLookback        = 5;
    float proximityThreshold = 150.0f;
    int   ballCls            = 1;
    int   jsonBallCls        = 1;

    // Parse flags
    for (int i = posIdx; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--stride" && i + 1 < argc)
            detectStride = std::atoi(argv[++i]);
        else if (arg == "--conf" && i + 1 < argc)
            detConf = std::atof(argv[++i]);
        else if (arg == "--lookback" && i + 1 < argc)
            maxLookback = std::atoi(argv[++i]);
        else if (arg == "--proximity" && i + 1 < argc)
            proximityThreshold = std::atof(argv[++i]);
        else if (arg == "--ball-cls" && i + 1 < argc)
            ballCls = std::atoi(argv[++i]);
        else if (arg == "--json-ball-cls" && i + 1 < argc)
            jsonBallCls = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    // Print config
    std::cout << "=== YoloTrackRefine C++ ===" << std::endl;
    std::cout << "  json:     " << jsonInput << std::endl;
    std::cout << "  video:    " << videoPath << std::endl;
    std::cout << "  engine:   " << enginePath << std::endl;
    std::cout << "  output:   " << outputJson << std::endl;
    std::cout << "  stride:   " << detectStride << std::endl;
    std::cout << "  conf:     " << detConf << std::endl;
    std::cout << "  lookback: " << maxLookback << std::endl;
    std::cout << "==========================" << std::endl;

    YoloTrackRefineConfig config;
    config.detectStride       = detectStride;
    config.detConf            = detConf;
    config.maxLookback        = maxLookback;
    config.proximityThreshold = proximityThreshold;
    config.ballCls            = ballCls;
    config.jsonBallCls        = jsonBallCls;

    int ret = YoloTrackRefine::run(
        jsonInput.c_str(),
        videoPath.c_str(),
        enginePath.c_str(),
        outputJson.c_str(),
        config);

    if (ret != 0) {
        std::fprintf(stderr, "Error: %d\n", ret);
        return 1;
    }
    return 0;
}
