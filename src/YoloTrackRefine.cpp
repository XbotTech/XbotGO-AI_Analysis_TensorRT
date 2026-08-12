/**
 * @file YoloTrackRefine.cpp
 * @brief Implementation of tracking-guided ball refinement.
 */

#include "YoloTrackRefine.h"
#include "YOLOv11.h"
#include "logging.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// nlohmann/json – header-only, included privately (not exposed in .h)
#include "json.hpp"
using json = nlohmann::json;

// ============================================================================
// Box utilities
// ============================================================================

float YoloTrackRefine::distance(const BoxXYXY& a, const BoxXYXY& b) {
    float dx = a.cx() - b.cx();
    float dy = a.cy() - b.cy();
    return std::sqrt(dx * dx + dy * dy);
}

float YoloTrackRefine::iou(const BoxXYXY& a, const BoxXYXY& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    if (ix1 >= ix2 || iy1 >= iy2) return 0.0f;
    float inter = (ix2 - ix1) * (iy2 - iy1);
    float areaA = a.w() * a.h();
    float areaB = b.w() * b.h();
    float uni = areaA + areaB - inter;
    return (uni > 0.0f) ? inter / uni : 0.0f;
}

std::vector<int> YoloTrackRefine::nmsIndices(const std::vector<BoxXYXY>& boxes,
                                              float iouThreshold) {
    // 标准贪心 NMS：按面积降序排列，依次保留未被高 IoU 抑制的框。
    // 注：这里用面积作为排序依据（而非置信度），因为某些场景下我们没有置信度信息。
    std::vector<int> keep;
    int n = static_cast<int>(boxes.size());
    if (n == 0) return keep;

    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int i, int j) {
        return boxes[i].w() * boxes[i].h() > boxes[j].w() * boxes[j].h();
    });

    std::vector<bool> suppressed(n, false);
    for (int i : order) {
        if (suppressed[i]) continue;
        keep.push_back(i);
        for (int j : order) {
            if (i == j || suppressed[j]) continue;
            if (iou(boxes[i], boxes[j]) > iouThreshold)
                suppressed[j] = true;
        }
    }
    return keep;
}

// ============================================================================
// Motion prediction (速度估计 + 线性外推)
// ============================================================================
//
// 思路：取当前框的前一检测帧（srcFid - detectStride），找到与其最近的框，
// 计算像素/帧的速度，然后线性外推到目标帧。
//
//   velocity = (current_position - prev_position) / detectStride
//   predicted = current_position + velocity * (targetFid - srcFid) / detectStride
//
// 若无法在前一帧找到匹配框（距离 > 2×proximityThreshold），回退到原框位置
//（即假设球没有移动）。
//

BoxXYXY YoloTrackRefine::predictPosition(
    const BoxXYXY& box, int srcFid, int targetFid, int detectStride,
    const std::map<int, std::vector<BoxXYXY>>& refinedBoxes,
    float proximityThreshold)
{
    int prevFid = srcFid - detectStride;
    auto it = refinedBoxes.find(prevFid);
    if (it == refinedBoxes.end()) return box;

    const auto& prevBoxes = it->second;
    float cx = box.cx(), cy = box.cy();
    float bw = box.w(), bh = box.h();

    float bestDist = std::numeric_limits<float>::max();
    const BoxXYXY* bestPrev = nullptr;
    for (const auto& pb : prevBoxes) {
        float d = distance(box, pb);
        if (d < bestDist) { bestDist = d; bestPrev = &pb; }
    }

    if (bestPrev == nullptr || bestDist > proximityThreshold * 2.0f)
        return box;

    float pcx = bestPrev->cx(), pcy = bestPrev->cy();
    float vx = (cx - pcx) / float(detectStride);
    float vy = (cy - pcy) / float(detectStride);
    float steps = float(targetFid - srcFid) / float(detectStride);

    float pcx2 = cx + vx * steps;
    float pcy2 = cy + vy * steps;
    return {pcx2 - bw * 0.5f, pcy2 - bh * 0.5f,
            pcx2 + bw * 0.5f, pcy2 + bh * 0.5f};
}

// ============================================================================
// Crop regions
// ============================================================================

std::vector<YoloTrackRefine::CropRegion>
YoloTrackRefine::computeCropRegions(
    const std::vector<BoxXYXY>& balls, int frameW, int frameH, int cropSize)
{
    int half = cropSize / 2;
    std::vector<CropRegion> regions;
    for (const auto& b : balls) {
        int cx = static_cast<int>(b.cx());
        int cy = static_cast<int>(b.cy());
        int x1 = cx - half, y1 = cy - half;
        int x2 = x1 + cropSize, y2 = y1 + cropSize;
        if (x1 < 0) { x1 = 0; x2 = cropSize; }
        if (y1 < 0) { y1 = 0; y2 = cropSize; }
        if (x2 > frameW) { x2 = frameW; x1 = std::max(0, frameW - cropSize); }
        if (y2 > frameH) { y2 = frameH; y1 = std::max(0, frameH - cropSize); }
        regions.push_back({x1, y1, x2, y2});
    }
    return regions;
}

std::vector<YoloTrackRefine::CropRegion>
YoloTrackRefine::mergeOverlappingRegions(std::vector<CropRegion> regions) {
    if (regions.size() <= 1) return regions;
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<CropRegion> merged;
        std::vector<bool> used(regions.size(), false);
        for (size_t i = 0; i < regions.size(); ++i) {
            if (used[i]) continue;
            auto& a = regions[i];
            for (size_t j = i + 1; j < regions.size(); ++j) {
                if (used[j]) continue;
                auto& b = regions[j];
                int ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
                int ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
                if (ix1 < ix2 && iy1 < iy2) {
                    a.x1 = std::min(a.x1, b.x1); a.y1 = std::min(a.y1, b.y1);
                    a.x2 = std::max(a.x2, b.x2); a.y2 = std::max(a.y2, b.y2);
                    used[j] = true; changed = true;
                }
            }
            merged.push_back(a); used[i] = true;
        }
        regions = std::move(merged);
    }
    return regions;
}

// ============================================================================
// JSON helpers (thin wrappers around nlohmann::json)
// ============================================================================

void* YoloTrackRefine::parseJson(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr, "Cannot open JSON: %s\n", path.c_str());
        return nullptr;
    }
    auto* j = new json(json::parse(f));
    return static_cast<void*>(j);
}

void YoloTrackRefine::writeJson(const std::string& path, const void* jsonData) {
    namespace fs = std::filesystem;
    auto parent = fs::path(path).parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }
    std::ofstream f(path);
    f << std::setw(2) << *static_cast<const json*>(jsonData) << std::endl;
}

void YoloTrackRefine::freeJson(void* jsonData) {
    delete static_cast<json*>(jsonData);
}

// ============================================================================
// Ball lookup from input JSON
// ============================================================================

std::map<int, std::vector<std::pair<BoxXYXY, float>>>
YoloTrackRefine::buildYoloBallLookup(void* eventJson, int jsonBallCls, float confThresh) {
    std::map<int, std::vector<std::pair<BoxXYXY, float>>> lookup;
    auto& event = *static_cast<json*>(eventJson);
    auto& objects = event["Objects"];
    if (!objects.is_array()) return lookup;

    for (const auto& frameObj : objects) {
        int frameId = frameObj.value("frame_id", -1);
        if (frameId < 0) continue;
        std::vector<std::pair<BoxXYXY, float>> balls;
        auto& objs = frameObj["objects"];
        if (objs.is_array()) {
            for (const auto& obj : objs) {
                if (obj.value("cls", -1) != jsonBallCls) continue;
                float conf = obj.value("conf", 0.0f);
                if (conf < confThresh) continue;
                auto& bbox = obj["bbox"];
                if (!bbox.is_array() || bbox.size() < 4) continue;
                balls.emplace_back(
                    BoxXYXY{bbox[0].get<float>(), bbox[1].get<float>(),
                            bbox[2].get<float>(), bbox[3].get<float>()},
                    conf);
            }
        }
        lookup[frameId] = std::move(balls);
    }
    return lookup;
}

// ============================================================================
// Crop re-detection (批量 GPU 推理)
// ============================================================================
//
// 将多个 640×640 裁剪区域打包为一个 batch 送进 GPU，
// 避免逐区域的 kernel launch 开销。这是 C++ 版相比 Python 版的关键加速点之一：
// Python 版逐区域调用 detect_batch，每次都是一次独立的 GPU 调用；
// C++ 版将所有裁剪区放在一个推理批次中。
//

std::vector<std::pair<BoxXYXY, float>>
YoloTrackRefine::cropRedetectBatch(
    const cv::Mat& frame,
    const std::vector<CropRegion>& regions,
    YOLOv11& model, int ballCls, float detConf)
{
    std::vector<std::pair<BoxXYXY, float>> allDetected;
    if (regions.empty()) return allDetected;

    int B = model.getBatchSize();

    for (size_t start = 0; start < regions.size(); start += B) {
        int batchActual = std::min(B, static_cast<int>(regions.size() - start));

        for (int b = 0; b < batchActual; ++b) {
            const auto& r = regions[start + b];
            cv::Mat crop = frame(cv::Rect(r.x1, r.y1, r.x2 - r.x1, r.y2 - r.y1)).clone();
            model.preprocess(crop, 0, b);
        }

        model.infer(0);

        for (int b = 0; b < batchActual; ++b) {
            const auto& r = regions[start + b];
            std::vector<Detection> objects;
            model.postprocess(objects, 0, b);
            for (const auto& det : objects) {
                if (det.class_id != ballCls) continue;
                allDetected.emplace_back(
                    BoxXYXY{r.x1 + det.bbox.x,
                            r.y1 + det.bbox.y,
                            r.x1 + det.bbox.x + det.bbox.width,
                            r.y1 + det.bbox.y + det.bbox.height},
                    det.conf);
            }
        }
    }

    // NMS on re-detected boxes
    if (allDetected.size() > 1) {
        std::vector<BoxXYXY> boxesOnly;
        for (auto& p : allDetected) boxesOnly.push_back(p.first);
        auto keep = nmsIndices(boxesOnly, 0.001f);
        std::vector<std::pair<BoxXYXY, float>> nmsResult;
        for (int i : keep) nmsResult.push_back(allDetected[i]);
        allDetected = std::move(nmsResult);
    }
    return allDetected;
}

// ============================================================================
// Per-event refinement (核心算法)
// ============================================================================
//
// 对单个事件帧段执行跟踪引导的细化，逐帧处理逻辑如下：
//
//   FOR each frame (按 detectStride 步进):
//     1. 从 YOLO 查找表获取当前帧已有的篮球检测
//     2. 回溯 maxLookback 帧，收集近期出现过的篮球位置
//     3. 对每个近期篮球，检查当前 YOLO 检测是否覆盖了它（中心距离 < proximityThreshold）
//     4. 未被覆盖的近期篮球：
//        a. 用速度估计预测当前位置
//        b. 在预测位置周围裁剪 640×640 区域
//        c. 合并重叠的裁剪区
//        d. 批量推理球检测模型，映射回全帧坐标
//     5. 合并三类框：YOLO 匹配的 + YOLO 新检测的 + 裁剪重检测找回的
//     6. 对所有结果框做 NMS（IoU=0.3）
//

YoloTrackRefine::EventResult
YoloTrackRefine::refineEvent(
    void* eventJson, cv::VideoCapture& cap, YOLOv11& model,
    const YoloTrackRefineConfig& cfg, int eventIdx, int frameW, int frameH)
{
    auto& event = *static_cast<json*>(eventJson);
    int startFrame = std::max(0, event.value("startFrame", 0));
    int endFrame   = event.value("EndFrame", event.value("endFrame", startFrame));
    if (endFrame < startFrame) endFrame = startFrame;

    auto yoloLookup = buildYoloBallLookup(eventJson, cfg.jsonBallCls, cfg.yoloConfThresh);
    std::map<int, std::vector<BoxXYXY>> refinedBoxes;

    std::printf("  Event %d: frames [%d, %d], stride=%d\n",
                eventIdx, startFrame, endFrame, cfg.detectStride);

    for (int globalFid = startFrame; globalFid <= endFrame; globalFid += cfg.detectStride) {
        cv::Mat frame;
        {
            cv::Mat tmp;
            if (!cap.read(tmp)) break;
            frame = tmp.clone();
        }

        // Current YOLO detections for this frame
        auto yoloIt = yoloLookup.find(globalFid);
        std::vector<BoxXYXY> curBalls;
        std::vector<float>   curConfs;
        if (yoloIt != yoloLookup.end()) {
            for (auto& p : yoloIt->second) {
                curBalls.push_back(p.first);
                curConfs.push_back(p.second);
            }
        }

        // Collect recent ball positions
        struct BallHist { BoxXYXY box; int srcFid; };
        std::vector<BallHist> recentBalls;
        for (int off = 1; off <= cfg.maxLookback; ++off) {
            int pf = globalFid - off * cfg.detectStride;
            auto it = refinedBoxes.find(pf);
            if (it != refinedBoxes.end())
                for (const auto& b : it->second)
                    recentBalls.push_back({b, pf});
        }

        std::set<int> matchedYolo;
        std::vector<BoxXYXY> unmatchedPred;

        if (recentBalls.empty()) {
            refinedBoxes[globalFid] = curBalls;
            continue;
        }

        // Match recent balls against current YOLO
        for (const auto& rb : recentBalls) {
            bool covered = false;
            for (size_t yi = 0; yi < curBalls.size(); ++yi) {
                if (distance(rb.box, curBalls[yi]) <= cfg.proximityThreshold) {
                    matchedYolo.insert(static_cast<int>(yi));
                    covered = true; break;
                }
            }
            if (!covered) {
                unmatchedPred.push_back(predictPosition(
                    rb.box, rb.srcFid, globalFid, cfg.detectStride,
                    refinedBoxes, cfg.proximityThreshold));
            }
        }

        // Collect results
        std::vector<BoxXYXY> resultBoxes;
        for (int yi : matchedYolo) resultBoxes.push_back(curBalls[yi]);
        for (size_t yi = 0; yi < curBalls.size(); ++yi)
            if (matchedYolo.count(static_cast<int>(yi)) == 0)
                resultBoxes.push_back(curBalls[yi]);

        // Crop re-detect for unmatched recent balls
        if (!unmatchedPred.empty()) {
            auto crops = computeCropRegions(unmatchedPred, frameW, frameH, cfg.cropSize);
            auto merged = mergeOverlappingRegions(std::move(crops));
            auto redet = cropRedetectBatch(frame, merged, model, cfg.ballCls, cfg.detConf);
            for (auto& p : redet) resultBoxes.push_back(p.first);
        }

        // NMS
        if (resultBoxes.size() > 1) {
            auto keep = nmsIndices(resultBoxes, 0.3f);
            std::vector<BoxXYXY> nmsR;
            for (int i : keep) nmsR.push_back(resultBoxes[i]);
            resultBoxes = std::move(nmsR);
        }

        refinedBoxes[globalFid] = resultBoxes;

        if ((globalFid - startFrame) % (cfg.detectStride * 50) == 0 || globalFid == endFrame) {
            std::printf("\r    frame %d / %d", globalFid, endFrame);
            std::fflush(stdout);
        }
    }
    std::printf("\n");

    return {event.value("EventID", eventIdx), startFrame, endFrame,
            std::move(refinedBoxes)};
}

// ============================================================================
// Aggregate output
// ============================================================================

void YoloTrackRefine::writeAggregateJson(
    const std::string& path,
    const std::string& eventsJsonPath,
    const std::vector<EventResult>& results,
    int frameW, int frameH, int detectStride)
{
    json aggregateEvents = json::array();
    for (size_t ei = 0; ei < results.size(); ++ei) {
        const auto& r = results[ei];

        // Per-event JSON
        json bbObj = json::object();
        for (const auto& kv : r.boxes) {
            json arr = json::array();
            for (const auto& b : kv.second)
                arr.push_back({b.x1, b.y1, b.x2, b.y2});
            bbObj[std::to_string(kv.first)] = arr;
        }

        json evJson = {
            {"EventID",       r.eventID},
            {"StartFrame",    r.startFrame},
            {"EndFrame",      r.endFrame},
            {"DetectStride",  detectStride},
            {"BasketballBoxes", bbObj}
        };

        aggregateEvents.push_back(evJson);
    }

    json aggregate = {
        {"frameWidth",  frameW},
        {"frameHeight", frameH},
        {"detectStride", detectStride},
        {"eventsJson",  eventsJsonPath},
        {"events",      aggregateEvents}
    };
    writeJson(path, &aggregate);
}

// ============================================================================
// Main entry point
// ============================================================================

int YoloTrackRefine::run(
    const char* eventsJsonPath,
    const char* videoPath,
    const char* ballEnginePath,
    const char* outputJsonPath,
    const YoloTrackRefineConfig& config)
{
    auto tStart = std::chrono::steady_clock::now();

    // Load input JSON
    void* inputJson = parseJson(eventsJsonPath);
    if (!inputJson) return -1;
    auto& events = *static_cast<json*>(inputJson);
    if (!events.is_array()) {
        std::fprintf(stderr, "Input JSON must be a list of event objects\n");
        freeJson(inputJson);
        return -1;
    }
    std::printf("Loaded %zu events from JSON\n", events.size());

    // Open video
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::fprintf(stderr, "Cannot open video: %s\n", videoPath);
        freeJson(inputJson);
        return -1;
    }
    int frameW = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameH = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    std::printf("Video: %dx%d\n", frameW, frameH);

    // Load ball detection engine
    Logger logger;
    std::printf("Loading engine: %s\n", ballEnginePath);
    YOLOv11 model(ballEnginePath, logger);

    // Process events
    std::vector<EventResult> results;
    for (size_t ei = 0; ei < events.size(); ++ei) {
        int startFrame = events[ei].value("startFrame", 0);
        cap.set(cv::CAP_PROP_POS_FRAMES, startFrame);
        results.push_back(refineEvent(
            static_cast<void*>(&events[ei]), cap, model,
            config, static_cast<int>(ei), frameW, frameH));
    }

    // Write output
    writeAggregateJson(outputJsonPath, eventsJsonPath, results,
                       frameW, frameH, config.detectStride);

    freeJson(inputJson);

    auto tEnd = std::chrono::steady_clock::now();
    double totalSec = std::chrono::duration<double>(tEnd - tStart).count();
    std::printf("\nDone. Output: %s\n", outputJsonPath);
    std::printf("Total time: %.2f s\n", totalSec);

    return 0;
}
