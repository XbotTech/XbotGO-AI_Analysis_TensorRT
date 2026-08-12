/**
 * @file YoloTrackRefine.h
 * @brief 跟踪引导的篮球检测细化 —— Tracking-guided ball refinement.
 *
 * ============================================================================
 * 背景
 * ============================================================================
 * 在篮球视频分析的流水线中，第一阶段的 YOLO 模型（1920x1088 四类检测器）会
 * 对全帧进行推理，产生包含 person/ball/backboard/hoop 的检测结果并写入
 * ``JsonForLLM_with_objects.json``。由于篮球体积小、运动快、常被遮挡，全帧
 * 检测器在某些帧会漏检篮球。
 *
 * ============================================================================
 * 算法思路
 * ============================================================================
 * 本模块利用「篮球不会瞬间消失」的先验，进行跟踪引导的裁剪重检测：
 *
 *   1. 从 ``JsonForLLM_with_objects.json`` 读取每帧已有 YOLO 检测，构建
 *      逐帧篮球查找表（frame_id → ball boxes）。
 *
 *   2. 对每个事件段，按 detectStride 步长逐帧处理：
 *
 *      a. 检查当前帧是否有 YOLO 检测到的球。
 *      b. 回溯 maxLookback 帧，找出「近期出现过但当前帧未检测到」的球。
 *      c. 对丢失的球，利用速度估计预测当前位置。
 *      d. 在预测位置周围裁剪 640×640 区域，用专用的 640 球检测模型
 *         （yolo11s_detect_ball_640_640）对该裁剪区重新推理。
 *      e. 将重检测到的框映射回全帧坐标。
 *      f. 合并三类框（YOLO 匹配的 + YOLO 新出现的 + 重检测找回的），
 *         通过 NMS 去重。
 *
 *   3. 输出 ``YoloRefineBallboxes.json``，包含每个事件每帧的精确篮球位置。
 *
 * ============================================================================
 * 性能对比（180 帧测试视频）
 * ============================================================================
 *   Python (PyTorch):  ~9 秒
 *   C++ (TensorRT):    ~2 秒（约 5x 加速）
 *
 * 主要加速来源：
 *   - TensorRT 引擎推理 vs PyTorch 推理
 *   - C++ 原生数值计算 vs Python 解释器 + 对象装箱
 *   - 紧凑内存布局、cache 友好的数据结构
 *
 * ============================================================================
 * 输入 / 输出
 * ============================================================================
 * Input:  JsonForLLM_with_objects.json  (流水线第一阶段产生的逐帧检测 JSON)
 * Model:  yolo11s_detect_ball_640_640.engine (640×640 单类篮球检测 TensorRT 引擎)
 * Output: YoloRefineBallboxes.json        (细化后的逐帧篮球框)
 */

#ifndef YOLO_TRACK_REFINE_H
#define YOLO_TRACK_REFINE_H

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

/* ========================================================================= */
/*  Types                                                                    */
/* ========================================================================= */

/** 轴对齐矩形框，使用左上角 + 右下角坐标表示。 */
struct BoxXYXY {
    float x1, y1, x2, y2;

    float cx() const { return (x1 + x2) * 0.5f; }  /**< 中心 x */
    float cy() const { return (y1 + y2) * 0.5f; }  /**< 中心 y */
    float w()  const { return x2 - x1; }            /**< 宽度 */
    float h()  const { return y2 - y1; }            /**< 高度 */
};

/** YoloTrackRefine 算法配置。 */
struct YoloTrackRefineConfig {
    int    detectStride        = 1;       /**< 帧步长：每隔 N 帧处理一次。 */
    float  detConf             = 0.25f;   /**< 裁剪重检测的置信度阈值。 */
    int    cropSize            = 640;     /**< 裁剪区域边长（像素），与球检测模型的输入尺寸一致。 */
    int    maxLookback         = 5;       /**< 最大回溯帧数（按步长计）。 */
    float  proximityThreshold  = 150.0f;  /**< 中心距离阈值（像素），用于判断两个框是否代表同一个球。 */
    float  yoloConfThresh      = 0.0f;    /**< 过滤输入 JSON 中低于此置信度的 YOLO 球检测。 */
    int    ballCls             = 1;       /**< 裁剪检测模型中篮球的类别 ID。 */
    int    jsonBallCls         = 1;       /**< 输入 JSON 的 Objects 字段中篮球的类别 ID（流水线约定始终为 1）。 */
};

/** 单个事件的细化结果。 */
struct YoloTrackRefineResult {
    int    eventID;
    int    startFrame;
    int    endFrame;
    int    detectStride;
    /** 逐帧篮球框：frame_id → [ [x1,y1,x2,y2], ... ] */
    std::map<int, std::vector<std::array<float, 4>>> basketballBoxes;
};

/* ========================================================================= */
/*  YoloTrackRefine                                                          */
/* ========================================================================= */

class YOLOv11;

class YoloTrackRefine {
public:
    /**
     * @brief 执行跟踪引导的篮球检测细化。
     *
     * 这是唯一的公开入口。内部完成 JSON 读取 → 视频读取 → 逐事件细化 →
     * 输出聚合 JSON 的全流程。
     *
     * @param eventsJsonPath    JsonForLLM_with_objects.json 的路径。
     * @param videoPath         输入视频路径（mp4/avi/mov）。
     * @param ballEnginePath    640×640 球检测 TensorRT 引擎路径。
     * @param outputJsonPath    输出 JSON 路径（聚合结果）。
     * @param config            算法配置，见 YoloTrackRefineConfig。
     * @return 0 成功，负数失败。
     */
    static int run(
        const char* eventsJsonPath,
        const char* videoPath,
        const char* ballEnginePath,
        const char* outputJsonPath,
        const YoloTrackRefineConfig& config);

private:
    YoloTrackRefine() = delete;

    // ---- 框操作工具函数 ----------------------------------------------------
    static float distance(const BoxXYXY& a, const BoxXYXY& b);
    static float iou(const BoxXYXY& a, const BoxXYXY& b);
    /** NMS：按面积降序贪心抑制，返回保留的框下标。 */
    static std::vector<int> nmsIndices(const std::vector<BoxXYXY>& boxes,
                                       float iouThreshold);

    // ---- 运动预测 ----------------------------------------------------------
    /**
     * @brief 根据历史框预估目标帧中球的位置。
     *
     * 取 srcFid 的前一帧（srcFid - detectStride）中最近的框，
     * 计算像素/帧的速度，然后线性外推到 targetFid。
     * 若无法估算速度，则回退到原框位置。
     */
    static BoxXYXY predictPosition(
        const BoxXYXY& box, int srcFid, int targetFid, int detectStride,
        const std::map<int, std::vector<BoxXYXY>>& refinedBoxes,
        float proximityThreshold);

    // ---- 裁剪区域计算 ------------------------------------------------------
    struct CropRegion { int x1, y1, x2, y2; };
    /** 以每个球为中心，生成 cropSize×cropSize 的裁剪区，并 clamp 到图像边界。 */
    static std::vector<CropRegion> computeCropRegions(
        const std::vector<BoxXYXY>& balls, int frameW, int frameH, int cropSize);
    /** 迭代合并有重叠的裁剪区，减少后续 YOLO 推理次数。 */
    static std::vector<CropRegion> mergeOverlappingRegions(
        std::vector<CropRegion> regions);

    // ---- JSON 辅助（nlohmann/json 的薄封装，void* 避免头文件泄漏）---------
    static void* parseJson(const std::string& path);
    static void  writeJson(const std::string& path, const void* jsonData);
    static void  freeJson(void* jsonData);

    // ---- 从输入 JSON 构建逐帧篮球查找表 ------------------------------------
    /**
     * @brief 提取事件的 Objects 字段，构造 frame_id → [(框, 置信度), ...] 映射。
     * @param confThresh 过滤低于此置信度的检测。
     */
    static std::map<int, std::vector<std::pair<BoxXYXY, float>>>
    buildYoloBallLookup(void* eventJson, int jsonBallCls, float confThresh);

    // ---- 裁剪重检测（批量推理）---------------------------------------------
    /**
     * @brief 对多个裁剪区域进行批量 YOLO 推理。
     *
     * 使用 YOLOv11 的 batch API，尽可能在一个 GPU kernel 调用中处理多个裁剪区，
     * 避免逐区 kernel launch 开销。检测结果自动映射回全帧坐标并做类内 NMS。
     */
    static std::vector<std::pair<BoxXYXY, float>>
    cropRedetectBatch(const cv::Mat& frame,
                      const std::vector<CropRegion>& regions,
                      YOLOv11& model, int ballCls, float detConf);

    // ---- 单事件细化（核心算法）---------------------------------------------
    struct EventResult {
        int eventID, startFrame, endFrame;
        std::map<int, std::vector<BoxXYXY>> boxes;
    };
    /** 对单个事件执行完整的跟踪细化流程。 */
    static EventResult refineEvent(
        void* eventJson, cv::VideoCapture& cap, YOLOv11& model,
        const YoloTrackRefineConfig& cfg, int eventIdx, int frameW, int frameH);

    // ---- 聚合输出 ----------------------------------------------------------
    /** 将多个事件的结果汇总写入聚合 JSON，同时写出每个事件的独立 JSON 到子目录。 */
    static void writeAggregateJson(
        const std::string& path,
        const std::string& eventsJsonPath,
        const std::vector<EventResult>& results,
        int frameW, int frameH, int detectStride);
};

#endif // YOLO_TRACK_REFINE_H
