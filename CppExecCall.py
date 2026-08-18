from XbotGoAI.strategy.CppExecCall import CppExecCall


CppExecCall().extract_info(
    video="/workspace/data/test.mp4",
    output_json="detect.json",
    detect_exec="/usr/local/bin/exec_detect_video",
    detect_engine="/workspace/model/yolo11s_person_basketball_backboard_hoop_1920_1088_AIAnalysi1_without_nms_batch4.engine",
    conf=0.4,
)
