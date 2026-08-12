"""Strategy that wraps C++ TensorRT executables for pipeline steps.

Provides two operations mirroring their Python counterparts:
- ``extract_info()`` — runs ``exec_detect_video`` to generate ``detect.json``
  (replaces Python InfoExtract step 1).
- ``refine_ball()`` — runs ``exec_yolo_refine`` to generate ball boxes JSON
  (replaces Python YoloTrackRefine step 6).
"""

from __future__ import annotations

import subprocess
import sys


class CppExecCall:
    """Call pre-built C++ TensorRT executables for detection and ball refinement.

    The constructor takes no arguments; executable and engine paths are passed
    directly to each method so callers control which binaries to use per step.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Step 1 — object detection (replaces InfoExtract)
    # ------------------------------------------------------------------

    def extract_info(
        self,
        video: str,
        output_json: str,
        detect_exec: str,
        detect_engine: str,
        conf: float = 0.5,
    ) -> None:
        """Run C++ detection on *video*, writing per-frame detections to *output_json*.

        The JSON format is compatible with ``detect.json`` produced by the Python
        ``InfoExtract`` stage.
        """
        cmd = [
            detect_exec,
            detect_engine,
            video,
            output_json,
            "None",  # skip output MP4
            str(conf),
        ]
        self._run(cmd, "exec_detect_video")

    # ------------------------------------------------------------------
    # Step 6 — ball refinement (replaces YoloTrackRefine)
    # ------------------------------------------------------------------

    def refine_ball(
        self,
        events_json: str,
        video: str,
        output_json: str,
        refine_exec: str,
        refine_engine: str,
    ) -> None:
        """Run C++ ball refinement, reading *events_json* and *video*, writing to *output_json*.

        *events_json* should be ``JsonForLLM_with_objects.json`` from the Python
        ``BasicInfoExtract`` stage.
        """
        cmd = [
            refine_exec,
            events_json,
            video,
            refine_engine,
            output_json,
        ]
        self._run(cmd, "exec_yolo_refine")

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run(cmd: list[str], label: str) -> None:
        print(f"[CPP] {label}: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"{label} exited with code {result.returncode}"
            )

if __name__ == "__main__":
    # Example usage:
    cpp_exec = CppExecCall()
    cpp_exec.extract_info(
        video="./test.mp4",
        output_json="detect.json",
        detect_exec="build/exec_detect_video",
        detect_engine="model/yolo11s_person_basketball_backboard_hoop_1920_1088_AIAnalysi1_without_nms_batch4.engine",
        conf=0.4,
    )
    # cpp_exec.refine_ball(
    #     events_json="/home/xiaodai/Code/XbotGo-AI_Analysis/Datasets/Faclon_AI_Analysis_2/Single_Event_Analysis_Result_For_CPP/2019018476048580608_man7_clips/2019018476048580608_man7_clip_001/JsonForLLM_with_objects.json",
    #     video="/home/xiaodai/Code/XbotGo-AI_Analysis/Datasets/Faclon_AI_Analysis_2/2019018476048580608_man7_clips/2019018476048580608_man7_clip_001.mp4",
    #     output_json="ball_boxes.json",
    #     refine_exec="/home/xiaodai/Code/yolov11-tensorrt/examples/demo/build/exec_yolo_refine",
    #     refine_engine="/home/xiaodai/Code/XbotGo-AI_Analysis/models/yolo11s_detect_ball_640_640_20260617_without_nms_batch4.engine",
    # )