#!/bin/bash
set -euo pipefail

show_help() {
    cat <<'EOF'
Usage:
  detect <engine> <video> [output.json] [output.mp4|None] [conf]
  refine <events.json> <video> <engine> [output.json] [refine options...]

Examples:
  detect /models/detect.engine /data/input.mp4 /output/detect.json None 0.3
  refine /data/events.json /data/input.mp4 /models/refine.engine /output/refine.json

Models and input/output data are not included in the image; mount them with
docker run -v. Any other command is executed directly.
EOF
}

if [ "$#" -eq 0 ]; then
    show_help
    exit 0
fi

case "$1" in
    help|-h|--help)
        show_help
        ;;
    detect)
        if [ "$#" -lt 3 ]; then
            echo "[ERROR] detect requires <engine> and <video>." >&2
            show_help >&2
            exit 2
        fi
        engine="$2"
        video="$3"
        output_json="${4:-/data/detect.json}"
        output_video="${5:-None}"
        confidence="${6:-0.3}"
        exec /usr/local/bin/exec_detect_video \
            "$engine" "$video" "$output_json" "$output_video" "$confidence"
        ;;
    refine)
        if [ "$#" -lt 4 ]; then
            echo "[ERROR] refine requires <events.json>, <video>, and <engine>." >&2
            show_help >&2
            exit 2
        fi
        events_json="$2"
        video="$3"
        engine="$4"
        shift 4
        exec /usr/local/bin/exec_yolo_refine \
            "$events_json" "$video" "$engine" "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
