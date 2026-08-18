# XbotGO TensorRT inference

This project builds TensorRT inference tools for an NVIDIA T4 GPU (CUDA
architecture `sm_75`). The Dockerfile uses a multi-stage build:

- `builder` contains CUDA/TensorRT development tools and can generate a
  TensorRT engine from ONNX.
- The final `runtime` image contains the CUDA/TensorRT/OpenCV runtime,
  a minimal Python virtual environment, `exec_detect_video`, and
  `exec_yolo_refine`.
- Models, videos, JSON files, compilers, headers, and CMake are not included in
  the final image.

The executables in the final image are installed at:

```text
/usr/local/bin/exec_detect_video
/usr/local/bin/exec_yolo_refine
```

Compatibility symlinks are also provided at the old paths:

```text
/workspace/build/exec_detect_video
/workspace/build/exec_yolo_refine
```

## Prerequisites

- An NVIDIA T4 host with a compatible NVIDIA driver
- Docker and NVIDIA Container Toolkit
- A writable local `model/` directory for generated engines
- Input video and JSON files under `data/`

## Build the engine generator

Build and retain the development stage when an engine must be generated from
ONNX on the T4:

```bash
sudo docker build --target builder \
  -t xbotgo-tensorrt:builder .
```

Generate an engine. The generator writes the `.engine` file next to the ONNX
file in the mounted `model/` directory:

```bash
sudo docker run --rm --gpus all \
  -v "$(pwd)/model:/models" \
  xbotgo-tensorrt:builder \
  /workspace/build/yolov11-tensorrt \
  /models/yolo11s_person_basketball_backboard_hoop_1920_1088_AIAnalysi1_without_nms_batch4.onnx
```

Confirm that the engine is not an empty placeholder before using it:

```bash
test -s model/yolo11s_person_basketball_backboard_hoop_1920_1088_AIAnalysi1_without_nms_batch4.engine
```

TensorRT engines should be generated with the same TensorRT version and GPU
class used for inference. This image uses TensorRT 10.7, CUDA 12.6, and targets
the T4 `sm_75` architecture.

## Build the runtime image

```bash
sudo docker build -t xbotgo-tensorrt:runtime .
```

The builder stage is used during compilation, but its layers are not part of
the final runtime image. Display the runtime image size and layers with:

```bash
sudo docker image inspect xbotgo-tensorrt:runtime \
  --format 'runtime size: {{.Size}} bytes'

sudo docker history xbotgo-tensorrt:runtime
```

Show the commands supported by the runtime entrypoint:

```bash
sudo docker run --rm xbotgo-tensorrt:runtime help
```

Running the image interactively without a command opens Bash:

```bash
sudo docker run --rm --gpus all -it xbotgo-tensorrt:runtime
```

## Python runtime and wheel testing

The runtime creates a virtual environment at `/opt/venv`. Its `python` and
`pip` commands are already on `PATH`, so packages can be installed without
modifying Ubuntu's system Python environment.

Mount and test a wheel with:

```bash
sudo docker run --rm --gpus all -it \
  -v "$(pwd)/model:/workspace/model:ro" \
  -v "$(pwd)/data:/workspace/data" \
  -v /tmp/xbotgoaianalysis-0.2.26-py3-none-any.whl:/workspace/xbotgoaianalysis-0.2.26-py3-none-any.whl:ro \
  -v "$(pwd)/CppExecCall.py:/workspace/CppExecCall.py:ro" \
  xbotgo-tensorrt:runtime \
  bash -c 'python -m pip install /workspace/xbotgoaianalysis-0.2.26-py3-none-any.whl && python /workspace/CppExecCall.py'
```

Installing the wheel this way happens inside the temporary container. With
`--rm`, the installed package is removed when the container exits. Create a
derived image and install the wheel during `docker build` when it must be
permanently included.

## Run full-frame detection

Command syntax:

```text
detect <engine> <video> [output.json] [output.mp4|None] [confidence]
```

Recommended invocation:

```bash
sudo docker run --rm --gpus all \
  -v "$(pwd)/model:/models:ro" \
  -v "$(pwd)/data:/data" \
  xbotgo-tensorrt:runtime \
  detect \
  /models/yolo11s_person_basketball_backboard_hoop_1920_1088_AIAnalysi1_without_nms_batch4.engine \
  /data/test.mp4 \
  /data/detect.json \
  None \
  0.3
```

The result is written to `data/detect.json` on the host. Replace `None` with a
path such as `/data/detect.mp4` when an annotated video is also required.

The executable can also be called directly using its runtime path:

```bash
sudo docker run --rm --gpus all \
  -v "$(pwd)/model:/models:ro" \
  -v "$(pwd)/data:/data" \
  xbotgo-tensorrt:runtime \
  /usr/local/bin/exec_detect_video \
  /models/yolo11s_person_basketball_backboard_hoop_1920_1088_AIAnalysi1_without_nms_batch4.engine \
  /data/test.mp4 \
  /data/detect.json \
  None \
  0.3
```

## Run tracking-guided refinement

Command syntax:

```text
refine <events.json> <video> <engine> [output.json] [refine options...]
```

Example:

```bash
sudo docker run --rm --gpus all \
  -v "$(pwd)/model:/models:ro" \
  -v "$(pwd)/data:/data" \
  xbotgo-tensorrt:runtime \
  refine \
  /data/JsonForLLM_with_objects.json \
  /data/test.mp4 \
  /models/refine.engine \
  /data/YoloRefineBallboxes.json \
  --stride 2
```

Supported refinement options include `--stride`, `--conf`, `--lookback`,
`--proximity`, `--ball-cls`, and `--json-ball-cls`.

## Reclaim builder disk space

After the engine and final runtime image have been tested, the tagged builder
image can be removed:

```bash
sudo docker image rm xbotgo-tensorrt:builder
```

Unused build cache can also be removed when it is no longer needed:

```bash
sudo docker builder prune
```

Removing build cache does not shrink or delete the tagged runtime image, but a
later rebuild may need to download dependencies and compile everything again.

## Troubleshooting

### An executable cannot be found

The preferred locations are:

```text
/usr/local/bin/exec_detect_video
/usr/local/bin/exec_yolo_refine
```

The old `/workspace/build/exec_*` locations are compatibility symlinks. Rebuild
the runtime image if those links do not exist in an older image.

### Image build reports `undefined reference to cv::dnn::NMSBoxes`

Ensure the current `CMakeLists.txt` includes the OpenCV `dnn` component and the
runtime installs `libopencv-dnn406t64`. Rebuild without reusing a stale source
checkout.
