cmake_minimum_required(VERSION 3.12)
project(yolov11-tensorrt)

find_package(CUDA REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_definitions(-DAPI_EXPORTS)

# Ensure NVCC also compiles with -fPIC (needed when a static lib is linked into a shared lib)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC")

# Include CUDA directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(src/)

# -------------------------------------------------------------------
# Source files shared by the executable and the shared library
# -------------------------------------------------------------------
set(LIB_SOURCES
    src/YOLOv11.cpp
    src/YoloTrackRefine.cpp
)

set(CUDA_SOURCES
    src/preprocess.cu
    src/postprocess.cu
)

set(HEADERS
    src/YOLOv11.h
    src/YoloTrackRefine.h
    src/macros.h
    src/logging.h
    src/cuda_utils.h
    src/preprocess.h
    src/postprocess.h
    src/common.h
    src/yolov11_api.h
)

# -------------------------------------------------------------------
# Dependencies（通过 cmake -D 传入）
# -------------------------------------------------------------------
# OpenCV_DIR 和 TENSORRT_DIR 由 cmake -D 传入，未传入则报错

# Only link the OpenCV modules used by the two runtime executables.  Keeping the
# component list explicit also lets the runtime image install just these shared
# libraries instead of the full OpenCV development package.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvonnxparser)

# -------------------------------------------------------------------
# Static library: compile common CUDA + C++ sources once
# -------------------------------------------------------------------
cuda_add_library(yolov11_common STATIC
    ${LIB_SOURCES}
    ${CUDA_SOURCES}
)

set_target_properties(yolov11_common PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(yolov11_common
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)

# -------------------------------------------------------------------
# Shared library: libyolov11_tensorrt.so (adds the C API layer)
# -------------------------------------------------------------------
cuda_add_library(yolov11_tensorrt SHARED
    src/yolov11_api.cpp
    ${HEADERS}
)

target_link_libraries(yolov11_tensorrt
    yolov11_common
)

# -------------------------------------------------------------------
# Standalone executable (for testing / CLI use)
# -------------------------------------------------------------------
cuda_add_executable(${PROJECT_NAME}
    main.cpp
    ${HEADERS}
)

# 设置 BUILD_RPATH 让可执行文件能找到 TensorRT 的 builder_resource 库
# libnvinfer.so 内部 dlopen 不会继承可执行文件的 RUNPATH，所以同时需要脚本中设置 LD_LIBRARY_PATH
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "${TENSORRT_DIR}/lib"
)

target_link_libraries(${PROJECT_NAME}
    yolov11_common
)

# -------------------------------------------------------------------
# exec_detect_video – 全帧视频目标检测 CLI
# -------------------------------------------------------------------
cuda_add_executable(exec_detect_video
    src/exec_detect_video.cpp
    src/yolov11_api.cpp
)
set_target_properties(exec_detect_video PROPERTIES
    BUILD_RPATH "${TENSORRT_DIR}/lib"
)
target_link_libraries(exec_detect_video yolov11_common)

# -------------------------------------------------------------------
# exec_yolo_refine – 跟踪引导篮球检测细化 CLI
# -------------------------------------------------------------------
cuda_add_executable(exec_yolo_refine
    src/exec_yolo_refine.cpp
)
set_target_properties(exec_yolo_refine PROPERTIES
    BUILD_RPATH "${TENSORRT_DIR}/lib"
)
target_link_libraries(exec_yolo_refine yolov11_common)
