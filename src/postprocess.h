#pragma once

#include <cuda_runtime.h>

/**
 * @brief GPU-accelerated postprocess: decode boxes + filter by confidence threshold.
 *        Processes all raw detections in parallel, outputs only those above threshold.
 *
 * @param raw_output       Device pointer to raw model output [det_attr_size * num_detections]
 * @param filtered_boxes   Device pointer to output: [x, y, w, h, conf, class_id] per detection
 * @param filtered_count   Device pointer to atomic counter (single int, must be zeroed before call)
 * @param num_detections   Total number of raw detections (e.g., 8400)
 * @param num_classes      Number of classes (e.g., 80 for COCO)
 * @param det_attr_size    Detection attribute size = 4 + num_classes
 * @param conf_threshold   Minimum confidence threshold
 * @param max_detections   Maximum output detections (safety bound for atomic writes)
 * @param stream           CUDA stream
 */
void cuda_postprocess_decode(
    const float* raw_output,
    float* filtered_boxes,
    int* filtered_count,
    int num_detections,
    int num_classes,
    int det_attr_size,
    float conf_threshold,
    int max_detections,
    cudaStream_t stream
);
