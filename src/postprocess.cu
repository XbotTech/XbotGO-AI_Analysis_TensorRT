#include "postprocess.h"
#include "cuda_utils.h"
#include <cstdio>

/**
 * @brief GPU kernel: decode YOLO output and filter by confidence.
 *
 * raw_output layout (row-major, detection_attr_size x num_detections):
 *   - Row 0: cx for all detections
 *   - Row 1: cy
 *   - Row 2: w
 *   - Row 3: h
 *   - Row 4..4+num_classes-1: class scores
 *
 * filtered_boxes layout: [x, y, w, h, conf, class_id] per valid detection
 */
__global__ void decode_filter_kernel(
    const float* __restrict__ raw_output,
    float* __restrict__ filtered_boxes,
    int* __restrict__ filtered_count,
    int num_detections,
    int num_classes,
    int det_attr_size,
    float conf_threshold,
    int max_detections
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_detections) return;

    // Find max class score and corresponding class
    float max_score = 0.0f;
    int max_class = 0;
    for (int c = 0; c < num_classes; c++) {
        float score = raw_output[(4 + c) * num_detections + idx];
        if (score > max_score) {
            max_score = score;
            max_class = c;
        }
    }

    if (max_score < conf_threshold) return;

    // Decode box: cx, cy, w, h -> x, y, w, h (top-left corner + dimensions)
    float cx = raw_output[0 * num_detections + idx];
    float cy = raw_output[1 * num_detections + idx];
    float w  = raw_output[2 * num_detections + idx];
    float h  = raw_output[3 * num_detections + idx];

    // Atomic increment to get a slot in the output buffer
    int out_idx = atomicAdd(filtered_count, 1);
    if (out_idx >= max_detections) {
        // Overflow: clamp counter (best-effort, won't write beyond buffer)
        atomicExch(filtered_count, max_detections);
        return;
    }

    // Store: [x, y, w, h, conf, class_id]
    filtered_boxes[out_idx * 6 + 0] = cx - 0.5f * w;   // x
    filtered_boxes[out_idx * 6 + 1] = cy - 0.5f * h;   // y
    filtered_boxes[out_idx * 6 + 2] = w;                 // width
    filtered_boxes[out_idx * 6 + 3] = h;                 // height
    filtered_boxes[out_idx * 6 + 4] = max_score;         // confidence
    filtered_boxes[out_idx * 6 + 5] = (float)max_class;  // class_id
}


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
)
{
    const int threads_per_block = 256;
    const int blocks = (num_detections + threads_per_block - 1) / threads_per_block;

    decode_filter_kernel<<<blocks, threads_per_block, 0, stream>>>(
        raw_output, filtered_boxes, filtered_count,
        num_detections, num_classes, det_attr_size,
        conf_threshold, max_detections
    );
}
