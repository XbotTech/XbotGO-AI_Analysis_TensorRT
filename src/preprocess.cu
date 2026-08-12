#include "preprocess.h"
#include "cuda_utils.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct PreprocessStagingBuffer {
    uint8_t* host = nullptr;
    uint8_t* device = nullptr;
    size_t capacity = 0;
};

static std::vector<PreprocessStagingBuffer> staging_buffers;
static size_t max_image_bytes = 0;
static int staging_num_streams = 0;
static int staging_batch_size = 0;

struct AffineMatrix {
    float value[6];
};

__global__ void warpaffine_kernel(
    uint8_t* src, int src_line_size, int src_width,
    int src_height, float* dst, int dst_width,
    int dst_height, uint8_t const_value_st,
    AffineMatrix d2s, int edge) {
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return;

    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    int dx = position % dst_width;
    int dy = position / dst_width;
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f;
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
    float c0, c1, c2;

    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        // out of range
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }
    else {
        int y_low = floorf(src_y);
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        uint8_t const_value[] = { const_value_st, const_value_st, const_value_st };
        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;

        if (y_low >= 0) {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3;

            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        if (y_high < src_height) {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3;

            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    // bgr to rgb 
    float t = c2;
    c2 = c0;
    c0 = t;

    // normalization
    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    // rgbrgbrgb to rrrgggbbb
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}

void cuda_preprocess(
    uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    cudaStream_t stream, int slot, int batch_idx) {

    if (slot < 0 || slot >= staging_num_streams ||
        batch_idx < 0 || batch_idx >= staging_batch_size) {
        std::fprintf(stderr,
                     "Invalid preprocess staging buffer index: slot=%d, batch_idx=%d\n",
                     slot, batch_idx);
        std::abort();
    }

    const size_t img_size = static_cast<size_t>(src_width) * src_height * 3;
    if (img_size > max_image_bytes) {
        std::fprintf(stderr,
                     "Source image requires %zu bytes, exceeding preprocess limit %zu bytes\n",
                     img_size, max_image_bytes);
        std::abort();
    }

    const int buffer_index = slot * staging_batch_size + batch_idx;
    PreprocessStagingBuffer& staging = staging_buffers[buffer_index];

    // Allocate to the actual source-frame size. Growth is rare, but the old
    // buffer may still be referenced by queued work on this stream, so wait
    // before replacing it.
    if (staging.capacity < img_size) {
        if (staging.capacity != 0) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaFree(staging.device));
            CUDA_CHECK(cudaFreeHost(staging.host));
        }
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&staging.host), img_size));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&staging.device), img_size));
        staging.capacity = img_size;
    }

    // copy data to pinned memory
    std::memcpy(staging.host, src, img_size);
    // copy data to device memory
    CUDA_CHECK(cudaMemcpyAsync(staging.device, staging.host, img_size,
                               cudaMemcpyHostToDevice, stream));

    AffineMatrix s2d, d2s;
    float scale = std::min(dst_height / (float)src_height, dst_width / (float)src_width);

    s2d.value[0] = scale;
    s2d.value[1] = 0;
    s2d.value[2] = -scale * src_width * 0.5 + dst_width * 0.5;
    s2d.value[3] = 0;
    s2d.value[4] = scale;
    s2d.value[5] = -scale * src_height * 0.5 + dst_height * 0.5;

    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s);

    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    int jobs = dst_height * dst_width;
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);

    warpaffine_kernel << <blocks, threads, 0, stream >> > (
        staging.device, src_width * 3, src_width,
        src_height, dst, dst_width,
        dst_height, 128, d2s, jobs);
}

void cuda_preprocess_init(int max_image_size, int num_streams, int batch_size) {
    max_image_bytes = static_cast<size_t>(max_image_size) * 3;
    staging_num_streams = num_streams;
    staging_batch_size = batch_size;
    staging_buffers.clear();
    staging_buffers.resize(static_cast<size_t>(num_streams) * batch_size);
}

void cuda_preprocess_destroy() {
    for (PreprocessStagingBuffer& staging : staging_buffers) {
        if (staging.device != nullptr) {
            CUDA_CHECK(cudaFree(staging.device));
        }
        if (staging.host != nullptr) {
            CUDA_CHECK(cudaFreeHost(staging.host));
        }
    }
    staging_buffers.clear();
    max_image_bytes = 0;
    staging_num_streams = 0;
    staging_batch_size = 0;
}
