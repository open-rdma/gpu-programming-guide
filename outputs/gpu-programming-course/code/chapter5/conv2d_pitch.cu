/**
 * conv2d_pitch.cu
 *
 * 第5章动手体验：使用 cudaMallocPitch 进行二维数组处理
 *
 * 展示：
 *   1. cudaMallocPitch() 的正确使用
 *   2. pitch 的概念及在 kernel 中如何使用 pitch 计算行偏移
 *   3. cudaMemcpy2D() 进行二维数据传输
 *
 * 编译：
 *   nvcc conv2d_pitch.cu -o conv2d_pitch -arch=sm_60
 *
 * 运行：
 *   ./conv2d_pitch
 */

#include <stdio.h>
#include <cuda_runtime.h>

// 错误检查宏
#define CUDA_CHECK(err)                                                        \
    do                                                                         \
    {                                                                          \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess)                                               \
        {                                                                      \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,            \
                   cudaGetErrorString(err_));                                   \
            exit(-1);                                                          \
        }                                                                      \
    } while (0)

/**
 * 二维卷积内核（3x3 均值滤波器）
 *
 * 重点：使用 pitch 计算每一行的起始地址
 *   float *row = (float *)((char *)basePtr + rowIndex * pitch);
 */
__global__ void Convolve2D(const float *input, float *output,
                           size_t pitch, int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        // 使用 pitch（而非 width * sizeof(float)）计算行偏移
        const float *inRow = (const float *)((const char *)input + row * pitch);
        float *outRow = (float *)((char *)output + row * pitch);

        // 3x3 均值滤波器
        float sum = 0.0f;
        int count = 0;
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int nr = row + dy;
                int nc = col + dx;
                if (nr >= 0 && nr < height && nc >= 0 && nc < width)
                {
                    const float *neighborRow =
                        (const float *)((const char *)input + nr * pitch);
                    sum += neighborRow[nc];
                    count++;
                }
            }
        }
        outRow[col] = sum / count;
    }
}

/**
 * 初始化二维输入（主机端）
 * 每元素值 = row + col，存储在 pitch 对齐的布局中
 */
void initHost2D(float *h_data, size_t pitch, int width, int height)
{
    for (int r = 0; r < height; r++)
    {
        float *row = (float *)((char *)h_data + r * pitch);
        for (int c = 0; c < width; c++)
        {
            row[c] = (float)(r + c);
        }
    }
}

/**
 * 验证二维输出
 * 3x3 均值滤波器应该使每个输出像素等于其 3x3 邻域的平均值
 */
int verifyOutput2D(const float *h_input, const float *h_output,
                   size_t pitch, int width, int height)
{
    int errors = 0;
    for (int r = 0; r < height; r++)
    {
        const float *inRow = (const float *)((const char *)h_input + r * pitch);
        const float *outRow =
            (const float *)((const char *)h_output + r * pitch);
        for (int c = 0; c < width; c++)
        {
            // 手动计算 3x3 均值
            float expected = 0.0f;
            int count = 0;
            for (int dy = -1; dy <= 1; dy++)
            {
                for (int dx = -1; dx <= 1; dx++)
                {
                    int nr = r + dy;
                    int nc = c + dx;
                    if (nr >= 0 && nr < height && nc >= 0 && nc < width)
                    {
                        const float *nRow =
                            (const float *)((const char *)h_input +
                                            nr * pitch);
                        expected += nRow[nc];
                        count++;
                    }
                }
            }
            expected /= count;

            if (fabsf(outRow[c] - expected) > 1e-5f)
            {
                if (errors < 5)
                {
                    printf("  Error at (%d,%d): expected %f, got %f\n", r, c,
                           expected, outRow[c]);
                }
                errors++;
            }
        }
    }
    return errors;
}

int main()
{
    // =========================================================================
    // 参数设置
    // =========================================================================
    int width = 1024;
    int height = 1024;

    printf("========================================\n");
    printf("2D Convolution using cudaMallocPitch\n");
    printf("========================================\n");
    printf("Image size: %d x %d\n", width, height);

    // =========================================================================
    // 使用 cudaMallocPitch 分配设备内存
    // =========================================================================
    size_t pitch;
    float *d_input, *d_output;

    CUDA_CHECK(cudaMallocPitch(&d_input, &pitch,
                                width * sizeof(float), height));
    CUDA_CHECK(cudaMallocPitch(&d_output, &pitch,
                                width * sizeof(float), height));

    printf("Requested line width: %zu bytes\n", width * sizeof(float));
    printf("Actual pitch:         %zu bytes\n", pitch);
    printf("Padding per row:      %zu bytes\n",
           pitch - width * sizeof(float));

    // =========================================================================
    // 分配主机内存（使用相同的 pitch 布局）
    // =========================================================================
    float *h_input = (float *)malloc(height * pitch);
    float *h_output = (float *)malloc(height * pitch);
    if (h_input == NULL || h_output == NULL)
    {
        printf("Error: Host memory allocation failed!\n");
        return -1;
    }

    // =========================================================================
    // 初始化输入数据
    // =========================================================================
    initHost2D(h_input, pitch, width, height);
    printf("Input initialized (value = row + col).\n");
    printf("  h_input[0][0] = %f\n", h_input[0]);
    printf("  h_input[%d][%d] = %f\n", height - 1, width - 1,
           ((float *)((char *)h_input + (height - 1) * pitch))[width - 1]);

    // =========================================================================
    // 使用 cudaMemcpy2D 进行二维数据拷贝
    // =========================================================================
    printf("Copying data Host -> Device (cudaMemcpy2D)...\n");
    CUDA_CHECK(cudaMemcpy2D(d_input, pitch,        // 设备端：dst + dpitch
                            h_input, pitch,        // 主机端：src + spitch
                            width * sizeof(float), // 行宽（逻辑宽度）
                            height,                // 行数
                            cudaMemcpyHostToDevice));

    // =========================================================================
    // 启动二维内核
    // =========================================================================
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);
    printf("Launching kernel: grid(%d,%d) x block(%d,%d)...\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    Convolve2D<<<gridDim, blockDim>>>(d_input, d_output, pitch,
                                       width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel time: %.3f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaGetLastError());

    // =========================================================================
    // 拷贝结果回主机
    // =========================================================================
    printf("Copying result Device -> Host...\n");
    CUDA_CHECK(cudaMemcpy2D(h_output, pitch, d_output, pitch,
                            width * sizeof(float), height,
                            cudaMemcpyDeviceToHost));

    // =========================================================================
    // 验证结果
    // =========================================================================
    printf("Verifying results...\n");
    int errors = verifyOutput2D(h_input, h_output, pitch, width, height);
    if (errors == 0)
    {
        printf("  All %d elements verified successfully!\n",
               width * height);
    }
    else
    {
        printf("  Found %d errors out of %d elements.\n", errors,
               width * height);
    }

    // 打印几个样本输出值
    printf("\nSample output values:\n");
    printf("  output[0][0] = %f\n", h_output[0]);
    printf("  output[%d][%d] = %f\n", height / 2, width / 2,
           ((float *)((char *)h_output + (height / 2) * pitch))[width / 2]);
    printf("  output[%d][%d] = %f\n", height - 1, width - 1,
           ((float *)((char *)h_output + (height - 1) * pitch))[width - 1]);

    // =========================================================================
    // 释放内存
    // =========================================================================
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    printf("========================================\n");
    printf("Done!\n");
    printf("========================================\n");

    return 0;
}
