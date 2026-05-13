/**
 * matrix_transpose.cu
 *
 * 第6章动手体验：矩阵转置——共享内存优化
 *
 * 比较两种实现的性能：
 *   1. TransposeNaive:  朴素实现（非合并写入）
 *   2. TransposeShared: 共享内存分块实现（合并读写 + padding 避免 bank conflict）
 *
 * 编译：
 *   nvcc matrix_transpose.cu -o matrix_transpose -arch=sm_60
 *
 * 运行：
 *   ./matrix_transpose
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

// 分块大小
#define TILE_DIM 32

// =========================================================================
// 朴素转置内核
// 问题：output 的写入不是合并访问（stride = height）
// =========================================================================
__global__ void TransposeNaive(const float *input, float *output,
                               int width, int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height)
    {
        // 读取是合并的（连续），但写入 stride 为 height，不是合并的
        output[col * height + row] = input[row * width + col];
    }
}

// =========================================================================
// 使用共享内存优化的转置内核
//
// 策略：
//   1. 合并读取：每个线程从全局内存读取一个元素到共享内存 tile
//   2. __syncthreads() 确保所有线程加载完成
//   3. 在共享内存中"转置"读取：threadIdx.x 和 threadIdx.y 互换
//   4. 合并写入：从共享内存写入全局内存
//
// 注意：tile[32][33] 多了一列 padding，避免 32 列一字对齐造成 bank conflict
// =========================================================================
__global__ void TransposeShared(const float *input, float *output,
                                int width, int height)
{
    // tile[32][33] 的 padding 避免了列访问时的 bank conflict
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    // 步骤 1：合并读取（连续的 threadIdx.x 对应连续的内存地址）
    if (col < width && row < height)
    {
        tile[threadIdx.y][threadIdx.x] = input[row * width + col];
    }

    // 步骤 2：屏障同步——确保 tile 中所有数据都加载完毕
    __syncthreads();

    // 步骤 3：计算转置后在输出矩阵中的位置
    // 注意：blockIdx.x 和 blockIdx.y 在转置中互换了角色
    int outRow = blockIdx.x * TILE_DIM + threadIdx.y;
    int outCol = blockIdx.y * TILE_DIM + threadIdx.x;

    // 步骤 4：合并写入（连续的 threadIdx.y 对应连续的内存地址）
    if (outRow < height && outCol < width)
    {
        // 在共享内存 tile 中做"转置读取"
        // threadIdx.x 和 threadIdx.y 互换
        output[outRow * width + outCol] = tile[threadIdx.x][threadIdx.y];
    }
}

// =========================================================================
// 初始化矩阵（行主序）
// 使用 row * width + col 作为值，方便验证
// =========================================================================
void initMatrix(float *mat, int width, int height)
{
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            mat[r * width + c] = (float)(r * width + c);
        }
    }
}

// =========================================================================
// 在 CPU 上计算转置（用于验证）
// =========================================================================
void transposeCPU(const float *input, float *output, int width, int height)
{
    for (int r = 0; r < height; r++)
    {
        for (int c = 0; c < width; c++)
        {
            output[c * height + r] = input[r * width + c];
        }
    }
}

// =========================================================================
// 验证结果
// =========================================================================
int verifyTranspose(const float *gpuResult, const float *cpuResult,
                    int width, int height)
{
    int errors = 0;
    for (int i = 0; i < width * height; i++)
    {
        if (fabsf(gpuResult[i] - cpuResult[i]) > 1e-5f)
        {
            if (errors < 10)
            {
                printf("  Error at %d: expected %f, got %f\n", i,
                       cpuResult[i], gpuResult[i]);
            }
            errors++;
        }
    }
    return errors;
}

int main()
{
    // =========================================================================
    // 参数设置
    // =========================================================================
    int width = 4096;
    int height = 4096;
    size_t size = (size_t)width * height * sizeof(float);

    printf("========================================\n");
    printf("Matrix Transpose with Shared Memory\n");
    printf("========================================\n");
    printf("Matrix size: %d x %d\n", width, height);
    printf("Memory size: %.2f MB\n", (float)size / (1024.0f * 1024.0f));
    printf("Tile dimension: %d x %d\n\n", TILE_DIM, TILE_DIM);

    // =========================================================================
    // 分配主机内存
    // =========================================================================
    float *h_input = (float *)malloc(size);
    float *h_output_naive = (float *)malloc(size);
    float *h_output_shared = (float *)malloc(size);
    float *h_verify = (float *)malloc(size);

    if (!h_input || !h_output_naive || !h_output_shared || !h_verify)
    {
        printf("Error: Host memory allocation failed!\n");
        return -1;
    }

    // 初始化输入
    printf("Initializing matrix...\n");
    initMatrix(h_input, width, height);

    // 在 CPU 上计算转置作为参考
    transposeCPU(h_input, h_verify, width, height);
    printf("CPU reference transpose completed.\n\n");

    // =========================================================================
    // 分配设备内存
    // =========================================================================
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));

    // 拷贝输入数据到设备
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // =========================================================================
    // 测试1：朴素转置
    // =========================================================================
    printf("Running naive transpose...\n");

    dim3 blockDim1(TILE_DIM, TILE_DIM);
    dim3 gridDim1((width + blockDim1.x - 1) / blockDim1.x,
                  (height + blockDim1.y - 1) / blockDim1.y);

    // GPU 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    TransposeNaive<<<gridDim1, blockDim1>>>(d_input, d_output,
                                              width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_naive;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));

    CUDA_CHECK(cudaMemcpy(h_output_naive, d_output, size,
                          cudaMemcpyDeviceToHost));

    int errors_naive = verifyTranspose(h_output_naive, h_verify,
                                        width, height);
    printf("  Naive transpose time: %.3f ms\n", ms_naive);
    printf("  Verification: %s\n",
           errors_naive == 0 ? "PASSED" : "FAILED");

    // =========================================================================
    // 测试2：共享内存优化转置
    // =========================================================================
    printf("Running shared memory transpose...\n");

    // 重置 d_output
    CUDA_CHECK(cudaMemset(d_output, 0, size));

    // 注意：共享内存版本的 grid 块索引互换（因为输出是转置后的）
    dim3 blockDim2(TILE_DIM, TILE_DIM);
    dim3 gridDim2((width + blockDim2.x - 1) / blockDim2.x,
                  (height + blockDim2.y - 1) / blockDim2.y);

    CUDA_CHECK(cudaEventRecord(start));
    TransposeShared<<<gridDim2, blockDim2>>>(d_input, d_output,
                                              width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_shared;
    CUDA_CHECK(cudaEventElapsedTime(&ms_shared, start, stop));

    CUDA_CHECK(cudaMemcpy(h_output_shared, d_output, size,
                          cudaMemcpyDeviceToHost));

    int errors_shared = verifyTranspose(h_output_shared, h_verify,
                                         width, height);
    printf("  Shared memory transpose time: %.3f ms\n", ms_shared);
    printf("  Verification: %s\n",
           errors_shared == 0 ? "PASSED" : "FAILED");

    // =========================================================================
    // 结果汇总
    // =========================================================================
    printf("\n========================================\n");
    printf("RESULTS SUMMARY\n");
    printf("========================================\n");
    printf("Naive transpose:         %.3f ms\n", ms_naive);
    printf("Shared memory transpose: %.3f ms\n", ms_shared);
    printf("Speedup:                 %.2fx\n",
           ms_naive / ms_shared);
    printf("========================================\n");

    // =========================================================================
    // 释放资源
    // =========================================================================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output_naive);
    free(h_output_shared);
    free(h_verify);

    printf("Done!\n");
    return 0;
}
