/**
 * vector_add.cu
 *
 * 第5章动手体验：完整的向量加法程序
 *
 * 展示 CUDA 内存管理的标准五步流程：
 *   cudaMalloc → cudaMemcpy(HtoD) → kernel → cudaMemcpy(DtoH) → cudaFree
 *
 * 编译：
 *   nvcc vector_add.cu -o vector_add -arch=sm_60
 *
 * 运行：
 *   ./vector_add
 */

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

/**
 * 错误检查宏
 * 对每个 CUDA API 调用进行错误检查
 */
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
 * 向量加法内核
 * 每个线程计算一个元素的 C[i] = A[i] + B[i]
 */
__global__ void VecAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * 初始化向量 h_A 和 h_B
 * h_A[i] = i, h_B[i] = i * 2
 */
void initVectors(float *h_A, float *h_B, int N)
{
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }
}

/**
 * 验证结果
 * 检查 h_C[i] == h_A[i] + h_B[i]
 * 返回错误个数
 */
int verifyResult(const float *h_A, const float *h_B, const float *h_C,
                 int N)
{
    int errors = 0;
    for (int i = 0; i < N; ++i)
    {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f)
        {
            if (errors < 10)
            { // 只打印前 10 个错误
                printf("  Error at %d: expected %f, got %f\n", i,
                       h_A[i] + h_B[i], h_C[i]);
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
    int N = 1 << 20; // 1M 元素（2^20）
    size_t size = N * sizeof(float);

    printf("========================================\n");
    printf("Vector Addition with CUDA\n");
    printf("========================================\n");
    printf("Vector length: %d elements\n", N);
    printf("Data size: %.2f MB per vector\n",
           (float)size / (1024.0f * 1024.0f));
    printf("\n");

    // =========================================================================
    // 第1步：分配主机内存
    // =========================================================================
    printf("[Step 1] Allocating host memory...\n");
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        printf("Error: Host memory allocation failed!\n");
        return -1;
    }

    // =========================================================================
    // 第2步：初始化输入数据
    // =========================================================================
    printf("[Step 2] Initializing input vectors...\n");
    initVectors(h_A, h_B, N);
    printf("  h_A[0]=%.2f, h_A[N-1]=%.2f\n", h_A[0], h_A[N - 1]);
    printf("  h_B[0]=%.2f, h_B[N-1]=%.2f\n", h_B[0], h_B[N - 1]);

    // =========================================================================
    // 第3步：分配设备内存
    // =========================================================================
    printf("[Step 3] Allocating device memory...\n");
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));

    // =========================================================================
    // 第4步：主机 → 设备 数据传输
    // =========================================================================
    printf("[Step 4] Copying data Host -> Device...\n");
    CUDA_CHECK(
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // =========================================================================
    // 第5步：启动内核
    // =========================================================================
    printf("[Step 5] Launching kernel...\n");
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("  Grid: %d blocks, Block: %d threads\n", blocksPerGrid,
           threadsPerBlock);

    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));

    // 等待内核执行完成
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("  Kernel execution time: %.3f ms\n", milliseconds);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 检查内核启动错误
    CUDA_CHECK(cudaGetLastError());

    // =========================================================================
    // 第6步：设备 → 主机 数据传输
    // =========================================================================
    printf("[Step 6] Copying result Device -> Host...\n");
    CUDA_CHECK(
        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // =========================================================================
    // 第7步：验证结果
    // =========================================================================
    printf("[Step 7] Verifying results...\n");
    int errors = verifyResult(h_A, h_B, h_C, N);
    if (errors == 0)
    {
        printf("  All %d elements verified successfully!\n", N);
    }
    else
    {
        printf("  Found %d errors out of %d elements.\n", errors, N);
    }

    // =========================================================================
    // 第8步：释放设备内存
    // =========================================================================
    printf("[Step 8] Freeing device memory...\n");
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // =========================================================================
    // 第9步：释放主机内存
    // =========================================================================
    free(h_A);
    free(h_B);
    free(h_C);

    printf("========================================\n");
    printf("Done!\n");
    printf("========================================\n");

    return 0;
}
