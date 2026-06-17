/**
 * 实验1：朴素矩阵乘法实现 - 参考解答
 *
 * 第一个 CUDA SGEMM kernel，使用 2D grid 和 2D block。
 * 每个线程计算结果矩阵 C 的一个元素。
 *
 * 预期性能：矩阵 4096x4096 时约 309 GFLOPS/s
 *
 * 编译: nvcc lab1_solution.cu -o lab1_solution -lcublas
 * 运行: ./lab1_solution
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// ===================== 宏定义 =====================
#define BLOCK_SIZE      32
#define CEIL_DIV(a, b)  (((a) + (b) - 1) / (b))

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error [%s:%d]: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ===================== 工具函数 =====================
// 随机初始化矩阵，值域 [0, 1]
void randomize_matrix(float *mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// ===================== 朴素 SGEMM Kernel =====================
__global__ void sgemm_naive(int M, int N, int K, float alpha,
                             const float *A, const float *B,
                             float beta, float *C) {
    // 计算当前线程在 C 矩阵中的全局坐标
    const unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int col = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查：处理 tile quantization 越界问题
    if (row < M && col < N) {
        float tmp = 0.0f;
        // 内积循环：A 的第 row 行 × B 的第 col 列
        for (int i = 0; i < K; ++i) {
            tmp += A[row * K + i] * B[i * N + col];
        }
        // 标准 GEMM 公式: C = alpha * A*B + beta * C
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

// ===================== 主函数 =====================
int main() {
    // 矩阵参数
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    const int num_warmup = 5;
    const int num_iter   = 10;

    printf("===== CUDA Naive SGEMM =====\n");
    printf("Matrix: A(%dx%d), B(%dx%d), C(%dx%d)\n", M, K, K, N, M, N);

    // 初始化随机种子
    srand((unsigned int)time(NULL));

    // 1. 分配主机内存
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    randomize_matrix(h_A, M * K);
    randomize_matrix(h_B, K * N);
    for (int i = 0; i < M * N; ++i) {
        h_C[i] = 0.0f;
    }

    // 2. 分配设备显存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // 3. 数据拷贝 Host -> Device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // 4. 配置 Kernel 启动参数
    dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    printf("Grid: (%d, %d), Block: (%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // 5. Kernel 预热（消除冷启动开销）
    for (int i = 0; i < num_warmup; ++i) {
        sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. 正式计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iter; ++i) {
        sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // 7. 计算性能
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / num_iter;
    double gflops = (2.0 * M * N * K) / (avg_ms * 1e6);

    printf("\nAverage Time: %.4f ms\n", avg_ms);
    printf("Performance:  %.1f GFLOPS\n", gflops);

    // 8. 资源释放
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Program finished.\n");
    return 0;
}