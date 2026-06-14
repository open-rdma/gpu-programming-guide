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
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// 随机初始化矩阵
void randomize_matrix(float *mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// 朴素 SGEMM kernel
__global__ void sgemm_naive(int M, int N, int K, float alpha,
                             const float *A, const float *B,
                             float beta, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int BLOCK_SIZE = 32;
    const int num_warmup = 5;
    const int num_iter = 10;

    printf("Matrix: M=%d, N=%d, K=%d\n", M, N, K);

    // 分配主机内存
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));
    if (!A || !B || !C) {
        printf("Host memory allocation failed\n");
        return 1;
    }

    // 初始化
    randomize_matrix(A, M * K);
    randomize_matrix(B, K * N);
    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // 拷贝到设备
    CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    printf("Grid: (%d,%d), Block: (%d,%d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // 预热
    for (int i = 0; i < num_warmup; ++i) {
        sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iter; ++i) {
        sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float avg_ms = elapsed_ms / num_iter;

    double gflops = (2.0 * M * N * K) / (avg_ms * 1e6);
    printf("Average time: %.4f ms\n", avg_ms);
    printf("Performance: %.1f GFLOPS\n", gflops);

    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}