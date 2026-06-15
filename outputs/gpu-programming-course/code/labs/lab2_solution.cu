/**
 * 实验2：全局内存合并访问 - 在线测评提交版
 * 无 cuBLAS 依赖，可直接在 CMake 环境编译运行
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// ===================== 宏定义 =====================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error [%s:%d]: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ===================== 辅助函数 =====================
void randomize_matrix(float *mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

void zero_init_matrix(float *mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = 0.0f;
    }
}

double calculate_gflops(int M, int N, int K, double time_ms) {
    // 2 * M * N * K 浮点操作（乘加算两个 FLOP）
    double flops = 2.0 * M * N * K;
    return flops / (time_ms * 1e6);
}

void print_device_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return;
    }
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
}

// ===================== Kernel: 合并访问 =====================
template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                           const float *A, const float *B,
                                           float beta, float *C) {
    // 重映射：threadIdx.x 连续 => cCol 连续
    const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (cRow < M && cCol < N) {
        float tmp = 0.0f;
        for (int i = 0; i < K; ++i) {
            // 合并访问：B[i * N + cCol] 地址连续
            tmp += A[cRow * K + i] * B[i * N + cCol];
        }
        C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
    }
}

// ===================== 主函数 =====================
int main() {
    srand((unsigned int)time(NULL));
    print_device_info();

    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int num_warmup = 5;
    const int num_iter = 10;

    printf("\n========================================\n");
    printf("实验2：全局内存合并访问\n");
    printf("矩阵大小: M=%d, N=%d, K=%d\n", M, N, K);
    printf("========================================\n");

    // 分配主机内存
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));

    // 初始化
    randomize_matrix(A, M * K);
    randomize_matrix(B, K * N);
    zero_init_matrix(C, M * N);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // 拷贝到设备
    CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel 配置
    const uint BLOCKSIZE = 32;
    dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);   // 1024 线程，1D
    printf("Kernel 配置: Grid(%d, %d), Block(%d) 1D\n", gridDim.x, gridDim.y, blockDim.x);

    // 预热
    for (int i = 0; i < num_warmup; ++i) {
        sgemm_global_mem_coalesce<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iter; ++i) {
        sgemm_global_mem_coalesce<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float avg_ms = elapsed_ms / num_iter;

    double gflops = calculate_gflops(M, N, K, avg_ms);

    printf("\n========================================\n");
    printf("实验结果:\n");
    printf("  平均耗时: %.4f ms\n", avg_ms);
    printf("  计算性能: %.1f GFLOPS/s\n", gflops);
    printf("========================================\n");

    // 清理
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);

    return 0;
}