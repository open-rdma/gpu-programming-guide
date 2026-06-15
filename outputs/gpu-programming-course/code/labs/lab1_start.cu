/**
 * 实验1：朴素矩阵乘法实现 - 起始代码
 *
 * 这是你的起点。你需要完成以下任务：
 * 1. 编写 sgemm_naive kernel，实现第一个 CUDA 矩阵乘法
 * 2. 理解线程索引计算
 * 3. 配置正确的 kernel 启动参数
 * 4. 验证结果正确性并与 cuBLAS 对比性能
 *
 * 编译: nvcc lab1_start.cu -o lab1_start -lcublas
 * 运行: ./lab1_start
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// ========== 宏定义 ==========
#define CEIL_DIV(a, b)  (((a) + (b) - 1) / (b))

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error [%s:%d]: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

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
// ==============================================================

// ============================================================================
// TODO: 在此处编写你的 sgemm_naive kernel
// 提示：
//   1. 使用 blockIdx, blockDim, threadIdx 计算全局位置 (x, y)
//   2. 检查越界: if (x < M && y < N)
//   3. 内积循环: for (int i = 0; i < K; ++i) tmp += A[x*K+i] * B[i*N+y]
//   4. 写回结果: C[x*N+y] = alpha * tmp + beta * C[x*N+y]
// ============================================================================

// TODO: 实现 sgemm_naive kernel
// __global__ void sgemm_naive(...) { ... }

int main() {

  // 矩阵参数
  const int M = 4096;
  const int N = 4096;
  const int K = 4096;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int num_warmup = 5;
  const int num_iter = 10;

  printf("\n矩阵大小: M=%d, N=%d, K=%d\n", M, N, K);

  // 在 CPU 端分配矩阵
  float *A = (float *)malloc(M * K * sizeof(float));
  float *B = (float *)malloc(K * N * sizeof(float));
  float *C = (float *)malloc(M * N * sizeof(float));
  float *C_ref = (float *)malloc(M * N * sizeof(float));

  // 初始化矩阵
  randomize_matrix(A, M * K);
  randomize_matrix(B, K * N);
  zero_init_matrix(C, M * N);
  zero_init_matrix(C_ref, M * N);

  // 在 GPU 端分配矩阵
  float *d_A, *d_B, *d_C, *d_C_ref;
  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(float)));

  // 将数据拷贝到 GPU
  CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C_ref, C_ref, M * N * sizeof(float), cudaMemcpyHostToDevice));

  // ========================================================================
  // 使用 cuBLAS 获取参考结果（··可选··）
  // ========================================================================
  // cublasHandle_t handle;
  // cublasCreate(&handle);
  // runCublasSgemm(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);

  // ========================================================================
  // TODO: 配置 kernel 启动参数并运行你的 kernel
  // 提示: dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  //       dim3 blockDim(32, 32);
  // ========================================================================

  // TODO: 计算 gridDim 和 blockDim
  // dim3 gridDim(...);
  // dim3 blockDim(...);

  // TODO: 调用你的 kernel
  // sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);

  printf("\n请在代码中完成 TODO 部分的实现，然后重新编译运行。\n");

  // 清理：删掉未定义的 cublasDestroy(handle)
  // cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_C_ref);
  free(A);
  free(B);
  free(C);
  free(C_ref);

  return 0;
}