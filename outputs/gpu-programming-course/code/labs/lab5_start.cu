/**
 * 实验5：二维 Block Tile - 起始代码
 *
 * 起点：实验4（1D Block Tile）
 * 目标：扩展到 2D Thread Tile (TM x TN)，使用 Register Blocking
 *       和 Strided Load 从 GMEM 加载多个元素
 *
 * 任务：
 * 1. 增加 TN 参数，每个线程计算 TM*TN 个结果
 * 2. 实现多元素加载（使用 loadOffset 循环）
 * 3. 实现 Register Blocking（regM[TM], regN[TN]，外积模式）
 * 4. 调整线程索引计算
 *
 * 编译: nvcc lab5_start.cu -o lab5_start -lcublas
 */

#include "sgemm_common.h"
#include <cublas_v2.h>

// TODO: 实现带 2D Block Tile 的 kernel
// template <const int BM, const int BN, const int BK, const int TM, const int TN>
// __global__ void sgemm2DBlocktiling(...)
//
// 提示：
//   - threadCol = threadIdx.x % (BN / TN); threadRow = threadIdx.x / (BN / TN)
//   - 多元素加载：for (loadOffset) As[offset] = A[offset]
//   - Register Blocking: regM[TM], regN[TN], threadResults[TM*TN]
//   - 外积: threadResults[M*TN+N] += regM[M] * regN[N]

int main() {
  CudaDeviceInfo();
  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f, beta = 0.0f;

  printf("\n========================================\n");
  printf("实验5：二维 Block Tile\n");
  printf("========================================\n");

  float *A = (float *)malloc(M * K * sizeof(float));
  float *B = (float *)malloc(K * N * sizeof(float));
  float *C = (float *)malloc(M * N * sizeof(float));
  float *C_ref = (float *)malloc(M * N * sizeof(float));

  randomize_matrix(A, M * K);
  randomize_matrix(B, K * N);
  zero_init_matrix(C, M * N);
  zero_init_matrix(C_ref, M * N);

  float *d_A, *d_B, *d_C, *d_C_ref;
  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C_ref, M * N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C_ref, C_ref, M * N * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  cublasCreate(&handle);
  runCublasSgemm(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);

  // TODO: 配置参数
  // const uint BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  // dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  // dim3 blockDim((BM * BN) / (TM * TN));

  printf("\n请在代码中完成 TODO 部分的实现。\n");

  cublasDestroy(handle);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
  free(A); free(B); free(C); free(C_ref);
  return 0;
}
