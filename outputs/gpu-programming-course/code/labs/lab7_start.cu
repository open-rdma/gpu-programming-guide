/**
 * 实验7：参数自动调优 - 起始代码
 *
 * 起点：实验6（向量化内存访问）
 * 目标：引入 Warp Tile 概念，试验不同 Tile 参数组合，
 *       使用 __launch_bounds__ 提示编译器
 *
 * 任务：
 * 1. 将 kernel 改为带 BM, BN, BK, TM, TN 的模板
 * 2. 添加 Warp Tile 划分（WMITER, WNITER）
 * 3. 实现 warp 内部的外积循环
 * 4. 尝试 BK=16 等不同参数
 *
 * 编译: nvcc lab7_start.cu -o lab7_start -lcublas
 */

#include "sgemm_common.h"
#include <cublas_v2.h>

const int NUM_THREADS = 256;

// TODO: 实现带 Warp Tile 划分的 autotuned kernel
// template <const int BM, const int BN, const int BK, const int TM, const int TN>
// __global__ void __launch_bounds__(NUM_THREADS) sgemmAutotuned(...)
//
// 提示：
//   - WM = TM * 16; WN = TN * 16
//   - WMITER = CEIL_DIV(BM, WM); WNITER = CEIL_DIV(BN, WN)
//   - threadCol = threadIdx.x % (WN / TN); threadRow = threadIdx.x / (WN / TN)
//   - 使用 float4 向量化加载（A 转置 + B 直接）
//   - 三层循环：wmIdx, wnIdx, dotIdx
//   - SMEM 加载使用 rowStrideA/rowStrideB 实现多元素加载

int main() {
  CudaDeviceInfo();
  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f, beta = 0.0f;

  printf("\n========================================\n");
  printf("实验7：参数自动调优\n");
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

  // TODO: 配置参数并启动 kernel
  // A6000 最优：BM=128, BN=128, BK=16, TM=8, TN=8
  // dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  // dim3 blockDim(NUM_THREADS);

  printf("\n请在代码中完成 TODO 部分的实现。\n");

  cublasDestroy(handle);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
  free(A); free(B); free(C); free(C_ref);
  return 0;
}
