/**
 * 实验8：Warp Tile、Bank Conflict 与性能对比 cuBLAS - 起始代码
 *
 * 起点：实验7（参数自动调优）
 * 目标：实现显式的 Warp Tiling，最终达到 cuBLAS 93.7% 性能
 *
 * 任务：
 * 1. 显式计算 Warp 索引 (warpIdx, warpCol, warpRow)
 * 2. 从 SMEM 批量加载 Warp Subtile 数据到寄存器
 * 3. 在 regM[WMITER*TM] 和 regN[WNITER*TN] 上执行密集外积
 * 4. 理解 Bank Conflict 及其解决方案
 *
 * 编译: nvcc lab8_start.cu -o lab8_start -lcublas
 */

#include "sgemm_common.h"
#include <cublas_v2.h>

// Warp size 是硬件常量，不是 C++ constexpr
#define WARPSIZE 32

// TODO: 实现 loadFromGmem 和 processFromSmem 两个 device 函数
// TODO: 实现 sgemmWarptiling kernel

int main() {
  CudaDeviceInfo();
  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f, beta = 0.0f;

  printf("\n========================================\n");
  printf("实验8：Warp Tile + 对比 cuBLAS\n");
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

  // TODO: 配置 A6000 最优参数
  // NUM_THREADS=128, BM=128, BN=128, BK=16, WM=64, WN=64, WNITER=4, TM=8, TN=4

  printf("\n请在代码中完成 TODO 部分的实现。\n");

  cublasDestroy(handle);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
  free(A); free(B); free(C); free(C_ref);
  return 0;
}
