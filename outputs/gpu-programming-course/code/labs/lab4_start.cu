/**
 * 实验4：一维 Block Tile - 起始代码
 *
 * 起点：实验3（共享内存缓存分块）
 * 目标：引入 1D Block Tile，每个线程计算 TM 个结果，
 *       提高算术强度，减少 SMEM 访问压力
 *
 * 任务：
 * 1. 引入模板参数 BM, BN, BK, TM
 * 2. 每个线程分配 float threadResults[TM]
 * 3. 重新组织循环结构（dotIdx 在外层）
 * 4. 写回 TM 个结果
 *
 * 编译: nvcc lab4_start.cu -o lab4_start -lcublas
 * 运行: ./lab4_start
 */

#include "sgemm_common.h"
#include <cublas_v2.h>

// TODO: 实现带 1D Block Tile 的 kernel
// template <const int BM, const int BN, const int BK, const int TM>
// __global__ void sgemm1DBlocktiling(...)
//
// 提示：
//   - threadCol = threadIdx.x % BN; threadRow = threadIdx.x / BN
//   - float threadResults[TM] = {0.0}
//   - 外循环沿 K/BK 分块，内循环 dotIdx 在外层（缓存 Btmp）
//   - 每个线程计算 TM 个结果（沿 M 方向）

int main() {
  CudaDeviceInfo();

  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f, beta = 0.0f;

  printf("\n========================================\n");
  printf("实验4：一维 Block Tile\n");
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

  // TODO: 配置参数和启动 kernel
  // const uint BM = 64, BN = 64, BK = 8, TM = 8;
  // dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  // dim3 blockDim((BM * BN) / TM);

  printf("\n请在代码中完成 TODO 部分的实现。\n");

  cublasDestroy(handle);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
  free(A); free(B); free(C); free(C_ref);
  return 0;
}
