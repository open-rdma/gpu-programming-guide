/**
 * 实验6：向量化内存访问 - 起始代码
 *
 * 起点：实验5（2D Block Tile）
 * 目标：使用 float4 向量化 GMEM 和 SMEM 的访问，
 *       对 A 进行转置加载以使 SMEM 中 As 的加载可被向量化
 *
 * 任务：
 * 1. 使用 float4 从 GMEM 加载 A（带转置写入 SMEM）
 * 2. 使用 float4 从 GMEM 加载 B（直接写入 SMEM）
 * 3. 调整 As 读取索引（转置后为列主序，读取连续）
 * 4. 使用 float4 向量化 C 的写回
 *
 * 编译: nvcc lab6_start.cu -o lab6_start -lcublas
 */

#include "sgemm_common.h"
#include <cublas_v2.h>

// TODO: 实现向量化内存访问的 kernel
// 提示：
//   1. A 加载并转置：
//      float4 tmp = reinterpret_cast<float4 *>(&A[innerRowA*K + innerColA*4])[0];
//      As[(innerColA*4+0)*BM + innerRowA] = tmp.x; ...
//   2. B 直接 float4 加载：
//      reinterpret_cast<float4 *>(&Bs[innerRowB*BN + innerColB*4])[0] =
//          reinterpret_cast<float4 *>(&B[innerRowB*N + innerColB*4])[0];
//   3. 注意 innerColA 改为 threadIdx.x % (BK/4)
//   4. 转置后 As 按列主序：regM[i] = As[dotIdx*BM + threadRow*TM + i]; // 连续！
//   5. C 写回用 float4

int main() {
  CudaDeviceInfo();
  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f, beta = 0.0f;

  printf("\n========================================\n");
  printf("实验6：向量化内存访问\n");
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

  printf("\n请在代码中完成 TODO 部分的实现。\n");

  cublasDestroy(handle);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
  free(A); free(B); free(C); free(C_ref);
  return 0;
}
