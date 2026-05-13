/**
 * 实验3：共享内存缓存分块 - 起始代码
 *
 * 起点：实验2（全局内存合并访问）
 * 目标：使用共享内存缓存 A 和 B 的子块，减少 GMEM 访问次数
 *
 * 任务：
 * 1. 声明 __shared__ float As[BLOCKSIZE * BLOCKSIZE] 和 Bs[BLOCKSIZE * BLOCKSIZE]
 * 2. 协作加载数据到共享内存
 * 3. 正确放置 __syncthreads() 同步点
 * 4. 在共享内存上执行分块内积计算
 *
 * 编译: nvcc lab3_start.cu -o lab3_start -lcublas
 * 运行: ./lab3_start
 */

#include "sgemm_common.h"
#include <cublas_v2.h>

// TODO: 实现使用共享内存缓存分块的 kernel
// 参考结构：
//   __shared__ float As[BLOCKSIZE * BLOCKSIZE];
//   __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];
//   外层循环：for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
//     协作加载 As, Bs
//     __syncthreads();
//     内积循环
//     __syncthreads();

int main() {
  CudaDeviceInfo();

  const int M = 4096;
  const int N = 4096;
  const int K = 4096;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int num_warmup = 5;
  const int num_iter = 10;

  printf("\n========================================\n");
  printf("实验3：共享内存缓存分块\n");
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

  // cuBLAS reference
  cublasHandle_t handle;
  cublasCreate(&handle);
  runCublasSgemm(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);

  // TODO: 配置 kernel 启动参数并调用你的 kernel
  // dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  // dim3 blockDim(32 * 32);
  // 提示：使用 cudaFuncSetAttribute 配置 SMEM carveout

  printf("\n请在代码中完成 TODO 部分的实现。\n");

  cublasDestroy(handle);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
  free(A); free(B); free(C); free(C_ref);
  return 0;
}
