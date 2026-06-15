/**
 * 实验2：全局内存合并访问 - 起始代码
 *
 * 起点：实验1（朴素实现，2D block）
 * 目标：修改线程到元素的映射关系，实现全局内存合并访问
 *
 * 任务：
 * 1. 分析实验1的内存访问模式，找出非合并访问的位置
 * 2. 重写 kernel，将 2D block 改为 1D block
 * 3. 修改线程索引计算，使 threadIdx.x 连续对应连续的 C 列坐标
 * 4. 对比实验1与实验2的性能提升
 *
 * 编译: nvcc lab2_start.cu -o lab2_start -lcublas
 * 运行: ./lab2_start
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// ============================================================================
// TODO: 将实验1的 2D kernel 改写为 1D kernel，实现合并访问
//
// 实验1的代码（需要改写）：
//   const uint x = blockIdx.x * blockDim.x + threadIdx.x;
//   const uint y = blockIdx.y * blockDim.y + threadIdx.y;
//
// 改写思路：
//   - blockDim 从 2D(32,32) 变为 1D(32*32)
//   - 使用 cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE)
//   - 使用 cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE)
//   - 注意需要模板参数 BLOCKSIZE
// ============================================================================

// TODO: 实现带合并访问的 kernel
// template <const uint BLOCKSIZE>
// __global__ void sgemm_global_mem_coalesce(...) { ... }

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
  printf("实验2：全局内存合并访问\n");
  printf("矩阵大小: M=%d, N=%d, K=%d\n", M, N, K);
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

  // // cuBLAS reference（可选）
  // cublasHandle_t handle;
  // cublasCreate(&handle);
  // runCublasSgemm(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);

  // TODO: 配置 1D block 启动参数
  // dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  // dim3 blockDim(32 * 32);
  // sgemm_global_mem_coalesce<32><<<gridDim, blockDim>>>(...);

  printf("\n请在代码中完成 TODO 部分的实现。\n");

  // cublasDestroy(handle);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
  free(A); free(B); free(C); free(C_ref);
  return 0;
}
