/**
 * 实验2：全局内存合并访问 - 参考解答
 *
 * 将 2D block 改为 1D block，重新映射线程到 C 矩阵的位置关系，
 * 使连续的 threadIdx.x 对应连续的 C 列坐标，实现全局内存合并访问。
 *
 * 预期性能：矩阵 4096x4096 时约 1986.5 GFLOPS/s（比实验1提升约 6.4x）
 */

#include "sgemm_common.h"
#include <cublas_v2.h>

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                           const float *A, const float *B,
                                           float beta, float *C) {
  // 重映射：threadIdx.x 连续 => cCol 连续
  // 因为 cCol 在 B 的访问中是连续的维度：B[i * N + cCol]
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      // 同一 Warp 内：cCol 连续 => B[i * N + cCol] 地址连续 => 合并访问
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}

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

  // cuBLAS reference
  cublasHandle_t handle;
  cublasCreate(&handle);
  runCublasSgemm(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);

  // 1D block 启动配置
  const uint BLOCKSIZE = 32;
  dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);  // 1024 threads, 1D

  printf("\nKernel 配置:\n");
  printf("  Grid:  (%d, %d)\n", gridDim.x, gridDim.y);
  printf("  Block: (%d) = 1D, %d threads\n", blockDim.x, blockDim.x);

  // Warmup
  for (int i = 0; i < num_warmup; ++i) {
    sgemm_global_mem_coalesce<BLOCKSIZE>
        <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < num_iter; ++i) {
    sgemm_global_mem_coalesce<BLOCKSIZE>
        <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  float avg_ms = elapsed_ms / num_iter;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  // Verification
  CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(C_ref, d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  bool correct = verify_matrix(C_ref, C, M * N);

  double gflops = calculate_gflops(M, N, K, avg_ms);

  printf("\n========================================\n");
  printf("实验结果:\n");
  printf("  正确性: %s\n", correct ? "通过" : "失败");
  printf("  平均耗时: %.4f ms\n", avg_ms);
  printf("  计算性能: %.1f GFLOPS/s\n", gflops);
  printf("  预期性能: ~1986.5 GFLOPS/s\n");
  printf("  相比实验1 (309 GFLOPS): %.1fx 提升\n", gflops / 309.0);
  printf("========================================\n");

  cublasDestroy(handle);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
  free(A); free(B); free(C); free(C_ref);
  return correct ? 0 : 1;
}
