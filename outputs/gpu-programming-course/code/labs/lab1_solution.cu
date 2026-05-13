/**
 * 实验1：朴素矩阵乘法实现 - 参考解答
 *
 * 第一个 CUDA SGEMM kernel，使用 2D grid 和 2D block。
 * 每个线程计算结果矩阵 C 的一个元素。
 *
 * 预期性能：矩阵 4096x4096 时约 309 GFLOPS/s
 *
 * 编译: nvcc lab1_solution.cu -o lab1_solution -lcublas
 * 运行: ./lab1_solution
 */

#include "sgemm_common.h"
#include <cublas_v2.h>

// ============================================================================
// Kernel: 朴素矩阵乘法
// 每个线程计算 C 的一个元素
// ============================================================================
__global__ void sgemm_naive(int M, int N, int K, float alpha,
                             const float *A, const float *B,
                             float beta, float *C) {
  // 计算当前线程在 C 矩阵中的全局位置
  // A 是 MxK, B 是 KxN, C 是 MxN
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;  // C 的行索引
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;  // C 的列索引

  // 边界检查：处理 tile quantization（当 M 或 N 不是 block 大小的整数倍）
  if (x < M && y < N) {
    float tmp = 0.0;
    // 内积：A 的第 x 行 与 B 的第 y 列点积
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // GEMM 更新：C = alpha * A*B + beta * C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

int main() {
  // 显示 GPU 信息
  CudaDeviceInfo();

  // 矩阵参数
  const int M = 4096;
  const int N = 4096;
  const int K = 4096;
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const int num_warmup = 5;
  const int num_iter = 10;

  printf("\n========================================\n");
  printf("实验1：朴素矩阵乘法实现\n");
  printf("矩阵大小: M=%d, N=%d, K=%d\n", M, N, K);
  printf("========================================\n");

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
  // 使用 cuBLAS 获取参考结果
  // ========================================================================
  cublasHandle_t handle;
  cublasCreate(&handle);
  runCublasSgemm(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);

  // ========================================================================
  // 运行自己的 kernel
  // ========================================================================
  const int BLOCK_SIZE = 32;
  dim3 gridDim(CEIL_DIV(M, BLOCK_SIZE), CEIL_DIV(N, BLOCK_SIZE));
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

  printf("\nKernel 配置:\n");
  printf("  Grid:  (%d, %d)\n", gridDim.x, gridDim.y);
  printf("  Block: (%d, %d) = %d threads\n", blockDim.x, blockDim.y,
         blockDim.x * blockDim.y);

  // Warmup
  printf("\n预热中...\n");
  for (int i = 0; i < num_warmup; ++i) {
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark
  printf("性能测试中...\n");
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < num_iter; ++i) {
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  float avg_ms = elapsed_ms / num_iter;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  // ========================================================================
  // 验证正确性
  // ========================================================================
  CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(C_ref, d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  bool correct = verify_matrix(C_ref, C, M * N);

  // ========================================================================
  // 性能报告
  // ========================================================================
  double gflops = calculate_gflops(M, N, K, avg_ms);

  printf("\n========================================\n");
  printf("实验结果:\n");
  printf("  正确性: %s\n", correct ? "通过" : "失败");
  printf("  平均耗时: %.4f ms\n", avg_ms);
  printf("  计算性能: %.1f GFLOPS/s\n", gflops);
  printf("  预期性能: ~309 GFLOPS/s\n");
  printf("========================================\n");

  // 清理
  cublasDestroy(handle);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaFree(d_C_ref);
  free(A);
  free(B);
  free(C);
  free(C_ref);

  return correct ? 0 : 1;
}
