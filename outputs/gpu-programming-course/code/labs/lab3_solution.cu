/**
 * 实验3：共享内存缓存分块 - 参考解答
 *
 * 使用共享内存缓存 A 和 B 的子块，将数据复用在快速 on-chip 内存中。
 * 外层循环沿 K 维度分块加载，内层循环在 SMEM 上执行分块内积。
 *
 * 预期性能：矩阵 4096x4096 时约 2980.3 GFLOPS/s（比实验2提升约 1.5x）
 */

#include "sgemm_common.h"
#include <cublas_v2.h>

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                        const float *A, const float *B,
                                        float beta, float *C) {
  // 当前 Block 要计算的 C 子块位置
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // 在共享内存中分配当前 Block 的缓存
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // 线程在 Block 内的行列位置
  const uint threadCol = threadIdx.x % BLOCKSIZE;
  const uint threadRow = threadIdx.x / BLOCKSIZE;

  // 指针移动到当前 Block 负责的区域
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // 协作加载数据到共享内存（保持合并访问）
    // threadCol 是连续的 threadIdx 维度 => 合并访问
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // 同步：确保所有线程都加载完成后再开始计算
    __syncthreads();

    // 前进到下一个 K 维度的块
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // 在共享内存上执行分块内积
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // 同步：防止快线程提前加载下一个块，覆盖仍在被慢线程读取的 SMEM
    __syncthreads();
  }

  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
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
  const int BLOCKSIZE = 32;

  printf("\n========================================\n");
  printf("实验3：共享内存缓存分块\n");
  printf("矩阵大小: M=%d, N=%d, K=%d, BLOCKSIZE=%d\n", M, N, K, BLOCKSIZE);
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

  // 配置 kernel
  dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);

  // 本 kernel 不会使用 L1 缓存，将所有 L1 让渡给 SMEM
  cudaFuncSetAttribute(sgemm_shared_mem_block<BLOCKSIZE>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);

  printf("\nKernel 配置:\n");
  printf("  Grid:  (%d, %d)\n", gridDim.x, gridDim.y);
  printf("  Block: (%d) = 1D, %d threads\n", blockDim.x, blockDim.x);
  printf("  SMEM/Block: %zu bytes\n", 2 * BLOCKSIZE * BLOCKSIZE * sizeof(float));

  // Warmup
  for (int i = 0; i < num_warmup; ++i) {
    sgemm_shared_mem_block<BLOCKSIZE>
        <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < num_iter; ++i) {
    sgemm_shared_mem_block<BLOCKSIZE>
        <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
  float avg_ms = elapsed_ms / num_iter;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  CUDA_CHECK(cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(C_ref, d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  bool correct = verify_matrix(C_ref, C, M * N);

  double gflops = calculate_gflops(M, N, K, avg_ms);

  printf("\n========================================\n");
  printf("实验结果:\n");
  printf("  正确性: %s\n", correct ? "通过" : "失败");
  printf("  平均耗时: %.4f ms\n", avg_ms);
  printf("  计算性能: %.1f GFLOPS/s\n", gflops);
  printf("  预期性能: ~2980.3 GFLOPS/s\n");
  printf("========================================\n");

  cublasDestroy(handle);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
  free(A); free(B); free(C); free(C_ref);
  return correct ? 0 : 1;
}
