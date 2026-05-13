/**
 * 实验4：一维 Block Tile - 参考解答
 *
 * 每个线程计算 TM 个结果（沿 M 方向排列）。
 * dotIdx 在外层，将 Bs 的一个元素缓存到寄存器后复用 TM 次。
 *
 * 预期性能：矩阵 4096x4096 时约 8474.7 GFLOPS/s（比实验3提升约 2.8x）
 */

#include "sgemm_common.h"
#include <cublas_v2.h>

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                    const float *A, const float *B,
                                    float beta, float *C) {
  // Block 在 C 矩阵中的位置
  // 交换 x 和 y 以获得更好的 L2 缓存命中率
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // 当前线程在 Block Tile 中的行列位置
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // 在 SMEM 中分配当前 Block Tile 的缓存
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 指针移动到当前 Block Tile 的起始位置
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // 每个线程负责从 GMEM 加载到 SMEM 时的索引
  // 保持合并访问：innerCol 维度对应连续的 threadIdx
  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  // 寄存器中的线程结果缓存（TM 个 float）
  float threadResults[TM] = {0.0};

  // 外层循环：沿 K 维度分块
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 协作加载当前子块到 SMEM
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // 前进到下一个 K 维度的块
    A += BK;
    B += BK * N;

    // 内积计算：dotIdx 在外层，缓存 Bs 元素
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float Btmp = Bs[dotIdx * BN + threadCol];  // 缓存到寄存器
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
      }
    }
    __syncthreads();
  }

  // 写回 TM 个结果
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}

int main() {
  CudaDeviceInfo();

  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f, beta = 0.0f;
  const int num_warmup = 5, num_iter = 10;

  printf("\n========================================\n");
  printf("实验4：一维 Block Tile\n");
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

  cublasHandle_t handle;
  cublasCreate(&handle);
  runCublasSgemm(handle, M, N, K, alpha, d_A, d_B, beta, d_C_ref);

  const uint BM = 64, BN = 64, BK = 8, TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);

  printf("\nKernel 配置:\n");
  printf("  BM=%d, BN=%d, BK=%d, TM=%d\n", BM, BN, BK, TM);
  printf("  Grid:  (%d, %d)\n", gridDim.x, gridDim.y);
  printf("  Block: (%d) threads\n", blockDim.x);
  printf("  SMEM:  %zu bytes\n", (BM * BK + BK * BN) * sizeof(float));

  for (int i = 0; i < num_warmup; ++i) {
    sgemm1DBlocktiling<BM, BN, BK, TM>
        <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < num_iter; ++i) {
    sgemm1DBlocktiling<BM, BN, BK, TM>
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
  printf("  预期性能: ~8474.7 GFLOPS/s\n");
  printf("========================================\n");

  cublasDestroy(handle);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
  free(A); free(B); free(C); free(C_ref);
  return correct ? 0 : 1;
}
