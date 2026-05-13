/**
 * 实验7：参数自动调优 - 参考解答
 *
 * 带 Warp Tile 划分的 kernel，使用编译时常量参数 BM, BN, BK, TM, TN。
 * 引入了 Warp 级别的子区域划分（WM = TM*16, WN = TN*16）。
 * 使用 float4 向量化加载和循环遍历 Warp Tile。
 *
 * 预期性能（A6000, BK=16）：约 19721.0 GFLOPS/s（达 cuBLAS 的 84.8%）
 */

#include "sgemm_common.h"
#include <cublas_v2.h>

const int NUM_THREADS = 256;

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(NUM_THREADS)
    sgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                   float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Warp Tile 的大小 = 16 * Thread Tile
  constexpr int WM = TM * 16;
  constexpr int WN = TN * 16;
  constexpr int WMITER = CEIL_DIV(BM, WM);
  constexpr int WNITER = CEIL_DIV(BN, WN);

  // 线程在 Warp Tile 中的位置
  const int threadCol = threadIdx.x % (WN / TN);
  const int threadRow = threadIdx.x / (WN / TN);

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // 向量化加载索引
  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  // 寄存器缓存（为所有 Warp Subtile 预分配）
  float threadResults[WMITER * WNITER * TM * TN] = {0.0};
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // 向量化多元素加载 A（带转置）
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
      float4 tmp = reinterpret_cast<float4 *>(
          &A[(innerRowA + offset) * K + innerColA * 4])[0];
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }
    // 向量化多元素加载 B
    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &B[(innerRowB + offset) * N + innerColB * 4])[0];
    }
    __syncthreads();

    // 遍历 Warp Tile：每个 Warp Tile 包含 WMITER * WNITER 个 Warp Subtile
    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
      for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
          // 加载 Thread Tile 到寄存器
          for (uint i = 0; i < TM; ++i) {
            regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
          }
          for (uint i = 0; i < TN; ++i) {
            regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
          }
          for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
              threadResults[(wmIdx * TM + resIdxM) * (WNITER * TN) +
                            wnIdx * TN + resIdxN] +=
                  regM[resIdxM] * regN[resIdxN];
            }
          }
        }
      }
    }
    __syncthreads();
    A += BK;
    B += BK * N;
  }

  // 向量化写回
  for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
    for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
      float *C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
          float4 tmp = reinterpret_cast<float4 *>(
              &C_interim[(threadRow * TM + resIdxM) * N + threadCol * TN +
                         resIdxN])[0];
          const int i =
              (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN;
          tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
          tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
          tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
          tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
          reinterpret_cast<float4 *>(
              &C_interim[(threadRow * TM + resIdxM) * N + threadCol * TN +
                         resIdxN])[0] = tmp;
        }
      }
    }
  }
}

int main() {
  CudaDeviceInfo();

  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f, beta = 0.0f;
  const int num_warmup = 5, num_iter = 10;

  printf("\n========================================\n");
  printf("实验7：参数自动调优\n");
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

  // A6000 最优参数
  const uint BM = 128, BN = 128, BK_PARAM = 16, TM = 8, TN = 8;

  // 编译时约束检查
  static_assert((NUM_THREADS * 4) % BK_PARAM == 0,
                "NUM_THREADS*4 must be multiple of BK");
  static_assert((NUM_THREADS * 4) % BN == 0,
                "NUM_THREADS*4 must be multiple of BN");
  static_assert(BN % (16 * TN) == 0,
                "BN must be multiple of 16*TN");
  static_assert(BM % (16 * TM) == 0,
                "BM must be multiple of 16*TM");
  static_assert((BM * BK_PARAM) % (4 * NUM_THREADS) == 0,
                "BM*BK must be multiple of 4*NUM_THREADS");
  static_assert((BN * BK_PARAM) % (4 * NUM_THREADS) == 0,
                "BN*BK must be multiple of 4*NUM_THREADS");

  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim(NUM_THREADS);

  printf("\nKernel 配置:\n");
  printf("  BM=%d, BN=%d, BK=%d, TM=%d, TN=%d\n", BM, BN, BK_PARAM, TM, TN);
  printf("  WarpTile: WM=%d, WN=%d\n", TM*16, TN*16);
  printf("  Grid:  (%d, %d)\n", gridDim.x, gridDim.y);
  printf("  Block: (%d) threads\n", blockDim.x);

  for (int i = 0; i < num_warmup; ++i) {
    sgemmAutotuned<BM, BN, BK_PARAM, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < num_iter; ++i) {
    sgemmAutotuned<BM, BN, BK_PARAM, TM, TN>
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
  printf("  预期性能 (BK=16): ~19721.0 GFLOPS/s\n");
  printf("========================================\n");

  cublasDestroy(handle);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
  free(A); free(B); free(C); free(C_ref);
  return correct ? 0 : 1;
}
