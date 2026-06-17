/**
 * 实验4：一维 Block Tile - 参考解答
 *
 * 每个线程计算 TM 个结果（沿 M 方向排列）。
 * dotIdx 在外层，将 Bs 的一个元素缓存到寄存器后复用 TM 次。
 *
 * 预期性能：矩阵 4096x4096 时约 8474.7 GFLOPS/s（比实验3提升约 2.8x）
 */


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// ===================== 宏定义 =====================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error [%s:%d]: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ===================== 辅助函数 =====================
void randomize_matrix(float *mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

void zero_init_matrix(float *mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = 0.0f;
    }
}

double calculate_gflops(int M, int N, int K, double time_ms) {
    double flops = 2.0 * M * N * K;
    return flops / (time_ms * 1e6);
}

void print_device_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return;
    }
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
}

// ===================== Kernel: 一维 Block Tile SGEMM =====================
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha,
                                    const float *A, const float *B,
                                    float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  const uint innerColA = threadIdx.x % BK;
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN;
  const uint innerRowB = threadIdx.x / BN;

  float threadResults[TM] = {0.0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float Btmp = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
      }
    }
    __syncthreads();
  }

  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}

// ===================== 主函数 =====================
int main() {
  srand((unsigned int)time(NULL));
  print_device_info();

  const int M = 4096, N = 4096, K = 4096;
  const float alpha = 1.0f, beta = 0.0f;
  const int num_warmup = 5, num_iter = 10;

  printf("\n========================================\n");
  printf("实验4：一维 Block Tile SGEMM\n");
  printf("矩阵大小: M=%d, N=%d, K=%d\n", M, N, K);
  printf("========================================\n");

  float *A = (float *)malloc(M * K * sizeof(float));
  float *B = (float *)malloc(K * N * sizeof(float));
  float *C = (float *)malloc(M * N * sizeof(float));

  randomize_matrix(A, M * K);
  randomize_matrix(B, K * N);
  zero_init_matrix(C, M * N);

  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));

  const uint BM = 64, BN = 64, BK = 8, TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);

  printf("Kernel 配置:\n");
  printf("  BM=%d, BN=%d, BK=%d, TM=%d\n", BM, BN, BK, TM);
  printf("  Grid:  (%d, %d)\n", gridDim.x, gridDim.y);
  printf("  Block: (%d) threads\n", blockDim.x);
  printf("  SMEM:  %zu bytes\n", (BM * BK + BK * BN) * sizeof(float));

  // 预热
  for (int i = 0; i < num_warmup; ++i) {
    sgemm1DBlocktiling<BM, BN, BK, TM>
        <<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // 计时
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
  double gflops = calculate_gflops(M, N, K, avg_ms);

  printf("\n========================================\n");
  printf("实验结果:\n");
  printf("  平均耗时: %.4f ms\n", avg_ms);
  printf("  计算性能: %.1f GFLOPS/s\n", gflops);
  printf("  预期性能: ~8474.7 GFLOPS/s\n");
  printf("========================================\n");

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
  free(A); free(B); free(C);
  return 0;
}