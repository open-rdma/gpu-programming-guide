#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sys/time.h>
#include <cassert>
#include <algorithm>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// ============================================================================
// Error checking
// ============================================================================

inline void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(err) cudaCheck(err, __FILE__, __LINE__)

// ============================================================================
// Device info
// ============================================================================

inline void CudaDeviceInfo() {
  int deviceId;
  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n", deviceId);
  printf("Name: %s\n", props.name);
  printf("Compute Capability: %d.%d\n", props.major, props.minor);
  printf("Memory Bus Width: %d bits\n", props.memoryBusWidth);
  printf("Max Threads Per Block: %d\n", props.maxThreadsPerBlock);
  printf("Max Threads Per Multiprocessor: %d\n", props.maxThreadsPerMultiProcessor);
  printf("Max Regs Per Block: %d\n", props.regsPerBlock);
  printf("Max Regs Per Multiprocessor: %d\n", props.regsPerMultiprocessor);
  printf("Total Global Mem: %zu MB\n", props.totalGlobalMem / 1024 / 1024);
  printf("Shared Mem Per Block: %zu KB\n", props.sharedMemPerBlock / 1024);
  printf("Shared Mem Per Multiprocessor: %zu KB\n",
         props.sharedMemPerMultiprocessor / 1024);
  printf("Total Const Mem: %zu KB\n", props.totalConstMem / 1024);
  printf("Multiprocessor Count: %d\n", props.multiProcessorCount);
  printf("Warp Size: %d\n", props.warpSize);
}

// ============================================================================
// Timing
// ============================================================================

inline float get_current_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

inline float cpu_elapsed_time(float &beg, float &end) {
  return 1.0e-6 * (end - beg);
}

// ============================================================================
// Matrix initialization
// ============================================================================

inline void randomize_matrix(float *mat, int N) {
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

inline void zero_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 0.0;
  }
}

inline void copy_matrix(const float *src, float *dest, int N) {
  int i;
  for (i = 0; src + i && dest + i && i < N; i++)
    *(dest + i) = *(src + i);
  if (i != N)
    printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

// ============================================================================
// Verification
// ============================================================================

inline bool verify_matrix(float *matRef, float *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(matRef[i] - matOut[i]);
    if (std::isnan(diff) || diff > 0.01) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

// ============================================================================
// cuBLAS reference
// ============================================================================

inline void runCublasSgemm(cublasHandle_t handle, int M, int N, int K,
                            float alpha, float *A, float *B, float beta,
                            float *C) {
  // cuBLAS uses column-major order. We swap A and B to compute C = alpha*A*B + beta*C
  // Since cuBLAS computes C^T = op(B)^T * op(A)^T = A * B when both are in row-major
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// ============================================================================
// Performance calculation
// ============================================================================

inline double calculate_gflops(int M, int N, int K, float elapsed_ms) {
  // FLOPS for SGEMM: 2 * M * N * K (one mul + one add per element)
  double flops = 2.0 * (double)M * (double)N * (double)K;
  return flops / (elapsed_ms * 1e6);  // GFLOPS/s
}

// ============================================================================
// Benchmarking helper
// ============================================================================

inline double benchmark_kernel(int M, int N, int K, float alpha, float *d_A,
                                float *d_B, float beta, float *d_C,
                                int num_warmup, int num_iter,
                                void (*launch_kernel)()) {
  // Warmup
  for (int i = 0; i < num_warmup; ++i) {
    launch_kernel();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // Benchmark
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < num_iter; ++i) {
    launch_kernel();
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return elapsed_ms / num_iter;
}
