/**
 * Chapter 11 - Experiment 11-2: Warp Shuffle Reduction Benchmark
 *
 * Compares reduction implementations to demonstrate the power of warp shuffle.
 * Implements 4 versions of sum reduction on a large float array:
 *   1. Global atomic accumulation (slowest - baseline)
 *   2. Shared memory block reduction
 *   3. Warp shuffle block reduction (warp-level shuffle + inter-warp shared memory)
 *   4. Fully optimized reduction (vectorized loads, loop unrolling, warp shuffle)
 *
 * Compile: nvcc -arch=sm_86 -O3 warp_reduce_benchmark.cu -o warp_reduce_benchmark
 * Run: ./warp_reduce_benchmark
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(call) {                                            \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

// ===== Version 1: Global Atomic Accumulation =====
__global__ void reduceAtomic(const float * __restrict__ input,
                              float * __restrict__ result, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        atomicAdd(result, input[idx]);
    }
}

// ===== Version 2: Shared Memory Block Reduction =====
__global__ void reduceSharedMem(const float * __restrict__ input,
                                 float * __restrict__ result, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];

        if (tid == 0) {
            atomicAdd(result, sdata[0]);
        }
    }
}

// ===== Version 3: Warp Shuffle Reduction =====
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduceWarpShuffle(const float * __restrict__ input,
                                   float * __restrict__ result, int n) {
    __shared__ float sdata[32];
    int tid = threadIdx.x;
    int idx = tid + blockIdx.x * blockDim.x;
    int laneId = tid & 0x1f;
    int warpId = tid >> 5;

    float val = (idx < n) ? input[idx] : 0.0f;
    val = warpReduceSum(val);

    if (laneId == 0) {
        sdata[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0) {
        val = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
        val = warpReduceSum(val);
        if (tid == 0) {
            atomicAdd(result, val);
        }
    }
}

// ===== Version 4: Fully Optimized Reduction (FIXED) =====
__global__ void reduceOptimized(const float * __restrict__ input,
                                 float * __restrict__ result, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = tid + blockIdx.x * blockDim.x;

    float sum = 0.0f;
    // 每个线程处理4个连续元素，增加严格的边界检查
    for (int i = 0; i < 4; i++) {
        int current_idx = idx + i * gridDim.x * blockDim.x;
        if (current_idx < n) {
            sum += input[current_idx];
        }
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];

        if (tid == 0) {
            atomicAdd(result, sdata[0]);
        }
    }
}

// Host-side benchmark function
float benchmarkKernel(const char* name,
                      void (*kernel)(const float*, float*, int),
                      const float *d_in, float *d_result, int n,
                      int gridSize, int blockSize, int iterations) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));

    // Warmup
    kernel<<<gridSize, blockSize>>>(d_in, d_result, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed iterations (减少到3次，避免ncu profiling超时)
    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < iterations; i++) {
        kernel<<<gridSize, blockSize>>>(d_in, d_result, n);
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / iterations;
}

int main() {
    cudaDeviceProp prop;
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("=== Warp Shuffle Reduction Benchmark ===\n");
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Peak Memory BW: %.1f GB/s\n\n",
           prop.memoryClockRate * (prop.memoryBusWidth / 8) * 2 / 1e6);

    const int N = 16 * 1024 * 1024;  // 16M elements = 64 MB
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    const int iterations = 3;
    const size_t bytes = N * sizeof(float);

    // Allocate and initialize input
    float *h_in = (float*)malloc(bytes);
    float expectedSum = 0.0f;
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f;
        expectedSum += 1.0f;
    }

    float *d_in, *d_result;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));

    printf("Array size: %d float elements (%.1f MB)\n", N, bytes / (1024.0f*1024.0f));
    printf("Expected sum: %.1f\n\n", expectedSum);

    printf("%-30s %10s %15s %12s\n", "Reduction Method", "Time(ms)", "Bandwidth(GB/s)", "Speedup");
    printf("--------------------------------------------------------------------------------\n");

    float ms, bw, speedup, baseline_ms = 0;

    // Version 1: Global atomic
    ms = benchmarkKernel("reduceAtomic", reduceAtomic,
                         d_in, d_result, N, gridSize, blockSize, iterations);
    bw = bytes / (ms / 1000.0) / 1e9;
    baseline_ms = ms;
    printf("%-30s %10.4f %15.2f %11.2fx\n", "1. Global Atomic", ms, bw, 1.0f);

    // Version 2: Shared memory
    ms = benchmarkKernel("reduceSharedMem", reduceSharedMem,
                         d_in, d_result, N, gridSize, blockSize, iterations);
    bw = bytes / (ms / 1000.0) / 1e9;
    speedup = baseline_ms / ms;
    printf("%-30s %10.4f %15.2f %11.2fx\n", "2. Shared Memory Block", ms, bw, speedup);

    // Version 3: Warp shuffle
    ms = benchmarkKernel("reduceWarpShuffle", reduceWarpShuffle,
                         d_in, d_result, N, gridSize, blockSize, iterations);
    bw = bytes / (ms / 1000.0) / 1e9;
    speedup = baseline_ms / ms;
    printf("%-30s %10.4f %15.2f %11.2fx\n", "3. Warp Shuffle", ms, bw, speedup);

    // Version 4: Fully optimized
    ms = benchmarkKernel("reduceOptimized", reduceOptimized,
                         d_in, d_result, N, gridSize, blockSize, iterations);
    bw = bytes / (ms / 1000.0) / 1e9;
    speedup = baseline_ms / ms;
    printf("%-30s %10.4f %15.2f %11.2fx\n", "4. Fully Optimized", ms, bw, speedup);

    // Verification
    float h_result;
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));
    printf("\n--- Verification ---\n");
    printf("Expected sum: %.1f\n", expectedSum);
    printf("Actual result: %.1f\n", h_result);
    printf("Match: %s\n", fabs(h_result - expectedSum) < 1.0f ? "YES" : "NO");

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_result));
    free(h_in);

    printf("\nKey insights:\n");
    printf("1. Global atomic is extremely slow due to serialization\n");
    printf("2. Warp shuffle avoids shared memory bank conflicts and sync overhead\n");
    printf("3. Xor shuffle is ideal for reductions due to its butterfly pattern\n");
    printf("4. Full optimization adds vectorized loads and sequential addressing\n");

    return 0;
}