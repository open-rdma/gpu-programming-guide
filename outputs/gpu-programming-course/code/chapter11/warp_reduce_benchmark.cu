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
// Every thread atomically adds to a single global variable
// Extremely slow due to atomic contention
__global__ void reduceAtomic(const float * __restrict__ input,
                              float * __restrict__ result, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        atomicAdd(result, input[idx]);
    }
}

// ===== Version 2: Shared Memory Block Reduction =====
// Classic shared-memory tree reduction within each block
__global__ void reduceSharedMem(const float * __restrict__ input,
                                 float * __restrict__ result, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Load into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // Tree reduction in shared memory (inter-warp reduction)
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Final warp-level reduction (still using shared memory)
    if (tid < 32) {
        // No need for sync within a single warp
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];

        // Thread 0 writes block result
        if (tid == 0) {
            atomicAdd(result, sdata[0]);
        }
    }
}

// ===== Version 3: Warp Shuffle Reduction =====
// Uses warp shuffle for intra-warp reduction, shared memory for inter-warp
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduceWarpShuffle(const float * __restrict__ input,
                                   float * __restrict__ result, int n) {
    __shared__ float sdata[32];  // Only 32 slots needed (one per warp)
    int tid = threadIdx.x;
    int idx = tid + blockIdx.x * blockDim.x;
    int laneId = tid & 0x1f;  // tid % 32
    int warpId = tid >> 5;    // tid / 32

    // Each thread loads one element and does warp-level reduction
    float val = (idx < n) ? input[idx] : 0.0f;

    // Warp-level reduction using shuffle (no shared memory needed!)
    val = warpReduceSum(val);

    // One thread per warp writes the warp result to shared memory
    if (laneId == 0) {
        sdata[warpId] = val;
    }
    __syncthreads();

    // Final reduction of warp results (only first warp is active)
    if (warpId == 0) {
        val = (tid < blockDim.x / 32) ? sdata[tid] : 0.0f;
        val = warpReduceSum(val);
        if (tid == 0) {
            atomicAdd(result, val);
        }
    }
}

// ===== Version 4: Fully Optimized Reduction =====
// - Vectorized loading (float4)
// - Loop unrolling
// - Sequential addressing (avoids bank conflicts)
// - Warp shuffle for final stages
// - Multiple elements per thread
__global__ void reduceOptimized(const float * __restrict__ input,
                                 float * __restrict__ result, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = tid + blockIdx.x * blockDim.x * 4;  // Each thread processes 4 elements initially

    // Load 4 elements per thread and accumulate
    float sum = 0.0f;
    if (idx < n) {
        // Unrolled accumulation with ILP
        float4 v = reinterpret_cast<const float4*>(input + idx)[0];
        sum = v.x + v.y + v.z + v.w;
    }

    // Handle remaining elements if any (for this simplified version, array is multiple of blockDim*4)
    sdata[tid] = sum;
    __syncthreads();

    // Tree reduction in shared memory with sequential addressing
    // Sequential addressing means threads access consecutive addresses
    // -> no bank conflicts
    for (int s = blockDim.x / 2; s >= 1; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

// Host-side final reduction of block results
float hostReduce(float *d_block_results, int numBlocks) {
    float *h_results = (float*)malloc(numBlocks * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_results, d_block_results,
                          numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    float total = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        total += h_results[i];
    }
    free(h_results);
    return total;
}

float benchmarkKernel(const char* name,
                      void (*kernel)(const float*, float*, int),
                      const float *d_in, float *d_result, int n,
                      int gridSize, int blockSize, int iterations,
                      float expectedSum) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Reset result
    CHECK_CUDA(cudaMemset(d_result, 0, gridSize * sizeof(float)));

    // Warmup
    kernel<<<gridSize, blockSize>>>(d_in, d_result, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed iterations
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

    // Use a size that's a multiple of blockDim*4 for the optimized kernel
    const int N = 16 * 1024 * 1024;  // 16M elements = 64 MB
    const int blockSize = 256;
    const int gridSize = 20480;  // Many blocks to saturate the GPU
    const int iterations = 100;
    const size_t bytes = N * sizeof(float);

    // Allocate and initialize input
    float *h_in = (float*)malloc(bytes);
    float expectedSum = 0.0f;
    for (int i = 0; i < N; i++) {
        h_in[i] = 1.0f;  // Simple constant to verify sum
        expectedSum += 1.0f;
    }

    float *d_in, *d_result;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&d_result, gridSize * sizeof(float)));

    printf("Array size: %d float elements (%.1f MB)\n", N, bytes / (1024.0f*1024.0f));
    printf("Expected sum: %.1f\n\n", expectedSum);

    printf("%-30s %10s %15s %12s\n", "Reduction Method", "Time(ms)", "Bandwidth(GB/s)", "Speedup");
    printf("--------------------------------------------------------------------------------\n");

    float ms, bw, speedup, baseline_ms = 0;

    // Version 1: Global atomic
    ms = benchmarkKernel("reduceAtomic", (void(*)(const float*, float*, int))reduceAtomic,
                         d_in, d_result, N, gridSize, blockSize, iterations, expectedSum);
    bw = bytes / (ms / 1000.0) / 1e9;
    baseline_ms = ms;
    printf("%-30s %10.4f %15.2f %11.2fx\n", "1. Global Atomic", ms, bw, 1.0f);

    // Version 2: Shared memory
    ms = benchmarkKernel("reduceSharedMem", (void(*)(const float*, float*, int))reduceSharedMem,
                         d_in, d_result, N, gridSize, blockSize, iterations, expectedSum);
    bw = bytes / (ms / 1000.0) / 1e9;
    speedup = baseline_ms / ms;
    printf("%-30s %10.4f %15.2f %11.2fx\n", "2. Shared Memory Block", ms, bw, speedup);

    // Version 3: Warp shuffle
    ms = benchmarkKernel("reduceWarpShuffle", (void(*)(const float*, float*, int))reduceWarpShuffle,
                         d_in, d_result, N, gridSize, blockSize, iterations, expectedSum);
    bw = bytes / (ms / 1000.0) / 1e9;
    speedup = baseline_ms / ms;
    printf("%-30s %10.4f %15.2f %11.2fx\n", "3. Warp Shuffle", ms, bw, speedup);

    // Version 4: Fully optimized
    float finalResult = 0;
    {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaMemset(d_result, 0, gridSize * sizeof(float)));

        reduceOptimized<<<gridSize, blockSize>>>(d_in, d_result, N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(start, 0));
        for (int i = 0; i < iterations; i++) {
            reduceOptimized<<<gridSize, blockSize>>>(d_in, d_result, N);
        }
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        ms /= iterations;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        finalResult = hostReduce(d_result, gridSize);
    }
    bw = bytes / (ms / 1000.0) / 1e9;
    speedup = baseline_ms / ms;
    printf("%-30s %10.4f %15.2f %11.2fx\n", "4. Fully Optimized", ms, bw, speedup);

    printf("\n--- Verification ---\n");
    printf("Expected sum: %.1f\n", expectedSum);
    printf("Optimized result: %.1f\n", finalResult);
    printf("Match: %s\n", fabs(finalResult - expectedSum) < 1.0f ? "YES" : "NO");

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_result));
    free(h_in);

    printf("\nKey insights:\n");
    printf("1. Global atomic is extremely slow due to serialization\n");
    printf("2. Warp shuffle avoids shared memory bank conflicts and sync overhead\n");
    printf("3. Xor shuffle is ideal for reductions due to its butterfly pattern\n");
    printf("4. Full optimization (in this case) adds vectorized loads and\n");
    printf("   sequential addressing for the shared memory phase\n");

    return 0;
}
