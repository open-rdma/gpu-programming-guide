/**
 * Chapter 11 - Experiment 11-1: Warp Divergence Benchmark
 *
 * Measures the performance impact of different branching patterns.
 * Compile: nvcc -arch=sm_86 -O3 divergence_benchmark.cu -o divergence_benchmark
 * Run: ./divergence_benchmark
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) {                                            \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

// Baseline: no branching at all
__global__ void noBranch(float * __restrict__ a, float * __restrict__ b,
                          float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Branch aligned with warp boundaries
// All threads in the same warp take the same branch path
__global__ void warpAlignedBranch(float * __restrict__ a, float * __restrict__ b,
                                    float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int warpId = threadIdx.x / 32;
        if (warpId % 2 == 0) {
            c[idx] = a[idx] + b[idx];
        } else {
            c[idx] = a[idx] * b[idx];
        }
    }
}

// Branch within warp - divergence on every warp!
// Half the threads in each warp take one path, half take the other
__global__ void innerWarpBranch(float * __restrict__ a, float * __restrict__ b,
                                  float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        if (threadIdx.x % 2 == 0) {
            c[idx] = a[idx] + b[idx];
        } else {
            c[idx] = a[idx] * b[idx];
        }
    }
}

// More complex divergence: 4-way branch within warp
__global__ void innerWarpBranch4Way(float * __restrict__ a, float * __restrict__ b,
                                      float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int laneId = threadIdx.x % 32;
        if (laneId < 8) {
            c[idx] = a[idx] + b[idx];
        } else if (laneId < 16) {
            c[idx] = a[idx] * b[idx];
        } else if (laneId < 24) {
            c[idx] = a[idx] - b[idx];
        } else {
            c[idx] = a[idx] / (b[idx] + 1e-8f);
        }
    }
}

// Warp-aligned 4-way branch (by warp ID instead of lane ID)
__global__ void warpAlignedBranch4Way(float * __restrict__ a, float * __restrict__ b,
                                        float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        int warpId = threadIdx.x / 32;
        int warpMod = warpId % 4;
        if (warpMod == 0) {
            c[idx] = a[idx] + b[idx];
        } else if (warpMod == 1) {
            c[idx] = a[idx] * b[idx];
        } else if (warpMod == 2) {
            c[idx] = a[idx] - b[idx];
        } else {
            c[idx] = a[idx] / (b[idx] + 1e-8f);
        }
    }
}

// Using branch predication via ternary operator (compiler may predicate)
__global__ void predicatedBranch(float * __restrict__ a, float * __restrict__ b,
                                  float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // Ternary operator is often predicated by the compiler
        c[idx] = (threadIdx.x % 2 == 0) ? (a[idx] + b[idx]) : (a[idx] * b[idx]);
    }
}

float benchmarkKernel(void (*kernel)(float*, float*, float*, int),
                      float *d_a, float *d_b, float *d_c, int n,
                      int gridSize, int blockSize, int iterations) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < iterations; i++) {
        kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
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
    printf("=== Warp Divergence Benchmark ===\n");
    printf("GPU: %s (SM %d.%d), Warp Size: %d\n",
           prop.name, prop.major, prop.minor, prop.warpSize);
    printf("\n");

    const int N = 32 * 1024 * 1024;  // 32M elements
    const int blockSize = 256;  // 8 warps per block
    const int gridSize = (N + blockSize - 1) / blockSize;
    const int iterations = 100;
    const size_t bytes = N * sizeof(float);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    // Initialize (on host, using pinned memory for speed)
    float *h_data;
    CHECK_CUDA(cudaMallocHost(&h_data, bytes));
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(i % 1000) + 1.0f;
    }
    CHECK_CUDA(cudaMemcpy(d_a, h_data, bytes, cudaMemcpyHostToDevice));
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)((i % 500) + 1.0f);
    }
    CHECK_CUDA(cudaMemcpy(d_b, h_data, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaFreeHost(h_data));

    printf("Array size: %d float elements (%.1f MB)\n", N, bytes / (1024.0f*1024.0f));
    printf("Block size: %d, Grid size: %d, Iterations: %d\n\n",
           blockSize, gridSize, iterations);

    printf("%-35s %10s %12s\n", "Branch Pattern", "Time(ms)", "vs Baseline");
    printf("-----------------------------------------------------------------\n");

    float baseline_ms, ms;

    // 1. No branching (baseline)
    baseline_ms = benchmarkKernel(noBranch, d_a, d_b, d_c, N, gridSize, blockSize, iterations);
    printf("%-35s %10.4f %11.2fx\n", "No branch (baseline)", baseline_ms, baseline_ms / baseline_ms);

    // 2. Warp-aligned 2-way
    ms = benchmarkKernel(warpAlignedBranch, d_a, d_b, d_c, N, gridSize, blockSize, iterations);
    printf("%-35s %10.4f %11.2fx\n", "Warp-aligned 2-way branch", ms, ms / baseline_ms);

    // 3. Inner-warp 2-way divergence
    ms = benchmarkKernel(innerWarpBranch, d_a, d_b, d_c, N, gridSize, blockSize, iterations);
    printf("%-35s %10.4f %11.2fx\n", "Inner-warp 2-way divergence", ms, ms / baseline_ms);

    // 4. Warp-aligned 4-way
    ms = benchmarkKernel(warpAlignedBranch4Way, d_a, d_b, d_c, N, gridSize, blockSize, iterations);
    printf("%-35s %10.4f %11.2fx\n", "Warp-aligned 4-way branch", ms, ms / baseline_ms);

    // 5. Inner-warp 4-way divergence
    ms = benchmarkKernel(innerWarpBranch4Way, d_a, d_b, d_c, N, gridSize, blockSize, iterations);
    printf("%-35s %10.4f %11.2fx\n", "Inner-warp 4-way divergence", ms, ms / baseline_ms);

    // 6. Predicated branch (ternary)
    ms = benchmarkKernel(predicatedBranch, d_a, d_b, d_c, N, gridSize, blockSize, iterations);
    printf("%-35s %10.4f %11.2fx\n", "Predicated (ternary op)", ms, ms / baseline_ms);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    printf("\nKey insights:\n");
    printf("1. Warp-aligned branches have minimal overhead vs no-branch baseline\n");
    printf("2. Inner-warp divergence serializes execution paths => significant slowdown\n");
    printf("3. 4-way inner-warp divergence is worse than 2-way (more path serialization)\n");
    printf("4. Ternary operator can use predication to avoid actual branching\n");

    return 0;
}
