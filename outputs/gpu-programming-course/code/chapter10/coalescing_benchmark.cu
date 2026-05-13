/**
 * Chapter 10 - Experiment 10-1: Global Memory Coalescing Benchmark
 *
 * Measures the impact of different global memory access patterns on bandwidth.
 * Compile: nvcc -arch=sm_86 -O3 coalescing_benchmark.cu -o coalescing_benchmark
 * Run: ./coalescing_benchmark
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

// Stride-1: perfectly coalesced
__global__ void readStride1(const float * __restrict__ input,
                              float * __restrict__ output, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

// Stride-2: every other element
__global__ void readStride2(const float * __restrict__ input,
                              float * __restrict__ output, int n) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

// Stride-8
__global__ void readStride8(const float * __restrict__ input,
                              float * __restrict__ output, int n) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 8;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

// Stride-32: worst case - all threads access different 128B segments
__global__ void readStride32(const float * __restrict__ input,
                               float * __restrict__ output, int n) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 32;
    if (idx < n) {
        output[idx] = input[idx];
    }
}

// Random access (pseudo-random pattern based on a deterministic shuffle)
__global__ void readRandom(const float * __restrict__ input,
                            float * __restrict__ output,
                            const int * __restrict__ indices, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        int idx = indices[tid];
        output[tid] = input[idx];
    }
}

// Vectorized load using float4 (best-case coalescing)
__global__ void readFloat4(const float * __restrict__ input,
                            float * __restrict__ output, int n) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    if (idx < n) {
        float4 val = reinterpret_cast<const float4*>(input)[threadIdx.x + blockIdx.x * blockDim.x];
        output[idx + 0] = val.x;
        output[idx + 1] = val.y;
        output[idx + 2] = val.z;
        output[idx + 3] = val.w;
    }
}

float benchmarkKernel(void (*kernel)(const float*, float*, int),
                      float *d_in, float *d_out, int n,
                      int gridSize, int blockSize, int iterations) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warmup
    kernel<<<gridSize, blockSize>>>(d_in, d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Timed iterations
    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < iterations; i++) {
        kernel<<<gridSize, blockSize>>>(d_in, d_out, n);
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
    // Get device properties
    cudaDeviceProp prop;
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("=== Coalescing Benchmark ===\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global Memory Bandwidth: %.1f GB/s\n",
           prop.memoryClockRate * (prop.memoryBusWidth / 8) * 2 / 1e6);
    printf("\n");

    const int N = 32 * 1024 * 1024;  // 32M elements = 128 MB
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    const int iterations = 100;
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    // Allocate device memory
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    printf("Array size: %d float elements (%.1f MB)\n", N, bytes / (1024.0*1024.0));
    printf("Block size: %d, Grid size: %d, Iterations: %d\n\n",
           blockSize, gridSize, iterations);

    printf("%-25s %10s %15s %20s\n", "Access Pattern", "Time(ms)", "Bandwidth(GB/s)", "% of Peak");
    printf("----------------------------------------------------------------------------\n");

    // Test each access pattern
    float ms, bw, pct;
    float peakBW = prop.memoryClockRate * (prop.memoryBusWidth / 8) * 2 / 1e6;

    // Stride 1
    ms = benchmarkKernel(readStride1, d_in, d_out, N, gridSize, blockSize, iterations);
    bw = (bytes * 2) / (ms / 1000.0) / 1e9;  // read + write
    pct = bw / peakBW * 100;
    printf("%-25s %10.4f %15.2f %20.1f%%\n", "Stride=1 (coalesced)", ms, bw, pct);

    // Stride 2
    ms = benchmarkKernel(readStride2, d_in, d_out, N/2, gridSize, blockSize, iterations);
    bw = (bytes) / (ms / 1000.0) / 1e9;  // approximate bytes loaded
    pct = bw / peakBW * 100;
    printf("%-25s %10.4f %15.2f %20.1f%%\n", "Stride=2", ms, bw, pct);

    // Stride 8
    ms = benchmarkKernel(readStride8, d_in, d_out, N/8, gridSize, blockSize, iterations);
    bw = (bytes / 4) / (ms / 1000.0) / 1e9;
    pct = bw / peakBW * 100;
    printf("%-25s %10.4f %15.2f %20.1f%%\n", "Stride=8", ms, bw, pct);

    // Stride 32
    ms = benchmarkKernel(readStride32, d_in, d_out, N/32, gridSize, blockSize, iterations);
    bw = (bytes / 16) / (ms / 1000.0) / 1e9;
    pct = bw / peakBW * 100;
    printf("%-25s %10.4f %15.2f %20.1f%%\n", "Stride=32 (worst)", ms, bw, pct);

    // Float4 vectorized
    int gridFloat4 = (N / 4 + blockSize - 1) / blockSize;
    ms = benchmarkKernel(readFloat4, d_in, d_out, N, gridFloat4, blockSize, iterations);
    bw = (bytes * 2) / (ms / 1000.0) / 1e9;
    pct = bw / peakBW * 100;
    printf("%-25s %10.4f %15.2f %20.1f%%\n", "Float4 vectorized", ms, bw, pct);

    // Cleanup
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    free(h_in);
    free(h_out);

    printf("\nConclusion: Stride-1 coalesced access achieves %.1f%% of peak bandwidth.\n", pct);
    printf("Increasing stride dramatically reduces effective bandwidth.\n");

    return 0;
}
