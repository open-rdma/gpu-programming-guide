/**
 * Chapter 10 - Experiment 10-1: Global Memory Coalescing Benchmark (FIXED for Online Judge)
 *
 * Measures the impact of different global memory access patterns on bandwidth.
 * Compile: nvcc -arch=sm_86 -O3 coalescing_benchmark.cu -o coalescing_benchmark
 * Run: ./coalescing_benchmark
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

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

// Vectorized load using float4 (best-case coalescing)
__global__ void readFloat4(const float * __restrict__ input,
                            float * __restrict__ output, int n) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    // Ensure we don't write beyond the array
    if (idx + 3 < n) {
        float4 val = reinterpret_cast<const float4*>(input)[threadIdx.x + blockIdx.x * blockDim.x];
        output[idx + 0] = val.x;
        output[idx + 1] = val.y;
        output[idx + 2] = val.z;
        output[idx + 3] = val.w;
    }
}

// Helper to benchmark a kernel
// n: number of elements to process (not including stride)
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
    double peakBW = prop.memoryClockRate * (prop.memoryBusWidth / 8) * 2 / 1e6;
    printf("Global Memory Bandwidth (theoretical peak): %.1f GB/s\n", peakBW);
    printf("\n");

    // 🔧 关键修复1：减小测试规模以适应在线评测系统
    const int N = 8 * 1024 * 1024;    // 8M元素 = 32MB（原32M）
    const int blockSize = 256;
    const int iterations = 20;        // 🔧 关键修复2：减少迭代次数（原100）
    const size_t totalBytes = N * sizeof(float);

    // Allocate host memory and initialize
    float *h_in = (float*)malloc(totalBytes);
    float *h_out = (float*)malloc(totalBytes);
    for (int i = 0; i < N; i++) h_in[i] = (float)i;

    // Allocate device memory
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, totalBytes));
    CHECK_CUDA(cudaMalloc(&d_out, totalBytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, totalBytes, cudaMemcpyHostToDevice));

    printf("Array size: %d float elements (%.1f MB)\n", N, totalBytes / (1024.0*1024.0));
    printf("Block size: %d, Iterations: %d\n\n", blockSize, iterations);

    printf("%-30s %12s %15s %15s\n", "Access Pattern", "Time(ms)", "BW(GB/s)", "% of Peak");
    printf("--------------------------------------------------------------------------------\n");

    // --- Test Stride=1 (coalesced) ---
    int effectiveN = N;               // stride 1: all elements
    int gridSize = (effectiveN + blockSize - 1) / blockSize;
    float ms = benchmarkKernel(readStride1, d_in, d_out, effectiveN, gridSize, blockSize, iterations);
    size_t bytesPerKernel = 2 * effectiveN * sizeof(float);  // read + write
    double bw = bytesPerKernel / (ms / 1000.0) / 1e9;
    double pct = bw / peakBW * 100.0;
    printf("%-30s %12.4f %15.2f %14.1f%%\n", "Stride=1 (coalesced)", ms, bw, pct);

    // --- Test Stride=2 ---
    effectiveN = N / 2;
    gridSize = (effectiveN + blockSize - 1) / blockSize;
    ms = benchmarkKernel(readStride2, d_in, d_out, effectiveN, gridSize, blockSize, iterations);
    bytesPerKernel = 2 * effectiveN * sizeof(float);
    bw = bytesPerKernel / (ms / 1000.0) / 1e9;
    pct = bw / peakBW * 100.0;
    printf("%-30s %12.4f %15.2f %14.1f%%\n", "Stride=2", ms, bw, pct);

    // --- Test Stride=8 ---
    effectiveN = N / 8;
    gridSize = (effectiveN + blockSize - 1) / blockSize;
    ms = benchmarkKernel(readStride8, d_in, d_out, effectiveN, gridSize, blockSize, iterations);
    bytesPerKernel = 2 * effectiveN * sizeof(float);
    bw = bytesPerKernel / (ms / 1000.0) / 1e9;
    pct = bw / peakBW * 100.0;
    printf("%-30s %12.4f %15.2f %14.1f%%\n", "Stride=8", ms, bw, pct);

    // --- Test Stride=32 (worst) ---
    effectiveN = N / 32;
    gridSize = (effectiveN + blockSize - 1) / blockSize;
    ms = benchmarkKernel(readStride32, d_in, d_out, effectiveN, gridSize, blockSize, iterations);
    bytesPerKernel = 2 * effectiveN * sizeof(float);
    bw = bytesPerKernel / (ms / 1000.0) / 1e9;
    pct = bw / peakBW * 100.0;
    printf("%-30s %12.4f %15.2f %14.1f%%\n", "Stride=32 (worst)", ms, bw, pct);

    // --- Test float4 vectorized (still coalesced, fewer transactions) ---
    effectiveN = N;   // all elements, but accessed in groups of 4
    int gridFloat4 = ((N/4) + blockSize - 1) / blockSize;
    ms = benchmarkKernel(readFloat4, d_in, d_out, effectiveN, gridFloat4, blockSize, iterations);
    bytesPerKernel = 2 * N * sizeof(float);   // full array read+write
    bw = bytesPerKernel / (ms / 1000.0) / 1e9;
    pct = bw / peakBW * 100.0;
    printf("%-30s %12.4f %15.2f %14.1f%%\n", "Float4 vectorized", ms, bw, pct);

    // Cleanup
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    free(h_in);
    free(h_out);

    printf("\nConclusion: Coalesced access (Stride-1) achieves high bandwidth.\n");
    printf("Increasing stride degrades performance due to increased memory transactions.\n");
    printf("Vectorized loads can further improve efficiency on some architectures.\n");

    return 0;
}