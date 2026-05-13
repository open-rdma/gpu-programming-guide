/**
 * Chapter 10 - Experiment 10-3: Occupancy Tuning
 *
 * Demonstrates how register usage, shared memory usage, and block size
 * affect occupancy and performance.
 *
 * Compile: nvcc -arch=sm_86 -O3 occupancy_tuning.cu -o occupancy_tuning
 * Run: ./occupancy_tuning
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

// Kernel with configurable register pressure
// By using many temporary variables, we force the compiler to use more registers
template<int NUM_TEMPS>
__global__ void registerPressure(float * __restrict__ a,
                                  float * __restrict__ b,
                                  float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float v0 = a[idx];
        float v1 = b[idx];
        float v2 = v0 * v1;
        float v3 = v2 + v0;
        float v4 = v3 * v1;
        float v5 = v4 - v0;
        float v6 = v5 * v2;
        float v7 = v6 + v3;
        float v8 = v7 * v4;

        // Use all temps to prevent compiler optimization
        if (NUM_TEMPS >= 1) c[idx] = v0;
        if (NUM_TEMPS >= 2) c[idx] += v1;
        if (NUM_TEMPS >= 3) c[idx] += v2;
        if (NUM_TEMPS >= 4) c[idx] += v3;
        if (NUM_TEMPS >= 5) c[idx] += v4;
        if (NUM_TEMPS >= 6) c[idx] += v5;
        if (NUM_TEMPS >= 7) c[idx] += v6;
        if (NUM_TEMPS >= 8) c[idx] += v7;
        if (NUM_TEMPS >= 9) c[idx] += v8;
    }
}

// Explicit instantiation
template __global__ void registerPressure<1>(float*, float*, float*, int);
template __global__ void registerPressure<3>(float*, float*, float*, int);
template __global__ void registerPressure<6>(float*, float*, float*, int);
template __global__ void registerPressure<9>(float*, float*, float*, int);

// Kernel with configurable shared memory usage
template<int SHMEM_BYTES>
__global__ void sharedMemPressure(float * __restrict__ a,
                                   float * __restrict__ b,
                                   float * __restrict__ c, int n) {
    __shared__ float smem[SHMEM_BYTES / sizeof(float)];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Initialize shared memory
    if (tid < SHMEM_BYTES / (int)sizeof(float)) {
        smem[tid] = (float)tid;
    }
    __syncthreads();

    if (idx < n) {
        float sum = a[idx] + b[idx];
        // Use shared memory data
        for (int i = 0; i < SHMEM_BYTES / (int)sizeof(float) && i < 256; i++) {
            sum += smem[i] * 0.001f;
        }
        c[idx] = sum;
    }
}

template __global__ void sharedMemPressure<0>(float*, float*, float*, int);
template __global__ void sharedMemPressure<1024>(float*, float*, float*, int);
template __global__ void sharedMemPressure<4096>(float*, float*, float*, int);
template __global__ void sharedMemPressure<8192>(float*, float*, float*, int);
template __global__ void sharedMemPressure<16384>(float*, float*, float*, int);

// Helper to calculate occupancy
float calculateOccupancy(int blockSize, int dynamicSmem,
                         void* kernel, const char* name) {
    int numBlocks;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, kernel, blockSize, dynamicSmem));

    cudaDeviceProp prop;
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    int activeWarps = numBlocks * blockSize / prop.warpSize;
    int maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    float occupancy = (float)activeWarps / maxWarps * 100.0f;

    printf("%-30s BlockSize=%4d, Smem=%5d: %2d blocks/SM, %3d warps/SM, Occupancy=%.1f%%\n",
           name, blockSize, dynamicSmem, numBlocks, activeWarps, occupancy);

    return occupancy;
}

// Simple benchmark function
float benchmarkKernelFloat(void (*kernel)(float*, float*, float*, int),
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

    printf("=== Occupancy Tuning Experiment ===\n");
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Shared memory per SM: %d KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Max block size: %d\n", prop.maxThreadsPerBlock);
    printf("Max blocks per SM: %d\n\n", prop.maxBlocksPerMultiProcessor);

    const int N = 16 * 1024 * 1024;  // 16M elements
    const int iterations = 50;
    const size_t bytes = N * sizeof(float);

    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    // ===== Part 1: Effect of block size on occupancy =====
    printf("--- Part 1: Block Size vs Occupancy ---\n");
    printf("(Kernel: registerPressure<3> - moderate register usage)\n\n");

    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    for (int i = 0; i < 6; i++) {
        int bs = blockSizes[i];
        calculateOccupancy(bs, 0, (void*)registerPressure<3>, "regPressure<3>");
        int gs = (N + bs - 1) / bs;
        float ms = benchmarkKernelFloat(registerPressure<3>, d_a, d_b, d_c, N, gs, bs, iterations);
        printf("   -> Time: %.4f ms\n\n", ms);
    }

    // ===== Part 2: Effect of register usage on occupancy =====
    printf("--- Part 2: Register Usage vs Occupancy ---\n");
    printf("(Block size fixed at 256)\n\n");

    calculateOccupancy(256, 0, (void*)registerPressure<1>, "regPressure<1> (few regs)");
    calculateOccupancy(256, 0, (void*)registerPressure<3>, "regPressure<3>");
    calculateOccupancy(256, 0, (void*)registerPressure<6>, "regPressure<6>");
    calculateOccupancy(256, 0, (void*)registerPressure<9>, "regPressure<9> (many regs)");

    // ===== Part 3: Effect of shared memory on occupancy =====
    printf("\n--- Part 3: Shared Memory Usage vs Occupancy ---\n");
    printf("(Block size fixed at 256)\n\n");

    calculateOccupancy(256, 0, (void*)sharedMemPressure<0>, "shmemPressure<0B>");
    calculateOccupancy(256, 1024, (void*)sharedMemPressure<1024>, "shmemPressure<1KB>");
    calculateOccupancy(256, 4096, (void*)sharedMemPressure<4096>, "shmemPressure<4KB>");
    calculateOccupancy(256, 8192, (void*)sharedMemPressure<8192>, "shmemPressure<8KB>");
    calculateOccupancy(256, 16384, (void*)sharedMemPressure<16384>, "shmemPressure<16KB>");

    // ===== Part 4: Auto-configuration =====
    printf("\n--- Part 4: Auto-configuration using cudaOccupancyMaxPotentialBlockSize ---\n\n");

    int minGridSize, blockSize;
    CHECK_CUDA(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize,
        (void*)registerPressure<3>, 0, N));

    printf("Recommended block size: %d\n", blockSize);
    printf("Minimum grid size for full occupancy: %d\n", minGridSize);

    int actualGrid = (N + blockSize - 1) / blockSize;
    printf("Actual grid size for N=%d: %d\n\n", N, actualGrid);

    // ===== Part 5: Performance with auto vs manual =====
    printf("--- Part 5: Performance Comparison ---\n\n");

    // Auto-configured
    float ms_auto = benchmarkKernelFloat(registerPressure<3>, d_a, d_b, d_c, N,
                                         actualGrid, blockSize, iterations);
    printf("Auto-configured (bs=%d, gs=%d): %.4f ms\n", blockSize, actualGrid, ms_auto);

    // Manual 256 threads
    int gs256 = (N + 255) / 256;
    float ms_256 = benchmarkKernelFloat(registerPressure<3>, d_a, d_b, d_c, N,
                                        gs256, 256, iterations);
    printf("Manual 256 (bs=256, gs=%d): %.4f ms\n", gs256, ms_256);

    // Manual 128 threads
    int gs128 = (N + 127) / 128;
    float ms_128 = benchmarkKernelFloat(registerPressure<3>, d_a, d_b, d_c, N,
                                        gs128, 128, iterations);
    printf("Manual 128 (bs=128, gs=%d): %.4f ms\n", gs128, ms_128);

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    printf("\nKey takeaways:\n");
    printf("1. Block size should be a multiple of 32 (warp size)\n");
    printf("2. More registers per thread = fewer blocks per SM = lower occupancy\n");
    printf("3. More shared memory per block = fewer blocks per SM = lower occupancy\n");
    printf("4. Use cudaOccupancyMaxPotentialBlockSize for automatic tuning\n");
    printf("5. The optimal block size depends on your specific kernel\n");

    return 0;
}
