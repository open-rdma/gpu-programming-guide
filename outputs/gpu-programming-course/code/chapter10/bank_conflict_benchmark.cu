/**
 * Chapter 10 - Experiment 10-2: Shared Memory Bank Conflict Benchmark
 *
 * Measures the impact of different shared memory access patterns on performance.
 * Compile: nvcc -arch=sm_86 -O3 bank_conflict_benchmark.cu -o bank_conflict_benchmark
 * Run: ./bank_conflict_benchmark
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

// 32 banks, each bank is 4 bytes wide.
// Address mapping: bank = (byte_address / 4) % 32

// No bank conflict: each thread accesses s[tid]
__global__ void noConflict(float *output) {
    __shared__ float sdata[256];  // 256 * 4 = 1024 bytes
    int tid = threadIdx.x;

    // Initialize shared memory - no conflict on write
    sdata[tid] = (float)tid;
    __syncthreads();

    // Read - no bank conflict: tid maps to bank[tid]
    // Each thread reads from a different bank
    float val = sdata[tid];

    // Repeat to amplify the effect
    for (int i = 0; i < 1000; i++) {
        val += sdata[tid];
        __syncthreads();
    }
    __syncthreads();

    output[tid] = val;
}

// 2-way bank conflict: each thread accesses s[2*tid]
// thread 0 -> bank 0, thread 1 -> bank 2, thread 16 -> bank 0 (conflict with thread 0!)
__global__ void conflict2Way(float *output) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    sdata[2 * tid] = (float)tid;
    __syncthreads();

    float val = sdata[2 * tid];
    for (int i = 0; i < 1000; i++) {
        val += sdata[2 * tid];
        __syncthreads();
    }
    __syncthreads();

    output[tid] = val;
}

// 4-way bank conflict
__global__ void conflict4Way(float *output) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    sdata[4 * tid] = (float)tid;
    __syncthreads();

    float val = sdata[4 * tid];
    for (int i = 0; i < 1000; i++) {
        val += sdata[4 * tid];
        __syncthreads();
    }
    __syncthreads();

    output[tid] = val;
}

// 32-way bank conflict: all threads access the same bank
// s[32*tid] always maps to bank 0
__global__ void conflict32Way(float *output) {
    __shared__ float sdata[1024];  // need larger array for 32*tid up to 31
    int tid = threadIdx.x;

    sdata[32 * tid] = (float)tid;
    __syncthreads();

    float val = sdata[32 * tid];
    for (int i = 0; i < 1000; i++) {
        val += sdata[32 * tid];
        __syncthreads();
    }
    __syncthreads();

    output[tid] = val;
}

// Fixed bank conflict using padding: s[32][33] instead of s[32][32]
// The extra column shifts elements so same row indices map to different banks
__global__ void noConflictWithPadding(float *output) {
    __shared__ float sdata[32][33];  // 32 rows, 33 cols (padded)
    int tid = threadIdx.x;
    int row = tid / 32;
    int col = tid % 32;

    // Write in row-major - uses padded layout
    sdata[row][col] = (float)tid;
    __syncthreads();

    // Read in column-major (transpose-like) - NO bank conflict due to padding!
    float val = sdata[col][row];  // Column major read, but padding avoids conflicts
    for (int i = 0; i < 1000; i++) {
        val += sdata[col][row];
        __syncthreads();
    }
    __syncthreads();

    output[tid] = val;
}

// Without padding - column-major read causes bank conflicts
__global__ void conflictTranspose(float *output) {
    __shared__ float sdata[32][32];  // 32 rows, 32 cols (no padding)
    int tid = threadIdx.x;
    int row = tid / 32;
    int col = tid % 32;

    sdata[row][col] = (float)tid;
    __syncthreads();

    float val = sdata[col][row];  // Bank conflict on each pair of threads!
    for (int i = 0; i < 1000; i++) {
        val += sdata[col][row];
        __syncthreads();
    }
    __syncthreads();

    output[tid] = val;
}

float measureKernel(void (*kernel)(float*), float *d_out, int gridSize, int blockSize, int iterations) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    kernel<<<gridSize, blockSize>>>(d_out);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < iterations; i++) {
        kernel<<<gridSize, blockSize>>>(d_out);
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
    printf("=== Shared Memory Bank Conflict Benchmark ===\n");
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Shared memory per SM: %d KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Shared memory per block: %d KB\n", prop.sharedMemPerBlock / 1024);
    printf("\n");

    const int blockSize = 32;  // One warp only for clear measurement
    const int gridSize = 10240;  // Many blocks to saturate GPU
    const int iterations = 20;

    float *d_out;
    CHECK_CUDA(cudaMalloc(&d_out, blockSize * gridSize * sizeof(float)));

    printf("%-35s %10s %12s\n", "Access Pattern", "Time(ms)", "Relative");
    printf("-----------------------------------------------------------------\n");

    float baseline;

    // No conflict - baseline
    float ms = measureKernel(noConflict, d_out, gridSize, blockSize, iterations);
    baseline = ms;
    printf("%-35s %10.4f %11.2fx\n", "No conflict (sequential)", ms, ms / baseline);

    // 2-way conflict
    ms = measureKernel(conflict2Way, d_out, gridSize, blockSize, iterations);
    printf("%-35s %10.4f %11.2fx\n", "2-way bank conflict", ms, ms / baseline);

    // 4-way conflict
    ms = measureKernel(conflict4Way, d_out, gridSize, blockSize, iterations);
    printf("%-35s %10.4f %11.2fx\n", "4-way bank conflict", ms, ms / baseline);

    // 32-way conflict
    ms = measureKernel(conflict32Way, d_out, gridSize, blockSize, iterations);
    printf("%-35s %10.4f %11.2fx\n", "32-way bank conflict", ms, ms / baseline);

    // Transpose without padding (bank conflicts)
    ms = measureKernel(conflictTranspose, d_out, gridSize, blockSize, iterations);
    printf("%-35s %10.4f %11.2fx\n", "Transpose w/o padding", ms, ms / baseline);

    // Transpose with padding (no bank conflicts)
    ms = measureKernel(noConflictWithPadding, d_out, gridSize, blockSize, iterations);
    printf("%-35s %10.4f %11.2fx\n", "Transpose with padding (+1)", ms, ms / baseline);

    CHECK_CUDA(cudaFree(d_out));

    printf("\nKey insight: Adding a padding column (+1) to shared memory arrays\n");
    printf("can eliminate bank conflicts in transpose-like access patterns.\n");
    printf("For CC 5.0+: 32 banks, 4-byte words -> bank = (addr/4) %% 32\n");

    return 0;
}
