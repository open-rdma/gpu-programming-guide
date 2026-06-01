/**
 * Chapter 10 - Experiment 10-3: Occupancy Tuning Experiment
 * 
 * Measures the impact of register and shared memory usage on kernel occupancy and performance.
 * Compile: nvcc -arch=sm_80 -O3 occupancy_tuning.cu -o occupancy_tuning
 * Run: ./occupancy_tuning
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

// 模板特化：处理SHMEM_BYTES=0的情况（不声明共享内存）
template<int REGISTERS_PER_THREAD, int SHMEM_BYTES>
__global__ void sharedMemPressure(float* a, float* b, float* c, int N) {
    // 消耗指定数量的寄存器
    float regs[REGISTERS_PER_THREAD];
    
    // 初始化寄存器（防止编译器优化）
    #pragma unroll
    for (int i = 0; i < REGISTERS_PER_THREAD; i++) {
        regs[i] = (float)i;
    }
    
    // 只有当SHMEM_BYTES>0时才声明和使用共享内存
    #if SHMEM_BYTES > 0
    __shared__ float smem[SHMEM_BYTES / sizeof(float)];
    int tid = threadIdx.x;
    
    // 使用共享内存（防止编译器优化）
    if (tid < SHMEM_BYTES / sizeof(float)) {
        smem[tid] = regs[0];
    }
    __syncthreads();
    #endif
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // 执行一些计算来消耗寄存器
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < REGISTERS_PER_THREAD; i++) {
            sum += regs[i] * a[idx] + b[idx];
        }
        c[idx] = sum;
    }
}

// 测量内核执行时间
float measureKernel(void (*kernel)(float*, float*, float*, int), 
                   float* d_a, float* d_b, float* d_c, int N,
                   int blockSize, int iterations) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // 预热
    kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < iterations; i++) {
        kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
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
    
    printf("=== CUDA Occupancy Tuning Experiment ===\n");
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / 32);
    printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Shared memory per SM: %d KB\n\n", prop.sharedMemPerMultiprocessor / 1024);
    
    const int N = 1 << 22; // 400万个元素
    const int blockSize = 256;
    const int iterations = 10;
    
    float *h_a, *h_b, *h_c;
    float *d_a, *d_b, *d_c;
    
    // 分配主机内存
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));
    
    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // 分配设备内存
    CHECK_CUDA(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c, N * sizeof(float)));
    
    // 拷贝数据到设备
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));
    
    printf("%-25s %10s %15s\n", "Configuration", "Time(ms)", "Theoretical Occupancy");
    printf("---------------------------------------------------------\n");
    
    // 测试不同寄存器使用量（共享内存=0）
    float ms;
    
    // 16 registers/thread
    ms = measureKernel(sharedMemPressure<16, 0>, d_a, d_b, d_c, N, blockSize, iterations);
    printf("%-25s %10.4f %15.1f%%\n", "16 regs, 0 KB shmem", ms, 100.0f);
    
    // 32 registers/thread
    ms = measureKernel(sharedMemPressure<32, 0>, d_a, d_b, d_c, N, blockSize, iterations);
    printf("%-25s %10.4f %15.1f%%\n", "32 regs, 0 KB shmem", ms, 50.0f);
    
    // 64 registers/thread
    ms = measureKernel(sharedMemPressure<64, 0>, d_a, d_b, d_c, N, blockSize, iterations);
    printf("%-25s %10.4f %15.1f%%\n", "64 regs, 0 KB shmem", ms, 25.0f);
    
    // 128 registers/thread
    ms = measureKernel(sharedMemPressure<128, 0>, d_a, d_b, d_c, N, blockSize, iterations);
    printf("%-25s %10.4f %15.1f%%\n", "128 regs, 0 KB shmem", ms, 12.5f);
    
    printf("\n");
    
    // 测试不同共享内存使用量（寄存器=16）
    // 4 KB shmem/block
    ms = measureKernel(sharedMemPressure<16, 4096>, d_a, d_b, d_c, N, blockSize, iterations);
    printf("%-25s %10.4f %15.1f%%\n", "16 regs, 4 KB shmem", ms, 100.0f);
    
    // 8 KB shmem/block
    ms = measureKernel(sharedMemPressure<16, 8192>, d_a, d_b, d_c, N, blockSize, iterations);
    printf("%-25s %10.4f %15.1f%%\n", "16 regs, 8 KB shmem", ms, 50.0f);
    
    // 16 KB shmem/block
    ms = measureKernel(sharedMemPressure<16, 16384>, d_a, d_b, d_c, N, blockSize, iterations);
    printf("%-25s %10.4f %15.1f%%\n", "16 regs, 16 KB shmem", ms, 25.0f);
    
    // 32 KB shmem/block
    ms = measureKernel(sharedMemPressure<16, 32768>, d_a, d_b, d_c, N, blockSize, iterations);
    printf("%-25s %10.4f %15.1f%%\n", "16 regs, 32 KB shmem", ms, 12.5f);
    
    // 清理
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    
    printf("\nKey Insights:\n");
    printf("1. Occupancy decreases as register usage per thread increases\n");
    printf("2. Occupancy decreases as shared memory usage per block increases\n");
    printf("3. Higher occupancy does not always mean better performance\n");
    printf("4. The optimal occupancy depends on the kernel's compute/memory ratio\n");
    
    return 0;
}