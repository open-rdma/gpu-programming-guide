/**
 * pinned_bandwidth.cu
 *
 * 第6章动手体验：页锁定内存 vs 可分页内存带宽对比
 *
 * 比较四种内存类型的 Host <-> Device 传输带宽：
 *   1. 可分页内存 (malloc)
 *   2. 页锁定内存 (cudaMallocHost)
 *   3. 页锁定 + 写结合 (cudaHostAllocWriteCombined)
 *   4. 页锁定 + 映射内存 (cudaHostAllocMapped)
 *
 * 编译：
 *   nvcc pinned_bandwidth.cu -o pinned_bandwidth -arch=sm_80
 *
 * 运行：
 *   ./pinned_bandwidth
 */

#include <stdio.h>
#include <cuda_runtime.h>

// 错误检查宏
#define CUDA_CHECK(err)                                                        \
    do                                                                         \
    {                                                                          \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess)                                               \
        {                                                                      \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,            \
                   cudaGetErrorString(err_));                                   \
            exit(-1);                                                          \
        }                                                                      \
    } while (0)

// 测试迭代次数（针对A800优化：大数据量+少迭代）
#define NUM_ITERATIONS 20

/**
 * 测试 Host -> Device 传输带宽
 */
float bandwidthHtoD(int N, const float *h_data, float *d_data,
                    cudaEvent_t start, cudaEvent_t stop)
{
    size_t size = (size_t)N * sizeof(float);

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    double totalBytes = (double)size * NUM_ITERATIONS;
    double bandwidth = totalBytes / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    return (float)bandwidth;
}

/**
 * 测试 Device -> Host 传输带宽
 */
float bandwidthDtoH(int N, float *h_data, const float *d_data,
                    cudaEvent_t start, cudaEvent_t stop)
{
    size_t size = (size_t)N * sizeof(float);

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    double totalBytes = (double)size * NUM_ITERATIONS;
    double bandwidth = totalBytes / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    return (float)bandwidth;
}

/**
 * 测试映射内存的零拷贝访问带宽（直接在GPU中访问主机内存）
 */
__global__ void zeroCopyKernel(const float *d_mapped, float *d_out, int N)
{
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < N)
    {
        d_out[idx] = d_mapped[idx] * 2.0f; // 简单的计算操作
    }
}

float bandwidthZeroCopy(int N, const float *d_mapped, float *d_out,
                        cudaEvent_t start, cudaEvent_t stop)
{
    size_t size = (size_t)N * sizeof(float);
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    // 暖身运行
    zeroCopyKernel<<<grid, block>>>(d_mapped, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        zeroCopyKernel<<<grid, block>>>(d_mapped, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // 零拷贝带宽计算：每次迭代读取N个float，写入N个float
    double totalBytes = (double)size * 2 * NUM_ITERATIONS;
    double bandwidth = totalBytes / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    return (float)bandwidth;
}

int main()
{
    // =========================================================================
    // 参数设置（针对A800优化：256MB数据量）
    // =========================================================================
    int N = 64 * 1024 * 1024; // 64M 元素 = 256 MB
    size_t size = (size_t)N * sizeof(float);

    printf("========================================\n");
    printf("Memory Transfer Bandwidth Comparison\n");
    printf("========================================\n");
    printf("Data size: %.2f MB\n", (float)size / (1024.0f * 1024.0f));
    printf("Iterations per test: %d\n", NUM_ITERATIONS);
    printf("Total data transferred per test: %.2f MB\n\n",
           (float)(size * NUM_ITERATIONS) / (1024.0f * 1024.0f));

    // 显示设备信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (CC %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("A800 PCIe 4.0 x16 Theoretical Bandwidth: 32.0 GB/s\n");
    printf("\n");

    // =========================================================================
    // 创建计时事件
    // =========================================================================
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 分配设备内存
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));

    // =========================================================================
    // 测试 1：可分页内存 (malloc)
    // =========================================================================
    printf("--- Test 1: Pageable Memory (malloc) ---\n");
    float *h_pageable = (float *)malloc(size);
    if (!h_pageable)
    {
        printf("  Failed to allocate pageable memory!\n");
        return -1;
    }
    for (int i = 0; i < N; i++)
    {
        h_pageable[i] = (float)i;
    }
    // 暖身传输
    CUDA_CHECK(cudaMemcpy(d_data, h_pageable, size, cudaMemcpyHostToDevice));

    float bw_h2d_pageable = bandwidthHtoD(N, h_pageable, d_data, start, stop);
    float bw_d2h_pageable = bandwidthDtoH(N, h_pageable, d_data, start, stop);
    printf("  Host -> Device: %.2f GB/s\n", bw_h2d_pageable);
    printf("  Device -> Host: %.2f GB/s\n", bw_d2h_pageable);

    // =========================================================================
    // 测试 2：页锁定内存 (cudaMallocHost)
    // =========================================================================
    printf("\n--- Test 2: Pinned Memory (cudaMallocHost) ---\n");
    float *h_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, size));
    for (int i = 0; i < N; i++)
    {
        h_pinned[i] = (float)i;
    }
    // 暖身传输
    CUDA_CHECK(cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice));

    float bw_h2d_pinned = bandwidthHtoD(N, h_pinned, d_data, start, stop);
    float bw_d2h_pinned = bandwidthDtoH(N, h_pinned, d_data, start, stop);
    printf("  Host -> Device: %.2f GB/s (%.2fx)\n",
           bw_h2d_pinned, bw_h2d_pinned / bw_h2d_pageable);
    printf("  Device -> Host: %.2f GB/s (%.2fx)\n",
           bw_d2h_pinned, bw_d2h_pinned / bw_d2h_pageable);

    // =========================================================================
    // 测试 3：页锁定 + 写结合 (cudaHostAllocWriteCombined)
    // =========================================================================
    printf("\n--- Test 3: Write-Combined Memory ---\n");
    float *h_wc;
    CUDA_CHECK(cudaHostAlloc(&h_wc, size, cudaHostAllocWriteCombined));
    for (int i = 0; i < N; i++)
    {
        h_wc[i] = (float)i;
    }
    __sync_synchronize(); // 内存屏障，确保数据完全写入
    // 暖身传输
    CUDA_CHECK(cudaMemcpy(d_data, h_wc, size, cudaMemcpyHostToDevice));

    float bw_h2d_wc = bandwidthHtoD(N, h_wc, d_data, start, stop);
    float bw_d2h_wc = bandwidthDtoH(N, h_wc, d_data, start, stop);
    printf("  Host -> Device: %.2f GB/s (%.2fx)\n",
           bw_h2d_wc, bw_h2d_wc / bw_h2d_pageable);
    printf("  Device -> Host: %.2f GB/s (%.2fx)\n",
           bw_d2h_wc, bw_d2h_wc / bw_d2h_pageable);
    printf("  H->D vs standard pinned: %.2f%%\n",
           ((bw_h2d_wc - bw_h2d_pinned) / bw_h2d_pinned) * 100.0f);

    // =========================================================================
    // 测试 4：页锁定 + 映射内存 (cudaHostAllocMapped)
    // =========================================================================
    printf("\n--- Test 4: Mapped (Zero-Copy) Memory ---\n");
    float *h_mapped;
    CUDA_CHECK(cudaHostAlloc(&h_mapped, size, cudaHostAllocMapped));
    for (int i = 0; i < N; i++)
    {
        h_mapped[i] = (float)i;
    }
    // 获取设备端指针
    float *d_mapped;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_mapped, h_mapped, 0));

    // 测试1：显式cudaMemcpy带宽
    CUDA_CHECK(cudaMemcpy(d_data, h_mapped, size, cudaMemcpyHostToDevice));
    float bw_h2d_mapped = bandwidthHtoD(N, h_mapped, d_data, start, stop);
    float bw_d2h_mapped = bandwidthDtoH(N, h_mapped, d_data, start, stop);
    printf("  Host -> Device (cudaMemcpy): %.2f GB/s\n", bw_h2d_mapped);
    printf("  Device -> Host (cudaMemcpy): %.2f GB/s\n", bw_d2h_mapped);

    // 测试2：零拷贝访问带宽（核心特性）
    float bw_zero_copy = bandwidthZeroCopy(N, d_mapped, d_data, start, stop);
    printf("  Zero-Copy Access Bandwidth: %.2f GB/s\n", bw_zero_copy);

    // =========================================================================
    // 结果汇总
    // =========================================================================
    printf("\n========================================\n");
    printf("RESULTS SUMMARY (Host -> Device)\n");
    printf("========================================\n");
    printf("%-30s %10s %10s\n", "Memory Type", "Bandwidth", "Speedup");
    printf("-----------------------------------------------\n");
    printf("%-30s %10.2f %10s\n", "Pageable (malloc)", bw_h2d_pageable, "1.00x");
    printf("%-30s %10.2f %10.2fx\n", "Pinned (cudaMallocHost)", bw_h2d_pinned, bw_h2d_pinned / bw_h2d_pageable);
    printf("%-30s %10.2f %10.2fx\n", "Write-Combined", bw_h2d_wc, bw_h2d_wc / bw_h2d_pageable);
    printf("%-30s %10.2f %10.2fx\n", "Mapped (cudaMemcpy)", bw_h2d_mapped, bw_h2d_mapped / bw_h2d_pageable);
    printf("========================================\n");

    // =========================================================================
    // 释放资源
    // =========================================================================
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    free(h_pageable);
    CUDA_CHECK(cudaFreeHost(h_pinned));
    CUDA_CHECK(cudaFreeHost(h_wc));
    CUDA_CHECK(cudaFreeHost(h_mapped));

    printf("\nDone!\n");
    return 0;
}