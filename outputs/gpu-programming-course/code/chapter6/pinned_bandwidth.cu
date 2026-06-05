/**
 * pinned_bandwidth.cu
 *
 * 第6章动手体验：页锁定内存 vs 可分页内存带宽对比
 *
 * 比较四种内存类型的 Host -> Device 传输带宽：
 *   1. 可分页内存 (malloc)
 *   2. 页锁定内存 (cudaMallocHost)
 *   3. 页锁定 + 写结合 (cudaHostAllocWriteCombined)
 *   4. 页锁定 + 映射内存 (cudaHostAllocMapped)
 *
 * 编译：
 *   nvcc pinned_bandwidth.cu -o pinned_bandwidth -arch=sm_60
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

// 测试迭代次数（多次迭代取平均，减少波动）
#define NUM_ITERATIONS 100

/**
 * 测试可分页内存的 Host -> Device 传输带宽
 *
 * @param N       元素个数
 * @param h_data  输入数据（主机端）
 * @param d_data  设备端缓冲区
 * @return        带宽 (GB/s)
 */
float bandwidthPageable(int N, float *h_data, float *d_data,
                        cudaEvent_t start, cudaEvent_t stop)
{
    size_t size = N * sizeof(float);

    // 将事件记录到默认流
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size,
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // 有效带宽 = 总传输字节数 / 总时间
    // 乘以 2 是因为每次传输包括读和写（这里只是 HtoD，所以乘以 1）
    // 实际公式：bandwidth = (size * NUM_ITERATIONS) / (ms / 1000) bytes/s
    float totalBytes = (float)(size) * NUM_ITERATIONS;
    float bandwidth = totalBytes / (ms / 1000.0f) / (1024.0f * 1024.0f *
                                                      1024.0f);
    return bandwidth;
}

/**
 * 测试页锁定内存的 Host -> Device 传输带宽
 */
float bandwidthPinned(int N, float *h_data, float *d_data,
                      cudaEvent_t start, cudaEvent_t stop)
{
    size_t size = N * sizeof(float);

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size,
                              cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    float totalBytes = (float)(size) * NUM_ITERATIONS;
    float bandwidth = totalBytes / (ms / 1000.0f) / (1024.0f * 1024.0f *
                                                      1024.0f);
    return bandwidth;
}

/**
 * 测试设备到主机的带宽（用于写结合和映射内存的可选测试）
 */
float bandwidthDeviceToHost(int N, float *h_data, float *d_data,
                            cudaEvent_t start, cudaEvent_t stop)
{
    size_t size = N * sizeof(float);

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        CUDA_CHECK(cudaMemcpy(h_data, d_data, size,
                              cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    float totalBytes = (float)(size) * NUM_ITERATIONS;
    float bandwidth = totalBytes / (ms / 1000.0f) / (1024.0f * 1024.0f *
                                                      1024.0f);
    return bandwidth;
}

int main()
{
    // =========================================================================
    // 参数设置
    // =========================================================================
    int N = 16 * 1024 * 1024; // 16M 元素 = 64 MB
    size_t size = (size_t)N * sizeof(float);

    printf("========================================\n");
    printf("Memory Transfer Bandwidth Comparison\n");
    printf("========================================\n");
    printf("Data size: %.2f MB\n",
           (float)size / (1024.0f * 1024.0f));
    printf("Iterations per test: %d\n", NUM_ITERATIONS);
    printf("Total data transferred per test: %.2f MB\n\n",
           (float)(size * NUM_ITERATIONS) / (1024.0f * 1024.0f));

    // 显示设备信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (CC %d.%d)\n", prop.name, prop.major,
           prop.minor);
    printf("\n");

    // =========================================================================
    // 创建计时事件
    // =========================================================================
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 分配设备内存（只用一次）
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
    // 初始化数据
    for (int i = 0; i < N; i++)
    {
        h_pageable[i] = (float)i;
    }
    // 先执行一次"暖身"传输（触发 CUDA 初始化）
    cudaMemcpy(d_data, h_pageable, size, cudaMemcpyHostToDevice);

    float bw_pageable = bandwidthPageable(N, h_pageable, d_data,
                                           start, stop);
    printf("  Host -> Device Bandwidth: %.2f GB/s\n", bw_pageable);

    // =========================================================================
    // 测试 2：页锁定内存 (cudaMallocHost)
    // =========================================================================
    printf("--- Test 2: Pinned Memory (cudaMallocHost) ---\n");
    float *h_pinned;
    CUDA_CHECK(cudaMallocHost(&h_pinned, size));
    for (int i = 0; i < N; i++)
    {
        h_pinned[i] = (float)i;
    }
    // 暖身传输
    cudaMemcpy(d_data, h_pinned, size, cudaMemcpyHostToDevice);

    float bw_pinned = bandwidthPinned(N, h_pinned, d_data,
                                       start, stop);
    printf("  Host -> Device Bandwidth: %.2f GB/s\n", bw_pinned);
    printf("  Speedup vs pageable:     %.2fx\n",
           bw_pinned / bw_pageable);

    // =========================================================================
    // 测试 3：页锁定 + 写结合 (cudaHostAllocWriteCombined)
    // =========================================================================
    printf("--- Test 3: Write-Combined Memory ---\n");
    float *h_wc;
    CUDA_CHECK(cudaHostAlloc(&h_wc, size, cudaHostAllocWriteCombined));
    for (int i = 0; i < N; i++)
    {
        h_wc[i] = (float)i;
    }
    // 暖身传输
    cudaMemcpy(d_data, h_wc, size, cudaMemcpyHostToDevice);

    float bw_wc = bandwidthPinned(N, h_wc, d_data, start, stop);
    printf("  Host -> Device Bandwidth: %.2f GB/s\n", bw_wc);
    printf("  Speedup vs pageable:     %.2fx\n",
           bw_wc / bw_pageable);
    printf("  Difference vs pinned:    %.2f%%\n",
           ((bw_wc - bw_pinned) / bw_pinned) * 100.0f);

    // =========================================================================
    // 测试 4：页锁定 + 映射内存 (cudaHostAllocMapped)
    // =========================================================================
    printf("--- Test 4: Mapped (Zero-Copy) Memory ---\n");
    float *h_mapped;
    CUDA_CHECK(
        cudaHostAlloc(&h_mapped, size, cudaHostAllocMapped));
    for (int i = 0; i < N; i++)
    {
        h_mapped[i] = (float)i;
    }
    // 获取设备端指针
    float *d_mapped;
    CUDA_CHECK(
        cudaHostGetDevicePointer(&d_mapped, h_mapped, 0));

    // 测试显式 cudaMemcpy 带宽（mapped 内存同样可以用于 cudaMemcpy）
    cudaMemcpy(d_data, h_mapped, size, cudaMemcpyHostToDevice);
    float bw_mapped = bandwidthPinned(N, h_mapped, d_data,
                                       start, stop);
    printf("  Host -> Device Bandwidth (cudaMemcpy): %.2f GB/s\n",
           bw_mapped);

    // =========================================================================
    // 测试 5：设备到主机的带宽
    // =========================================================================
    printf("--- Test 5: Device-to-Host Bandwidth ---\n");

    float bw_d2h_pageable = bandwidthDeviceToHost(
        N, h_pageable, d_data, start, stop);
    printf("  Pageable (Device->Host): %.2f GB/s\n", bw_d2h_pageable);

    float bw_d2h_pinned = bandwidthDeviceToHost(N, h_pinned, d_data,
                                                 start, stop);
    printf("  Pinned   (Device->Host): %.2f GB/s (%.2fx)\n",
           bw_d2h_pinned, bw_d2h_pinned / bw_d2h_pageable);

    // =========================================================================
    // 结果汇总
    // =========================================================================
    printf("\n========================================\n");
    printf("RESULTS SUMMARY\n");
    printf("========================================\n");
    printf("%-30s %10s %10s\n", "Memory Type", "H->D (GB/s)",
           "Speedup");
    printf("-----------------------------------------------\n");
    printf("%-30s %10.2f %10s\n", "Pageable (malloc)", bw_pageable,
           "1.00x");
    printf("%-30s %10.2f %10.2fx\n", "Pinned (cudaMallocHost)",
           bw_pinned, bw_pinned / bw_pageable);
    printf("%-30s %10.2f %10.2fx\n",
           "Write-Combined", bw_wc, bw_wc / bw_pageable);
    printf("%-30s %10.2f %10.2fx\n", "Mapped (Zero-Copy)",
           bw_mapped, bw_mapped / bw_pageable);
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

    printf("Done!\n");
    return 0;
}
