/**
 * device_query.cu
 *
 * 第4章动手体验：CUDA 设备查询与 __CUDA_ARCH__ 宏演示
 *
 * 编译示例：
 *   nvcc device_query.cu -o device_query -arch=sm_60
 *   nvcc device_query.cu -o device_query_fat \
 *       -gencode arch=compute_60,code=sm_60 \
 *       -gencode arch=compute_70,code=sm_70 \
 *       -gencode arch=compute_80,code="compute_80,sm_80"
 */

#include <stdio.h>
#include <cuda_runtime.h>

/**
 * 演示 __CUDA_ARCH__ 宏的内核
 * 根据编译时的目标计算能力打印不同信息
 */
__global__ void archCheckKernel()
{
#if __CUDA_ARCH__ >= 900
    printf("Device code compiled for CC 9.0+ (__CUDA_ARCH__ = %d)\n",
           __CUDA_ARCH__);
#elif __CUDA_ARCH__ >= 800
    printf("Device code compiled for CC 8.0+ (__CUDA_ARCH__ = %d)\n",
           __CUDA_ARCH__);
#elif __CUDA_ARCH__ >= 700
    printf("Device code compiled for CC 7.0+ (__CUDA_ARCH__ = %d)\n",
           __CUDA_ARCH__);
#elif __CUDA_ARCH__ >= 600
    printf("Device code compiled for CC 6.0+ (__CUDA_ARCH__ = %d)\n",
           __CUDA_ARCH__);
#elif __CUDA_ARCH__ >= 500
    printf("Device code compiled for CC 5.0+ (__CUDA_ARCH__ = %d)\n",
           __CUDA_ARCH__);
#elif __CUDA_ARCH__ >= 300
    printf("Device code compiled for CC 3.0+ (__CUDA_ARCH__ = %d)\n",
           __CUDA_ARCH__);
#else
    printf("Device code compiled for CC < 3.0 (__CUDA_ARCH__ = %d)\n",
           __CUDA_ARCH__);
#endif
}

/**
 * 使用不同架构条件编译路径的内核（示例：warp reduce 操作）
 * 计算能力 8.0+ 使用 __reduce_add_sync，否则使用共享内存规约
 */
__global__ void conditionalArchKernel(float *input, float *output, int N)
{
    // 计算能力 8.0+ 可以使用 warp reduce
#if __CUDA_ARCH__ >= 800
    // 简化的 warp reduce 示例（仅作演示）
    float val = (blockIdx.x * blockDim.x + threadIdx.x < N)
                    ? input[blockIdx.x * blockDim.x + threadIdx.x]
                    : 0.0f;
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(mask, val, offset);
    }
    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = val;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("Using warp shuffle reduction path (CC 8.0+).\n");
    }
#else
    // 兼容路径：使用共享内存规约
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    sdata[tid] = (blockIdx.x * blockDim.x + tid < N)
                     ? input[blockIdx.x * blockDim.x + tid]
                     : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        printf("Using shared memory reduction path (CC < 8.0).\n");
    }
#endif
}

int main()
{
    // =========================================================================
    // 第1部分：查询系统中的 CUDA 设备
    // =========================================================================
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        printf("Error: Cannot get CUDA device count: %s\n",
               cudaGetErrorString(err));
        return -1;
    }
    printf("========================================\n");
    printf("Number of CUDA-capable devices: %d\n", deviceCount);
    printf("========================================\n\n");

    // 遍历所有设备并打印详细属性
    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability:          %d.%d\n", prop.major, prop.minor);
        printf("  Multiprocessors (SMs):       %d\n",
               prop.multiProcessorCount);
        printf("  Max Threads per Block:       %d\n",
               prop.maxThreadsPerBlock);
        printf("  Max Threads per SM:          %d\n",
               prop.maxThreadsPerMultiProcessor);
        printf("  Warp Size:                   %d\n", prop.warpSize);
        printf("  Global Memory:               %.2f GB\n",
               (float)prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f));
        printf("  Shared Memory per Block:     %zu KB\n",
               prop.sharedMemPerBlock / 1024);
        printf("  Registers per Block:         %d\n", prop.regsPerBlock);
        printf("  Max Threads per SM:          %d\n",
               prop.maxThreadsPerMultiProcessor);
        printf("  Concurrent Kernels:          %s\n",
               prop.concurrentKernels ? "Yes" : "No");
        printf("  Unified Addressing:          %s\n",
               prop.unifiedAddressing ? "Yes" : "No");
        printf("  Can Map Host Memory:         %s\n",
               prop.canMapHostMemory ? "Yes" : "No");
        printf("  Integrated:                  %s\n",
               prop.integrated ? "Yes" : "No");
        printf("  Max Grid Dim:                %d x %d x %d\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("\n");
    }

    // =========================================================================
    // 第2部分：启动 archCheckKernel 展示 __CUDA_ARCH__
    // =========================================================================
    printf("========================================\n");
    printf("Launching __CUDA_ARCH__ check kernel...\n");
    printf("========================================\n");
    archCheckKernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    // =========================================================================
    // 第3部分：演示条件编译——不同架构使用不同代码路径
    // =========================================================================
    printf("\n========================================\n");
    printf("Launching conditional architecture kernel...\n");
    printf("========================================\n");

    int N = 256;
    int numBlocks = 4;
    size_t inputSize = N * numBlocks * sizeof(float);
    size_t outputSize = numBlocks * sizeof(float);

    // 分配并初始化主机内存
    float *h_input = (float *)malloc(inputSize);
    float *h_output = (float *)malloc(outputSize);
    for (int i = 0; i < N * numBlocks; i++)
    {
        h_input[i] = 1.0f;
    }

    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_output, outputSize);

    // 拷贝输入数据到设备
    cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);

    // 启动条件架构内核
    conditionalArchKernel<<<numBlocks, N>>>(d_input, d_output,
                                             N * numBlocks);
    cudaDeviceSynchronize();

    // 拷贝结果回主机
    cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);

    printf("Reduction results (one per block):\n");
    for (int i = 0; i < numBlocks; i++)
    {
        printf("  Block %d: %.2f (expected: %.2f)\n", i, h_output[i],
               (float)N);
    }

    // 释放内存
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    printf("\nDone!\n");
    return 0;
}
