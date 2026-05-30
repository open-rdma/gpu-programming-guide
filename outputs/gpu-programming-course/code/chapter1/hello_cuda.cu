/**
 * hello_cuda.cu - Chapter 1: First CUDA Kernel
 * 演示 __global__, threadIdx, blockIdx 和 <<<...>>> 启动语法
 */
#include <cuda_runtime.h>
#include <stdio.h>

/**
 * 内核函数：每个执行此函数的线程打印一条问候
 * 使用 __global__ 声明，从CPU端调用，在GPU上执行
 */
__global__ void helloFromGPU()
{
    // threadIdx.x: 当前线程在其线程块内的索引（从0开始）
    // blockIdx.x:  当前线程块在网格中的索引（从0开始）
    printf("来自GPU的问候! 我是线程块[%d]中的线程[%d]\n",
           blockIdx.x, threadIdx.x);
}

int main()
{
    // 步骤1：从CPU端打印（这是普通的C/C++代码）
    printf("来自CPU的问候!\n\n");

    // 步骤2：配置内核启动参数
    int numBlocks = 2;         // 启动2个线程块
    int threadsPerBlock = 4;   // 每个线程块包含4个线程
    // 总共 2 * 4 = 8 个CUDA线程

    printf("启动内核：%d个线程块 x %d个线程/块 = %d个线程\n\n",
           numBlocks, threadsPerBlock,
           numBlocks * threadsPerBlock);

    // 步骤3：使用 <<<numBlocks, threadsPerBlock>>> 语法启动内核
    // 注意：内核启动是异步的——CPU不会等待GPU完成就继续执行
    helloFromGPU<<<numBlocks, threadsPerBlock>>>();

    // 步骤4：等待GPU完成所有提交的工作
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        printf("内核启动后出现CUDA错误: %s\n",
               cudaGetErrorString(error));
        return 1;
    }

    printf("\n内核执行成功完成!\n");

    return 0;
}