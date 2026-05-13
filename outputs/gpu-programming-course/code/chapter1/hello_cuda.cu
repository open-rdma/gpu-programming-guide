/**
 * hello_cuda.cu - Chapter 1: First CUDA Program
 *
 * A minimal CUDA kernel that each thread prints its greeting.
 * Demonstrates: __global__, threadIdx, and the <<<...>>> launch syntax.
 *
 * Compile: nvcc hello_cuda.cu -o hello_cuda
 * Run:     ./hello_cuda
 */

#include <cuda_runtime.h>
#include <stdio.h>

/**
 * @brief  A simple kernel where each thread prints a greeting.
 *
 * Each of the N threads that execute this kernel prints one line.
 * The built-in variable threadIdx.x identifies the thread within the block.
 */
__global__ void helloFromGPU()
{
    // Each thread prints a greeting with its unique thread index
    printf("Hello World from GPU! I am thread [%d] in block [%d]\n",
           threadIdx.x, blockIdx.x);
}

int main()
{
    // 1. Print a greeting from the CPU (host)
    printf("Hello World from CPU!\n\n");

    // 2. Launch the kernel with 2 blocks, each with 4 threads
    //    Syntax: kernel<<<numBlocks, threadsPerBlock>>>(args)
    int numBlocks = 2;
    int threadsPerBlock = 4;

    printf("Launching kernel with %d block(s) and %d thread(s) per block\n\n",
           numBlocks, threadsPerBlock);

    helloFromGPU<<<numBlocks, threadsPerBlock>>>();

    // 3. Wait for the GPU to finish before accessing the results
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        printf("CUDA error after kernel launch: %s\n",
               cudaGetErrorString(error));
        return 1;
    }

    printf("\nKernel execution completed successfully!\n");

    return 0;
}
