/**
 * device_query.cu - Chapter 1: GPU Device Query Example
 *
 * This program demonstrates how to query CUDA device properties.
 * It prints detailed information about all CUDA-capable GPUs in the system.
 *
 * Compile: nvcc device_query.cu -o device_query
 * Run:     ./device_query
 */

#include <cuda_runtime.h>
#include <stdio.h>

int main()
{
    int deviceCount;
    cudaError_t error;

    // 1. Get the number of CUDA-capable devices
    error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess)
    {
        printf("Error getting device count: %s\n", cudaGetErrorString(error));
        return 1;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA-capable devices found. Exiting.\n");
        return 0;
    }

    printf("Found %d CUDA-capable device(s)\n\n", deviceCount);

    // 2. Iterate through all devices and print their properties
    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;

        error = cudaGetDeviceProperties(&deviceProp, dev);
        if (error != cudaSuccess)
        {
            printf("Error getting properties for device %d: %s\n",
                   dev, cudaGetErrorString(error));
            continue;
        }

        printf("=== Device %d: %s ===\n", dev, deviceProp.name);
        printf("  Compute Capability:          %d.%d\n",
               deviceProp.major, deviceProp.minor);
        printf("  Total Global Memory:         %.2f GB\n",
               deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Number of SMs:               %d\n",
               deviceProp.multiProcessorCount);
        printf("  Max Threads per Block:       %d\n",
               deviceProp.maxThreadsPerBlock);
        printf("  Max Threads Dimension:       (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max Grid Size:               (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Shared Memory per Block:     %zu KB\n",
               deviceProp.sharedMemPerBlock / 1024);
        printf("  Registers per Block:         %d\n",
               deviceProp.regsPerBlock);
        printf("  Warp Size:                   %d\n",
               deviceProp.warpSize);
        printf("  Max Threads per SM:          %d\n",
               deviceProp.maxThreadsPerMultiProcessor);
        printf("  Clock Rate:                  %.2f GHz\n",
               deviceProp.clockRate / 1e6);
        printf("  Memory Bus Width:            %d bits\n",
               deviceProp.memoryBusWidth);
        printf("  Memory Clock Rate:           %.2f GHz\n",
               deviceProp.memoryClockRate / 1e6);
        printf("  ECC Enabled:                 %s\n",
               deviceProp.ECCEnabled ? "Yes" : "No");
        printf("  Unified Addressing:          %s\n",
               deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  Managed Memory:              %s\n",
               deviceProp.managedMemory ? "Yes" : "No");
        printf("  Concurrent Kernels:          %s\n",
               deviceProp.concurrentKernels ? "Yes" : "No");
        printf("\n");
    }

    return 0;
}
