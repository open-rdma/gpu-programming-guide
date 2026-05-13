#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    printf("Found %d CUDA device(s)\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("=== Device %d: %s ===\n", i, prop.name);
        printf("  Compute Capability:        %d.%d\n", prop.major, prop.minor);
        printf("  Multiprocessors (SM):      %d\n", prop.multiProcessorCount);
        printf("  GPU Clock Rate:            %.2f GHz\n",
               prop.clockRate * 1e-6f);
        printf("  Global Memory:             %.2f GB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared Memory per Block:   %zu KB\n",
               prop.sharedMemPerBlock / 1024);
        printf("  Registers per Block:       %d\n", prop.regsPerBlock);
        printf("  Max Threads per Block:     %d\n", prop.maxThreadsPerBlock);
        printf("  Warp Size:                 %d\n", prop.warpSize);
        printf("  Concurrent Kernels:        %s\n",
               prop.concurrentKernels ? "Yes" : "No");
        printf("  Async Engine Count:        %d\n", prop.asyncEngineCount);
        printf("  Unified Addressing:        %s\n",
               prop.unifiedAddressing ? "Yes" : "No");
        printf("  Can Map Host Memory:       %s\n",
               prop.canMapHostMemory ? "Yes" : "No");
        printf("  L2 Cache Size:             %d KB\n", prop.l2CacheSize / 1024);
        printf("\n");
    }

    return 0;
}
