#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("=== Device %d: %s ===\n", i, prop.name);
        printf("  Compute Capability:      %d.%d\n", prop.major, prop.minor);

        // Check concurrency capabilities
        printf("\n  Concurrency Capabilities:\n");
        printf("    Concurrent Kernels:    %s\n",
               prop.concurrentKernels ? "YES" : "NO");
        printf("    Async Engine Count:    %d\n", prop.asyncEngineCount);

        // Explain asyncEngineCount meaning
        if (prop.asyncEngineCount >= 2) {
            printf("      -> Can overlap H2D copy with D2H copy\n");
        }
        if (prop.asyncEngineCount >= 1) {
            printf("      -> Can overlap data copy with kernel execution\n");
        } else {
            printf("      -> Cannot overlap data copy with kernel execution\n");
        }

        // Check deviceOverlap property (compatibility for older code)
        printf("    Device Overlap:        %s\n",
               prop.deviceOverlap ? "YES" : "NO");

        // Max concurrent kernels (for devices that support it)
        if (prop.concurrentKernels) {
            printf("    Max Concurrent Kernels: typically > 1\n");
            printf("      (Depends on compute capability, see CUDA Guide)\n");
        }

        // Recommendations
        printf("\n  Recommendations:\n");
        if (prop.asyncEngineCount >= 1) {
            printf("    * Use page-locked memory for overlapped transfers\n");
            printf("    * Use multiple streams to hide data transfer latency\n");
        }
        if (prop.concurrentKernels) {
            printf("    * Consider launching multiple small kernels concurrently\n");
            printf("    * Be mindful of resource constraints (registers, shared memory)\n");
        }
        if (prop.asyncEngineCount >= 2 && prop.concurrentKernels) {
            printf("    * Full concurrency: can overlap H2D + D2H + Kernel\n");
        }
        printf("\n");
    }

    return 0;
}
