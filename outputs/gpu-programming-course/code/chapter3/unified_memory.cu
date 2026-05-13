/**
 * unified_memory.cu - Chapter 3: Unified Memory Basic Example
 *
 * Demonstrates CUDA Unified Memory (managed memory) where the same pointer
 * can be accessed by both host and device without explicit cudaMemcpy calls.
 *
 * Compile: nvcc unified_memory.cu -o unified_memory
 * Run:     ./unified_memory
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * @brief  Kernel to double each element of a vector in place.
 *         This kernel accesses managed memory directly — no explicit copy needed.
 */
__global__ void vecScale(double *data, int N, double factor)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        data[i] *= factor;
    }
}

/**
 * @brief  Kernel performing vector addition using managed memory.
 */
__global__ void vecAdd(const double *A, const double *B, double *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int N = 1 << 16;  // 65536 elements
    const size_t size = N * sizeof(double);

    printf("=== CUDA Unified Memory Demo ===\n");
    printf("Vector size: %d elements (%.2f MB)\n\n",
           N, (float)size / (1024.0f * 1024.0f));

    // --- Approach 1: Explicit Memory Management (for comparison) ---
    printf("--- Approach 1: Explicit Memory Management ---\n");

    // Allocate host memory
    double *h_A = (double *)malloc(size);
    double *h_B = (double *)malloc(size);
    double *h_C1 = (double *)malloc(size);

    // Initialize on host
    for (int i = 0; i < N; i++)
    {
        h_A[i] = (double)i;
        h_B[i] = (double)(N - i);
    }

    // Allocate device memory
    double *d_A, *d_B, *d_C1;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C1, size);

    // Explicit copy to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C1, N);

    // Explicit copy back to host
    cudaMemcpy(h_C1, d_C1, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("  Explicit approach: cudaMalloc + cudaMemcpy (3 steps)\n");
    printf("  Sample: C[0] = A[0] + B[0] = %.0f + %.0f = %.0f\n\n",
           h_A[0], h_B[0], h_C1[0]);

    // --- Approach 2: Unified Memory (Managed Memory) ---
    printf("--- Approach 2: Unified Memory (Managed Memory) ---\n");

    double *um_A, *um_B, *um_C2;

    // Allocate managed memory accessible from both host and device
    cudaMallocManaged((void **)&um_A, size);
    cudaMallocManaged((void **)&um_B, size);
    cudaMallocManaged((void **)&um_C2, size);

    // Initialize directly on host — no cudaMemcpy needed!
    for (int i = 0; i < N; i++)
    {
        um_A[i] = (double)i;
        um_B[i] = (double)(N - i);
    }

    // Launch kernel using the same pointers — the driver handles migration
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(um_A, um_B, um_C2, N);
    cudaDeviceSynchronize();

    // Access result directly on host — no cudaMemcpy needed!
    printf("  Unified approach: cudaMallocManaged (no explicit copies)\n");
    printf("  Sample: C[0] = A[0] + B[0] = %.0f + %.0f = %.0f\n\n",
           um_A[0], um_B[0], um_C2[0]);

    // Verify
    bool passed = true;
    for (int i = 0; i < N; i++)
    {
        if (fabs(um_C2[i] - (um_A[i] + um_B[i])) > 1e-10)
        {
            passed = false;
            break;
        }
    }
    printf("Verification: %s\n\n", passed ? "PASSED" : "FAILED");

    // --- Approach 3: In-place Operation with Unified Memory ---
    printf("--- Approach 3: In-place Scaling with Unified Memory ---\n");

    double *um_D;
    cudaMallocManaged((void **)&um_D, size);

    for (int i = 0; i < N; i++)
    {
        um_D[i] = (double)(i + 1);
    }

    printf("  Before scaling: um_D[0] = %.0f, um_D[%d] = %.0f\n",
           um_D[0], N-1, um_D[N-1]);

    vecScale<<<blocksPerGrid, threadsPerBlock>>>(um_D, N, 2.0);
    cudaDeviceSynchronize();

    // Result available immediately on host
    printf("  After scaling:  um_D[0] = %.0f, um_D[%d] = %.0f\n\n",
           um_D[0], N-1, um_D[N-1]);

    // Verify scaling
    passed = true;
    for (int i = 0; i < N; i++)
    {
        if (fabs(um_D[i] - 2.0 * (i + 1)) > 1e-10)
        {
            passed = false;
            break;
        }
    }
    printf("  Scaling verification: %s\n\n", passed ? "PASSED" : "FAILED");

    // Cleanup - both explicit and managed memory use cudaFree
    free(h_A); free(h_B); free(h_C1);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C1);
    cudaFree(um_A); cudaFree(um_B); cudaFree(um_C2); cudaFree(um_D);

    // Print device info for managed memory support
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device \"%s\" (Compute Capability %d.%d):\n",
           prop.name, prop.major, prop.minor);
    printf("  Managed Memory supported: %s\n",
           prop.managedMemory ? "Yes" : "No");
    printf("  Unified Addressing supported: %s\n",
           prop.unifiedAddressing ? "Yes" : "No");

    printf("\nDone!\n");
    return 0;
}
