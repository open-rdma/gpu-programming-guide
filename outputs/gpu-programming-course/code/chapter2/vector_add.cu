/**
 * vector_add.cu - Chapter 2: Vector Addition (1D) with Multi-Block
 *
 * Implements vector addition C = A + B using CUDA.
 * Each thread computes one element of the output vector.
 * Multiple blocks are used to handle vectors of any size.
 *
 * Compile: nvcc vector_add.cu -o vector_add
 * Run:     ./vector_add
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * @brief  Kernel for vector addition: C[i] = A[i] + B[i]
 *
 * Each thread computes one element. The global index is computed from
 * blockIdx.x, blockDim.x, and threadIdx.x.
 *
 * @param A  Input vector A (device memory)
 * @param B  Input vector B (device memory)
 * @param C  Output vector C (device memory)
 * @param N  Total number of elements
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    // Compute global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: ensure the thread does not access out of bounds
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * @brief  Initialize a vector with values: A[i] = sin(i) * scale
 */
void initVector(float *vec, int N)
{
    for (int i = 0; i < N; i++)
    {
        vec[i] = sinf((float)i) * 100.0f;
    }
}

/**
 * @brief  Verify the results on the CPU for correctness
 */
bool verifyResult(const float *A, const float *B, const float *C, int N)
{
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        float expected = A[i] + B[i];
        float error = fabsf(C[i] - expected);
        if (error > maxError)
        {
            maxError = error;
        }
    }
    printf("  Max error: %e\n", maxError);
    return maxError < 1e-5f;
}

int main()
{
    // 1. Problem configuration
    const int N = 1 << 20;  // 1M elements (2^20 = 1,048,576)
    const size_t size = N * sizeof(float);

    printf("=== CUDA Vector Addition ===\n");
    printf("Vector size: %d elements (%.2f MB per vector)\n\n",
           N, (float)size / (1024.0f * 1024.0f));

    // 2. Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        printf("Error: Host memory allocation failed\n");
        return 1;
    }

    // 3. Initialize input vectors on host
    initVector(h_A, N);
    initVector(h_B, N);

    // 4. Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaError_t error;

    error = cudaMalloc((void **)&d_A, size);
    if (error != cudaSuccess)
    {
        printf("Error allocating d_A: %s\n", cudaGetErrorString(error));
        return 1;
    }
    error = cudaMalloc((void **)&d_B, size);
    if (error != cudaSuccess)
    {
        printf("Error allocating d_B: %s\n", cudaGetErrorString(error));
        return 1;
    }
    error = cudaMalloc((void **)&d_C, size);
    if (error != cudaSuccess)
    {
        printf("Error allocating d_C: %s\n", cudaGetErrorString(error));
        return 1;
    }

    // 5. Copy input data from host to device
    error = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("Error copying h_A to d_A: %s\n", cudaGetErrorString(error));
        return 1;
    }
    error = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("Error copying h_B to d_B: %s\n", cudaGetErrorString(error));
        return 1;
    }

    // 6. Configure kernel launch parameters
    //    Use 256 threads per block (a common choice)
    int threadsPerBlock = 256;
    //    Calculate number of blocks needed (ceil division)
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Kernel configuration:\n");
    printf("  Threads per block: %d\n", threadsPerBlock);
    printf("  Blocks per grid:   %d\n", blocksPerGrid);
    printf("  Total threads:     %d\n\n", blocksPerGrid * threadsPerBlock);

    // 7. Launch the kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 8. Check for kernel launch errors
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("Kernel launch error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    // 9. Wait for GPU to finish and copy result back to host
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess)
    {
        printf("Synchronization error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    error = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        printf("Error copying d_C to h_C: %s\n", cudaGetErrorString(error));
        return 1;
    }

    // 10. Verify results
    printf("Verification:\n");
    bool passed = verifyResult(h_A, h_B, h_C, N);
    if (passed)
    {
        printf("  Result: PASSED\n");
    }
    else
    {
        printf("  Result: FAILED\n");
    }

    // 11. Print a few sample values
    printf("\nSample values (first 5 elements):\n");
    for (int i = 0; i < 5; i++)
    {
        printf("  C[%d] = A[%d] + B[%d] = %.4f + %.4f = %.4f\n",
               i, i, i, h_A[i], h_B[i], h_C[i]);
    }

    // 12. Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("\nDone!\n");
    return 0;
}
