/**
 * matrix_add.cu - Chapter 2: Matrix Addition (2D) with Multi-Block
 *
 * Implements matrix addition C = A + B using CUDA with 2D thread blocks.
 * Each thread computes one element of the output matrix.
 * Uses dim3 data types for 2D block and grid dimensions.
 *
 * Compile: nvcc matrix_add.cu -o matrix_add
 * Run:     ./matrix_add
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/**
 * @brief  Kernel for 2D matrix addition: C[i][j] = A[i][j] + B[i][j]
 *
 * Uses 2D thread and block indices to map threads to matrix elements.
 * The global row index i and column index j are computed from
 * blockIdx, blockDim, and threadIdx in both x and y dimensions.
 *
 * @param A      Input matrix A (device memory, row-major)
 * @param B      Input matrix B (device memory, row-major)
 * @param C      Output matrix C (device memory, row-major)
 * @param width  Number of columns in each matrix
 * @param height Number of rows in each matrix
 */
__global__ void matrixAdd(const float *A, const float *B, float *C,
                          int width, int height)
{
    // Compute global 2D index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // column index
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // row index

    // Boundary check
    if (row < height && col < width)
    {
        int idx = row * width + col;  // row-major index
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * @brief  Initialize a matrix with values: M[i][j] = sin(i+j) * 10
 */
void initMatrix(float *mat, int width, int height)
{
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            mat[row * width + col] = sinf((float)(row + col)) * 10.0f;
        }
    }
}

/**
 * @brief  Verify the results on the CPU
 */
bool verifyResult(const float *A, const float *B, const float *C,
                  int width, int height)
{
    float maxError = 0.0f;
    for (int row = 0; row < height; row++)
    {
        for (int col = 0; col < width; col++)
        {
            int idx = row * width + col;
            float expected = A[idx] + B[idx];
            float error = fabsf(C[idx] - expected);
            if (error > maxError)
            {
                maxError = error;
            }
        }
    }
    printf("  Max error: %e\n", maxError);
    return maxError < 1e-5f;
}

int main()
{
    // 1. Problem configuration
    const int WIDTH  = 2048;  // 2048 columns
    const int HEIGHT = 1024;  // 1024 rows
    const int totalElements = WIDTH * HEIGHT;
    const size_t size = totalElements * sizeof(float);

    printf("=== CUDA 2D Matrix Addition ===\n");
    printf("Matrix dimensions: %d x %d (%d elements, %.2f MB per matrix)\n\n",
           HEIGHT, WIDTH, totalElements,
           (float)size / (1024.0f * 1024.0f));

    // 2. Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        printf("Error: Host memory allocation failed\n");
        return 1;
    }

    // 3. Initialize input matrices on host
    initMatrix(h_A, WIDTH, HEIGHT);
    initMatrix(h_B, WIDTH, HEIGHT);

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

    // 6. Configure 2D kernel launch parameters
    //    Use 16x16 threads per block (256 total), a common choice
    dim3 threadsPerBlock(16, 16);

    //    Calculate grid dimensions (ceil division)
    dim3 blocksPerGrid(
        (WIDTH  + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    printf("Kernel configuration:\n");
    printf("  Thread block size:  (%d, %d) = %d threads\n",
           threadsPerBlock.x, threadsPerBlock.y,
           threadsPerBlock.x * threadsPerBlock.y);
    printf("  Grid size:          (%d, %d) = %d blocks\n",
           blocksPerGrid.x, blocksPerGrid.y,
           blocksPerGrid.x * blocksPerGrid.y);
    printf("  Total threads:      %d\n\n",
           blocksPerGrid.x * blocksPerGrid.y *
           threadsPerBlock.x * threadsPerBlock.y);

    // 7. Launch the kernel
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C,
                                                   WIDTH, HEIGHT);

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
    bool passed = verifyResult(h_A, h_B, h_C, WIDTH, HEIGHT);
    if (passed)
    {
        printf("  Result: PASSED\n");
    }
    else
    {
        printf("  Result: FAILED\n");
    }

    // 11. Print a few sample values
    printf("\nSample values:\n");
    for (int row = 0; row < 3; row++)
    {
        for (int col = 0; col < 3; col++)
        {
            int idx = row * WIDTH + col;
            printf("  C[%d][%d] = A[%d][%d] + B[%d][%d] = "
                   "%.4f + %.4f = %.4f\n",
                   row, col, row, col, row, col,
                   h_A[idx], h_B[idx], h_C[idx]);
        }
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
