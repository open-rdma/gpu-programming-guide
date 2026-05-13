#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// Version 1: Kernel with severe warp divergence
// The if-else condition alternates between adjacent threads
__global__ void divergentKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Condition alternates between adjacent threads: 0, 1, 0, 1, ...
        if (idx % 2 == 0) {
            // Path A: complex computation
            float val = input[idx];
            for (int i = 0; i < 100; i++) {
                val = val * 0.99f + 0.01f;
            }
            output[idx] = val;
        } else {
            // Path B: different computation
            float val = input[idx];
            for (int i = 0; i < 100; i++) {
                val = sqrtf(val + 1.0f);
            }
            output[idx] = val;
        }
    }
}

// Version 2: Avoid warp divergence via data reorganization
// All even-index elements go to the first half, odd to the second half
__global__ void coalescedKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int halfN = N / 2;
        if (idx < halfN) {
            // First half: process original even indices
            int origIdx = idx * 2;
            float val = input[origIdx];
            for (int i = 0; i < 100; i++) {
                val = val * 0.99f + 0.01f;
            }
            output[origIdx] = val;
        } else {
            // Second half: process original odd indices
            int origIdx = (idx - halfN) * 2 + 1;
            float val = input[origIdx];
            for (int i = 0; i < 100; i++) {
                val = sqrtf(val + 1.0f);
            }
            output[origIdx] = val;
        }
    }
}

// Version 3: Baseline with no divergence (same computation for all threads)
__global__ void baselineKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = input[idx];
        for (int i = 0; i < 100; i++) {
            val = val * 0.99f + 0.01f;
        }
        output[idx] = val;
    }
}

int main() {
    const int N = 1 << 20;  // 1M elements
    const size_t size = N * sizeof(float);

    float *h_input, *h_output;
    float *d_input, *d_output;

    // Allocate host memory
    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);

    // Initialize
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(i % 100) / 100.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float timeBaseline, timeDivergent, timeCoalesced;

    // Warmup
    baselineKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    // === Baseline kernel (no divergence) ===
    cudaEventRecord(start, 0);
    baselineKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeBaseline, start, stop);

    // === Divergent kernel ===
    cudaEventRecord(start, 0);
    divergentKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeDivergent, start, stop);

    // === Coalesced kernel (avoids intra-warp divergence) ===
    cudaEventRecord(start, 0);
    coalescedKernel<<<blocks, threadsPerBlock>>>(d_input, d_output, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&timeCoalesced, start, stop);

    // Output results
    printf("Warp Divergence Impact Analysis (N = %d)\n", N);
    printf("=========================================\n");
    printf("Baseline (no branch):        %.3f ms\n", timeBaseline);
    printf("Divergent kernel:            %.3f ms  (%.2fx slower vs baseline)\n",
           timeDivergent, timeDivergent / timeBaseline);
    printf("Coalesced kernel:            %.3f ms  (%.2fx vs baseline)\n",
           timeCoalesced, timeCoalesced / timeBaseline);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
