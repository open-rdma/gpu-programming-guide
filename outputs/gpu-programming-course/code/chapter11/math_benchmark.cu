/**
 * Chapter 11 - Experiment 11-3: Fast Math Functions Benchmark
 *
 * Compares throughput of standard math functions vs CUDA intrinsics.
 * Compile: nvcc -arch=sm_86 -O3 math_benchmark.cu -o math_benchmark
 * Run: ./math_benchmark
 *
 * Note: Using --use_fast_math flag enables more aggressive compiler
 * optimizations. You can also compare with/without this flag.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(call) {                                            \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

// Test 1: Standard division vs __fdividef
__global__ void standardDiv(const float * __restrict__ a,
                             const float * __restrict__ b,
                             float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] / b[idx];
    }
}

__global__ void fastDiv(const float * __restrict__ a,
                         const float * __restrict__ b,
                         float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = __fdividef(a[idx], b[idx]);
    }
}

// Test 2: Standard reciprocal sqrt vs rsqrtf
__global__ void standardRsqrt(const float * __restrict__ a,
                               float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = 1.0f / sqrtf(a[idx]);
    }
}

__global__ void fastRsqrt(const float * __restrict__ a,
                           float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = rsqrtf(a[idx]);
    }
}

// Test 3: Standard sin/cos vs __sinf/__cosf (small arguments - fast path)
__global__ void standardSinCos(const float * __restrict__ a,
                                float * __restrict__ s,
                                float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        s[idx] = sinf(a[idx]);
        c[idx] = cosf(a[idx]);
    }
}

__global__ void fastSinCos(const float * __restrict__ a,
                            float * __restrict__ s,
                            float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        s[idx] = __sinf(a[idx]);
        c[idx] = __cosf(a[idx]);
    }
}

// Test 4: Sin/Cos with large arguments (forces slow path)
__global__ void standardSinCosLarge(const float * __restrict__ a,
                                     float * __restrict__ s,
                                     float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // Large value ~1e6 forces slow path argument reduction
        s[idx] = sinf(a[idx] + 1000000.0f);
        c[idx] = cosf(a[idx] + 1000000.0f);
    }
}

__global__ void fastSinCosLarge(const float * __restrict__ a,
                                 float * __restrict__ s,
                                 float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        s[idx] = __sinf(a[idx] + 1000000.0f);
        c[idx] = __cosf(a[idx] + 1000000.0f);
    }
}

// Test 5: Standard sqrt vs __fsqrt_rn
__global__ void standardSqrt(const float * __restrict__ a,
                              float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = sqrtf(a[idx]);
    }
}

__global__ void fastLogExp(const float * __restrict__ a,
                            float * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = __log2f(a[idx]) + exp2f(a[idx] * 0.0001f);
    }
}

// Test 6: Integer division vs bit shift
__global__ void intDivision(const int * __restrict__ a,
                             int * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] / 32;
    }
}

__global__ void intBitShift(const int * __restrict__ a,
                             int * __restrict__ c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] >> 5;  // Same as a[idx] / 32
    }
}

// ===== Benchmark Utility =====

template<typename KernelFunc>
float benchmark1out(KernelFunc kernel, int n, int gridSize, int blockSize,
                    int iterations, float *d_a, float *d_c) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    kernel<<<gridSize, blockSize>>>(d_a, d_c, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < iterations; i++) {
        kernel<<<gridSize, blockSize>>>(d_a, d_c, n);
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / iterations;
}

template<typename KernelFunc>
float benchmark2out(KernelFunc kernel, int n, int gridSize, int blockSize,
                    int iterations, float *d_a, float *d_s, float *d_c) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    kernel<<<gridSize, blockSize>>>(d_a, d_s, d_c, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < iterations; i++) {
        kernel<<<gridSize, blockSize>>>(d_a, d_s, d_c, n);
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return ms / iterations;
}

int main() {
    cudaDeviceProp prop;
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("=== Math Function Benchmark ===\n");
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    const int N = 8 * 1024 * 1024;  // 8M elements
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    const int iterations = 100;
    const size_t bytes = N * sizeof(float);

    // Allocate device memory
    float *d_a, *d_b, *d_c, *d_s;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));
    CHECK_CUDA(cudaMalloc(&d_s, bytes));

    int *d_ia, *d_ic, iBytes = N * sizeof(int);
    CHECK_CUDA(cudaMalloc(&d_ia, iBytes));
    CHECK_CUDA(cudaMalloc(&d_ic, iBytes));

    // Initialize data on host
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    int *h_ia = (int*)malloc(iBytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(i % 100) + 1.0f;       // 1.0 - 100.0
        h_b[i] = (float)(i % 50) + 1.0f;        // 1.0 - 50.0
        h_ia[i] = i * 64;                        // For integer tests
    }

    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ia, h_ia, iBytes, cudaMemcpyHostToDevice));

    printf("%-40s %10s %10s\n", "Function", "Time(ms)", "Speedup");
    printf("-----------------------------------------------------------------\n");

    float std_ms, fast_ms;

    // === Test 1: Division ===
    std_ms = benchmark1out(standardDiv, N, gridSize, blockSize, iterations, d_a, d_b, d_c);
    fast_ms = benchmark1out(fastDiv, N, gridSize, blockSize, iterations, d_a, d_b, d_c);
    printf("%-40s %10.4f %10s\n", "Standard / (division)", std_ms, "baseline");
    printf("%-40s %10.4f %10.2fx\n", "__fdividef()", fast_ms, std_ms / fast_ms);

    // === Test 2: Reciprocal Sqrt ===
    std_ms = benchmark1out(standardRsqrt, N, gridSize, blockSize, iterations, d_a, d_c);
    fast_ms = benchmark1out(fastRsqrt, N, gridSize, blockSize, iterations, d_a, d_c);
    printf("%-40s %10.4f %10s\n", "1.0f/sqrtf()", std_ms, "baseline");
    printf("%-40s %10.4f %10.2fx\n", "rsqrtf()", fast_ms, std_ms / fast_ms);

    // === Test 3: Sin/Cos small arguments ===
    std_ms = benchmark2out(standardSinCos, N, gridSize, blockSize, iterations, d_a, d_s, d_c);
    fast_ms = benchmark2out(fastSinCos, N, gridSize, blockSize, iterations, d_a, d_s, d_c);
    printf("%-40s %10.4f %10s\n", "sinf()+cosf() (small args)", std_ms, "baseline");
    printf("%-40s %10.4f %10.2fx\n", "__sinf()+__cosf() (small)", fast_ms, std_ms / fast_ms);

    // === Test 4: Sin/Cos large arguments ===
    std_ms = benchmark2out(standardSinCosLarge, N, gridSize, blockSize, iterations, d_a, d_s, d_c);
    fast_ms = benchmark2out(fastSinCosLarge, N, gridSize, blockSize, iterations, d_a, d_s, d_c);
    printf("%-40s %10.4f %10s\n", "sinf()+cosf() (large args)", std_ms, "baseline");
    printf("%-40s %10.4f %10.2fx\n", "__sinf()+__cosf() (large)", fast_ms, std_ms / fast_ms);

    // === Test 5: Log/Exp intrinsics ===
    std_ms = benchmark1out(standardSqrt, N, gridSize, blockSize, iterations, d_a, d_c);
    fast_ms = benchmark1out(fastLogExp, N, gridSize, blockSize, iterations, d_a, d_c);
    printf("%-40s %10.4f %10s\n", "sqrtf()", std_ms, "baseline");
    printf("%-40s %10.4f %10.2fx\n", "__log2f()+exp2f()", fast_ms, std_ms / fast_ms);

    // === Test 6: Integer division vs bit shift ===
    {
        // Need separate benchmark for integer kernels
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        intDivision<<<gridSize, blockSize>>>(d_ia, d_ic, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start, 0));
        for (int i = 0; i < iterations; i++) {
            intDivision<<<gridSize, blockSize>>>(d_ia, d_ic, N);
        }
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&std_ms, start, stop));
        std_ms /= iterations;

        intBitShift<<<gridSize, blockSize>>>(d_ia, d_ic, N);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaEventRecord(start, 0));
        for (int i = 0; i < iterations; i++) {
            intBitShift<<<gridSize, blockSize>>>(d_ia, d_ic, N);
        }
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&fast_ms, start, stop));
        fast_ms /= iterations;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));

        printf("%-40s %10.4f %10s\n", "Integer: val/32", std_ms, "baseline");
        printf("%-40s %10.4f %10.2fx\n", "Integer: val>>5", fast_ms, std_ms / fast_ms);
    }

    // Cleanup
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_s));
    CHECK_CUDA(cudaFree(d_ia));
    CHECK_CUDA(cudaFree(d_ic));
    free(h_a);
    free(h_b);
    free(h_ia);

    printf("\nKey insights:\n");
    printf("1. __fdividef() is faster than standard / operator\n");
    printf("2. rsqrtf() is significantly faster than 1.0f/sqrtf()\n");
    printf("3. __sinf/__cosf are fast for small arguments (< ~105615 for float)\n");
    printf("4. Large arguments force slow-path trig evaluation (much slower!)\n");
    printf("5. Integer bit shift >> is much faster than division (when divisor is power of 2)\n");
    printf("6. Use --use_fast_math flag for automatic intrinsic substitution\n");

    return 0;
}
