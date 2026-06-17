/*
 * 第15章 代码示例：编程式依赖启动（PDL）
 * 硬件要求：CC 9.0+ (Hopper H100+)
 * 编译：nvcc -arch=sm_90 pdl_example.cu -o pdl_example
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024
#define BLOCK_SIZE 256

__global__ void primary_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = (float)idx;
        data[idx] *= 2.0f;
    }
    // 【修正1】确保所有线程完成计算，且每个线程块仅调用一次
    __syncthreads();
    if (threadIdx.x == 0) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

__global__ void secondary_kernel(float *data, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 不依赖primary的工作可提前完成
        result[idx] = 0.0f;
    }
    // 等待primary grid完全结束，数据可见
    cudaGridDependencySynchronize();
    if (idx < n) {
        // 依赖primary数据的操作
        result[idx] = data[idx] + 1.0f;
    }
}

#define CHECK_CUDA(err) do{auto e=err;if(e!=cudaSuccess){printf("CUDA ERR:%s at line %d\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

int main() {
    float *d_data, *d_result;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, N * sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    int gridDim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 开启PDL属性
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t configSecondary = {0};
    configSecondary.gridDim = dim3(gridDim, 1, 1);
    configSecondary.blockDim = dim3(BLOCK_SIZE, 1, 1);
    configSecondary.dynamicSmemBytes = 0;
    configSecondary.stream = stream;
    configSecondary.attrs = attribute;
    configSecondary.numAttrs = 1;

    // 提交primary kernel
    primary_kernel<<<gridDim, BLOCK_SIZE, 0, stream>>>(d_data, N);
    CHECK_CUDA(cudaGetLastError());

    // 【修正2】参数数组必须传递参数变量的地址，即指针的地址
    void *args[] = {&d_data, &d_result, &N};
    CHECK_CUDA(cudaLaunchKernelEx(&configSecondary, secondary_kernel, args));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 结果验证
    float *h_result = (float*)malloc(N * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = (float)i * 2.0f + 1.0f;
        if (fabsf(h_result[i] - expected) > 1e-5f) {
            errors++;
        }
    }

    if (errors == 0) {
        printf("PDL example: PASSED\n");
    } else {
        printf("PDL example: FAILED (%d errors)\n", errors);
    }

    free(h_result);
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_result));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return errors;
}