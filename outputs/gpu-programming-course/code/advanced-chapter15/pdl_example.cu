/*
 * 第15章 代码示例：编程式依赖启动（PDL）
 * 硬件要求：CC 9.0+ (Hopper H100+)
 * 编译：nvcc -arch=sm_90 pdl_example.cu -o pdl_example
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024
#define BLOCK_SIZE 256

__global__ void primary_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 初始工作：初始化数据
        data[idx] = (float)idx;
    }

    // 触发secondary kernel的启动
    // 所有线程块都需要调用此函数
    cudaTriggerProgrammaticLaunchCompletion();

    // 与secondary kernel并发执行的工作
    // （此处为示例，实际中可能是更复杂的计算）
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

__global__ void secondary_kernel(float *data, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 独立工作——不依赖primary kernel的结果
    if (idx < n) {
        result[idx] = 0.0f;
    }

    // 等到primary kernel的结果对当前kernel可见
    cudaGridDependencySynchronize();

    // 依赖的工作——使用primary kernel产生的结果
    if (idx < n) {
        result[idx] = data[idx] + 1.0f;
    }
}

int main() {
    float *d_data, *d_result;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, N * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int gridDim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 配置secondary kernel的启动属性
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

    // 在同一stream中启动两个kernel
    primary_kernel<<<gridDim, BLOCK_SIZE, 0, stream>>>(d_data, N);

    // secondary kernel通过extensible launch API启动
    // 参数需要通过指针数组传递
    void *args[] = {&d_data, &d_result, &N};
    cudaLaunchKernelEx(&configSecondary, secondary_kernel);

    cudaStreamSynchronize(stream);

    // 验证结果
    float *h_result = (float*)malloc(N * sizeof(float));
    cudaMemcpy(h_result, d_result, N * sizeof(float), cudaMemcpyDeviceToHost);

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
    cudaFree(d_data);
    cudaFree(d_result);
    cudaStreamDestroy(stream);

    return errors;
}
