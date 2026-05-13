#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// 简单的向量加法核函数
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// 初始化主机数据
void initData(float* data, int N, float val) {
    for (int i = 0; i < N; i++) {
        data[i] = val;
    }
}

// 验证结果
bool verifyResult(const float* data, int N, float expected) {
    for (int i = 0; i < N; i++) {
        if (fabs(data[i] - expected) > 1e-5) {
            printf("Mismatch at index %d: expected %f, got %f\n", i, expected, data[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // 参数设置
    const int N = 1 << 22;  // 约 4M 元素，16MB
    const int numStreams = 2;
    const int chunkSize = N / numStreams;
    const size_t bytesPerChunk = chunkSize * sizeof(float);

    // 线程块配置
    const int threadsPerBlock = 256;
    const int blocksPerChunk = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;

    // 分配页锁定主机内存
    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, N * sizeof(float));
    cudaMallocHost(&h_B, N * sizeof(float));
    cudaMallocHost(&h_C, N * sizeof(float));

    // 初始化数据
    initData(h_A, N, 1.0f);
    initData(h_B, N, 2.0f);

    // 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // 创建流和事件
    cudaStream_t streams[numStreams];
    cudaEvent_t startEvent, stopEvent;
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // 记录开始事件
    cudaEventRecord(startEvent, 0);

    // === 方法1: 使用多流实现数据传输与计算的流水线重叠 ===
    for (int i = 0; i < numStreams; i++) {
        int offset = i * chunkSize;
        // 异步拷贝 Host -> Device
        cudaMemcpyAsync(d_A + offset, h_A + offset, bytesPerChunk,
                        cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_B + offset, h_B + offset, bytesPerChunk,
                        cudaMemcpyHostToDevice, streams[i]);
        // 启动核函数
        vectorAdd<<<blocksPerChunk, threadsPerBlock, 0, streams[i]>>>(
            d_A + offset, d_B + offset, d_C + offset, chunkSize);
        // 异步拷贝 Device -> Host
        cudaMemcpyAsync(h_C + offset, d_C + offset, bytesPerChunk,
                        cudaMemcpyDeviceToHost, streams[i]);
    }

    // 等待所有流完成
    cudaDeviceSynchronize();

    // 记录结束事件并计算时间
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float elapsedTimePipeline;
    cudaEventElapsedTime(&elapsedTimePipeline, startEvent, stopEvent);

    // === 方法2: 使用单流（默认流）作为对照 ===
    // 先重置数据
    initData(h_C, N, 0.0f);
    cudaMemset(d_C, 0, N * sizeof(float));

    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2, 0);

    // 单流：所有操作按顺序执行
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);
    vectorAdd<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaEventRecord(stop2, 0);
    cudaEventSynchronize(stop2);
    float elapsedTimeSingle;
    cudaEventElapsedTime(&elapsedTimeSingle, start2, stop2);

    // 输出结果
    printf("Stream Pipeline VectorAdd Example (N = %d, data = %.2f MB)\n",
           N, (float)(N * 3 * sizeof(float)) / (1024 * 1024));
    printf("===============================================\n");
    printf("2-stream pipeline time: %.3f ms\n", elapsedTimePipeline);
    printf("Single stream time:     %.3f ms\n", elapsedTimeSingle);
    printf("Speedup:                %.2fx\n", elapsedTimeSingle / elapsedTimePipeline);

    // 验证结果
    if (verifyResult(h_C, N, 3.0f)) {
        printf("Result verification: PASS\n");
    } else {
        printf("Result verification: FAIL\n");
    }

    // 清理资源
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    return 0;
}
