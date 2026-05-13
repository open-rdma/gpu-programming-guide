// 文件: async_pipeline_demo.cu
// 编译: nvcc -arch=sm_80 async_pipeline_demo.cu -o async_pipeline_demo
// 硬件要求: NVIDIA Ampere A100 或更新 (CC 8.0+)
// 第13章 异步SIMT编程模型 - 多阶段Pipeline示例

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>

#define CUDA_CHECK(call)                                             \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

constexpr int stages = 3;         // 3阶段流水线
constexpr int threads_per_block = 256;

__global__ void pipeline_demo_kernel(
    const float *__restrict__ input,
    float *__restrict__ output,
    size_t total_elements,
    float scale)
{
    extern __shared__ float shared_buffers[];
    auto block = cooperative_groups::this_thread_block();

    // 每个阶段一个缓冲区
    float *buffer[stages];
    for (int s = 0; s < stages; s++) {
        buffer[s] = shared_buffers + s * threads_per_block;
    }

    // 创建 3 阶段 pipeline
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block, stages> pipe_state;
    auto pipe = cuda::make_pipeline(block, &pipe_state);

    size_t total_blocks = total_elements / threads_per_block;
    size_t block_id = block.group_index().x;

    // 只有第一个块执行（简化演示）
    if (block_id != 0) return;

    // 预热流水线：填充前 (stages-1) 个阶段
    for (int s = 0; s < stages - 1; s++) {
        pipe.producer_acquire();
        cuda::memcpy_async(block, buffer[s],
                           input + s * threads_per_block,
                           sizeof(float) * threads_per_block, pipe);
        pipe.producer_commit();
    }

    // 流水线稳态
    for (size_t i = 0; i < total_blocks - (stages - 1); i++) {
        int stage = i % stages;

        // 生产者：发起下一批数据的异步拷贝
        pipe.producer_acquire();
        size_t next_batch = i + stages - 1;
        if (next_batch < total_blocks) {
            cuda::memcpy_async(block, buffer[stage],
                               input + next_batch * threads_per_block,
                               sizeof(float) * threads_per_block, pipe);
        }
        pipe.producer_commit();

        // 消费者：处理当前阶段的数据
        pipe.consumer_wait();
        int tid = threadIdx.x;
        buffer[stage][tid] = buffer[stage][tid] * scale + 1.0f;
        __syncthreads();
        // 写回
        output[i * threads_per_block + tid] = buffer[stage][tid];
        pipe.consumer_release();
    }

    // 排空流水线：处理最后 (stages-1) 个阶段
    for (size_t i = total_blocks - (stages - 1); i < total_blocks; i++) {
        int stage = i % stages;
        pipe.consumer_wait();
        int tid = threadIdx.x;
        buffer[stage][tid] = buffer[stage][tid] * scale + 1.0f;
        __syncthreads();
        output[i * threads_per_block + tid] = buffer[stage][tid];
        pipe.consumer_release();
    }
}

int main() {
    const size_t N = threads_per_block * 100; // 100个批次
    const size_t bytes = N * sizeof(float);
    const float scale = 2.0f;

    // 主机内存
    float *h_input = (float *)malloc(bytes);
    float *h_output = (float *)malloc(bytes);
    for (size_t i = 0; i < N; i++) {
        h_input[i] = (float)(i % 100) / 100.0f;
    }

    // 设备内存
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // 启动 pipeline 核函数
    size_t shared_mem = stages * threads_per_block * sizeof(float);
    pipeline_demo_kernel<<<1, threads_per_block, shared_mem>>>(
        d_input, d_output, N, scale);

    CUDA_CHECK(cudaDeviceSynchronize());

    // 验证
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (size_t i = 0; i < N; i++) {
        float expected = h_input[i] * scale + 1.0f;
        if (fabsf(h_output[i] - expected) > 1e-5f) {
            printf("Mismatch at %zu: GPU %f vs CPU %f\n",
                   i, h_output[i], expected);
            correct = false;
            break;
        }
    }
    printf("Result: %s\n", correct ? "PASS" : "FAIL");

    free(h_input); free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return correct ? 0 : 1;
}
