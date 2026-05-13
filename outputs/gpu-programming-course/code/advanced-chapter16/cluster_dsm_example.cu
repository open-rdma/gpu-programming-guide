/*
 * 第16章 代码示例：Cluster Group + 分布式共享内存
 * 硬件要求：CC 9.0+ (Hopper H100+)
 * 编译：nvcc -arch=sm_90 -rdc=true cluster_dsm_example.cu -o cluster_dsm_example
 *
 * 演示：集群内多个线程块使用分布式共享内存进行协同归约
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

#define CLUSTER_X 2
#define CLUSTER_Y 1
#define CLUSTER_Z 1
#define BLOCK_SIZE 256

/*
 * 使用集群维度属性在编译期指定集群大小
 * __cluster_dims__(X, Y, Z) 表示集群包含 X*Y*Z 个线程块
 */
__global__ void __cluster_dims__(CLUSTER_X, CLUSTER_Y, CLUSTER_Z)
clusterReduceKernel(const float *input, float *output, int n) {
    // 获取集群组
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int numBlocks = cluster.num_blocks();
    unsigned int blockRank = cluster.block_rank();

    // 每个块的共享内存
    __shared__ float s_partialSum[BLOCK_SIZE];
    __shared__ float s_blockSum;

    // 阶段1：每个线程加载数据并做块内reduce
    int tid = threadIdx.x;
    float localSum = 0.0f;

    // 数据分配：集群中所有线程共同处理数据
    unsigned int threadsTotal = numBlocks * blockDim.x;
    unsigned int globalTid = blockRank * blockDim.x + tid;

    for (int i = globalTid; i < n; i += threadsTotal) {
        localSum += input[i];
    }

    // 块内归约（使用共享内存）
    s_partialSum[tid] = localSum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_partialSum[tid] += s_partialSum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        s_blockSum = s_partialSum[0];
    }
    __syncthreads();

    // 阶段2：使用DSM进行跨块归约
    cluster.sync();

    // 使用二叉树归约合并所有块的部分和
    for (int stride = 1; stride < numBlocks; stride <<= 1) {
        float otherBlockSum = 0.0f;
        if (tid == 0) {
            int otherBlock = blockRank ^ stride;
            if (otherBlock < numBlocks) {
                // 使用map_shared_rank访问远程块的共享内存
                float *remoteBlockSum =
                    cluster.map_shared_rank(&s_blockSum, otherBlock);
                otherBlockSum = *remoteBlockSum;
            }
        }
        // 广播otherBlockSum到块内所有线程
        __syncthreads();
        s_partialSum[tid] = otherBlockSum;
        __syncthreads();

        if (tid == 0) {
            s_blockSum += s_partialSum[0];
        }
        __syncthreads();

        cluster.sync();
    }

    // 块0写入最终结果
    if (blockRank == 0 && tid == 0) {
        *output = s_blockSum;
    }
}

int main() {
    const int N = 1048576;  // 1M elements
    float *h_input = (float*)malloc(N * sizeof(float));
    float h_expected = 0.0f;

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
        h_expected += h_input[i];
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // 使用集群启动
    // grid必须包含恰好一个集群（因为__cluster_dims__编译期指定了集群大小）
    dim3 clusterDim(CLUSTER_X, CLUSTER_Y, CLUSTER_Z);
    dim3 blockDim(BLOCK_SIZE);

    cudaLaunchConfig_t config = {0};
    config.gridDim = clusterDim;       // 网格维度 = 集群维度
    config.blockDim = blockDim;
    config.dynamicSmemBytes = 0;
    config.stream = 0;

    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CLUSTER_X;
    attr[0].val.clusterDim.y = CLUSTER_Y;
    attr[0].val.clusterDim.z = CLUSTER_Z;
    config.attrs = attr;
    config.numAttrs = 1;

    void *args[] = {&d_input, &d_output, &N};
    cudaLaunchKernelEx(&config, clusterReduceKernel);

    float h_output;
    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Cluster DSM Reduce:\n");
    printf("  Expected sum: %.1f\n", h_expected);
    printf("  Computed sum: %.1f\n", h_output);
    printf("  Match: %s\n",
           fabsf(h_output - h_expected) < 1e-3f ? "YES" : "NO");

    free(h_input);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
