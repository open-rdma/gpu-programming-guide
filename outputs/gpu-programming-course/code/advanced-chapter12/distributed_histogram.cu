// 文件: distributed_histogram.cu
// 编译: nvcc -arch=sm_90 distributed_histogram.cu -o distributed_histogram
// 硬件要求: NVIDIA Hopper H100 或更新 (CC 9.0+)
// 第12章 Thread Block Clusters与分布式共享内存 - 分布式直方图示例

#include <stdio.h>
#include <stdlib.h>
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

// 分布式共享内存直方图核函数
__global__ void clusterHist_kernel(int *bins, const int nbins,
                                   const int bins_per_block,
                                   const int *__restrict__ input,
                                   size_t array_size)
{
    extern __shared__ int smem[];
    namespace cg = cooperative_groups;
    int tid = cg::this_grid().thread_rank();

    cg::cluster_group cluster = cg::this_cluster();
    unsigned int clusterBlockRank = cluster.block_rank();

    // 初始化本地共享内存直方图
    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
    {
        smem[i] = 0;
    }

    // 确保所有块的共享内存都已初始化
    cluster.sync();

    // 分布式直方图计算
    for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
    {
        int ldata = input[i];

        int binid = ldata;
        if (ldata < 0)        binid = 0;
        if (ldata >= nbins)   binid = nbins - 1;

        int dst_block_rank = binid / bins_per_block;
        int dst_offset     = binid % bins_per_block;

        int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);
        atomicAdd(dst_smem + dst_offset, 1);
    }

    // 确保所有分布式操作完成
    cluster.sync();

    // 归约到全局内存
    int *lbins = bins + cluster.block_rank() * bins_per_block;
    for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
    {
        if (smem[i] > 0) {
            atomicAdd(&lbins[i], smem[i]);
        }
    }
}

int main()
{
    // 参数设置
    const size_t array_size = 1024 * 1024;  // 1M 个元素
    const int nbins = 512;                   // 512 个 bin
    const int threads_per_block = 256;
    const int cluster_size = 4;              // 簇大小：4 个线程块
    const int bins_per_block = nbins / cluster_size; // 每个块 128 个 bin

    // 分配并初始化输入数据（随机整数值 0..nbins-1）
    int *h_input = (int *)malloc(array_size * sizeof(int));
    for (size_t i = 0; i < array_size; i++) {
        h_input[i] = rand() % nbins;
    }

    // 分配设备内存
    int *d_input, *d_bins;
    CUDA_CHECK(cudaMalloc(&d_input, array_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bins, nbins * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, array_size * sizeof(int),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_bins, 0, nbins * sizeof(int)));

    // 使用 cudaLaunchKernelEx 启动簇核函数
    {
        cudaLaunchConfig_t config = {0};
        config.gridDim = dim3(array_size / threads_per_block);
        config.blockDim = dim3(threads_per_block);
        config.dynamicSmemBytes = bins_per_block * sizeof(int);

        CUDA_CHECK(cudaFuncSetAttribute(
            (void *)clusterHist_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            config.dynamicSmemBytes));

        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = cluster_size;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;

        config.numAttrs = 1;
        config.attrs = attr;

        CUDA_CHECK(cudaLaunchKernelEx(&config, clusterHist_kernel,
                                       d_bins, nbins, bins_per_block,
                                       d_input, array_size));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // 复制结果回主机
    int *h_bins = (int *)malloc(nbins * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_bins, d_bins, nbins * sizeof(int),
                           cudaMemcpyDeviceToHost));

    // 验证结果（与 CPU 串行版对比）
    int *cpu_bins = (int *)calloc(nbins, sizeof(int));
    for (size_t i = 0; i < array_size; i++) {
        cpu_bins[h_input[i]]++;
    }

    bool correct = true;
    for (int i = 0; i < nbins; i++) {
        if (h_bins[i] != cpu_bins[i]) {
            printf("Bin %d: GPU %d vs CPU %d\n", i, h_bins[i], cpu_bins[i]);
            correct = false;
            break;
        }
    }
    printf("Result: %s\n", correct ? "PASS" : "FAIL");

    // 清理
    free(h_input); free(h_bins); free(cpu_bins);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_bins));

    return correct ? 0 : 1;
}
