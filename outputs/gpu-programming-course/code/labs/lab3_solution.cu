/**
 * 实验3：共享内存缓存分块 - 参考解答
 *
 * 使用共享内存缓存 A 和 B 的子块，将数据复用在快速 on-chip 内存中。
 * 外层循环沿 K 维度分块加载，内层循环在 SMEM 上执行分块内积。
 *
 * 预期性能：矩阵 4096x4096 时约 2980.3 GFLOPS/s（比实验2提升约 1.5x）
 */

/**
 * 实验3：共享内存缓存分块 - 在线测评提交版
 * 无 cuBLAS 依赖，完全对齐 lab2 代码结构，CMake/ncu 环境直接编译运行
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

// ===================== 宏定义 =====================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error [%s:%d]: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

// ===================== 辅助函数 =====================
void randomize_matrix(float *mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

void zero_init_matrix(float *mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = 0.0f;
    }
}

double calculate_gflops(int M, int N, int K, double time_ms) {
    // 2 * M * N * K 浮点操作（乘+加各算1个FLOP）
    double flops = 2.0 * M * N * K;
    return flops / (time_ms * 1e6);
}

void print_device_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return;
    }
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
}

// ===================== Kernel: 共享内存分块SGEMM =====================
template <const uint BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
    // 当前Block对应的C矩阵起始行列
    const int blockRow = blockIdx.x * BLOCKSIZE;
    const int blockCol = blockIdx.y * BLOCKSIZE;

    // Block内线程索引 1D 展开
    const int tid = threadIdx.x;
    const int tidRow = tid / BLOCKSIZE;
    const int tidCol = tid % BLOCKSIZE;

    // 线程最终输出C的全局坐标
    const int cRow = blockRow + tidRow;
    const int cCol = blockCol + tidCol;

    // 共享内存缓存A、B子块
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    float accum = 0.0f;

    // K维度分块循环，逐块加载到共享内存计算
    for (int bk = 0; bk < K; bk += BLOCKSIZE) {
        // 1. 协作加载A子块到共享内存
        int aRow = cRow;
        int aCol = bk + tidCol;
        if (aRow < M && aCol < K) {
            As[tidRow * BLOCKSIZE + tidCol] = A[aRow * K + aCol];
        } else {
            As[tidRow * BLOCKSIZE + tidCol] = 0.0f;
        }

        // 2. 协作加载B子块到共享内存
        int bRow = bk + tidRow;
        int bCol = cCol;
        if (bRow < K && bCol < N) {
            Bs[tidRow * BLOCKSIZE + tidCol] = B[bRow * N + bCol];
        } else {
            Bs[tidRow * BLOCKSIZE + tidCol] = 0.0f;
        }

        // 同步：确保整块A、B加载完成再计算
        __syncthreads();

        // 分块内积计算，复用共享内存减少全局内存访问
        for (int k = 0; k < BLOCKSIZE; k++) {
            accum += As[tidRow * BLOCKSIZE + k] * Bs[k * BLOCKSIZE + tidCol];
        }

        // 同步：防止下一轮加载覆盖未计算完成的数据
        __syncthreads();
    }

    // 写回最终结果到全局内存C
    if (cRow < M && cCol < N) {
        int cIdx = cRow * N + cCol;
        C[cIdx] = alpha * accum + beta * C[cIdx];
    }
}

// ===================== 主函数（和lab2逻辑完全对齐） =====================
int main() {
    srand((unsigned int)time(NULL));
    print_device_info();

    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int num_warmup = 5;
    const int num_iter = 10;

    printf("\n========================================\n");
    printf("实验3：共享内存缓存分块 SGEMM\n");
    printf("矩阵大小: M=%d, N=%d, K=%d\n", M, N, K);
    printf("========================================\n");

    // 分配主机内存
    float *A = (float*)malloc(M * K * sizeof(float));
    float *B = (float*)malloc(K * N * sizeof(float));
    float *C = (float*)malloc(M * N * sizeof(float));

    // 矩阵初始化
    randomize_matrix(A, M * K);
    randomize_matrix(B, K * N);
    zero_init_matrix(C, M * N);

    // 分配设备显存
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    // 主机数据拷贝到GPU
    CUDA_CHECK(cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel 启动参数配置
    const uint BLOCKSIZE = 32;
    dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);   // 单Block 1024线程，一维展开
    printf("Kernel 配置: Grid(%d, %d), Block(%d) 1D\n", gridDim.x, gridDim.y, blockDim.x);

    // 预热迭代，消除启动开销
    for (int i = 0; i < num_warmup; ++i) {
        sgemm_shared_mem_block<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 性能计时
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iter; ++i) {
        sgemm_shared_mem_block<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    float avg_ms = elapsed_ms / num_iter;

    double gflops = calculate_gflops(M, N, K, avg_ms);

    // 输出性能结果
    printf("\n========================================\n");
    printf("实验结果:\n");
    printf("  平均耗时: %.4f ms\n", avg_ms);
    printf("  计算性能: %.1f GFLOPS/s\n", gflops);
    printf("========================================\n");

    // 资源释放
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);

    return 0;
}