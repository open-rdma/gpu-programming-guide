// 文件: tma_2d_demo.cu
// 演示：2D TMA 分块加载、处理和写回
// 编译: nvcc -arch=sm_90a -lcuda tma_2d_demo.cu -o tma_2d_demo
// 硬件要求: NVIDIA Hopper H100 (CC 9.0+)，推荐 sm_90a 以启用完整 TMA 特性
//
// 重要概念说明：
// 1. Global → Shared 使用 mbarrier 完成机制，TMA 指令**自动**更新 barrier 的事务计数，
//    所有线程调用普通 bar.arrive()
// 2. Shared → Global 使用 bulk async-group 完成机制，只有发起线程需要 commit 和 wait
// 3. __grid_constant__ 提示编译器该参数在整个 grid 执行期间不变，放入常量缓存加速访问

#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>
#include <cuda/experimental/__pipeline>

// ========== CUDA 错误检查宏 ==========
#define CUDA_CHECK(call)                                             \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// ========== 常量定义 ==========
constexpr int GMEM_WIDTH  = 256;              // 全局内存宽度（元素数）
constexpr int GMEM_HEIGHT = 256;              // 全局内存高度（元素数）
constexpr int SMEM_WIDTH  = 16;               // 共享内存 tile 宽度（元素数）
constexpr int SMEM_HEIGHT = 16;               // 共享内存 tile 高度（元素数）
constexpr int TILE_SIZE   = SMEM_WIDTH * SMEM_HEIGHT;  // 每个 tile 的元素总数

// 编译时检查对齐要求
static_assert(GMEM_WIDTH % SMEM_WIDTH == 0, "GMEM_WIDTH must be multiple of SMEM_WIDTH");
static_assert(GMEM_HEIGHT % SMEM_HEIGHT == 0, "GMEM_HEIGHT must be multiple of SMEM_HEIGHT");

// ========== 主机端：获取 cuTensorMapEncodeTiled 函数指针 ==========
// 注意：cuTensorMapEncodeTiled 是 Driver API 函数，需要通过 cudaGetDriverEntryPointByVersion 获取

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
    cudaError_t err = cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &ptr, 12000, cudaEnableDefault, &status);
    if (err != cudaSuccess || status != cudaDriverEntryPointSuccess) {
        fprintf(stderr, "Failed to get cuTensorMapEncodeTiled: %s\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
}

// ========== 主机端：创建 2D Tensor Map ==========
// Tensor Map 描述了全局内存中张量的布局，供 TMA 硬件使用
// 注意：维度顺序是 fastest-changing dimension 在索引 0

CUtensorMap create_2d_tensor_map(int* d_data) {
    CUtensorMap tmap{};
    constexpr uint32_t rank = 2;

    // globalDim: 全局张量各维度尺寸（元素数）
    uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};

    // globalStrides: 全局张量各维度步长（字节）
    // rank-1 个步长：最快维度（dim0）无 stride，dim1 的 stride = width * sizeof(int)
    uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};

    // boxDim: 每次 TMA 拷贝的 tile 大小（元素数）
    uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};

    // elementStride: 元素步长（以 sizeof(datatype) 为单位），通常设为 1
    uint32_t elem_stride[rank] = {1, 1};

    auto encode = get_cuTensorMapEncodeTiled();
    CUresult res = encode(
        &tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,  // 数据类型
        rank,                                                 // 张量维度数
        d_data,                                               // 全局内存基地址
        size,                                                 // 各维度尺寸
        stride,                                               // 各维度步长
        box_size,                                             // tile 尺寸
        elem_stride,                                          // 元素步长
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,       // 本章先不开启 Swizzle
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuTensorMapEncodeTiled failed with code %d\n", res);
        exit(EXIT_FAILURE);
    }
    return tmap;
}

// ========== 设备端：核函数 ==========
// 使用 __grid_constant__ 提示编译器将 tensor_map 放入常量缓存以提高访问效率
// 每个线程块处理一个 (x, y) 坐标对应的 tile

__global__ void tma_copy_kernel(const __grid_constant__ CUtensorMap tensor_map,
                                int* __restrict__ output,
                                int tiles_per_row) {
    // 多维 TMA 操作的共享内存目标缓冲区需要 128 字节对齐
    __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

    // ---------- 1. 初始化 mbarrier ----------
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    // 只有线程 0 初始化 barrier
    // barrier 以 blockDim.x（块内所有线程）为参与线程数
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        // fence_proxy_async_shared_cta 使 barrier 对 async proxy 可见
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // ---------- 2. TMA 读取：Global → Shared ----------
    // 计算当前 block 要处理的 tile 在全局数组中的坐标
    int global_x = (blockIdx.x % tiles_per_row) * SMEM_WIDTH;   // 最内层维度（column）
    int global_y = (blockIdx.x / tiles_per_row) * SMEM_HEIGHT;  // 外层维度（row）

    barrier::arrival_token token;
    if (threadIdx.x == 0) {
        // 发起 2D TMA 拷贝
        // 重要：硬件指令 cp_async_bulk_tensor 会**自动**向 barrier 提交预期的事务计数
        // 因此这里不需要（也不应该）调用 barrier_arrive_tx
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &smem_buffer, &tensor_map, global_x, global_y, bar);
    }
    // 所有线程到达 barrier（包括发起线程）
    // barrier 会自动等待 TMA 传输完成后才让 wait 返回
    token = bar.arrive();
    bar.wait(std::move(token));
    // 此时数据已在共享内存中，所有线程均可安全访问

    // ---------- 3. 计算：对数据加 1 ----------
    // 使用块内所有线程并行处理 tile 数据
    for (int idx = threadIdx.x; idx < TILE_SIZE; idx += blockDim.x) {
        int row = idx / SMEM_WIDTH;
        int col = idx % SMEM_WIDTH;
        smem_buffer[row][col] += 1;
    }

    // ---------- 4. TMA 写入：Shared → Global ----------
    // 写回前需要确保共享内存写入对 TMA 引擎可见
    cde::fence_proxy_async_shared_cta();
    __syncthreads();

    if (threadIdx.x == 0) {
        // 发起 2D TMA 写回
        cde::cp_async_bulk_tensor_2d_shared_to_global(
            &tensor_map, global_x, global_y, &smem_buffer);

        // Shared → Global 方向使用 bulk async-group 完成机制
        // 将当前操作提交到 group
        cde::cp_async_bulk_commit_group();
        // 等待 group 完成（0 表示等待所有之前的操作）
        cde::cp_async_bulk_wait_group_read<0>();
    }

    // ---------- 5. 清理 ----------
    if (threadIdx.x == 0) {
        (&bar)->~barrier();   // 手动销毁 barrier，释放共享内存
    }
}

// ========== 主机端：验证函数 ==========
bool verify_result(const int* h_output, size_t size) {
    for (size_t i = 0; i < size; i++) {
        // 输入是 1，经过加 1 后应为 2
        if (h_output[i] != 2) {
            printf("Verification failed at index %zu: expected 2, got %d\n",
                   i, h_output[i]);
            return false;
        }
    }
    return true;
}

// ========== 主函数 ==========
int main() {
    // 1. 检查设备计算能力
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    if (props.major < 9) {
        fprintf(stderr, "Error: TMA requires Compute Capability 9.0+ (NVIDIA Hopper). "
                        "Current device: sm_%d%d\n", props.major, props.minor);
        return 1;
    }
    printf("Device: %s (Compute Capability %d.%d)\n", props.name, props.major, props.minor);
    if (props.major == 9 && props.minor == 0) {
        printf("Note: For best TMA performance, compile with -arch=sm_90a\n");
    }

    // 2. 分配全局内存
    size_t bytes = static_cast<size_t>(GMEM_WIDTH) * GMEM_HEIGHT * sizeof(int);
    int* d_input = nullptr;
    int* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    // 3. 初始化输入数据为全 1
    CUDA_CHECK(cudaMemset(d_input, 1, bytes));
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));

    // 4. 创建 Tensor Map（TMA 硬件描述符）
    CUtensorMap tensor_map = create_2d_tensor_map(d_input);

    // 5. 计算 launch 配置
    int tiles_per_row = GMEM_WIDTH / SMEM_WIDTH;
    int tiles_per_col = GMEM_HEIGHT / SMEM_HEIGHT;
    int total_tiles = tiles_per_row * tiles_per_col;
    int threads_per_block = 256;
    // 共享内存大小：使用常量计算，不能用 sizeof(smem_buffer)（核函数内局部变量）
    size_t smem_size = SMEM_HEIGHT * SMEM_WIDTH * sizeof(int);

    printf("Global matrix: %d x %d (%zu bytes)\n", GMEM_WIDTH, GMEM_HEIGHT, bytes);
    printf("Tile: %d x %d (%zu bytes)\n", SMEM_WIDTH, SMEM_HEIGHT, smem_size);
    printf("Blocks: %d (%d x %d), Threads per block: %d\n",
           total_tiles, tiles_per_row, tiles_per_col, threads_per_block);

    // 6. 启动核函数
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    tma_copy_kernel<<<total_tiles, threads_per_block, smem_size>>>(tensor_map, d_output, tiles_per_row);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    printf("Kernel execution time: %.3f ms\n", elapsed_ms);

    // 检查 kernel 执行错误
    CUDA_CHECK(cudaGetLastError());

    // 7. 验证结果
    int* h_output = static_cast<int*>(malloc(bytes));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    bool correct = verify_result(h_output, static_cast<size_t>(GMEM_WIDTH) * GMEM_HEIGHT);
    printf("\n========== Result ==========\n");
    printf("TMA 2D Demo: %s\n", correct ? "PASS ✓" : "FAIL ✗");

    if (correct) {
        // 打印前几个元素作为示例
        printf("First 16 elements of output:\n");
        for (int i = 0; i < 16 && i < GMEM_WIDTH * GMEM_HEIGHT; i++) {
            printf("%d ", h_output[i]);
        }
        printf("\n");
    }

    // 8. 清理资源
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return correct ? 0 : 1;
}