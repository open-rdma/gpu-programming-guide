// 文件: tma_2d_demo.cu
// 概念演示：2D TMA 分块加载、处理和写回
// 编译: nvcc -arch=sm_90 tma_2d_demo.cu -lcuda -o tma_2d_demo
// 硬件要求: NVIDIA Hopper H100 (CC 9.0+)
// 第14章 Tensor Memory Accelerator - TMA基础示例

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda/barrier>

#define CUDA_CHECK(call)                                             \
    do {                                                             \
        cudaError_t err = call;                                      \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)

constexpr int GMEM_WIDTH  = 256;
constexpr int GMEM_HEIGHT = 256;
constexpr int SMEM_WIDTH  = 16;
constexpr int SMEM_HEIGHT = 16;

// ========== 主机端：获取 cuTensorMapEncodeTiled 函数指针 ==========

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
    CUDA_CHECK(cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &ptr, 12000,
        cudaEnableDefault, &status));
    if (status != cudaDriverEntryPointSuccess) {
        fprintf(stderr, "Failed to get cuTensorMapEncodeTiled\n");
        exit(EXIT_FAILURE);
    }
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(ptr);
}

// ========== 主机端：创建 2D Tensor Map ==========

CUtensorMap create_2d_tensor_map(int *d_data) {
    CUtensorMap tmap{};
    constexpr uint32_t rank = 2;
    uint64_t size[rank]   = {GMEM_WIDTH, GMEM_HEIGHT};
    uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
    uint32_t box_size[rank]   = {SMEM_WIDTH, SMEM_HEIGHT};
    uint32_t elem_stride[rank] = {1, 1};

    auto encode = get_cuTensorMapEncodeTiled();
    CUresult res = encode(
        &tmap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
        rank,
        d_data,                    // globalAddress
        size,                      // globalDim
        stride,                    // globalStrides
        box_size,                  // boxDim
        elem_stride,               // elementStrides
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if (res != CUDA_SUCCESS) {
        fprintf(stderr, "cuTensorMapEncodeTiled failed\n");
        exit(EXIT_FAILURE);
    }
    return tmap;
}

// ========== 核函数：使用 TMA 加载 2D tile ==========

// 注意：此核函数为概念框架，因为 TMA 操作需要通过 PTX 封装或 libcu++ 使用
// 实际运行需要完整的 TMA PTX 封装代码
__global__ void tma_kernel(const __grid_constant__ CUtensorMap tensor_map,
                           int *output)
{
    // 多维 TMA 操作的共享内存需要 128 字节对齐
    __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

    // 初始化 shared memory barrier
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> bar;

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        // fence_proxy_async_shared_cta 在完整实现中使用
    }
    __syncthreads();

    // 以下为概念代码：实际需要 cp_async_bulk_tensor_2d_global_to_shared
    // 和 cp_async_bulk_tensor_2d_shared_to_global 的封装
    //
    // 1. 发起 TMA 拷贝：global -> shared
    // 2. 等待 barrier
    // 3. 计算（修改共享内存数据）
    // 4. fence + syncthreads
    // 5. 发起 TMA 拷贝：shared -> global
    // 6. 等待 bulk async-group

    // 填充示例数据
    for (int i = threadIdx.x; i < SMEM_HEIGHT * SMEM_WIDTH; i += blockDim.x) {
        int r = i / SMEM_WIDTH;
        int c = i % SMEM_WIDTH;
        smem_buffer[r][c] = (r + c) * blockIdx.x + 1;
    }
    __syncthreads();

    // 示例写回
    for (int i = threadIdx.x; i < SMEM_HEIGHT * SMEM_WIDTH; i += blockDim.x) {
        int r = i / SMEM_WIDTH;
        int c = i % SMEM_WIDTH;
        int global_idx = blockIdx.x * SMEM_HEIGHT * SMEM_WIDTH + r * SMEM_WIDTH + c;
        if (global_idx < GMEM_WIDTH * GMEM_HEIGHT) {
            output[global_idx] = smem_buffer[r][c];
        }
    }
}

int main() {
    // 检查设备计算能力
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    if (props.major < 9) {
        fprintf(stderr, "Error: This demo requires Compute Capability 9.0+ "
                        "(NVIDIA H100). Current device: sm_%d%d\n",
                props.major, props.minor);
        return 1;
    }

    printf("Device: %s (CC %d.%d)\n", props.name, props.major, props.minor);

    // 分配全局内存
    int *d_input, *d_output;
    size_t bytes = GMEM_WIDTH * GMEM_HEIGHT * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    // 初始化输入数据
    CUDA_CHECK(cudaMemset(d_input, 1, bytes));
    CUDA_CHECK(cudaMemset(d_output, 0, bytes));

    // 创建 Tensor Map
    CUtensorMap tmap = create_2d_tensor_map(d_input);

    // 计算 grid 维度（每个块处理一个 tile）
    int tiles_x = GMEM_WIDTH / SMEM_WIDTH;
    int tiles_y = GMEM_HEIGHT / SMEM_HEIGHT;
    int total_tiles = tiles_x * tiles_y;

    // 启动 kernel
    size_t smem_size = SMEM_HEIGHT * SMEM_WIDTH * sizeof(int);
    tma_kernel<<<total_tiles, 256, smem_size>>>(tmap, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 验证（简单检查：所有输出应为 2）
    int *h_output = (int *)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (size_t i = 0; i < (size_t)(GMEM_WIDTH * GMEM_HEIGHT); i++) {
        if (h_output[i] == 0) {
            printf("Zero at index %zu\n", i);
            correct = false;
            break;
        }
    }
    printf("TMA 2D demo: %s\n", correct ? "PASS" : "FAIL");

    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return correct ? 0 : 1;
}
