# 第14章 Tensor Memory Accelerator (TMA)

<strong>硬件要求</strong>：Compute Capability 9.0+（NVIDIA Hopper H100+）；多播特性建议 `sm_90a`

## 14.1 引言

在上一章中，我们学习了 `cuda::memcpy_async` 和 `cuda::pipeline`——它们将数据从全局内存异步拷贝到共享内存，绕过了寄存器中转。这些 API 本质上是对硬件"cp.async"指令的抽象，能够高效地完成连续一维数据块的拷贝。

然而，许多高性能计算和深度学习工作负载涉及<strong>多维数组</strong>的<strong>不规则访问模式</strong>——例如矩阵乘法中的分块（tiling）、卷积中的滑动窗口、以及张量转置等。在这些场景中，地址计算本身就成为了一项昂贵的开销：你需要计算源地址、目标地址、步长、边界检查等。

这就是 <strong>Tensor Memory Accelerator (TMA)</strong> 大显身手的地方。CUDA Programming Guide 对 TMA 的定位非常清晰：

> "The primary goal of TMA is to provide an efficient data transfer mechanism from global memory to shared memory for multi-dimensional arrays."

TMA 是 NVIDIA Hopper 架构引入的一个<strong>硬件加速数据拷贝单元</strong>。它不仅仅在做数据搬运，更关键的是：<strong>它将地址计算从 CUDA Core 卸载到了专用硬件</strong>。这意味着：

1. <strong>减少寄存器压力</strong>：不再需要寄存器来存储地址计算中的中间值；
2. <strong>零地址计算开销</strong>：硬件根据"张量映射"（Tensor Map）自动完成多维地址生成；
3. <strong>硬件管理的 Swizzle</strong>：TMA 可以自动重排共享内存中的数据布局，消除 bank conflict；
4. <strong>异步执行</strong>：与 `memcpy_async` 一样，TMA 拷贝是异步的；
5. <strong>Cluster 多播</strong>：一次拷贝可以同时将数据广播到簇中多个块的共享内存。

本章将带你深入理解 TMA 的工作机制、Tensor Map 的创建与使用、TMA 的完成机制，以及 Swizzle 模式如何优化数据访问。最后我们将给出 TMA 与手动 `memcpy_async` 的性能对比分析。

## 14.2 TMA 概述

### 14.2.1 TMA 的命名约定

CUDA Programming Guide 在命名上做了明确的区分：

> "Naming. Tensor memory accelerator (TMA) is a broad term used to refer to the features described in this section. For the purpose of forward-compatibility and to reduce discrepancies with the PTX ISA, the text in this section refers to TMA operations as either bulk-asynchronous copies or bulk tensor asynchronous copies, depending on the specific type of copy used. The term 'bulk' is used to contrast these operations with the asynchronous memory operations described in the previous sections."

简单来说：

| 术语 | 含义 | 使用场景 |
|------|------|---------|
| <strong>bulk-asynchronous copy</strong> | 一维连续数据的批量异步拷贝 | 不需要 Tensor Map，直接使用指针+大小 |
| <strong>bulk tensor asynchronous copy</strong> | 多维张量的批量异步拷贝 | 需要 Tensor Map 描述数据布局 |

### 14.2.2 TMA 的关键特性

CUDA Programming Guide 归纳了 TMA 的以下关键特性：

<strong>维度支持</strong>：

> "Dimensions. TMA supports copying both one-dimensional and multi-dimensional arrays (up to 5-dimensional)."

TMA 支持 1D 到 5D 的数组拷贝。1D 拷贝不需要 Tensor Map，而多维拷贝需要通过 Tensor Map 描述全局内存中的数据布局。

<strong>源和目标</strong>：

TMA 支持的拷贝方向非常灵活：

| 源 | 目标 | 完成机制 |
|-----|------|---------|
| Global | Shared::cta | Shared Memory Barrier (mbarrier) |
| Shared::cta | Global | Bulk async-group |
| Global | Shared::cluster（多播） | Shared Memory Barrier (mbarrier) |
| Shared::cta | Shared::cluster | Shared Memory Barrier (mbarrier) |

CUDA Programming Guide 原表（Table 8）：

> "Asynchronous copies with possible source and destinations memory spaces and completion mechanisms."

<strong>异步性</strong>：

> "Asynchronous. Data transfers using TMA are asynchronous. This allows the initiating thread to continue computing while the hardware asynchronously copies the data."

关键的是 TMA 拷贝的异步性取决于硬件实现，未来可能发生变化：

> "Whether the data transfer occurs asynchronously in practice is up to the hardware implementation and may change in the future."

### 14.2.3 TMA 的优势总结

| 特性 | 传统 memcpy_async | TMA |
|------|------------------|-----|
| 地址计算 | 由 CUDA Core 计算（占用寄存器） | 硬件自动生成（零开销） |
| 维度支持 | 仅 1D | 1D ~ 5D |
| 边界检查 | 手动实现 | 硬件自动处理（越界填充零） |
| Swizzle | 不支持 | 硬件支持（4种模式） |
| 多播 | 不支持 | 支持 Cluster 多播 |
| 拷贝大小 | 4/8/16 字节对齐 | 16 字节对齐 |
| 共享内存对齐 | 16 字节(最大) | 128 字节（多维） |

## 14.3 一维 TMA 拷贝

一维 TMA 拷贝（bulk-asynchronous copy）不需要 Tensor Map，直接使用指针和大小参数。下面是从 CUDA Programming Guide 10.29.1 节提取的完整示例：

### 14.3.1 一维 TMA 核函数

```cuda
#include &lt;cuda/barrier&gt;
#include &lt;cuda/ptx&gt;
using barrier = cuda::barrier&lt;cuda::thread_scope_block&gt;;
namespace ptx = cuda::ptx;

static constexpr size_t buf_len = 1024;
__global__ void add_one_kernel(int* data, size_t offset)
{
  // 共享内存缓冲区——bulk 操作的目标缓冲区需要 16 字节对齐
  __shared__ alignas(16) int smem_data[buf_len];

  // 1. 初始化共享内存 barrier
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);                      // a) 初始化为参与线程数
    ptx::fence_proxy_async(ptx::space_shared);   // b) 使 barrier 对 async proxy 可见
  }
  __syncthreads();

  // 2. 发起 TMA 传输：global → shared
  if (threadIdx.x == 0) {
    // cuda::memcpy_async 自动对 barrier 执行 arrive，
    // 并告知预期的传输字节数（transaction count）
    cuda::memcpy_async(
        smem_data,
        data + offset,
        cuda::aligned_size_t&lt;16&gt;(sizeof(smem_data)),
        bar
    );
  }
  // 3b. 所有线程到达 barrier
  barrier::arrival_token token = bar.arrive();

  // 3c. 等待数据到达
  bar.wait(std::move(token));

  // 4. 计算：对共享内存数据加一
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
    smem_data[i] += 1;
  }

  // 5. 等待共享内存写入对 TMA 引擎可见
  ptx::fence_proxy_async(ptx::space_shared);
  __syncthreads();
  // syncthreads 之后，所有线程的写入对 TMA 引擎可见

  // 6. 发起 TMA 传输：shared → global
  if (threadIdx.x == 0) {
    ptx::cp_async_bulk(
        ptx::space_global,
        ptx::space_shared,
        data + offset, smem_data, sizeof(smem_data));
    // 7. 等待 TMA 传输完成读取共享内存
    // 创建 bulk async-group
    ptx::cp_async_bulk_commit_group();
    // 等待 group 完成读取共享内存
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t&lt;0&gt;());
  }
}
```

### 14.3.2 步骤详解

CUDA Programming Guide 对这个一维 TMA 核函数的每个步骤都有详细说明：

<strong>Barrier 初始化</strong>：barrier 以参与线程数初始化。使用 `fence.proxy.async.shared::cta` 指令确保后续 bulk-asynchronous copy 操作看到的是已初始化的 barrier。

<strong>TMA 读取</strong>：

> "The bulk-asynchronous copy instruction directs the hardware to copy a large chunk of data into shared memory, and to update the transaction count of the shared memory barrier after completing the read. In general, issuing as few bulk copies with as big a size as possible results in the best performance. Because the copy can be performed asynchronously by the hardware, it is not necessary to split the copy into smaller chunks."

关键理解：`cuda::memcpy_async` 在发起 TMA 拷贝时会自动调用 `mbarrier.expect_tx`，告诉 barrier 预期接收多少字节。barrier 只有在<strong>所有线程都到达 且 所有字节都已到达</strong>时才会翻转。

<strong>SMEM 写入与同步</strong>：在共享内存上完成计算后，需要 `fence.proxy.async.shared::cta` + `__syncthreads()` 确保所有线程的写入在 async proxy 中排序到后续的 bulk 操作之前。

<strong>TMA 写入与同步</strong>：从共享内存写回全局内存时，使用 `cp_async_bulk_commit_group` + `cp_async_bulk_wait_group_read` 的完成机制。这是<strong>线程局部</strong>的机制——只有发起线程需要等待。

### 14.3.3 一维 TMA 对齐要求

| 地址/大小 | 对齐要求 |
|-----------|---------|
| 全局内存地址 | 16 字节对齐 |
| 共享内存地址 | 16 字节对齐 |
| Barrier 地址 | 8 字节对齐（`cuda::barrier` 内部保证） |
| 传输大小 | 16 字节的倍数 |

## 14.4 多维 TMA 拷贝与 Tensor Map

### 14.4.1 Tensor Map 的概念

多维 TMA 拷贝需要一个 <strong>Tensor Map</strong>（张量映射）来描述全局内存中多维数组的布局。CUDA Programming Guide 指出：

> "To perform a bulk tensor asynchronous copy of a multi-dimensional array, the hardware requires a tensor map. This object describes the layout of the multi-dimensional array in global and shared memory."

Tensor Map 是一个 <strong>`CUtensorMap`</strong> 结构体，包含以下信息：
- 数据指针（base address）
- 各维度的尺寸（size）
- 各维度的步长（stride，以字节为单位）
- 共享内存缓冲区尺寸（box size）
- 元素步长（element stride）
- Swizzle 模式
- L2 缓存策略
- 越界填充模式

### 14.4.2 主机端创建 Tensor Map

Tensor Map 通过 CUDA Driver API 的 `cuTensorMapEncodeTiled` 函数创建：

```cpp
#include &lt;cudaTypedefs.h&gt; // PFN_cuTensorMapEncodeTiled, CUtensorMap

// 通过 Driver Entry Point API 获取函数指针
PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  CUDA_CHECK(cudaGetDriverEntryPointByVersion(
      "cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr,
      12000, cudaEnableDefault, &driver_status));
  assert(driver_status == cudaDriverEntryPointSuccess);
  return reinterpret_cast&lt;PFN_cuTensorMapEncodeTiled_v12000&gt;(
      cuTensorMapEncodeTiled_ptr);
}

// 创建 2D Tensor Map
CUtensorMap tensor_map{};
constexpr uint32_t rank = 2;
uint64_t size[rank]   = {GMEM_WIDTH, GMEM_HEIGHT};
// stride 是从一行移动到下一行所需的字节数，必须是 16 的倍数
uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
// box_size 是共享内存缓冲区的大小
uint32_t box_size[rank]   = {SMEM_WIDTH, SMEM_HEIGHT};
uint32_t elem_stride[rank] = {1, 1};

auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

CUresult res = cuTensorMapEncodeTiled(
    &tensor_map,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
    rank,                       // 张量维度
    tensor_ptr,                 // 全局内存基地址
    size,                       // 全局内存各维度尺寸
    stride,                     // 全局内存各维度步长（字节）
    box_size,                   // 共享内存 box 尺寸
    elem_stride,                // 元素步长
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);

//检查返回值
if (res != CUDA_SUCCESS) {
    const char* errStr = "unknown";
    cuGetErrorString(res, &errStr);
    fprintf(stderr, "cuTensorMapEncodeTiled failed with error %d: %s\n", res, errStr);
    exit(EXIT_FAILURE);
}
```

`cuTensorMapEncodeTiled` 的参数含义：
- <strong>tensorRank</strong>：张量的维度数（1~5）
- <strong>globalAddress</strong>：全局内存中的起始地址
- <strong>globalDim</strong>：各维度的元素数量
- <strong>globalStrides</strong>：各维度的步长（字节），注意 <strong>最快移动维度不需要 stride</strong>
- <strong>boxDim</strong>：共享内存 box 各维度的尺寸
- <strong>elementStrides</strong>：元素间的步长（以 sizeof(element) 为单位）
- <strong>Interleave</strong>：交错模式（用于加载小于 4 字节的元素）
- <strong>Swizzle</strong>：Swizzle 模式
- <strong>L2promotion</strong>：L2 缓存策略
- <strong>OOBfill</strong>：越界填充模式

### 14.4.3 传递 Tensor Map 到设备

CUDA Programming Guide 推荐三种方式将 Tensor Map 传递给核函数：

<strong>方式一：const __grid_constant__ 参数（推荐）</strong>

```cuda
#include &lt;cuda.h&gt;

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map)
{
   // 使用 tensor_map
}
int main() {
  CUtensorMap map;
  // [ ..初始化 map.. ]
  kernel&lt;&lt;&lt;1, 1&gt;&gt;&gt;(map);
}
```

`__grid_constant__` 注解告诉编译器这个参数在整个 grid 执行期间不变，可以被放入特殊的常量缓存中，从而获得更快的访问速度。

<strong>方式二：__constant__ 变量</strong>

```cuda
__constant__ CUtensorMap global_tensor_map;
__global__ void kernel()
{
  // 使用 global_tensor_map
}
int main() {
  CUtensorMap local_tensor_map;
  // [ ..初始化 map.. ]
  cudaMemcpyToSymbol(global_tensor_map, &local_tensor_map,
                     sizeof(CUtensorMap));
  kernel&lt;&lt;&lt;1, 1&gt;&gt;&gt;();
}
```

<strong>方式三：全局内存（最灵活但最慢）</strong>

```cuda
#include &lt;cuda/ptx&gt;
namespace ptx = cuda::ptx;

__device__ CUtensorMap global_tensor_map;
__global__ void kernel(CUtensorMap *tensor_map)
{
  // Fence acquire tensor map（因为 tensor map 可能被 host 修改）
  ptx::fence_proxy_tensormap_generic(
     ptx::sem_acquire, ptx::scope_sys, tensor_map,
     ptx::n32_t&lt;128&gt;());
  // 在 fence 之后安全使用 tensor_map
}
```

### 14.4.4 二维 TMA 核函数

以下是从 CUDA Programming Guide 10.29.2 节提取的 2D TMA 示例：

```cuda
#include &lt;cuda.h&gt;         // CUtensormap
#include &lt;cuda/barrier&gt;
using barrier = cuda::barrier&lt;cuda::thread_scope_block&gt;;
namespace cde = cuda::device::experimental;

__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map,
                       int x, int y) {
  // 多维 TMA 操作的共享内存目标缓冲区需要 128 字节对齐
  __shared__ alignas(128) int smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

  // 初始化 barrier
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
    // 使初始化的 barrier 对 async proxy 可见
    cde::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    // 发起二维批量张量拷贝：global → shared
    cde::cp_async_bulk_tensor_2d_global_to_shared(
        &smem_buffer, &tensor_map, x, y, bar);
    // 到达 barrier 并告知预期接收的字节数
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    // 其他线程仅到达 barrier
    token = bar.arrive();
  }
  // 等待数据到达
  bar.wait(std::move(token));

  // 对共享内存数据做计算（示例：修改第一个元素）
  smem_buffer[0][threadIdx.x] += threadIdx.x;

  // 等待共享内存写入对 TMA 引擎可见
  cde::fence_proxy_async_shared_cta();
  __syncthreads();

  // 发起 TMA 传输：shared → global
  if (threadIdx.x == 0) {
    cde::cp_async_bulk_tensor_2d_shared_to_global(
        &tensor_map, x, y, &smem_buffer);
    // 创建 bulk async-group
    cde::cp_async_bulk_commit_group();
    // 等待 group 完成读取共享内存
    cde::cp_async_bulk_wait_group_read&lt;0&gt;();
  }

}
```

### 14.4.5 多维 TMA 的越界处理

TMA 的一个重要特性是硬件级别的越界处理：

> "Negative indices and out of bounds. When part of the tile that is being read from global to shared memory is out of bounds, the shared memory that corresponds to the out of bounds area is zero-filled. The top-left corner indices of the tile may also be negative."

从全局内存<strong>读取</strong>时：越界区域自动填充零，且左上角坐标可以为负数。

从共享内存<strong>写入</strong>全局内存时：部分 tile 可以越界，但左上角坐标不能为负数。

### 14.4.6 尺寸与步长约定

CUDA Programming Guide 说明了尺寸和步长的约定：

> "The size of a tensor is the number of elements along one dimension. All sizes must be greater than one. The stride is the number of bytes between elements of the same dimension."

例如，一个 4x4 的整数矩阵：
- 尺寸：4 和 4
- 步长：4 字节（行内）和 16 字节（行间，4 x 4 bytes）

对于 4x3 的行优先整数矩阵，由于对齐要求：
- 步长：4 字节和 16 字节（padding 到 16 字节对齐）

### 14.4.7 多维 TMA 对齐要求

| 地址/尺寸 | 对齐要求 |
|-----------|---------|
| 全局内存地址 | 16 字节对齐 |
| 全局内存尺寸 | >= 1，不需要是 16 字节的倍数 |
| 全局内存步长 | 必须是 16 字节的倍数 |
| 共享内存地址 | 128 字节对齐 |
| Barrier 地址 | 8 字节对齐（`cuda::barrier` 保证） |
| 传输大小 | 16 字节的倍数 |

## 14.5 TMA 完成机制

TMA 操作的不同方向对应不同的完成机制。CUDA Programming Guide 的 Table 8 清晰总结了这些规则。

### 14.5.1 方向与完成机制对照表

| 目标 | 源 | 异步拷贝 | Bulk 异步拷贝 (TMA) |
|------|-----|---------|-------------------|
| Global | Global | -- | -- |
| Global | Shared::cta | -- | <strong>Bulk async-group</strong> |
| Shared::cta | Global | Async-group, mbarrier | <strong>Mbarrier</strong> |
| Shared::cluster | Global | -- | <strong>Mbarrier (multicast)</strong> |
| Shared::cta | Shared::cluster | -- | <strong>Mbarrier</strong> |
| Shared::cta | Shared::cta | -- | -- |

### 14.5.2 Shared Memory Barrier (mbarrier)

当 TMA 从全局内存<strong>读取</strong>到共享内存时，使用 <strong>Shared Memory Barrier (mbarrier)</strong> 作为完成机制。

工作流程：
1. 单个线程发起 `cp.async.bulk.tensor` 指令；
2. 该指令自动更新 barrier 的 <strong>transaction count</strong>（预期接收的字节数）；
3. 所有线程调用 `bar.arrive()`，或发起线程通过 `barrier_arrive_tx` 到达；
4. 当所有线程到达 且 所有字节到达时，barrier 翻转；
5. `bar.wait()` 返回后，数据在共享内存中可读。

### 14.5.3 Bulk Async-Group

当 TMA 从共享内存<strong>写入</strong>到全局内存（或分布式共享内存）时，使用 <strong>bulk async-group</strong> 作为完成机制。

工作流程：
1. 单个线程发起 `cp.async.bulk` 或 `cp.async.bulk.tensor` 指令；
2. 该线程将操作提交到一个线程局部的 bulk async-group：`cp_async_bulk_commit_group()`；
3. 该线程等待 group 完成：`cp_async_bulk_wait_group_read&lt;N&gt;()`。

注意：bulk async-group 是<strong>线程局部</strong>的——只有发起线程可以等待。这与 mbarrier 的块内多线程协同不同。

### 14.5.4 TMA 多播（Multicast）

在 Thread Block Cluster 中，TMA 支持<strong>多播</strong>——一次拷贝可以将数据同时广播到簇中多个线程块的共享内存。

CUDA Programming Guide 指出：

> "In addition, when in a cluster, a bulk-asynchronous operation can be specified as being multicast. In this case, data can be transferred from global memory to the shared memory of multiple blocks within the cluster. The multicast feature is optimized for target architecture `sm_90a` and may have significantly reduced performance on other targets."

多播的方向：Global → Shared::cluster，完成机制为 Mbarrier。

## 14.6 TMA Swizzle 模式

### 14.6.1 为什么需要 Swizzle

CUDA Programming Guide 解释了 Swizzle 的动机：

> "By default, the TMA engine loads data to shared memory in the same order as it is laid out in global memory. However, this layout may not be optimal for certain shared memory access patterns, as it could cause shared memory bank conflicts."

回忆一下共享内存 Bank 的组织方式：32 个 bank，每个连续 4 字节映射到连续的 bank。如果多个线程同时访问同一个 bank，就会产生 bank conflict。

Swizzle 的核心思想是：<strong>让 TMA 硬件在写入共享内存时重新排列数据，以消除后续访问中的 bank conflict</strong>。

> "To ensure that data is laid out in shared memory in such a way that user code can avoid shared memory bank conflicts, the TMA engine can be instructed to 'swizzle' the data before storing it in shared memory and 'unswizzle' it when copying the data back from shared memory to global memory."

### 14.6.2 矩阵转置案例

以下案例来自 CUDA Programming Guide 10.29.3.1：

<strong>问题</strong>：一个 8x8 的 `int4` 矩阵（行优先存储）从全局内存加载到共享内存。8 个线程将每行加载到转置缓冲区的对应列。在普通布局下，列方向存储会导致 8-way bank conflict。

<strong>解决方案</strong>：使用 `CU_TENSOR_MAP_SWIZZLE_128B` Swizzle 模式。该模式按 128 字节行为单位进行重排，使得行和列方向的访问都不会产生 bank conflict。

```cuda
__global__ void kernel_tma(const __grid_constant__ CUtensorMap tensor_map) {
   // 使用 128 字节 swizzle 模式时，共享内存需要 1024 字节对齐
   __shared__ alignas(1024) int4 smem_buffer[8][8];
   __shared__ alignas(1024) int4 smem_buffer_tr[8][8];

   // 初始化 barrier
   #pragma nv_diag_suppress static_var_with_dynamic_init
   __shared__ barrier bar;
   if (threadIdx.x == 0) {
     init(&bar, blockDim.x);
     cde::fence_proxy_async_shared_cta();
   }
   __syncthreads();

   barrier::arrival_token token;
   if (threadIdx.x == 0) {
     // 发起 TMA 拷贝（使用 swizzled tensor map）
     cde::cp_async_bulk_tensor_2d_global_to_shared(
         &smem_buffer, &tensor_map, 0, 0, bar);
     token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
   } else {
     token = bar.arrive();
   }
   bar.wait(std::move(token));

   /* 矩阵转置
    * 使用普通布局时，存储到转置缓冲区有 8 路 bank conflict
    * 使用 128 字节 swizzle 模式时，行和列访问都消除了 bank conflict */
   for(int sidx_j = threadIdx.x; sidx_j < 8; sidx_j += blockDim.x){
       // 转置操作（省略具体索引计算）
   }

   // 写回全局内存
   cde::fence_proxy_async_shared_cta();
   __syncthreads();
   if (threadIdx.x == 0) {
      cde::cp_async_bulk_tensor_2d_shared_to_global(
          &tensor_map, 0, 0, &smem_buffer_tr);
      cde::cp_async_bulk_commit_group();
      cde::cp_async_bulk_wait_group_read&lt;0&gt;();
   }
}
```

### 14.6.3 Swizzle 模式一览

CUDA Programming Guide 定义了四种 Swizzle 模式：

| 模式 | Swizzle 宽度 | Box 内维度要求 | 重复周期 | 共享内存对齐 | 全局内存对齐 |
|------|-------------|--------------|---------|------------|------------|
| `CU_TENSOR_MAP_SWIZZLE_NONE` | -- | -- | -- | 128 字节 | 16 字节 |
| `CU_TENSOR_MAP_SWIZZLE_32B` | 32 字节 | &lt;= 32 字节 | 256 字节 | 128 字节 | 128 字节 |
| `CU_TENSOR_MAP_SWIZZLE_64B` | 64 字节 | &lt;= 64 字节 | 512 字节 | 128 字节 | 128 字节 |
| `CU_TENSOR_MAP_SWIZZLE_128B` | 128 字节 | &lt;= 128 字节 | 1024 字节 | 128 字节 | 128 字节 |

### 14.6.4 Swizzle 索引计算

当共享内存缓冲区不是完全对齐到 Swizzle 模式周期时，需要计算偏移量：

```cpp
// 以 128B Swizzle 模式为例
data_t* smem_ptr = &smem[0][0];
int offset = (reinterpret_cast&lt;uintptr_t&gt;(smem_ptr)/128)%8;
// 访问 swizzled 共享内存：
smem[y][((y+offset)%8)^x] = ...;
```

CUDA Programming Guide 提供了各模式的偏移公式：

| Swizzle 模式 | 偏移公式 | 索引关系 |
|-------------|---------|---------|
| `CU_TENSOR_MAP_SWIZZLE_128B` | `(smem_ptr/128)%8` | `smem[y][((y+offset)%8)^x]` |
| `CU_TENSOR_MAP_SWIZZLE_64B` | `(smem_ptr/128)%4` | `smem[y][((y+offset)%4)^x]` |
| `CU_TENSOR_MAP_SWIZZLE_32B` | `(smem_ptr/128)%2` | `smem[y][((y+offset)%2)^x]` |

### 14.6.5 Swizzle 使用注意事项

CUDA Programming Guide 列出了以下关键要求：

> - Global memory must be aligned to 128 bytes.
> - Shared memory should be aligned according to the number of bytes after which the swizzle pattern repeats.
> - The inner dimension of the shared memory block must meet the size requirements.
> - The granularity of swizzle mapping is fixed at 16 bytes.

## 14.7 设备端 Tensor Map 编码

### 14.7.1 为什么需要设备端编码

CUDA Programming Guide (10.30) 介绍了在设备端编码 Tensor Map 的能力：

> "This section explains how to encode a tiled-type tensor map on device. This is useful in situations where the typical way of transferring the tensor map (using `const __grid_constant__` kernel parameters) is undesirable, for instance, when processing a batch of tensors of various sizes in a single kernel launch."

简单来说，当你需要在一个 kernel launch 中处理不同大小/形状的张量批次时，手动为每个张量在主机端创建 Tensor Map 传递是不现实的。设备端编码允许你在 GPU 上动态创建或修改 Tensor Map。

### 14.7.2 推荐模式

CUDA Programming Guide 推荐的三步模式：

> 1. Create a tensor map "template", `template_tensor_map`, using the Driver API on the host.
> 2. In a device kernel, copy the `template_tensor_map`, modify the copy, store in global memory, and appropriately fence.
> 3. Use the tensor map in a kernel with appropriate fencing.

```cpp
// 1. 主机端创建模板 tensor map
CUtensorMap template_tensor_map = make_tensormap_template();

// 2. 分配全局内存存放 tensor map
CUtensorMap* global_tensor_map;
cudaMalloc(&global_tensor_map, sizeof(CUtensorMap));

// 3. 设备端编码（在 kernel 中修改模板）
tensormap_params p{};
p.global_address = global_buf;
p.rank = 2;
p.box_dim[0] = 128; p.box_dim[1] = 4;
p.global_dim[0] = 256; p.global_dim[1] = 8;
// ...

encode_tensor_map&lt;&lt;&lt;1, 32&gt;&gt;&gt;(
    template_tensor_map, p, global_tensor_map);

// 4. 使用编码后的 tensor map
consume_tensor_map&lt;&lt;&lt;1, 1&gt;&gt;&gt;(global_tensor_map);
```

### 14.7.3 tensormap.replace PTX 指令

设备端修改 Tensor Map 通过 `tensormap.replace` PTX 指令实现，该指令可以修改 tiled-type tensor map 的任何字段，包括基地址、尺寸、步长等。这些功能通过 `cuda::ptx::tensormap_replace` 函数暴露。

### 14.7.4 相应同步要求

使用设备端编码的 Tensor Map 时需要注意同步：

1. 编码 kernel 和消费 kernel 之间需要适当的 fence（如 `fence_proxy_tensormap_generic`）确保 Tensor Map 的修改对 TMA 引擎可见。

2. 在消费 kernel 中，首次使用 Tensor Map 前需要 acquire fence。

## 14.8 多维 TMA 坐标与 Tile 加载

### 14.8.1 Tile 坐标系统

当使用多维 TMA 加载一个 tile 时，需要指定 tile 在全局数组中的"锚点"坐标（top-left corner）。TMA 硬件根据 Tensor Map 中描述的各维度尺寸自动生成 tile 中每个元素的全局地址。

例如，对于一个 256x256 的 2D 全局数组，如果要加载一个 16x16 的 tile，坐标为 (32, 64)：

```
全局数组 (256 x 256):
     0    16   32   48   ...  255
  0  +----+----+----+----+-----+
     |    |    |    |    |     |
 16  +----+----+----+----+-----+
     |    |    |    |    |     |
 32  +----+----+====+====+-----+
     |    |    |Tile|    |     |
 48  +----+----+====+====+-----+
     |    |    |    |    |     |
 ... |    |    |    |    |     |
255  +----+----+----+----+-----+

Tile 大小为 16x16，锚点为 (32, 64)
从 row=64, col=32 开始加载 16 行，每行 16 个元素
```

### 14.8.2 多维 TMA 的地址计算

TMA 硬件使用以下信息自动计算地址：
1. `base_address`（来自 Tensor Map 的 globalAddress 字段）
2. `global_dim[0..rank-1]`：各维度的尺寸
3. `global_stride[0..rank-2]`：除最快维度外各维度的步长（字节）
4. `box_dim[0..rank-1]`：要加载的 tile 尺寸
5. 锚点坐标 `(c0, c1, ..., c_{rank-1})`

对于 2D 数组，全局内存地址 = `base_address + c1 * stride[0] + c0 * element_size`

### 14.8.3 批量处理多个 Tile

在一个 kernel launch 中处理 2D 数组的多个 tile 的典型模式：

```cuda
__global__ void process_tiles(
    const __grid_constant__ CUtensorMap tensor_map,
    float *output, int num_tiles_x, int num_tiles_y)
{
    // 每个线程块处理一个 tile
    int tile_idx_x = blockIdx.x % num_tiles_x;
    int tile_idx_y = blockIdx.x / num_tiles_x;

    int global_x = tile_idx_x * SMEM_WIDTH;
    int global_y = tile_idx_y * SMEM_HEIGHT;

    // 检查边界
    if (global_x >= GMEM_WIDTH || global_y >= GMEM_HEIGHT) return;

    // 剩余的 TMA 加载-处理-写回流程与之前相同
    // ...
}
```

### 14.8.4 负坐标与越界处理

TMA 的一个独特能力是支持负的左上角坐标以处理边界条件（如卷积中的 padding）：

- <strong>读取时</strong>：左上角坐标可以为负数，越界部分自动填充零
- <strong>写入时</strong>：左上角坐标不能为负数，但部分 tile 可以越界（越界部分被静默丢弃）

CUDA Programming Guide 原文：

> "When part of the tile that is being read from global to shared memory is out of bounds, the shared memory that corresponds to the out of bounds area is zero-filled."

## 14.9 TMA 多播深度分析

### 14.9.1 多播工作流程

TMA 多播允许一次数据读取将相同数据广播到 Cluster 中多个块的共享内存。这在以下场景中极为有用：
- <strong>卷积权重广播</strong>：将相同的 filter 权重加载到所有块的共享内存；
- <strong>归约输入广播</strong>：将相同的输入数据发送给多个块进行并行归约；
- <strong>查找表分发</strong>：将索引表广播给所有块。

```cuda
// 使用 TMA 多播将相同 tile 加载到 Cluster 中所有块的共享内存
__global__ void multicast_kernel(
    const __grid_constant__ CUtensorMap tensor_map)
{
    __shared__ alignas(128) float smem_buffer[SMEM_HEIGHT][SMEM_WIDTH];

    // 初始化 barrier
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;
    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();

    // 多播 TMA 拷贝：全局 -> 簇中所有块的共享内存
    if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &smem_buffer, &tensor_map, x, y, bar);
        // 可选：添加 multicast 标记
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
    } else {
        token = bar.arrive();
    }
    bar.wait(std::move(token));

    // 所有块现在有相同的数据
}
```

### 14.9.2 多播的限制

- 多播使用共享内存 barrier 作为完成机制；
- 所有参与块必须属于同一个 Cluster；
- 在 `sm_90a` 目标上性能最佳；
- 多播的总数据量不能超过共享内存 barrier 的事务计数能力。

## 14.10 TMA Swizzle 进阶

### 14.10.1 理解 Swizzle 的物理含义

TMA Swizzle 本质上是在 TMA 硬件写入共享内存时，按照预定义的模式重新排列数据在共享内存中的物理位置。这个重排是透明的——写回全局内存时 TMA 会自动"反 Swizzle"。

以 `CU_TENSOR_MAP_SWIZZLE_128B` 为例：
- Swizzle 以 128 字节（即 32 个 int 元素）为宽度
- 数据被重新排列到 8 个子组（每组 4 个 bank）
- 每 1024 字节重复一次 Swizzle 模式

### 14.10.2 选择正确的 Swizzle 模式

选择 Swizzle 模式的依据：

1. <strong>无 Swizzle</strong>：访问模式与全局内存布局相同（无 bank conflict 时不需要）
2. <strong>32 字节 Swizzle</strong>：适用于内维度 &lt;= 32 字节的小 tile（例如 8 个 float）
3. <strong>64 字节 Swizzle</strong>：适用于内维度 &lt;= 64 字节的中等 tile（例如 16 个 float）
4. <strong>128 字节 Swizzle</strong>：适用于内维度 &lt;= 128 字节的较大 tile（例如 32 个 float 或 8 个 int4）

### 14.10.3 Swizzle 偏移的运行时计算

```cuda
__global__ void swizzled_access(const __grid_constant__ CUtensorMap tensor_map)
{
    __shared__ alignas(1024) int4 smem[8][8]; // 128B swizzle 要求 1024 字节对齐

    // 计算偏移量
    int4 *smem_ptr = &smem[0][0];
    int offset = (reinterpret_cast<uintptr_t>(smem_ptr) / 128) % 8;

    // 使用正确的索引访问 Swizzled 数据
    // 对于 128B Swizzle：
    // smem[y][x] 的实际物理位置 -> smem[y][((y+offset)%8)^x]
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int swizzled_x = ((y + offset) % 8) ^ x;
            int4 val = smem[y][swizzled_x];
            // ... 使用 val
        }
    }

    // 写回时同样需要正确的索引
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            int swizzled_x = ((y + offset) % 8) ^ x;
            smem[y][swizzled_x] = compute(val);
        }
    }
}
```

### 14.10.4 Bank Conflict 对比

对于 8x8 int4 矩阵的转置操作：

| 访问模式 | 无 Swizzle | 128B Swizzle |
|---------|-----------|-------------|
| 行读取 | 0 bank conflicts | 0 bank conflicts |
| 列写入 | 8-way bank conflict（串行化） | 0 bank conflicts |
| 总存储事务 | 64 (8x8) | 8 (1x8) |

对于列写入场景，128B Swizzle 将 64 个存储事务减少到 8 个，理论加速比高达 8x。

## 14.11 TMA 与 Pipeline 的结合

TMA 可以与第 13 章介绍的 `cuda::pipeline` 无缝结合，实现多维数据的异步流水线处理：

```cuda
// 伪代码：TMA + Pipeline 的多阶段流水线
__global__ void tma_pipeline_kernel(
    const __grid_constant__ CUtensorMap tensor_map,
    float *output)
{
    extern __shared__ float shared_buf[];
    // 分配 N 个阶段的共享内存缓冲区

    // 创建 pipeline 和 barrier
    __shared__ cuda::pipeline_shared_state&lt;
        cuda::thread_scope::thread_scope_block, stages&gt; pipe_state;
    auto pipe = cuda::make_pipeline(block, &pipe_state);

    // 预热流水线
    for (int s = 0; s < stages - 1; s++) {
        pipe.producer_acquire();
        // 使用 TMA 加载第 s 个 tile
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &shared_buf[s * tile_size], &tensor_map,
            tile_x[s], tile_y[s], bar);
        pipe.producer_commit();
    }

    // 稳态流水线
    for (int i = 0; i < num_tiles; i++) {
        int stage = i % stages;

        // 生产：加载下一个 tile
        pipe.producer_acquire();
        int next = i + stages - 1;
        if (next < num_tiles) {
            cde::cp_async_bulk_tensor_2d_global_to_shared(
                &shared_buf[stage * tile_size], &tensor_map,
                tile_x[next], tile_y[next], bar);
        }
        pipe.producer_commit();

        // 消费：处理当前 tile
        pipe.consumer_wait();
        compute(&shared_buf[stage * tile_size], output + i * tile_size);
        pipe.consumer_release();
    }
}
```

## 14.12 TMA vs 手动 memcpy_async 性能对比

| 维度 | 手动 memcpy_async | TMA |
|------|-----------------|-----|
| <strong>地址生成</strong> | 占用寄存器，需手动计算 | 硬件自动生成，零寄存器压力 |
| <strong>1D 连续拷贝</strong> | 性能接近 TMA | 性能接近 memcpy_async |
| <strong>多维分块拷贝</strong> | 需逐个元素或逐行手动拷贝 | 单指令完成整个 tile |
| <strong>Bank conflict</strong> | 手动处理（难） | 硬件 Swizzle 自动消除 |
| <strong>越界处理</strong> | 手动条件判断 | 硬件自动零填充 |
| <strong>多播</strong> | 不支持 | 支持 Cluster 多播 |
| <strong>编程复杂度</strong> | 中等 | 较高（需 Tensor Map） |
| <strong>CC 最低要求</strong> | CC 7.0+ | CC 9.0+ |

对于简单的 1D 连续数据拷贝，TMA 和 `memcpy_async` 性能接近。但当涉及多维分块、不规则步长、或需要消除 bank conflict 的场景时，TMA 的优势就非常显著——因为它将复杂的地址计算和布局转换卸载到了专用硬件。

## 14.13 TMA 的限制与兼容性

### 14.13.1 硬件限制

1. <strong>CC 9.0+ 独占</strong>：TMA 是 Hopper 架构的特性，无法在旧硬件上使用；
2. <strong>共享内存对齐严格</strong>：多维 TMA 要求共享内存 128 字节对齐，Swizzle 模式下要求 1024 字节对齐；
3. <strong>异步性不保证</strong>：CUDA Programming Guide 明确指出 TMA 传输的异步性取决于硬件实现；
4. <strong>多播性能</strong>：`sm_90a` 目标上优化最佳，其他目标可能性能显著下降。

### 14.13.2 编程限制

1. Tensor Map 的创建需要 Driver API（`cuTensorMapEncodeTiled`），不能完全在 Runtime API 中完成；
2. 设备端编码依赖 `tensormap.replace` PTX 指令，API 封装尚在 experimental 阶段；
3. 调试困难：TMA 是硬件黑盒，无法像手动 `memcpy_async` 那样直接跟踪数据流；
4. 编译器要求：需要 `nvcc -arch=sm_90` 或更高。

### 14.13.3 兼容性

- TMA 代码在 CC < 9.0 的设备上<strong>无法运行</strong>；
- 可以在编译时检查：

```cuda
#if __CUDA_ARCH__ < 900
static_assert(false,
    "Device code compiled with older architectures incompatible with TMA.");
#endif
```

- 可移植代码应提供回退路径：

```cuda
#if __CUDA_ARCH__ >= 900
    // TMA 路径
    cde::cp_async_bulk_tensor_2d_global_to_shared(...)
#else
    // 回退路径：使用 memcpy_async 手动逐行拷贝
    for (int row = 0; row < SMEM_HEIGHT; row++) {
        cuda::memcpy_async(block,
            &smem[row * SMEM_WIDTH],
            &global[(global_y + row) * GMEM_WIDTH + global_x],
            sizeof(float) * SMEM_WIDTH, pipe);
    }
#endif
```

## 14.14 TMA 实际应用场景

### 14.14.1 深度学习卷积

在深度学习框架中，卷积操作的 im2col 或直接卷积实现是 TMA 的典型应用：

```cuda
// 使用 TMA 加载卷积的 3D 输入 tile
__global__ void conv3d_tma(
    const __grid_constant__ CUtensorMap input_map,
    const __grid_constant__ CUtensorMap filter_map,
    float *output)
{
    // 输入维度: [C, H, W]
    // 每个块加载一个 3D tile: [C_TILE, H_TILE, W_TILE]
    __shared__ alignas(128) float smem_input[C_TILE][H_TILE][W_TILE];
    __shared__ alignas(128) float smem_filter[C_TILE][K_TILE][R][S];

    // 加载输入 tile
    cde::cp_async_bulk_tensor_3d_global_to_shared(
        &smem_input, &input_map, c0, h0, w0, bar);

    // 加载 filter tile
    cde::cp_async_bulk_tensor_4d_global_to_shared(
        &smem_filter, &filter_map, k0, c0, 0, 0, bar);

    // 等待数据就绪，然后执行卷积计算...
}
```

### 14.14.2 矩阵乘法分块

```cuda
__global__ void gemm_tma(
    const __grid_constant__ CUtensorMap A_map,
    const __grid_constant__ CUtensorMap B_map,
    float *C, int M, int N, int K)
{
    // 每个块计算一个 tile: C[TILE_M x TILE_N]
    __shared__ alignas(128) float As[TILE_M][TILE_K];
    __shared__ alignas(128) float Bs[TILE_K][TILE_N];

    int block_m = blockIdx.y * TILE_M;
    int block_n = blockIdx.x * TILE_N;

    float accum[TILE_M][TILE_N] = {0.0f};

    for (int k = 0; k < K; k += TILE_K) {
        // TMA 加载 A 的 tile: [block_m:block_m+TILE_M, k:k+TILE_K]
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &As, &A_map, block_m, k, bar);
        // TMA 加载 B 的 tile: [k:k+TILE_K, block_n:block_n+TILE_N]
        cde::cp_async_bulk_tensor_2d_global_to_shared(
            &Bs, &B_map, k, block_n, bar);

        barrier_arrive_tx_and_wait();

        // 计算 C += A * B (手动或通过 Tensor Core)
        for (int i = 0; i < TILE_M; i++) {
            for (int j = 0; j < TILE_N; j++) {
                for (int kk = 0; kk < TILE_K; kk++) {
                    accum[i][j] += As[i][kk] * Bs[kk][j];
                }
            }
        }
    }

    // 写回 C
    for (int i = 0; i < TILE_M; i++) {
        for (int j = 0; j < TILE_N; j++) {
            C[(block_m + i) * N + (block_n + j)] = accum[i][j];
        }
    }
}
```

### 14.14.3 图像处理中的边界处理

利用 TMA 的越界零填充机制，可以简化图像处理中的边界处理：

```cuda
__global__ void convolution_with_padding(
    const __grid_constant__ CUtensorMap input_map,
    float *output, int width, int height, int kernel_size)
{
    int pad = kernel_size / 2;

    // 故意将 tile 坐标延伸到负值——TMA 自动填充零
    int x_start = blockIdx.x * TILE_W - pad;
    int y_start = blockIdx.y * TILE_H - pad;

    // TMA 自动处理越界，越界区域填充零
    cde::cp_async_bulk_tensor_2d_global_to_shared(
        &smem, &input_map, x_start, y_start, bar);
}
```

## 14.15 TMA 性能调优指南

### 14.15.1 拷贝大小的选择

TMA 的最佳拷贝大小取决于多个因素：

| 拷贝大小 | 延迟 | 吞吐量 | 适用场景 |
|---------|------|-------|---------|
| 小 (&lt; 512B) | 低 | 低 | 细粒度 tile |
| 中 (512B - 4KB) | 中等 | 中等 | 典型 tile |
| 大 (&gt; 4KB) | 高 | 高 | 大 tile、整个行 |

指导原则：

> "In general, issuing as few bulk copies with as big a size as possible results in the best performance."

### 14.15.2 对齐的严格性

下表总结了不同 TMA 模式的对齐要求严重程度：

| 未满足的对齐 | 后果 |
|------------|------|
| 全局内存 16B 对齐 | <strong>硬错误</strong>（未定义行为） |
| 共享内存 128B 对齐 | <strong>硬错误</strong>（未定义行为） |
| 步长 16B 倍数 | <strong>硬错误</strong>（未定义行为） |
| Swizzle 共享内存 1024B 对齐 | <strong>硬错误</strong>（128B Swizzle 模式） |

### 14.15.3 使用 Nsight Compute 分析 TMA 性能

NVIDIA Nsight Compute 提供 TMA 相关指标：

- `smsp__inst_executed_pipe_tensor_op_hmma`：Tensor pipe 指令执行数
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld`：全局加载 sector 数
- `l1tex__t_sectors_pipe_lsu_mem_local_op_st`：共享内存存储 sector 数

通过这些指标可以量化 TMA 实际节省的指令数和带宽。

## 14.16 TMA 与手动 memcpy_async 的迁移指南

如果你的代码目前使用 `cuda::memcpy_async` 进行多维分块拷贝，迁移到 TMA 的步骤如下：

1. <strong>识别多维分块模式</strong>：找出代码中使用嵌套循环逐行拷贝的 `memcpy_async` 调用；

2. <strong>确保硬件支持</strong>：添加 CC 9.0+ 的条件编译；

3. <strong>创建 Tensor Map</strong>：在主机端使用 `cuTensorMapEncodeTiled` 描述数组布局；

4. <strong>替换拷贝调用</strong>：将多个逐行 `memcpy_async` 调用替换为单个 `cp_async_bulk_tensor` 调用；

5. <strong>调整同步</strong>：确保 barrier 的事务计数正确（TMA 拷贝的总字节数）；

6. <strong>验证正确性</strong>：在 H100 上测试，同时确保回退路径在旧硬件上继续工作。



## 14.17 动手体验：完整的 2D TMA 分块处理程序

下面是一个完整的、概念展示性的 2D TMA 程序（注意：实际运行需要 H100 硬件和 Driver API 支持）：

```cuda
// 文件: tma_2d_demo.cu
// 概念演示：2D TMA 分块加载、处理和写回
// 编译: nvcc -arch=sm_90 tma_2d_demo.cu -lcuda -o tma_2d_demo
// 硬件要求: NVIDIA Hopper H100 (CC 9.0+)

#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;cuda.h&gt;
#include &lt;cudaTypedefs.h&gt;
#include &lt;cuda/barrier&gt;

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

// ========== 主机端：创建 Tensor Map ==========

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    void* ptr = nullptr;
    cudaDriverEntryPointQueryResult status;
    CUDA_CHECK(cudaGetDriverEntryPointByVersion(
        "cuTensorMapEncodeTiled", &ptr, 12000,
        cudaEnableDefault, &status));
    return reinterpret_cast&lt;PFN_cuTensorMapEncodeTiled_v12000&gt;(ptr);
}

CUtensorMap create_2d_tensor_map(int *d_data) {
    CUtensorMap tmap{};
    constexpr uint32_t rank = 2;
    uint64_t size[rank]   = {GMEM_WIDTH, GMEM_HEIGHT};
    uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
    uint32_t box_size[rank]   = {SMEM_WIDTH, SMEM_HEIGHT};
    uint32_t elem_stride[rank] = {1, 1};

    auto encode = get_cuTensorMapEncodeTiled();
    encode(&tmap,
           CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
           rank,
           d_data,
           size,
           stride,
           box_size,
           elem_stride,
           CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
           CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
           CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
           CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

     if (res != CUDA_SUCCESS) {
        const char* errStr = "unknown";
        cuGetErrorString(res, &errStr);
        fprintf(stderr, "cuTensorMapEncodeTiled failed with error %d: %s\n", res, errStr);
        exit(EXIT_FAILURE);
    }
      
    return tmap;
}

int main() {
    // 分配全局内存
    int *d_input, *d_output;
    size_t bytes = GMEM_WIDTH * GMEM_HEIGHT * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    // 初始化输入数据（这里省略，实际需要主机端填充并拷贝到设备）
    CUDA_CHECK(cudaMemset(d_input, 1, bytes));

    // 创建 Tensor Map
    CUtensorMap tmap = create_2d_tensor_map(d_input);

    // 启动 kernel（概念演示——实际需要带有 TMA 操作的 kernel）
    // kernel&lt;&lt;&lt;grid, block, smem&gt;&gt;&gt;(tmap, d_output);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("TMA 2D demo setup complete.\n");
    printf("Run on H100 hardware with full kernel for actual execution.\n");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    return 0;
}
```

## 14.18 常见问题与故障排除

### 14.18.1 TMA 传输返回错误或崩溃

<strong>常见原因</strong>：
1. 全局内存地址未 16 字节对齐；
2. 共享内存地址未 128 字节对齐（多维 TMA）；
3. 全局内存步长不是 16 字节的倍数；
4. 在 CC < 9.0 的硬件上运行。

<strong>检查清单</strong>：
- 全局内存使用 `cudaMalloc`（默认 256 字节对齐）或 `alignas(16)`；
- 共享内存使用 `__shared__ alignas(128)` 或 `alignas(1024)`（Swizzle）；
- 使用 `static_assert` 在编译时检查计算能力。

### 14.18.2 Tensor Map 创建失败

<strong>cuTensorMapEncodeTiled 返回错误</strong>：

常见原因包括：
1. `box_dim` 大于 `global_dim`（共享内存 tile 大于全局数组）；
2. `stride` 不是 16 字节的倍数；
3. Driver API 版本不匹配（需要 CUDA 12.0+ 驱动）。

### 14.18.3 Swizzle 索引计算错误

<strong>症状</strong>：使用 Swizzle 后数据正确但位置错误。

<strong>诊断</strong>：
1. 检查共享内存对齐是否满足 Swizzle 模式要求；
2. 正确计算偏移量：`offset = (reinterpret_cast<uintptr_t>(smem_ptr)/128)%8`；
3. 正确使用索引关系：`smem[y][((y+offset)%8)^x]`。

### 14.18.4 设备端编码的 fence 问题

<strong>症状</strong>：修改后的 Tensor Map 不可见或使用了旧值。

<strong>解决方案</strong>：
- 编码 kernel 结束后添加 `__threadfence_system()` 或使用适当的 scope fence；
- 消费 kernel 中使用 `fence_proxy_tensormap_generic` 的 acquire fence；
- 确保编码和消费在不同的 kernel launch 中（或在同一 launch 中使用适当的内存顺序）。

### 14.18.5 TMA 多播性能不佳

<strong>可能原因</strong>：
1. 未使用 `sm_90a` 目标编译；
2. Cluster 大小超过实际的 GPC 容量；
3. 多播范围太大（多播所有块 vs 多播部分块）。

<strong>优化建议</strong>：
- 使用 `-arch=sm_90a` 编译以启用多播硬件优化；
- 将多播限制在较小的 cluster（2-4 个块）；
- 评估是否真的需要多播——有时单块加载 + DSM 分发更高效。

## 14.19 性能基准参考

以下是在 H100 (CC 9.0) 上对不同数据搬运方案的性能对比（2D 256x256 数组分块，tile 16x16）：

| 方案 | Tensor Map | Swizzle | 时间 (us) | 相对性能 |
|------|-----------|---------|----------|---------|
| 手动逐行 memcpy_async | N/A | N/A | 48 | 1.0x |
| TMA 1D bulk copy | 否 | 否 | 32 | 1.5x |
| TMA 2D tensor copy | 是 | 否 | 24 | 2.0x |
| TMA 2D + 128B Swizzle (转置) | 是 | 是 | 18 | 2.7x |

（注：实际性能因数据规模、访问模式、硬件配置而异。上述为示意性参考。）

从参考数据可以看出：
1. 一维 TMA 比手动 memcpy_async 快约 50%，主要来自减少了地址计算开销；
2. 二维 TMA tensor copy 比一维快约 33%，因为一次指令处理了整个 tile；
3. Swizzle 模式在转置场景提供了额外 33% 的加速，来自消除 bank conflict。

## 14.20 迁移指南：从 memcpy_async 到 TMA

如果你的代码目前使用 `cuda::memcpy_async` 进行多维分块拷贝，以下是迁移到 TMA 的步骤：

### 步骤 1：识别候选模式

适合 TMA 迁移的代码模式：
- 对二维或更高维度数组的分块访问；
- 每个块逐行或逐元素通过 `memcpy_async` 进行多次拷贝；
- 存在手工的地址计算和边界检查；
- 存在共享内存 bank conflict。

### 步骤 2：创建 Tensor Map

```cuda
// 在主机端创建 Tensor Map 替代手工地址计算
CUtensorMap tmap{};
// 填充尺寸、步长、box 大小等信息
cuTensorMapEncodeTiled(&tmap, ...);
```

### 步骤 3：替换拷贝调用

```cuda
// 之前：逐行 memcpy_async
for (int row = 0; row < TILE_H; row++) {
    cuda::memcpy_async(block,
        &smem[row * TILE_W],
        &global[(start_y + row) * GMEM_W + start_x],
        sizeof(float) * TILE_W, pipe);
}

// 之后：单个 TMA 指令
cde::cp_async_bulk_tensor_2d_global_to_shared(
    &smem, &tmap, start_x, start_y, bar);
```

### 步骤 4：调整同步

```cuda
// 添加 fence 和 barrier 初始化
if (threadIdx.x == 0) {
    init(&bar, blockDim.x);
    cde::fence_proxy_async_shared_cta();
}
__syncthreads();
```

### 步骤 5：验证与回退

```cuda
#if __CUDA_ARCH__ >= 900
    cde::cp_async_bulk_tensor_2d_global_to_shared(...);
#else
    // 回退：逐行 memcpy_async
    for (int row = 0; row < TILE_H; row++) { ... }
#endif
```

## 14.21 本章小结

本章深入介绍了 NVIDIA Hopper 架构的 Tensor Memory Accelerator (TMA)，要点总结如下：

1. <strong>TMA 是什么</strong>：一个硬件加速的多维数据拷贝单元，将地址计算从 CUDA Core 卸载到专用硬件，减少寄存器压力并消除手动地址计算开销。

2. <strong>Tensor Map</strong>：描述多维数组在全局和共享内存中布局的数据结构。通过 Driver API `cuTensorMapEncodeTiled` 在主机端创建，通过 `const __grid_constant__` 参数或 `__constant__` 变量传递给设备。

3. <strong>1D TMA 拷贝</strong>：不需要 Tensor Map，使用指针和大小参数。Global→Shared 方向使用 mbarrier 完成机制，Shared→Global 使用 bulk async-group。

4. <strong>多维 TMA 拷贝</strong>：需要 Tensor Map，支持最多 5 维。硬件自动处理越界（零填充）和边界条件。支持 Cluster 多播。

5. <strong>Swizzle</strong>：TMA 硬件可以在写入共享内存时自动重排数据布局，消除后续访问中的 bank conflict。支持 32B/64B/128B 三种 Swizzle 模式。

6. <strong>设备端编码</strong>：通过 `tensormap.replace` PTX 指令，可以在设备端动态修改 Tensor Map，适用于处理不同形状的张量批次。

7. <strong>与 Pipeline 结合</strong>：TMA 可以与 `cuda::pipeline` 结合实现多维数据的多阶段异步流水线，将数据搬运与计算完全重叠。

8. <strong>限制</strong>：仅限 CC 9.0+，硬件依赖性强，Tensor Map 创建需要 Driver API。

TMA 代表了 GPU 编程从"软件管理数据搬运"到"硬件加速数据搬运"的重要演进。对于矩阵乘法、卷积、转置等核心计算模式，TMA 能够显著简化代码并提升性能。

## 14.22 习题

1. 对比 TMA 的 bulk-asynchronous copy 和上一章的 `cuda::memcpy_async`，说明在什么场景下 TMA 有显著优势，什么场景下两者性能接近。

2. 解释为什么 Global→Shared::cta 的 TMA 拷贝使用 mbarrier 完成机制，而 Shared::cta→Global 使用 bulk async-group。这两种完成机制的设计原理是什么？

3. 创建一个 3D Tensor Map（例如尺寸为 `[D, H, W]` 的体积数据），写出 `cuTensorMapEncodeTiled` 的调用代码。

4. 对于 4x4 的整数矩阵，计算 `CU_TENSOR_MAP_SWIZZLE_128B` 模式下的 Swizzle 偏移量（假设共享内存地址为 0x100）。

5. TMA 多播（Multicast）是如何与 Thread Block Cluster 配合使用的？设计一个场景，利用 Multicast 一次性将数据广播到簇中所有块的共享内存。

6. 讨论设备端 Tensor Map 编码相比主机端创建的优势和局限性。什么场景下必须使用设备端编码？

## 14.23 参考文献

1. CUDA C++ Programming Guide 13.0, Section 10.29 "Asynchronous Data Copies using the Tensor Memory Accelerator (TMA)"
2. CUDA C++ Programming Guide 13.0, Section 10.30 "Encoding a Tensor Map on Device"
3. CUDA Driver API Documentation — cuTensorMapEncodeTiled
4. PTX ISA Documentation — cp.async.bulk and cp.async.bulk.tensor instructions
5. PTX ISA Documentation — tensormap.replace instruction
6. NVIDIA H100 Tensor Core GPU Architecture Whitepaper
7. libcu++ API Documentation — cuda::ptx namespace
