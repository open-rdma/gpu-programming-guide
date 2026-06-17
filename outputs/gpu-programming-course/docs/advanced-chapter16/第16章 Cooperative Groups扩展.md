# 第16章 Cooperative Groups扩展：Cluster Group与高级集合操作

<strong>硬件要求</strong>：CC 7.0+（Basic Cooperative Groups），CC 9.0+（Cluster Group），CC 8.0+（异步Reduce/Scan硬件加速）

> "Cooperative Groups is an extension to the CUDA programming model, introduced in CUDA 9, for organizing groups of communicating threads. Cooperative Groups allows developers to express the granularity at which threads are communicating, helping them to express richer, more efficient parallel decompositions."
> -- CUDA C++ Programming Guide 13.0

---

## 16.1 引言与回顾

Cooperative Groups（协作组，简称CG）是CUDA 9引入的一个重要编程模型扩展。在CG出现之前，CUDA编程模型只提供了一种简单的线程同步构造：`__syncthreads()`（一个线程块内所有线程的屏障）。程序员如果想要在warp级别或其他粒度上进行同步和协作，只能编写自己不可移植、不安全的原语。

CG通过将"线程组"抽象为<strong>一等程序对象</strong>，解决了这一问题。这种抽象不仅使程序员意图更加明确，还消除了脆弱的架构假设，使得代码更易于维护且跨GPU代际兼容。

### 16.1.1 回顾：基础CG概念

在基础课程中，你已经学习了以下CG核心概念：

- <strong>隐式组（Implicit Groups）</strong>：由kernel启动配置确定的组，包括 `thread_block` 和 `grid_group`。
- <strong>显式组（Explicit Groups）</strong>：通过划分操作创建的细粒度组，包括：
  - `coalesced_group`：warp中活跃线程组成的组
  - `thread_block_tile`：编译期确定大小的tile组（通过 `tiled_partition` 创建）
- <strong>集合操作（Collective Operations）</strong>：`group.sync()`、`group.thread_rank()`、`group.num_threads()` 等。

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void exampleKernel() {
    // 获取线程块组
    cg::thread_block block = cg::this_thread_block();

    // 将线程块按warp大小划分为tile
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

    // 进一步划分为大小为4的tile
    cg::thread_block_tile<4, cg::thread_block> tile4 =
        cg::tiled_partition<4>(block);

    if (tile4.thread_rank() == 0) {
        // 每个tile的leader执行的操作
    }
}
```

### 16.1.2 本章学习目标

- 掌握 `cluster_group` 的构造和使用，包括分布式共享内存操作
- 理解 `barrier_arrive()` / `barrier_wait()` 在cluster级别的作用
- 了解新增的分区方法：`labeled_partition` 和 `binary_partition`
- 掌握异步集合操作（reduce和scan）
- 理解 `invoke_one()` 和 `invoke_one_broadcast()` 新API
- 掌握grid同步和协作启动

---

## 16.2 CUDA 11.5以来的新功能

CG自CUDA 11.5以来经历了显著扩展。以下是各版本的关键新增功能：

### 16.2.1 CUDA 12.0（从实验性转为正式）

- 异步reduce和scan操作从实验性命名空间移入主命名空间
- 大于32的 `thread_block_tile` 在CC 8.0+上不再需要 `block_tile_memory` 对象

### 16.2.2 CUDA 12.1

- 新增 `invoke_one()` 和 `invoke_one_broadcast()` API

### 16.2.3 CUDA 12.2

- 为 `grid_group` 和 `thread_block` 新增 `barrier_arrive()` / `barrier_wait()` 成员函数

### 16.2.4 CUDA 13.0

- 移除了 `multi_grid_group`（多网格组已不再支持）

### 16.2.5 组类型层次体系

CG的组类型形成了一个层次结构：

```

thread_group（抽象基类）
    ├── coalesced_group（warp中活跃线程）
    ├── thread_block_tile<N>（编译期大小的tile）
    ├── thread_block（线程块中所有线程）
    ├── cluster_group（集群中所有线程/块）[CC 9.0+]
    └── grid_group（网格中所有线程）[需要cooperative launch]
```

---

## 16.3 Cluster Group 详解

> "This group object represents all the threads launched in a single cluster."
> -- CUDA C++ Programming Guide 13.0

`cluster_group` 是CG在Hopper架构（CC 9.0+）上引入的最重要的新组类型。它代表了<strong>一个线程块簇（Thread Block Cluster）中所有线程</strong>的集合。

<strong>兼容性说明</strong>：`cluster_group` 的API在所有CC 9.0+硬件上可用。当以非集群方式启动网格时，这些API假定一个1x1x1的集群。

### 16.3.1 构造Cluster Group

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void clusterKernel() {
    // 获取当前集群组
    cg::cluster_group cluster = cg::this_cluster();

    // 集群内同步
    cluster.sync();

    // ... 使用分布式共享内存
}
```

### 16.3.2 Cluster Group成员函数详解

以下是 `cluster_group` 的完整成员函数列表：

| 成员函数 | 返回类型 | 说明 |
|---------|---------|------|
| `sync()` | `void` | 集群级别同步，等价于 `barrier_wait(barrier_arrive())` |
| `barrier_arrive()` | `arrival_token` | 到达集群屏障，返回token |
| `barrier_wait(token&&)` | `void` | 等待集群屏障，接收arrive返回的token |
| `thread_rank()` | `static unsigned int` | 调用线程在集群中的排名 [0, num_threads) |
| `block_rank()` | `static unsigned int` | 调用线程所在块在集群中的排名 [0, num_blocks) |
| `num_threads()` | `static unsigned int` | 集群中的总线程数 |
| `num_blocks()` | `static unsigned int` | 集群中的总线程块数 |
| `dim_blocks()` | `static dim3` | 集群的线程块维度 |
| `block_index()` | `static dim3` | 调用块在集群中的3D索引 |
| `query_shared_rank(const void *addr)` | `static unsigned int` | 查询共享内存地址属于哪个块 |
| `map_shared_rank(T *addr, int rank)` | `static T*` | 获取集群中另一个块的共享内存地址映射 |

### 16.3.3 分布式共享内存操作

`cluster_group` 的两个独特成员函数是 `query_shared_rank` 和 `map_shared_rank`，它们支持<strong>分布式共享内存（Distributed Shared Memory, DSM）</strong>操作：

#### query_shared_rank

查询给定共享内存地址属于集群中哪个线程块：

```cuda
__global__ void dsmKernel() {
    __shared__ float myData[256];
    cg::cluster_group cluster = cg::this_cluster();

    // 查询myData的首地址属于哪个块
    unsigned int ownerBlock = cluster.query_shared_rank(&myData[0]);

    // ownerBlock 现在包含拥有该地址的块的排名
}
```

#### map_shared_rank

将当前块中的共享内存地址映射为集群中另一个块的对应地址，使得可以直接读写远程块的共享内存：

```cuda
__global__ void dsmAccessKernel() {
    __shared__ unsigned int localCounter;
    cg::cluster_group cluster = cg::this_cluster();

    unsigned int myBlockRank = cluster.block_rank();

    if (myBlockRank == 0) {
        // 块0：初始化计数器
        localCounter = 0;
    }
    cluster.sync();

    // 每个块递增块0中的计数器
    // 需要获取块0中localCounter的映射地址
    float *remoteCounter = cluster.map_shared_rank(
        &localCounter, 0  // 目标块rank = 0
    );

    // 跨块原子加操作
    atomicAdd(remoteCounter, 1);
    cluster.sync();

    if (myBlockRank == 0) {
        printf("Total increments: %u (should be %u)\n",
            localCounter, cluster.num_blocks());
    }
}
```

### 16.3.4 barrier_arrive / barrier_wait 在Cluster级别

与 `thread_block` 的 `barrier_arrive()/barrier_wait()` 类似，`cluster_group` 也支持分离式的屏障操作。这允许在集群级别实现<strong>重叠计算与同步</strong>的模式：

```cuda
__global__ void clusterAsyncKernel() {
    cg::cluster_group cluster = cg::this_cluster();
    __shared__ float stage1Data[256];
    __shared__ float stage2Data[256];

    // 阶段1：计算
    stage1Data[threadIdx.x] = computeStage1(threadIdx.x);

    // 到达屏障但不等待——开始阶段2的独立工作
    auto token = cluster.barrier_arrive();

    // 阶段2的独立准备工作（不依赖其他块的阶段1结果）
    prepareStage2Locally();

    // 现在必须等待所有块完成阶段1
    cluster.barrier_wait(std::move(token));

    // 使用其他块的阶段1结果
    float peerData = *cluster.map_shared_rank(
        &stage1Data[0],
        (cluster.block_rank() + 1) % cluster.num_blocks()
    );
    stage2Data[threadIdx.x] = computeStage2(stage1Data[threadIdx.x], peerData);
}
```

这种分离式屏障使得我们可以在等待集群同步的同时执行不依赖于同步结果的本地工作。

---

## 16.4 新增分区方法

除了你已经熟悉的 `tiled_partition`，CG还引入了两种新的分区方法。

### 16.4.1 labeled_partition

`labeled_partition` 允许基于<strong>标签</strong>将父组划分为子组。具有相同标签的线程被分到同一组中：

```cuda
__global__ void labeledPartitionKernel() {
    cg::thread_block block = cg::this_thread_block();

    // 基于线程rank的奇偶性进行分区
    // 标签0：偶数rank线程；标签1：奇数rank线程
    unsigned int label = threadIdx.x % 2;

    auto subgroup = cg::labeled_partition(block, label);

    // subgroup现在包含所有具有相同label的线程
    if (subgroup.thread_rank() == 0) {
        printf("Label %u, group size: %u\n",
               label, subgroup.num_threads());
    }
}
```

`labeled_partition` 对于实现基于数据特征的动态分组特别有用。

### 16.4.2 binary_partition

`binary_partition` 基于<strong>谓词（predicate）</strong>将父组划分为两个子组——谓词为真的线程属于一个组，谓词为假的线程属于另一个组：

```cuda
__global__ void binaryPartitionKernel(float *data, int n) {
    cg::thread_block block = cg::this_thread_block();

    // 基于数据值是否大于阈值进行二分
    bool predicate = (data[threadIdx.x] > THRESHOLD);

    auto subgroup = cg::binary_partition(block, predicate);

    // 每个子组独立执行
    if (predicate) {
        // 高值组——执行操作A
        processHighValue(subgroup, data);
    } else {
        // 低值组——执行操作B
        processLowValue(subgroup, data);
    }
}
```

<strong>注意</strong>：`labeled_partition` 和 `binary_partition` 是更灵活的组划分方式，它们允许基于运行时条件而非仅基于线程索引进行分组。这在实现自适应算法时特别有价值。

---

## 16.5 异步Reduce和Scan集合操作

> CUDA 11.7引入了异步reduce和scan的实验性API，CUDA 12.0将它们移入主命名空间。

异步reduce和scan操作允许<strong>计算与归约/扫描重叠</strong>，这与同步版本（所有线程必须等待操作完成）不同。异步操作使用 `cuda::barrier` 或 `cuda::pipeline` 作为完成机制。

### 16.5.1 同步 vs 异步Reduce

<strong>传统的同步Reduce</strong>：

```cuda
#include <cooperative_groups/reduce.h>

__global__ void syncReduceKernel(float *input, float *output, int n) {
    cg::thread_block block = cg::this_thread_block();
    float threadVal = input[threadIdx.x];

    // 所有线程必须等待reduce完成
    float sum = cg::reduce(block, threadVal, cg::plus<float>());

    if (block.thread_rank() == 0) {
        *output = sum;
    }
}
```

<strong>异步Reduce</strong>：

```cuda
#include <cooperative_groups/reduce.h>
#include <cuda/barrier>

__global__ void asyncReduceKernel(float *input, float *output, int n) {
    extern __shared__ float s_data[];

    cg::thread_block block = cg::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;

    if (block.thread_rank() == 0) {
        init(&barrier, block.size());
    }
    block.sync();

    float threadVal = input[threadIdx.x];
    s_data[threadIdx.x] = threadVal;

    // 发起异步reduce——不阻塞
    cg::async_reduce(block, barrier,
                     s_data,         // 目标（共享内存）
                     s_data,         // 源（共享内存）
                     block.size(),   // 元素数量
                     cg::plus<float>());

    // 在等待reduce完成的同时，执行其他独立计算
    float localResult = doSomeIndependentCalculation(threadIdx.x);
    barrier.arrive_and_wait();
    if (block.thread_rank() == 0) {
        *output = s_data[0];
    }
}
```

### 16.5.2 异步Scan（前缀和）

```cuda
#include <cooperative_groups/scan.h>

__global__ void asyncScanKernel(int *input, int *output, int n) {
    cg::thread_block block = cg::this_thread_block();
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    __shared__ int s_data[256];

    if (block.thread_rank() == 0) {
        init(&barrier, block.size());
    }
    block.sync();

    s_data[threadIdx.x] = input[threadIdx.x];

    // 异步exclusive_scan
    cg::async_exclusive_scan(block, barrier,
                             s_data,
                             s_data,
                             block.size(),
                             cg::plus<int>());

    // 在scan完成前执行其他工作
    preprocessData(s_data, threadIdx.x);

    barrier.arrive_and_wait();

    output[threadIdx.x] = s_data[threadIdx.x];
}
```

### 16.5.3 异步操作的优势与适用场景

| 方面 | 同步操作 | 异步操作 |
|------|---------|---------|
| 线程阻塞 | 所有线程必须等待 | 可以在等待期间执行其他工作 |
| 计算重叠 | 不支持 | 支持计算与归约/扫描重叠 |
| 编程复杂度 | 简单 | 需要管理barrier生命周期 |
| 性能 | 在无独立工作时高效 | 当有独立工作可执行时更优 |
| 硬件要求 | 所有CC版本 | 推荐CC 8.0+ |

---

## 16.6 invoke_one 和 invoke_one_broadcast

CUDA 12.1引入了两个新的集合操作API，用于在组内选择一个线程执行特定操作。

### 16.6.1 invoke_one

`invoke_one` 在组内<strong>选择一个线程</strong>执行给定的函数：

```cuda
__global__ void invokeOneExample(int *globalData, int value) {
    cg::thread_block block = cg::this_thread_block();

    // 只选一个线程来执行全局内存写入
    cg::invoke_one(block, [&]() {
        *globalData = value;
        printf("Written by thread %d\n", threadIdx.x);
    });

    // 其他线程可以继续执行
    // ...
}
```

### 16.6.2 invoke_one_broadcast

`invoke_one_broadcast` 选择一个线程执行函数，然后将<strong>返回值广播</strong>给组内所有线程：

```cuda
__global__ void invokeOneBroadcastExample(int *data, int n) {
    cg::thread_block block = cg::this_thread_block();

    // 选一个线程计算除数（例如计算全局最大值）
    int divisor = cg::invoke_one_broadcast(block, [&]() -> int {
        int maxVal = data[0];
        for (int i = 1; i < n; i++) {
            if (data[i] > maxVal) maxVal = data[i];
        }
        return maxVal;
    });

    // 所有线程现在都有相同的divisor值
    float normalized = (float)data[threadIdx.x] / divisor;
    // ...
}
```

<strong>硬件加速</strong>：在CC 9.0+设备上，当使用<strong>显式组类型</strong>（如 `thread_block_tile`）时，`invoke_one` 可能会使用硬件加速来选择执行线程。

<strong>代码生成要求</strong>：最低CC 5.0，CC 9.0用于硬件加速，需要C++11。

---

## 16.7 Grid同步与协作启动

### 16.7.1 grid_group.sync()

`grid_group` 代表了单次kernel启动中<strong>所有线程块的所有线程</strong>。与 `thread_block` 和 `cluster_group` 不同，跨整个网格的同步不是"免费"的——它需要使用<strong>协作启动（cooperative launch）API</strong>。

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void gridSyncKernel(int *data, int n) {
    cg::grid_group grid = cg::this_grid();

    // 阶段1：每个块处理自己的一段数据
    int blockStart = blockIdx.x * blockDim.x;
    int blockEnd = min(blockStart + blockDim.x, n);
    for (int i = blockStart + threadIdx.x; i < blockEnd; i += blockDim.x) {
        data[i] = processStage1(data[i]);
    }

    // 全局同步——所有线程块必须在此汇合
    grid.sync();

    // 阶段2：所有块可以读取任何其他块在阶段1写入的数据
    processStage2(data, n, grid);
}
```

<strong>协作启动</strong>：

```cuda
// 主机端代码
cudaLaunchConfig_t config = {0};
config.gridDim = gridDim;
config.blockDim = blockDim;
config.dynamicSmemBytes = sharedMemSize;

// 必须使用协同启动API
void *args[] = {&data, &n};
cudaLaunchCooperativeKernel((void*)gridSyncKernel,
                            config.gridDim, config.blockDim,
                            args, config.dynamicSmemBytes, stream);
```

### 16.7.2 grid_group的barrier_arrive / barrier_wait

CUDA 12.2为 `grid_group` 新增了 `barrier_arrive()` 和 `barrier_wait()` 成员函数：

```cuda
__global__ void gridAsyncKernel() {
    cg::grid_group grid = cg::this_grid();

    // ... 阶段1计算 ...

    auto token = grid.barrier_arrive();

    // 本地独立工作
    localIndependentWork();

    grid.barrier_wait(std::move(token));

    // ... 阶段2计算 ...
}
```

### 16.7.3 协作启动的限制和要求

- 协作启动需要确保GPU上有足够的资源同时驻留所有线程块。
- 可以通过 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 查询每个SM能容纳的最大活跃块数。
- 网格中的线程块总数不能超过设备支持的最大活跃块数。
- 使用 `cudaLaunchCooperativeKernel` 是必须的——普通的 `<<<>>>` 启动语法不支持网格同步。

### 16.7.4 Multi-Grid同步（已移除）

CUDA 13.0移除了 `multi_grid_group`。在需要跨多个GPU同步的场景中，建议使用NCCL或其他通信库。CUDA的核心CG库专注于单GPU内的线程组织。

---

## 16.8 最佳实践与使用建议

### 16.8.1 何时使用Cooperative Groups

<strong>使用CG的场景</strong>：

1. <strong>需要warp级操作</strong>：当需要 `shfl`、`ballot`、`match_any` 等warp内置函数时，使用 `thread_block_tile<32>` 比原始的 `__shfl_sync` 更安全。
2. <strong>集体算法</strong>：实现reduce、scan等需要多线程协作的算法时，使用CG的集合操作可以自动选择最优实现。
3. <strong>分布式共享内存</strong>：在CC 9.0+上使用 `cluster_group` 实现跨块数据共享。
4. <strong>网格同步</strong>：需要跨线程块同步的算法（如全规约）。

<strong>不推荐CG的场景</strong>：

1. <strong>简单的单线程操作</strong>：每线程独立计算且无需同步的场景，使用CG反而增加复杂度。
2. <strong>性能极度敏感的热路径</strong>：在某些情况下，直接使用PTX内置函数可能比CG的通用接口略快（但差异通常很小）。

### 16.8.2 性能建议

> "To write efficient code, its best to use specialized groups (going generic loses a lot of compile time optimizations), and pass these group objects by reference to functions that intend to use these threads in some cooperative fashion."
> -- CUDA C++ Programming Guide 13.0

1. <strong>使用特化的组类型</strong>：尽量使用 `thread_block_tile<32>` 而不是通用的 `thread_group`，因为编译器可以针对特化类型做更多优化。

2. <strong>按引用传递组对象</strong>：

```cuda
// 好——按引用传递
__device__ float blockSum(const cg::thread_block &block,
                          float val) {
    // ...
}

// 不好——按值传递（会拷贝组对象）
__device__ float blockSum(cg::thread_block block, float val) {
    // ...
}
```

3. <strong>尽早创建隐式组</strong>：在kernel开头（任何分支之前）创建隐式组句柄，因为创建隐式组是一个<strong>集合操作</strong>——如果并非所有线程都参与，可能导致死锁或数据损坏。

```cuda
__global__ void goodPractice() {
    // 好——在kernel开头创建
    cg::thread_block block = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    // 现在可以在分支中使用这些句柄
    if (condition) {
        blockSum(block, data[threadIdx.x]);
    }
}
```

4. <strong>CC 8.0+上大于32的tile不需要block_tile_memory</strong>：在CC 8.0+硬件上，`block_tile_memory` 是可选的，不会消耗共享内存。这意味着你可以写一份无需条件编译的代码。

### 16.8.3 CG vs 原始原语的对比

| 特性 | 原始CUDA原语 | Cooperative Groups |
|------|-------------|-------------------|
| 块内同步 | `__syncthreads()` | `thread_block::sync()` |
| Warp shuffle | `__shfl_sync()` | `tile.shfl()` |
| Warp投票 | `__ballot_sync()` | `tile.ballot()` |
| Warp匹配 | `__match_any_sync()` | `tile.match_any()` |
| 大于32的tile | 手动实现（复杂） | `tiled_partition<64>(block)` |
| 集群同步 | 不支持 | `cluster_group::sync()` |
| 网格同步 | 需要间接方式 | `grid_group::sync()` |
| 分布式共享内存 | 不支持 | `map_shared_rank` / `query_shared_rank` |
| 异步集合操作 | 需要手动实现 | `async_reduce` / `async_scan` |
| 跨架构可移植性 | 需要条件编译 | 自动适配 |

---

## 16.9 综合案例：分布式直方图计算

让我们通过一个综合示例来展示 `cluster_group` 和分布式共享内存的实际应用。我们将实现一个<strong>分布式直方图</strong>计算——利用集群内多个线程块的共享内存共同构建一个更大的直方图。

```cuda
#include <cooperative_groups.h>
#include <cuda/barrier>
namespace cg = cooperative_groups;

#define NUM_BINS 256
#define BLOCK_SIZE 256

__global__ void distributedHistogram(
    const unsigned int *input, int N,
    unsigned int *globalHistogram)
{
    // 获取集群组
    cg::cluster_group cluster = cg::this_cluster();
    unsigned int numBlocks = cluster.num_blocks();
    unsigned int myBlockRank = cluster.block_rank();

    // 每个块分配相同大小的共享内存用于直方图
    __shared__ unsigned int localBins[NUM_BINS];

    // 初始化本地共享内存
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        localBins[i] = 0;
    }
    cluster.sync();

    // 分布式全局索引分配
    // 每个块处理一部分数据
    unsigned int itemsPerBlock = (N + numBlocks - 1) / numBlocks;  // 向上取整，避免数据丢失
    unsigned int blockStart = myBlockRank * itemsPerBlock;
    unsigned int blockEnd = min(blockStart + itemsPerBlock, (unsigned int)N);

    // 阶段1：本地直方图计算
    for (int i = blockStart + threadIdx.x; i < blockEnd; i += blockDim.x) {
        unsigned int bin = input[i] % NUM_BINS;
        atomicAdd(&localBins[bin], 1);
    }
    cluster.sync();

    // 预先获取所有块的共享内存地址（避免循环中重复调用map_shared_rank）
    __shared__ unsigned int *remoteBins[32]; // 集群最大32个块
    if (threadIdx.x == 0) {
        for (unsigned int b = 0; b < numBlocks; b++) {
            remoteBins[b] = cluster.map_shared_rank(localBins, b);
        }
    }
    cluster.sync();

    // 阶段2：使用分布式共享内存进行全局合并
    // 每个块负责合并一定数量的bin
    unsigned int binsPerBlock = (NUM_BINS + numBlocks - 1) / numBlocks;  // 向上取整
    unsigned int myBinStart = myBlockRank * binsPerBlock;
    unsigned int myBinEnd = min(myBinStart + binsPerBlock, (unsigned int)NUM_BINS);

    for (unsigned int bin = myBinStart + threadIdx.x;
         bin < myBinEnd; bin += blockDim.x) {
        unsigned int total = 0;

        // 从集群中所有块收集该bin的计数
        for (unsigned int b = 0; b < numBlocks; b++) {
            total += remoteBins[b][bin];
        }

        // 写入全局内存
        globalHistogram[bin] = total;
    }
}
```

<strong>关键分析</strong>：

1. <strong>分布式数据分配</strong>：每个线程块处理输入数据的一部分（`itemsPerBlock`），实现了集群级别的数据并行。
2. <strong>本地直方图</strong>：每个块在自己的共享内存中构建局部直方图，使用原子操作处理冲突。
3. <strong>分布式合并</strong>：使用 `cluster.map_shared_rank` 访问其他块的共享内存，实现跨块的直方图合并。每个块只负责最终直方图中一段bin的合并。
4. <strong>集群同步</strong>：`cluster.sync()` 确保在访问远程共享内存之前，所有块的本地直方图已经完成。

---

## 16.10 本章小结

本章深入介绍了Cooperative Groups在CUDA 11.5之后的重要扩展。我们重点讲解了：

1. <strong>Cluster Group（`cluster_group`）</strong>：CC 9.0+引入的集群级组类型，支持集群内同步、分布式共享内存操作（`map_shared_rank` / `query_shared_rank`）以及分离式屏障（`barrier_arrive` / `barrier_wait`）。

2. <strong>新增分区方法</strong>：`labeled_partition` 和 `binary_partition` 提供了基于标签和谓词的灵活分组能力。

3. <strong>异步Reduce和Scan</strong>：允许计算与归约/扫描重叠，利用 `cuda::barrier` 作为完成机制，在CC 8.0+上有硬件加速。

4. <strong>invoke_one 和 invoke_one_broadcast</strong>：在组内选一个线程执行特定操作，并可选择将结果广播给全组。

5. <strong>Grid同步</strong>：`grid_group.sync()` 配合协作启动实现跨整个网格的全局同步。

6. <strong>最佳实践</strong>：使用特化组类型、按引用传递、尽早创建隐式组等关键建议。

Cooperative Groups是现代CUDA编程中实现高效、可维护、跨架构兼容的协作代码的基础设施。随着GPU架构的不断发展，CG的抽象能力将越来越重要。

---

## 16.11 习题

1. 编写一个使用 `cluster_group` 和分布式共享内存的程序，实现集群内多个线程块的<strong>归约求和</strong>。对比使用集群与不使用集群（每个块独立做归约后用原子操作合并到全局内存）的性能差异。

2. 使用 `labeled_partition` 实现一个基于数据特征（如元素奇偶性）的分组操作。每个子组执行不同的计算，最后将结果写回。

3. 将课堂上学习的一个同步归约算法改写为使用 `async_reduce` 的异步版本。分析哪些计算可以在等待归约完成的同时执行。

4. 使用 `invoke_one_broadcast` 在一个线程块中找到<strong>中位数</strong>，并将结果广播给所有线程。与手动实现相比，代码复杂度有何变化？

5. 阅读CUDA Programming Guide第11章中关于"Grid Synchronization"的协作启动限制说明。编写一个需要网格同步的简单kernel，并验证在使用普通启动API时 `grid.is_valid()` 的返回值。

---

## 16.12 参考文献

1. NVIDIA CUDA C++ Programming Guide 13.0, Chapter 11 — Cooperative Groups
   - 11.2 — What's New in Cooperative Groups
   - 11.4.1.2 — Cluster Group
   - 11.4.1.3 — Grid Group
   - 11.5 — Asynchronous Data Copies (memcpy_async within CG)
   - 11.6 — Asynchronous Reduce and Scan
2. NVIDIA CUDA C++ Programming Guide 13.0, Chapter 5.2.1 — Thread Block Clusters
3. NVIDIA CUDA C++ Programming Guide 13.0, Chapter 6.2.5 — Distributed Shared Memory
4. NVIDIA Developer Blog — "Cooperative Groups: Flexible CUDA Thread Programming"
