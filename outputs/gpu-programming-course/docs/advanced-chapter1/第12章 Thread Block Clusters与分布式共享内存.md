# 第12章 Thread Block Clusters与分布式共享内存

<strong>硬件要求</strong>：Compute Capability 9.0+（NVIDIA Hopper H100 及更新架构）

## 12.1 引言

在之前的章节中，我们学习了 CUDA 编程模型中最核心的线程层次结构：线程（Thread）组成线程束（Warp），线程束组成线程块（Thread Block），线程块组成网格（Grid）。这套层次结构自 CUDA 诞生以来一直是 GPU 编程的基础。

然而，随着 NVIDIA Hopper 架构（Compute Capability 9.0）的推出，CUDA 编程模型引入了一个全新的可选层次——<strong>线程块簇（Thread Block Cluster）</strong>。CUDA Programming Guide 对此给出了精炼的定义：

> "With the introduction of NVIDIA Compute Capability 9.0, the CUDA programming model introduces an optional level of hierarchy called Thread Block Clusters that are made up of thread blocks. Similar to how threads in a thread block are guaranteed to be co-scheduled on a streaming multiprocessor, thread blocks in a cluster are also guaranteed to be co-scheduled on a GPU Processing Cluster (GPC) in the GPU."

这一新层次的引入不仅仅是"多了一个层级"那么简单，它带来了三个革命性的能力：

1. <strong>簇内线程块同步</strong>：通过 `cluster.sync()` 实现跨线程块的硬件级同步；
2. <strong>分布式共享内存（Distributed Shared Memory）</strong>：一个簇中所有线程块的共享内存组成统一的分布式地址空间，任何线程块都可以读写其他块的共享内存；
3. <strong>跨块原子操作</strong>：在分布式共享内存地址空间中执行原子操作。

在本章中，我们将深入探究 Thread Block Clusters 的方方面面——从概念、启动方式，到分布式共享内存的使用，最后通过实战案例展示这一特性的强大威力。

## 12.2 CUDA 层次结构的扩展

### 12.2.1 从 Grid 到 Cluster 的演进

回顾我们熟悉的 CUDA 线程层次：

<div align="center"><img src="../images/advanced-chapter1-figures/grid-of-clusters.png" /><p>图 12.1 Grid of Thread Block Clusters（来源：CUDA Programming Guide）</p></div>

在引入 Cluster 之前，CUDA 的层次结构是严格的三层：

```
Grid → Block → Warp → Thread
```

在引入 Cluster 之后，层次结构扩展为四层：

```
Grid → Cluster → Block → Warp → Thread
```

与线程块类似，簇也可以在三个维度上组织：

- <strong>簇维度（Cluster Dimensions）</strong>：定义每个簇包含多少个线程块（最大可移植值为 8）；
- <strong>簇网格（Grid of Clusters）</strong>：簇本身在一个一维、二维或三维的网格中排列。

### 12.2.2 共调度保证（Co-scheduling Guarantee）

Cluster 最核心的硬件保证是<strong>共调度</strong>（co-scheduling）。CUDA Programming Guide 明确指出：

> "In GPUs with compute capability 9.0, all the thread blocks in the cluster are guaranteed to be co-scheduled on a single GPU Processing Cluster (GPC) and allow thread blocks in the cluster to perform hardware-supported synchronization using the Cluster Group API `cluster.sync()`."

这意味着：

1. 同一个 Cluster 中的所有线程块会被调度到单个 GPC 上；
2. 它们会<strong>同时存在</strong>（co-resident），而不仅仅是先后调度；
3. 这使得跨线程块的同步成为可能——如果没有共调度保证，`cluster.sync()` 将可能导致死锁。

类比：传统的 Grid 启动中，线程块可能分布在整个 GPU 的不同 SM 上，它们之间的执行顺序和存在时间是不可预测的。而 Cluster 提供了"同在一个屋檐下"的保证——所有簇内线程块在同一个 GPC 内同时运行。

### 12.2.3 可移植簇大小与查询

CUDA 定义的最大可移植簇大小为 <strong>8 个线程块</strong>。这个限制是因为并非所有 GPU 配置都有足够多的 SM 来支持更大的簇。在 MIG（Multi-Instance GPU）配置或物理规模较小的 GPU 上，最大簇大小还会进一步减小。

可以通过 `cudaOccupancyMaxPotentialClusterSize` API 查询当前设备支持的最大簇大小：

```cpp
int maxClusterSize = 0;
cudaOccupancyMaxPotentialClusterSize(&maxClusterSize, myKernel, config);
```

CUDA Programming Guide 原文指出：

> "Note that on GPU hardware or MIG configurations which are too small to support 8 multiprocessors the maximum cluster size will be reduced accordingly. Identification of these smaller configurations, as well as of larger configurations supporting a thread block cluster size beyond 8, is architecture-specific and can be queried using the `cudaOccupancyMaxPotentialClusterSize` API."

## 12.3 簇的启动方式

Thread Block Cluster 提供了两种启动方式：编译期固定大小和运行时动态配置。灵活选择这两种方式可以满足不同场景的需求。

### 12.3.1 编译期簇大小：__cluster_dims__

第一种方式是在核函数定义时通过 `__cluster_dims__(X, Y, Z)` 属性直接指定簇的大小。这种方式简单直接，类似于我们使用 `&lt;&lt;&lt;grid, block&gt;&gt;&gt;` 启动核函数的传统语法：

```cuda
// 编译期簇大小：X维度2，Y和Z维度各1
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float *output)
{
    // 核函数体
}

int main()
{
    float *input, *output;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // 网格维度不受簇启动影响，仍以线程块数量表示
    // 网格维度必须是簇大小的整数倍
    cluster_kernel&lt;&lt;&lt;numBlocks, threadsPerBlock&gt;&gt;&gt;(input, output);
}
```

使用编译期簇大小时需要注意几点：

1. 簇大小在编译时就固定了，后续启动时无法修改；
2. 网格维度（`gridDim`）仍然表示线程块的数量，与不使用 Cluster 时保持一致；
3. 网格维度必须是簇大小的整数倍——例如如果簇在 X 方向大小为 2，那么 `numBlocks.x` 必须是 2 的倍数。

CUDA Programming Guide 特别提到：

> "In a kernel launched using cluster support, the gridDim variable still denotes the size in terms of number of thread blocks, for compatibility purposes. The rank of a block in a cluster can be found using the Cluster Group API."

### 12.3.2 运行时簇大小：cudaLaunchKernelEx

第二种方式是通过可扩展的启动 API `cudaLaunchKernelEx` 在运行时动态指定簇大小。这种方式更加灵活，允许程序根据问题规模或硬件能力动态调整簇配置：

```cuda
// 注意：核函数不带 __cluster_dims__ 编译期属性
__global__ void cluster_kernel(float *input, float *output)
{
    // 核函数体
}

int main()
{
    float *input, *output;
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    {
        cudaLaunchConfig_t config = {0};
        config.gridDim = numBlocks;       // 仍以线程块数量表示
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // 运行时指定 X 方向簇大小
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, cluster_kernel, input, output);
    }
}
```

使用 `cudaLaunchKernelEx` 时：

1. 核函数不带有 `__cluster_dims__` 属性；
2. 通过 `cudaLaunchAttributeClusterDimension` 属性指定运行时簇维度；
3. 同样要求网格维度是簇大小的整数倍；
4. 可以更方便地根据运行时条件动态调整簇大小。

### 12.3.3 __block_size__ 属性——以簇数量启动

CUDA 还提供了另一种编译期属性 `__block_size__`，它允许你以簇的数量（而非线程块的数量）来指定启动配置。这种视角在某些场景下更加直观：

```cuda
// 第一个元组：每个线程块的维度（1024, 1, 1）
// 第二个元组：每个簇包含的线程块维度（2, 2, 2）
__block_size__((1024, 1, 1), (2, 2, 2)) __global__ void foo();

// 启动时以簇的数量指定 —— 8x8x8 个簇
foo&lt;&lt;&lt;dim3(8, 8, 8)&gt;&gt;&gt;();
```

CUDA Programming Guide 的解释：

> "`__block_size__` requires two fields each being a tuple of 3 elements. The first tuple denotes block dimension and second cluster size. The second tuple is assumed to be `(1,1,1)` if it's not passed."

需要注意的是：

1. `__block_size__` 和 `__cluster_dims__` 不能同时使用——它们是互斥的；
2. 当 `__block_size__` 的第二个元组被指定时，编译器会将 `&lt;&lt;&lt;&gt;&gt;&gt;` 的第一个参数理解为簇的数量而非线程块的数量；
3. 要指定流，必须在 `&lt;&lt;&lt;&gt;&gt;&gt;` 中传递 `1` 和 `0` 作为第二、第三个参数，然后才是流对象。

### 12.3.4 两种启动方式的对比

| 特性 | `__cluster_dims__` | `cudaLaunchKernelEx` |
|------|-------------------|---------------------|
| 簇大小确定时机 | 编译期 | 运行时 |
| API 复杂度 | 低（传统 `&lt;&lt;&lt;&gt;&gt;&gt;` 语法） | 中（需配置 `cudaLaunchConfig_t`） |
| 灵活性 | 低（编译期固定） | 高（可动态调整） |
| `__block_size__` 支持 | 互斥，但可用 `__block_size__` 替代 | 互斥 |
| 兼容性 | CC 9.0+ | CC 9.0+ |

## 12.4 Cluster Group API

在核函数内部，我们通过 Cooperative Groups 库中的 `cluster_group` 来与 Cluster 交互。以下是构造和使用 Cluster Group 的基本方式：

```cuda
#include &lt;cooperative_groups.h&gt;

__global__ void __cluster_dims__(2, 1, 1) my_kernel() {
    namespace cg = cooperative_groups;

    // 获取当前簇的 Cluster Group 对象
    cg::cluster_group cluster = cg::this_cluster();

    // 查询信息
    unsigned int blockRank = cluster.block_rank();    // 当前块在簇中的线性排名
    dim3 blockIndex = cluster.block_index();           // 当前块在簇中的3D索引
    unsigned int numBlocks = cluster.num_blocks();      // 簇中总块数
    dim3 dims = cluster.dim_blocks();                   // 簇的3D维度
    unsigned int numThreads = cluster.num_threads();    // 簇中总线程数
    dim3 threadDims = cluster.dim_threads();            // 簇的线程3D维度

    // 簇内同步
    cluster.sync();  // 确保簇中所有线程块都已启动并到达此处
}
```

CUDA Programming Guide 给出了 Cluster Group API 的适用范围：

> "The APIs are available on all hardware with Compute Capability 9.0+. In such cases, when a non-cluster grid is launched, the APIs assume a 1x1x1 cluster."

这意味着即使你不显式启动 Cluster，`this_cluster()` 也会返回一个 1x1x1 的 "单例簇"，这保证了代码的可移植性。

## 12.5 分布式共享内存（Distributed Shared Memory）

分布式共享内存是 Thread Block Cluster 带来的最强大的特性之一。它允许簇中每个线程块访问其他所有线程块的共享内存，就好像它们属于一个统一的地址空间。

### 12.5.1 概念与地址空间

CUDA Programming Guide 对分布式共享内存的定义：

> "Thread block clusters introduced in compute capability 9.0 provide the ability for threads in a thread block cluster to access shared memory of all the participating thread blocks in a cluster. This partitioned shared memory is called Distributed Shared Memory, and the corresponding address space is called Distributed shared memory address space."

分布式共享内存的关键特性：

1. <strong>统一地址空间</strong>：簇中所有线程块的共享内存在逻辑上组成统一的分布式地址空间，但各块的共享内存在物理上是独立的。访问远程线程块的共享内存必须通过 `map_shared_rank()` 进行地址映射，不能简单地做线性偏移寻址。
2. <strong>总大小</strong> = 每个线程块的共享内存大小 × 簇中线程块的数量；
3. <strong>完全可访问</strong>：任何线程块可以读、写、或在分布式共享内存的任何地址上执行原子操作，无论该地址属于本地线程块还是远程线程块；
4. <strong>每块共享内存大小不变</strong>：无论是否使用分布式共享内存，静态或动态共享内存的大小规格仍然是<strong>每线程块</strong>的。

### 12.5.2 query_shared_rank 与 map_shared_rank

分布式共享内存提供了两个关键的 API 来实现跨块访问：

<strong>`query_shared_rank(addr)`</strong>：给定分布式共享内存中的一个地址，查询该地址属于簇中哪个线程块的共享内存。返回该线程块的排名（rank）。

<strong>`map_shared_rank(addr, rank)`</strong>：给定一个本地共享内存地址和一个目标线程块的 rank，返回该地址映射到目标块共享内存中的对应指针。这是实现跨块访问的核心函数。

使用模式：

```cuda
cg::cluster_group cluster = cg::this_cluster();

// smem 是当前线程块的共享内存指针
extern __shared__ int smem[];

// 将 smem 映射到 rank=1 的远程线程块的共享内存
int *remote_smem = cluster.map_shared_rank(smem, 1);

// 现在 remote_smem 指向远程块的共享内存，可以直接读写
remote_smem[threadIdx.x] = value;  // 写入远程块的共享内存

// 也可以用原子操作
atomicAdd(remote_smem + offset, 1); // 对远程块共享内存做原子加
```

### 12.5.3 同步要求

使用分布式共享内存时，有两个关键的同步点必须注意：

1. <strong>访问前同步</strong>：在访问分布式共享内存之前，必须确保簇中所有线程块都已经启动并且并发存在。这通过调用 `cluster.sync()` 实现。

2. <strong>退出前同步</strong>：在线程块退出之前，必须确保所有对本地共享内存的远程访问已经完成。同样通过 `cluster.sync()` 来实现。

CUDA Programming Guide 原文：

> "Accessing data in distributed shared memory requires all the thread blocks to exist. A user can guarantee that all thread blocks have started executing using `cluster.sync()` from Cluster Group API. The user also needs to ensure that all distributed shared memory operations happen before the exit of a thread block, e.g., if a remote thread block is trying to read a given thread block's shared memory, user needs to ensure that the shared memory read by remote thread block is completed before it can exit."

## 12.6 跨块原子操作

在分布式共享内存中，原子操作同样可以跨线程块执行。这意味着一个线程块中的线程可以使用 `atomicAdd`、`atomicExch`、`atomicCAS` 等原子函数直接操作另一个线程块的共享内存。

这在以下场景中特别有用：

- <strong>分布式直方图</strong>：将直方图的 bin 分散到多个块的共享内存中，减少共享内存容量限制；
- <strong>分布式归约</strong>：将部分和分布到多个块的共享内存中，减少同步开销；
- <strong>工作窃取</strong>：多个块共享一个任务队列（通过原子操作分配任务）。

跨块原子操作的语法与普通原子操作完全一致，唯一的区别是目标地址需要经过 `map_shared_rank` 映射：

```cuda
int *remote_smem = cluster.map_shared_rank(smem, target_rank);
atomicAdd(&remote_smem[offset], value);  // 对远程块共享内存做原子加法
```

## 12.7 实战案例：分布式直方图计算

直方图（Histogram）计算是一个经典的并行计算问题。传统的 GPU 实现受限于单个线程块的共享内存容量——当直方图 bin 的数量超出单个块的共享内存大小时，就不得不退回到全局内存原子操作，性能急剧下降。分布式共享内存提供了一个完美的中间方案。

### 12.7.1 传统方法的局限

传统 GPU 直方图计算有三种策略：

1. <strong>共享内存直方图</strong>（bin 数少）：每个线程块在本地共享内存中维护一个直方图副本，最后归约到全局内存。受限于每块共享内存容量（通常最大 228KB）。

2. <strong>全局内存直方图</strong>（bin 数多）：所有线程直接通过原子操作更新全局内存中的直方图。原子操作的高延迟成为瓶颈。

3. <strong>分布式共享内存直方图</strong>（新增）：利用 Cluster 的多块共享内存，将直方图 bin 分布到多个块的共享内存中。既避免了全局内存原子操作的延迟，又突破了单块共享内存的容量限制。

### 12.7.2 分布式直方图核函数

下面的代码来自 CUDA Programming Guide 6.2.5 节的官方示例，展示了如何使用分布式共享内存实现直方图计算：

```cuda
#include &lt;cooperative_groups.h&gt;

// 分布式共享内存直方图核函数
__global__ void clusterHist_kernel(int *bins, const int nbins,
                                   const int bins_per_block,
                                   const int *__restrict__ input,
                                   size_t array_size)
{
  extern __shared__ int smem[];
  namespace cg = cooperative_groups;
  int tid = cg::this_grid().thread_rank();

  // 簇初始化：获取簇大小和当前块在簇中的排名
  cg::cluster_group cluster = cg::this_cluster();
  unsigned int clusterBlockRank = cluster.block_rank();
  int cluster_size = cluster.dim_blocks().x;

  // 将本地共享内存直方图初始化为零
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    smem[i] = 0;
  }

  // 簇同步：确保所有线程块都已启动，且共享内存已初始化
  cluster.sync();

  // 遍历输入数据，更新分布式直方图
  for (int i = tid; i < array_size; i += blockDim.x * gridDim.x)
  {
    int ldata = input[i];

    // 确定直方图 bin 归属
    int binid = ldata;
    if (ldata < 0)
      binid = 0;
    else if (ldata >= nbins)
      binid = nbins - 1;

    // 确定目标线程块排名和偏移
    int dst_block_rank = (int)(binid / bins_per_block);
    int dst_offset = binid % bins_per_block;

    // 获取目标块的共享内存指针
    int *dst_smem = cluster.map_shared_rank(smem, dst_block_rank);

    // 对远程块的共享内存执行原子更新
    atomicAdd(dst_smem + dst_offset, 1);
  }

  // 簇同步：确保所有分布式共享内存操作完成
  cluster.sync();

  // 将本地分布式直方图归约到全局内存
  int *lbins = bins + cluster.block_rank() * bins_per_block;
  for (int i = threadIdx.x; i < bins_per_block; i += blockDim.x)
  {
    atomicAdd(&lbins[i], smem[i]);
  }
}
```

### 12.7.3 运行时簇启动

这个核函数可以通过 `cudaLaunchKernelEx` 根据直方图大小动态选择簇大小：

```cuda
// 根据直方图 bin 数量动态决定簇大小
{
  cudaLaunchConfig_t config = {0};
  config.gridDim = dim3((array_size + thr`eads_per_block - 1) / threads_per_block);
  config.blockDim = threads_per_block;

  // 簇大小取决于直方图 bin 数量
  // cluster_size == 1 表示不使用分布式共享内存，退化为传统单块方案
  int cluster_size = 2; // 这里以2为例

  int nbins_per_block = nbins / cluster_size;

  // 动态共享内存大小仍然是每块的
  // 分布式共享内存总大小 = cluster_size * nbins_per_block * sizeof(int)
  config.dynamicSmemBytes = nbins_per_block * sizeof(int);

  CUDA_CHECK(::cudaFuncSetAttribute(
      (void *)clusterHist_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      config.dynamicSmemBytes));

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = cluster_size;
  attribute[0].val.clusterDim.y = 1;
  attribute[0].val.clusterDim.z = 1;

  config.numAttrs = 1;
  config.attrs = attribute;

  cudaLaunchKernelEx(&config, clusterHist_kernel, bins, nbins,
                     nbins_per_block, input, array_size);
}
```

### 12.7.4 设计思路分析

让我们逐步分析这个核函数的设计思路：

1. <strong>共享内存初始化</strong>：每个线程块将其本地共享内存区域置零。

2. <strong>`cluster.sync()`</strong>：第一个同步点。确保所有块都已启动，且共享内存初始化完成。

3. <strong>分布式更新</strong>：每个线程读取全局内存中的输入数据，计算它属于哪个 bin（`binid`），然后：
   - `dst_block_rank = binid / bins_per_block`：确定该 bin 存储在哪个线程块的共享内存中；
   - `dst_offset = binid % bins_per_block`：确定在该块共享内存中的偏移；
   - `cluster.map_shared_rank(smem, dst_block_rank)`：获取目标块共享内存的本地指针；
   - `atomicAdd(dst_smem + dst_offset, 1)`：对远程块共享内存执行原子更新。

4. <strong>`cluster.sync()`</strong>：第二个同步点。确保所有远程访问完成，没有线程块会提前退出。

5. <strong>全局归约</strong>：每个块将其本地共享内存中的直方图部分通过原子操作累加到全局内存。

### 12.7.5 性能分析

分布式共享内存直方图相比传统方案的优势：

| 方法 | 容量限制 | 原子操作延迟 | 适用场景 |
|------|---------|------------|---------|
| 共享内存 | 受限于单块SMEM（~228KB） | 低（SMEM原子） | bin数量少 |
| 全局内存 | 几乎无限（HBM） | 高（GMEM原子） | bin数量多 |
| 分布式共享内存 | 中等（N×SMEM每块，N≤8） | 中（远程SMEM原子） | bin数量适中 |

当 bin 数量超出单块共享内存但不超过 N 倍单块共享内存（N≤8）时，分布式共享内存方案可以同时享受大容量和低延迟的优势。

## 12.8 深入理解：分布式归约（Distributed Reduction）

除了直方图计算，分布式共享内存的另一个重要应用场景是<strong>分布式归约</strong>。在传统的并行归约中，每个线程块独立完成归约，然后通过全局内存进行跨块合并。使用 DSM 后，我们可以让簇中的所有块协作完成归约，减少全局内存访问。

### 12.8.1 传统归约的限制

传统块内归约（使用共享内存的分治策略）的流程：

1. 每个线程块将数据从全局内存加载到共享内存；
2. 在共享内存内进行分治归约（每轮将数据量减半）；
3. 每个块得到一个部分和；
4. 通过全局内存原子操作将所有块的部分和累加。

问题在于步骤 4——当块数量很大时，全局内存原子操作成为瓶颈。

### 12.8.2 分布式归约的设计

使用 DSM，我们可以让簇中的块协作完成归约：

```cuda
__global__ void distributed_reduce(float *data, float *result, int N)
{
    extern __shared__ float smem[];
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

    int tid = threadIdx.x;
    int block_id = blockIdx.x;
    int cluster_blocks = cluster.dim_blocks().x;
    int block_rank = cluster.block_rank();

    // 1. 每个块加载属于它的数据段到共享内存
        int elements_per_block = N / gridDim.x;
    int start = block_id * elements_per_block;

    float local_sum = 0.0f;
    for (int i = tid; i < elements_per_block; i += blockDim.x) {
        local_sum += data[start + i];
    }

    // 2. 块内归约（使用 warp shuffle 或共享内存）
    smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    // 3. 每个块的部分和存储在 smem[0] 中
    float block_sum = smem[0];

    // 4. DSM：将每个块的部分和写入 rank=0 的块中
    //    先同步确保所有块的部分和就绪
    cluster.sync();

    float *root_smem = cluster.map_shared_rank(smem, 0);
    if (tid == 0) {
        // 每个块将自己的部分和写入根块的共享内存
        // 注意：需要不同的偏移以避免竞争
        root_smem[block_rank] = block_sum;
    }

    cluster.sync();

    // 5. 根块（rank=0）汇总所有部分和
    if (block_rank == 0 && tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < cluster_blocks; i++) {
            total += smem[i];
        }
        *result = total;
    }
}
```

### 12.8.3 分布式归约的优势

| 阶段 | 传统方法 | 分布式方法 |
|------|---------|-----------|
| 块内归约 | 共享内存归约 | 共享内存归约（相同） |
| 跨块合并 | 全局内存原子操作 | DSM 直接写入 root block |
| 总原子操作次数 | gridDim.x（可能数百） | 0（DSM内完成） |
| 同步开销 | 仅在块内 | 两次 cluster.sync() |

当 `gridDim.x` 很大时（比如数千个块），DSM 方案避免了大量的全局内存原子操作，性能提升非常显著。

## 12.9 深入理解：Cluster 大小选择策略

选择合适的 Cluster 大小对性能有重要影响。本节讨论影响簇大小的因素和选择策略。

### 12.9.1 影响因素

1. <strong>共享内存使用量</strong>：如果每个块使用大量共享内存（接近 Hopper 的 228KB 限制），那么簇大小必须较小，否则 GPC 上的 SM 资源不够。

2. <strong>每块线程数</strong>：更多线程意味着更大的 warp 占用。如果每块线程数接近最大值（1024），单个 SM 可能只能容纳一个块。

3. <strong>寄存器使用量</strong>：寄存器压力大的核函数可能无法在一个 SM 上容纳多个线程块。

4. <strong>GPC 上 SM 数量</strong>：Hopper H100 的每个 GPC 大约有 16 个 SM（具体取决于配置）。虽然最大可移植簇大小为 8，但实际可用簇大小取决于 GPC 上的 SM 资源。

5. <strong>问题规模</strong>：对于分布式共享内存应用，簇大小应使得每个块的共享内存能够容纳其"那份"数据。

### 12.9.2 动态选择簇大小

以下代码展示了如何在运行时根据直方图 bin 数量动态选择最优簇大小：

```cuda
int choose_cluster_size(int nbins, int smem_per_block, int max_cluster_size) {
    // 策略：让每个块处理尽可能多的 bin，同时簇中块数不超过限制
    int min_blocks_needed = (nbins + smem_per_block - 1) / smem_per_block;
    int cluster_size = min(min_blocks_needed, max_cluster_size);

    // 确保网格维度是簇大小的整数倍
    cluster_size = max(1, cluster_size);
    return cluster_size;
}
```

### 12.9.3 性能调优建议

1. <strong>从小开始</strong>：从 cluster_size=2 开始测试性能，逐步增大观察性能变化；
2. <strong>监控占用率</strong>：使用 `cudaOccupancyMaxPotentialClusterSize` 查询设备支持的最大簇大小；
3. <strong>权衡同步开销</strong>：`cluster.sync()` 涉及跨块同步，簇越大同步开销越大，确保收益超过开销；
4. <strong>避免过度分布式</strong>：如果数据能放入单块共享内存，使用 cluster_size=1（等同于传统模式）通常更快。

## 12.10 Cluster Group 成员函数参考

CUDA Programming Guide 中 Cluster Group 提供的完整 API 如下表所示。这些 API 在 CC 9.0+ 的所有硬件上可用——即使你在非 Cluster 的 Grid 上启动，API 假定为 1x1x1 的 Cluster。

| 成员函数 | 返回值 | 描述 |
|---------|-------|------|
| `this_cluster()` | `cluster_group` | 获取当前线程所在的簇组 |
| `sync()` | `void` | 簇级别同步——确保所有块到达此处 |
| `block_rank()` | `unsigned int` | 当前块在簇中的线性排名 |
| `block_index()` | `dim3` | 当前块在簇中的 3D 索引 |
| `num_blocks()` | `unsigned int` | 簇中总块数 |
| `dim_blocks()` | `dim3` | 簇的 3D 维度 |
| `num_threads()` | `unsigned int` | 簇中总线程数 |
| `dim_threads()` | `dim3` | 簇的线程 3D 维度 |
| `thread_rank()` | `unsigned int` | 当前线程在簇中的排名 |
| `query_shared_rank(const void *addr)` | `unsigned int` | 查询地址属于哪个块的共享内存 |
| `map_shared_rank(void *addr, int rank)` | `void*` | 映射地址到目标块的共享内存 |

## 12.11 内存一致性模型

在使用分布式共享内存时，理解内存一致性模型非常重要。与单个线程块内的共享内存不同，分布式共享内存的访问涉及多个块的共享内存物理分区。

### 12.11.1 DSM 写入的可见性

当一个线程块向另一个块的共享内存写入数据时：
1. 写入被提交到远程块的共享内存；
2. 远程块中的线程需要适当的同步才能看到写入；
3. `cluster.sync()` 在两个方向上都确保了可见性——写入线程块确认写入完成，读取线程块确认写入可见。

### 12.11.2 原子操作的语义

跨块原子操作（如 `atomicAdd`）在 DSM 上的语义与本地共享内存原子操作完全一致：
- 原子性：操作是不可分割的；
- 可见性：原子操作的结果对所有块立即可见（在 `cluster.sync()` 之后）。

## 12.12 硬件限制与注意事项

### 12.12.1 网格维度约束

在使用 Thread Block Cluster 时，有一个关键的约束：

> <strong>网格维度必须是簇大小的整数倍。</strong>

例如，如果簇在 X 方向大小为 2，那么启动核函数时 `numBlocks.x`（也即 `gridDim.x`）必须能被 2 整除。这个约束确保了所有簇都是"完整的"——不会出现一个簇只有部分线程块被启动的情况。

### 12.12.2 gridDim 的语义

如前所述，`gridDim` 仍然表示线程块的数量（而非簇的数量），这是为了兼容性考虑。CUDA Programming Guide 明确：

> "In a kernel launched using cluster support, the gridDim variable still denotes the size in terms of number of thread blocks, for compatibility purposes."

如果你需要知道当前的簇排名或簇网格维度，应使用 Cluster Group API 提供的 `block_rank()`、`block_index()`、`dim_blocks()` 等函数。

### 12.12.3 最大簇大小与 MIG

可以同时被 GPC 调度的线程块数量受限于 GPC 中的 SM 数量。在以下场景中最大簇大小会受到限制：

- <strong>MIG（Multi-Instance GPU）配置</strong>：每个 MIG 实例只拥有 GPU 的一部分资源，SM 数量减少；
- <strong>小型 GPU</strong>：物理 SM 数量本身就较少的 GPU。

建议通过 `cudaOccupancyMaxPotentialClusterSize` API 查询实际可用的最大簇大小，而不是硬编码为 8。

### 12.12.4 共享内存限制

在使用分布式共享内存时，共享内存的限制仍然是<strong>每线程块</strong>的。每个线程块最多可寻址的共享内存容量受硬件限制（Hopper 上为 227KB）。分布式共享内存的总大小是每块大小乘以簇中块数，但你不能在单个块中访问超出其自身共享内存容量限制的地址。

### 12.12.5 与 __block_size__ 的互斥性

CUDA Programming Guide 特别指出：

> "Note that it is illegal for the second tuple of `__block_size__` and `__cluster_dims__` to be specified at the same time."

这两个编译期属性是互斥的——你不能在同一个核函数上同时使用它们。

### 12.12.6 端口簇大小 vs 实际簇大小

在实际开发中，我们常常需要区分"端口簇大小"（代码中设计的簇大小）和"实际簇大小"（运行时可用的簇大小）。良好的设计应遵循以下原则：

1. 将簇大小设计为可配置的参数（而非硬编码）；
2. 在运行时通过 `cudaOccupancyMaxPotentialClusterSize` 查询实际可用大小；
3. 如果查询到的最大大小小于代码期望的大小，优雅降级（使用较小的簇或回退到单块模式）；
4. 通过预处理宏在编译时检查计算能力：

```cuda
#if __CUDA_ARCH__ >= 900
    // 使用 Cluster 和 DSM 的优化路径
#else
    // 回退路径（传统共享内存或全局内存方法）
#endif
```

### 12.12.7 动态共享内存与 DSM 的交互

当使用动态共享内存（`extern __shared__`）时，DSM 的行为值得特别说明：

1. 每个线程块的动态共享内存大小由 `config.dynamicSmemBytes` 指定；
2. 分布式共享内存的总动态大小 = `dynamicSmemBytes * cluster_size`；
3. 通过 `map_shared_rank` 访问远程块的动态共享内存时，偏移量计算与静态共享内存相同——因为动态共享内存也是每块独立的连续区域；
4. `map_shared_rank` 返回的指针的有效范围仅限于目标块共享内存的大小。

### 12.12.8 Cluster Grid 的维度布局考量

在设计 Cluster Grid 的维度时，应考虑到：

1. <strong>数据局部性</strong>：如果相邻数据块之间有数据交换需求，将它们在 X 方向上组织在同一个 Cluster 中可以减少跨 Cluster 通信。

2. <strong>GPC 资源限制</strong>：每个 GPC 的 SM 数量有限。如果 Cluster 太大（超过 GPC 支持的块数），启动会失败。

3. <strong>二维/三维 Cluster</strong>：对于二维/三维问题领域（如矩阵、体积数据），使用二维或三维 Cluster 可以更好地映射问题拓扑到硬件拓扑：

```cuda
// 二维问题 -> 二维 Cluster
__global__ void __cluster_dims__(4, 2, 1) kernel_2d(...);
dim3 grid(16, 8, 1);  // 16x8 个簇（共 128 个块）
```

## 12.13 Cluster 与 Warp Specialization 结合

Thread Block Cluster 可以与异步 SIMT 编程模型中的 Warp Specialization 模式结合，创造出更强大的数据流水线：

```cuda
// Cluster + Warp Specialization 伪代码
__global__ void __cluster_dims__(4, 1, 1)
cluster_warp_specialized(float *input, float *output, int N)
{
    extern __shared__ float smem[];
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

    int block_rank = cluster.block_rank();
    int tid = threadIdx.x;

    // Warp split: 前一半 warps 是生产者，后一半是消费者
    bool is_producer = (tid / warpSize) < (blockDim.x / (2 * warpSize));

    float *local_buf = smem;
    float *remote_buf = cluster.map_shared_rank(smem, (block_rank + 1) % 4);

    cluster.sync();

    if (is_producer) {
        for (int i = tid; i < N/4; i += blockDim.x/2) {
            float val = input[block_rank * N/4 + i];
            local_buf[i] = val;
            remote_buf[i] = val * 2.0f;  // 写入邻居块的共享内存
        }
    }

    cluster.sync();

    if (!is_producer) {
        for (int i = tid; i < N/4; i += blockDim.x/2) {
            output[block_rank * N/4 + i] = smem[i] + 1.0f;
        }
    }
}
```

## 12.14 Cluster 与 Cooperative Groups 的内部关系

Thread Block Cluster 的实现建立在 Cooperative Groups 框架之上。理解两者的关系有助于更深入地理解这一特性。

### 12.14.1 cooperative_groups 中的 Cluster Group 实现

当你在核函数中调用 `cooperative_groups::this_cluster()` 时：

1. CUDA 运行时检查当前网格是否以 Cluster 模式启动；
2. 如果是，返回包含簇中所有块信息的 `cluster_group` 对象；
3. 如果不是，返回一个 1x1x1 的"退化" `cluster_group`（所有 API 仍然可用，行为和单块一致）。

```cuda
namespace cooperative_groups {
    class cluster_group {
        // 内部实现细节（用户不可见）
        unsigned int block_rank_in_cluster;
        dim3 cluster_dimensions;
        // ...
    public:
        void sync();                      // 硬件支持的簇同步
        unsigned int block_rank();        // 当前块在簇中的排名
        dim3 block_index();               // 当前块在簇中的3D索引
        unsigned int num_blocks();        // 簇中块总数
        dim3 dim_blocks();                // 簇的维度
        unsigned int num_threads();       // 簇中线程总数
        dim3 dim_threads();               // 簇的线程维度
        unsigned int thread_rank();       // 当前线程在簇中排名
        unsigned int query_shared_rank(const void* addr);  // 查询共享内存地址归属
        void* map_shared_rank(void* addr, int rank);       // 映射到目标块共享内存
    };
}
```

### 12.14.2 cluster.sync() 的内部实现

`cluster.sync()` 是硬件支持的簇级同步操作，依赖于 Hopper 架构的 Cluster Barrier 机制（PTX 中通常涉及` barrier.cluster `相关指令，具体实现由编译器决定）。

`cluster.sync()` 是一个块级集体操作（collective operation），要求块内所有线程都必须到达该调用点。它本身已隐式完成块内同步，用户不应在 `cluster.sync() `前后额外调用` __syncthreads()`。

执行流程：

块内所有线程到达 `cluster.sync()` 后，硬件自动完成块内同步；

各块的代表通过 GPC 上的硬件 barrier 进行跨块同步；

所有块到达后，barrier 翻转，所有块继续执行。

`cluster.sync()` 的开销取决于簇大小和块在 GPC 上的分布，实际延迟由硬件 barrier 同步主导。


### 12.14.3 不能用 cluster.sync() 做什么

虽然 `cluster.sync()` 提供了簇级别的同步，但它<strong>不能</strong>用于：

1. <strong>跨簇同步</strong>：只同步同一簇内的块，不同簇之间没有同步；
2. <strong>网格级同步</strong>：如果需要整个网格的同步，需要使用 Cooperative Groups 的 `grid.sync()`（有更多限制）；
3. <strong>细粒度 Warp 同步</strong>：`cluster.sync()` 是块级别的，不适用于 warp 级同步。

## 12.15 实践建议与最佳实践

### 12.15.1 何时使用 Thread Block Cluster

Thread Block Cluster 和分布式共享内存不是银弹。只有在以下场景中才值得使用：

1. <strong>共享内存容量不足</strong>：单个块的共享内存装不下所需的直方图/缓冲区/查找表；
2. <strong>需要跨块细粒度协作</strong>：算法天然需要块之间交换数据；
3. <strong>可以减少全局内存原子操作</strong>：通过 DSM 将跨块操作保留在共享内存域内。

### 12.15.2 何时不应使用

1. <strong>数据完全独立</strong>：每个块独立处理自己的数据，不需要块间通信；
2. <strong>单块共享内存已足够</strong>：如果数据大小适合单块共享内存，使用传统方法更简单；
3. <strong>CC < 9.0 的硬件</strong>：Cluster 是 Hopper+ 独占特性。

### 12.15.3 调试分布式共享内存的常见问题

1. <strong>忘记 cluster.sync()</strong>：这是最常见的错误。在访问 DSM 之前和退出之前都必须同步。

2. <strong>网格维度不是簇大小的整数倍</strong>：这会导致启动失败或未定义行为。

3. <strong>远程共享内存地址越界</strong>：`map_shared_rank` 返回的指针只在目标块的共享内存范围内有效。访问超出目标块共享内存大小的地址会导致未定义行为。

4. <strong>簇大小过大</strong>：如果簇大小超过 GPC 支持的最大值，启动会失败。始终先查询 `cudaOccupancyMaxPotentialClusterSize`。

5. <strong>共享内存分配不足</strong>：动态共享内存大小仍然是每块的。如果每块需要 X 字节共享内存，而簇大小是 4，总分布式共享内存大小是 4X——但启动时只需指定 X。

### 12.15.4 迁移现有代码到 Cluster 的步骤

1. <strong>评估收益</strong>：确定算法是否真正能从跨块共享内存中受益；
2. <strong>添加条件编译</strong>：使用 `#if __CUDA_ARCH__ >= 900` 保护 Cluster 代码；
3. <strong>重构共享内存使用</strong>：将原来单体共享内存的使用改为 DSM 感知的使用；
4. <strong>添加 cluster.sync()</strong>：在访问 DSM 前后插入同步；
5. <strong>测试回退路径</strong>：确保在旧硬件上回退路径正确工作；
6. <strong>性能对比</strong>：使用 Nsight Compute 对比新旧方案的性能。

## 12.16 动手体验
### 12.16.1 扩展练习——分布式矩阵乘法预处理

以下是一个扩展练习，展示如何使用 DSM 进行矩阵乘法的预处理步骤（数据重组/格式转换）。这个例子展示了一个实际场景：在 GEMM 的分块预处理中，使用 DSM 在簇内协同重组数据布局。

```cuda
// 分布式矩阵重排：将 row-major 全局数据重组为 block-optimized 共享内存布局
__global__ void __cluster_dims__(2, 2, 1)
distributed_matrix_reorder(
    const float *__restrict__ global_A,
    float *__restrict__ global_B,
    int M, int N, int tile_size)
{
    extern __shared__ float smem[];
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();

    int block_rank = cluster.block_rank();
    int cluster_size = cluster.dim_blocks().x * cluster.dim_blocks().y;

    // 每个块负责 tile_size/cluster_size 行
    int rows_per_block = tile_size / cluster_size;
    int my_start_row = block_rank * rows_per_block;
    int my_end_row = my_start_row + rows_per_block;

    // 加载数据到本地共享内存
    int global_row_offset = blockIdx.x * tile_size + my_start_row;
    for (int i = threadIdx.x; i < rows_per_block * N; i += blockDim.x) {
        int r = i / N;
        int c = i % N;
        smem[r * N + c] = global_A[(global_row_offset + r) * N + c];
    }
    __syncthreads();

    // DSM: 将本地数据分散到簇中所有块的共享内存
    // 每个块负责最终输出的一列
    cluster.sync();

    // 将数据重新排列：块 k 收集所有行中属于列 k 的元素
    for (int r = 0; r < rows_per_block; r++) {
        for (int c = 0; c < N; c++) {
            int dst_block = c % cluster_size;
            int dst_offset = r * (N / cluster_size) + (c / cluster_size);
            float *dst_smem = cluster.map_shared_rank(smem, dst_block);
            dst_smem[dst_offset] = smem[r * N + c];
        }
    }

    cluster.sync();

    // 写回全局内存（省略）
}

// 主机端动态启动
void launch_reorder(float *d_A, float *d_B, int M, int N, int tile_size)
{
    dim3 threads(256);
    int tiles = M / tile_size;

    int cluster_x = 2, cluster_y = 2;
    int rows_per_block = tile_size / (cluster_x * cluster_y);  

    cudaLaunchConfig_t config = {0};
    config.gridDim = dim3(tiles);
    config.blockDim = threads;
    config.dynamicSmemBytes = (tile_size / (cluster_x * cluster_y)) * N * sizeof(float);
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = cluster_x;
    attrs[0].val.clusterDim.y = cluster_y;
    attrs[0].val.clusterDim.z = 1;

    config.numAttrs = 1;
    config.attrs = attrs;

    cudaLaunchKernelEx(&config, distributed_matrix_reorder,
                       d_A, d_B, M, N, tile_size);
}
```

### 12.16.2动手体验：完整的分布式直方图程序

下面是一个完整的、可编译的分布式直方图示例程序。它包含了主机端准备数据、动态选择簇大小、核函数执行和结果验证的全流程。

```cuda
// 文件: distributed_histogram.cu
// 编译: nvcc -arch=sm_90 distributed_histogram.cu -o distributed_histogram
// 硬件要求: NVIDIA Hopper H100 或更新 (CC 9.0+)

#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;cooperative_groups.h&gt;

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
```

## 12.17 常见问题与故障排除

### 12.17.1 启动失败：网格维度不是簇大小的整数倍

<strong>症状</strong>：`cudaLaunchKernelEx` 返回 `cudaErrorInvalidConfiguration`。

<strong>原因</strong>：`gridDim.x * gridDim.y * gridDim.z` 不是 `clusterDim.x * clusterDim.y * clusterDim.z` 的整数倍。

<strong>解决</strong>：向上取整网格维度使其能整除：

```cuda
int blocks_x = (N + threads_x - 1) / threads_x;
int cluster_x = 4;
blocks_x = ((blocks_x + cluster_x - 1) / cluster_x) * cluster_x; // 向上取整
```

### 12.17.2 DSM 访问返回垃圾值

<strong>症状</strong>：从远程块的共享内存读取到未初始化或错误的数据。

<strong>可能原因</strong>：
1. 忘记在访问 DSM 前调用 `cluster.sync()`；
2. `map_shared_rank` 的 rank 参数超出簇大小；
3. 访问的偏移超出目标块的共享内存容量。

<strong>解决方法</strong>：仔细检查 `cluster.sync()` 的位置和 `map_shared_rank` 的参数。

### 12.17.3 簇同步死锁

<strong>症状</strong>：kernel 无限期挂起，不返回。

<strong>可能原因</strong>：
1. 某些线程块因条件分支未到达 `cluster.sync()`；
2. `__syncthreads()` 在条件分支中使用（块内线程发散）。

<strong>解决方法</strong>：确保 `cluster.sync()` 不在条件分支内，所有块都能到达。

```cuda
// 错误
if (block_rank == 0) {
    cluster.sync();  // 其他块不会到达！
}

// 正确
cluster.sync();  // 所有块都到达
if (block_rank == 0) {
    // 仅 rank 0 执行的代码
}
```

### 12.17.4 性能不如预期

<strong>可能原因和解决方案</strong>：
1. <strong>簇大小太大</strong>：减少簇大小，测试不同配置；
2. <strong>共享内存不足</strong>：减少每块的共享内存使用量；
3. <strong>同步开销</strong>：评估是否真的需要 DSM——如果单块共享内存够用，去掉 Cluster；
4. <strong>MIG 限制</strong>：检查是否在 MIG 实例上运行，MIG 减少可用 SM 数量。

### 12.17.5 编译错误：__cluster_dims__ 与 __block_size__ 冲突

<strong>症状</strong>：`error: "__block_size__" and "__cluster_dims__" cannot be used together`

<strong>解决</strong>：选择其中一个。通常使用 `__cluster_dims__` 更简洁，除非需要以簇数为单位启动。

## 12.18 性能基准参考

以下是在 NVIDIA H100 上使用不同方案进行直方图计算的性能对比（512 bins, 1M elements）：

| 方案 | 时间 (us) | 相对性能 | 共享内存使用 |
|------|----------|---------|------------|
| 全局内存原子 | 156 | 1.0x (基准) | 0 KB |
| 单块共享内存 (cluster_size=1) | 98 | 1.6x | 2 KB |
| DSM cluster_size=2 | 72 | 2.2x | 2 KB x 2 |
| DSM cluster_size=4 | 58 | 2.7x | 2 KB x 4 |
| DSM cluster_size=8 | 52 | 3.0x | 2 KB x 8 |

（注：实际性能因 bin 数量、数据规模、具体硬件配置而异。上述数据为示意性参考。）

从这个参考数据可以看出：
1. 单块共享内存方案相比全局原子操作有 1.6x 加速；
2. DSM 在 cluster_size=2 时比单块再加速 36%；
3. DSM 从 4 到 8 的收益递减（2.7x → 3.0x），这是因为较大的簇带来更多的同步开销。

## 12.19 GPU 架构演进中的 Cluster 设计哲学

Thread Block Cluster 的引入不仅仅是增加了一个编程层次——它反映了 GPU 硬件架构的深层演进趋势。

### 12.19.1 从 SM 到 GPC 的演进

回顾 GPU 架构的发展：

- <strong>Kepler/Maxwell (2012-2014)</strong>：GPC 概念首次引入，但主要用于图形管线。计算任务中 GPC 对程序员不可见。

- <strong>Pascal/Volta (2016-2018)</strong>：GPC 内的 SM 数量增加，但线程块的调度仍然是"平面"的——所有块在全局调度器上竞争 SM 资源。线程块之间无法安全通信。

- <strong>Ampere (2020)</strong>：异步 SIMT 编程模型引入，允许块内同步的 arrive/wait 分离。但跨块通信仍然只能通过全局内存。

- <strong>Hopper (2022)</strong>：Thread Block Cluster + DSM 首次将 GPC 概念暴露给程序员，使跨块协作成为一等公民。

### 12.19.2 为什么是现在？

为什么 Cluster 在 Hopper 架构上才被引入？有几个技术原因：

1. <strong>SM 数量增长</strong>：H100 有 132 个 SM（SXM5 版本），每个 GPC 有十几个 SM。这种规模使得 GPC 内部的协作变得有意义。

2. <strong>共享内存容量增长</strong>：H100 每个 SM 有 256KB 的统一数据缓存+共享内存，单块最多可寻址 228KB。更大的共享内存使得 DSM 更有价值。

3. <strong>硬件同步原语成熟</strong>：Ampere 引入的硬件 barrier 加速为 Cluster 的 `barrier.cluster` PTX 指令奠定了基础。

4. <strong>工作负载需求</strong>：深度学习和大规模模拟对跨块协作的需求日益增长（如分布式 softmax、batch normalization 等）。

### 12.19.3 展望：Blackwell 及以后

NVIDIA Blackwell 架构（CC 10.0/12.0）在 Cluster 的基础上进一步引入了 <strong>Cluster Launch Control</strong>（集群启动控制），支持线程块之间的工作窃取（work stealing）。这表明 Cluster 将成为未来 GPU 编程模型的核心组成部分，而不仅仅是 Hopper 的特色功能。

理解 Thread Block Cluster 和分布式共享内存，不仅让你能在 H100 上写出更好的程序，更让你为未来的 GPU 架构做好准备。

## 12.20 本章小结

本章我们深入探讨了 CUDA Hopper 架构引入的两个紧密相关的特性：Thread Block Clusters 和 Distributed Shared Memory。让我们回顾关键要点：

1. <strong>Thread Block Cluster</strong> 是在 Grid 和 Block 之间新增的层次，一个簇中最多 8 个线程块被共调度在一个 GPC 上。

2. <strong>两种启动方式</strong>：编译期 `__cluster_dims__`（简单直接）和运行时 `cudaLaunchKernelEx`（灵活动态），以及 `__block_size__` 属性允许以簇数为单位进行启动。

3. <strong>`cluster.sync()`</strong> 是簇内跨块同步的核心，必须在对分布式共享内存进行访问前和退出前调用。

4. <strong>分布式共享内存</strong> 将簇中所有块的共享内存统一为一个地址空间，通过 `map_shared_rank()` 和 `query_shared_rank()` 实现跨块访问。

5. <strong>分布式直方图</strong> 是 DSM 的典型应用，在单块共享内存容量不足时提供比全局内存原子操作更优的性能。

6. 使用 Cluster 时需要牢记：<strong>网格维度必须是簇大小的整数倍</strong>，且 `gridDim` 仍按线程块计数。

Thread Block Clusters 和分布式共享内存在 NVIDIA Hopper 架构上开启了 GPU 编程的新维度。它们使得跨线程块的细粒度协作成为可能，为许多以前需要全局内存回退的算法提供了更高效的选择。

## 12.21 习题

1. 解释 Thread Block Cluster 的共调度保证为什么是实现 `cluster.sync()` 的前提条件。如果没有共调度保证会发生什么？

2. 比较 `__cluster_dims__` 和 `cudaLaunchKernelEx` 两种启动方式的优缺点。在什么场景下你会选择其中一种？

3. 给出一个使用场景，其中 `__block_size__` 属性比 `__cluster_dims__` 更直观方便。

4. 编写一个核函数，使用分布式共享内存实现对两个大数组的逐元素求和。假设每个块处理一部分数据，然后将所有块的部分和通过 DSM 进行跨块归约。

5. 修改 12.9 节的完整程序，使其能自动根据 bin 数量和共享内存大小选择最优的簇大小。

6. 为什么在使用分布式共享内存时，需要在退出前再次调用 `cluster.sync()`？如果不调用会发生什么情况？

## 12.22 参考文献

1. CUDA C++ Programming Guide 13.0, Section 5.2.1 "Thread Block Clusters"
2. CUDA C++ Programming Guide 13.0, Section 5.2.2 "Blocks as Clusters"
3. CUDA C++ Programming Guide 13.0, Section 6.2.5 "Distributed Shared Memory"
4. CUDA C++ Programming Guide 13.0, Chapter 11 "Cooperative Groups" — Section 11.4.1.2 "Cluster Group"
5. NVIDIA H100 Tensor Core GPU Architecture Whitepaper
6. PTX ISA — barrier.cluster instruction documentation
7. NVIDIA CUDA Samples — clusterHierarchy and distributedSharedMemory
