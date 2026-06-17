# 第13章 异步SIMT编程模型

<strong>硬件要求</strong>：Compute Capability 8.0+（NVIDIA Ampere A100+）提供硬件加速；CC 7.0+ 提供软件支持（无硬件加速）

## 13.1 引言

在 CUDA 编程中，我们长期以来习惯了这样的模式：从全局内存加载数据到共享内存，使用 `__syncthreads()` 同步所有线程确保数据就绪，然后在共享内存上执行计算，再用 `__syncthreads()` 同步确保计算完成，最后将结果写回全局内存。

这种"加载-同步-计算-同步-写回"的模式虽然正确，但存在一个根本性的效率问题：<strong>同步屏障强制所有线程相互等待</strong>。当一些线程已经完成了加载，它们必须空闲等待其他线程完成加载。同样的等待发生在计算阶段。

随着 NVIDIA Ampere 架构（Compute Capability 8.0）的推出，CUDA 引入了<strong>异步 SIMT 编程模型（Asynchronous SIMT Programming Model）</strong>。CUDA Programming Guide 对此的定义是：

> "In the CUDA programming model a thread is the lowest level of abstraction for doing a computation or a memory operation. Starting with devices based on the NVIDIA Ampere GPU Architecture, the CUDA programming model provides acceleration to memory operations via the asynchronous programming model."

异步 SIMT 编程模型的核心思想是：<strong>将异步操作的发起与完成分离</strong>。一个 CUDA 线程发起一个操作（如数据拷贝），该操作由"另一个逻辑线程"（as-if thread）异步执行，而发起线程可以<strong>在操作完成之前</strong>继续执行其他工作。

本章将全面介绍异步 SIMT 编程模型的各个组件：线程作用域（Thread Scope）、异步屏障（`cuda::barrier`）、异步数据拷贝（`cuda::memcpy_async`）以及多阶段流水线（`cuda::pipeline`）。这些工具让你能够将计算与数据传输重叠执行，显著提升 GPU 程序的性能。

## 13.2 异步SIMT编程模型概述

### 13.2.1 核心思想

异步编程模型的关键突破在于：<strong>将同步屏障拆分为 arrive 和 wait 两个独立的操作</strong>。

在传统模型中（如 `__syncthreads()`），线程到达屏障后立即被阻塞，直到所有其他线程也到达。这是一个原子性的"到达并等待"操作。在异步模型中：

1. <strong>`arrive()`</strong>：线程声明自己已经到达屏障，但<strong>不会被阻塞</strong>。线程可以继续执行与屏障之后结果无关的工作。
2. <strong>`wait()`</strong>：线程在需要屏障之后的数据时，显式等待屏障翻转。

这两个操作之间存在一段<strong>重叠区</strong>——线程可以在 arrive 之后、wait 之前执行不依赖于同步结果的额外计算。这就是"重叠"的来源。

### 13.2.2 异步操作的定义

CUDA Programming Guide 对异步操作给出了精确的定义：

> "An asynchronous operation is defined as an operation that is initiated by a CUDA thread and is executed asynchronously as-if by another thread. In a well formed program one or more CUDA threads synchronize with the asynchronous operation. The CUDA thread that initiated the asynchronous operation is not required to be among the synchronizing threads."

关键点：

1. 异步操作由某个 CUDA 线程发起；
2. 它由一个"as-if 线程"异步执行——你可以理解为硬件中的一个独立执行单元；
3. 发起线程不需要在同步线程之中——这为 Warp Specialization 等高级模式奠定了基础。

### 13.2.3 同步对象

异步操作使用<strong>同步对象（synchronization object）</strong>来同步完成状态。CUDA Programming Guide 提到：

> "A synchronization object could be a `cuda::barrier` or a `cuda::pipeline`."

- `cuda::barrier`：异步屏障，提供 arrive/wait 分离式同步；
- `cuda::pipeline`：多阶段流水线，管理异步数据拷贝的生产者-消费者模式。

## 13.3 线程作用域（Thread Scope）

在使用异步操作时，你需要指定同步对象的<strong>线程作用域</strong>。作用域定义了哪些线程可以参与该同步对象的同步。以下是 CUDA 定义的四种线程作用域：

| 线程作用域 | 描述 |
|-----------|------|
| `cuda::thread_scope::thread_scope_thread` | 仅发起异步操作的 CUDA 线程参与同步 |
| `cuda::thread_scope::thread_scope_block` | 与发起线程在同一线程块中的所有 CUDA 线程参与同步 |
| `cuda::thread_scope::thread_scope_device` | 与发起线程在同一 GPU 设备上的所有 CUDA 线程参与同步 |
| `cuda::thread_scope::thread_scope_system` | 同一系统中的所有 CUDA 或 CPU 线程参与同步 |

作用域的选择直接影响性能和正确性。较小的作用域（如 `thread_scope_block`）通常具有更低的同步延迟，但只能在同一线程块内使用。较大的作用域支持更广的同步范围，但开销也更大。

通常在核函数内部，我们使用 `thread_scope_block`：

```cuda
using barrier = cuda::barrier&lt;cuda::thread_scope::thread_scope_block&gt;;
```

## 13.4 异步屏障：cuda::barrier

### 13.4.1 传统屏障 vs 异步屏障

在介绍 `cuda::barrier` 之前，让我们回顾传统同步模式的问题：

```cuda
#include &lt;cooperative_groups.h&gt;

__global__ void simple_sync(int iteration_count) {
    auto block = cooperative_groups::this_thread_block();

    for (int i = 0; i < iteration_count; ++i) {
        /* 同步前的代码 */
        block.sync();  /* 等待所有线程到达此处——线程被阻塞 */
        /* 同步后的代码 */
    }
}
```

这种模式有三个阶段：
1. 同步<strong>前</strong>的代码执行内存更新（将在同步后被读取）；
2. 同步点——所有线程被阻塞；
3. 同步<strong>后</strong>的代码（可以看到同步前的内存更新）。

问题在于：<strong>同步期间没有任何有用的工作被完成</strong>。所有线程都在等待最慢的那个线程。

### 13.4.2 时间分割（Temporal Splitting）：arrive/wait 分离

`cuda::barrier` 将同步点拆分为 arrive 和 wait，创造了五个阶段：

```cuda
#include &lt;cuda/barrier&gt;
#include &lt;cooperative_groups.h&gt;

__device__ void compute(float* data, int curr_iteration);

__global__ void split_arrive_wait(int iteration_count, float *data) {
    using barrier = cuda::barrier&lt;cuda::thread_scope::thread_scope_block&gt;;
    __shared__  barrier bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size()); // 初始化屏障，指定期望到达数
    }
    block.sync();

    for (int curr_iter = 0; curr_iter < iteration_count; ++curr_iter) {
        /* 阶段1：arrive 前的代码 */

        barrier::arrival_token token = bar.arrive(); // 阶段2：arrive 点（含隐式内存屏障）
        /* 注意：arrive() 不会阻塞线程！*/

        compute(data, curr_iter); // 阶段3：arrive 和 wait 之间的代码

        bar.wait(std::move(token)); // 阶段4：wait 点

        /* 阶段5：wait 后的代码 */
    }
}
```

CUDA Programming Guide 总结了这五个阶段：

> 1. Code <strong>before</strong> arrive performs memory updates that will be read <strong>after</strong> the wait.
> 2. Arrive point with implicit memory fence.
> 3. Code <strong>between</strong> arrive and wait.
> 4. Wait point.
> 5. Code <strong>after</strong> the wait, with visibility of updates that were performed <strong>before</strong> the arrive.

<strong>阶段3——arrive 和 wait 之间的代码</strong>——是异步屏障相比传统屏障的真正优势所在。在这个区间内，线程可以执行不依赖于同步结果的计算，从而<strong>将计算与等待重叠</strong>。

### 13.4.3 屏障初始化

在使用 `cuda::barrier` 之前，必须先初始化。初始化指定<strong>期望到达数（expected arrival count）</strong>——在屏障翻转之前需要调用 `arrive()` 的次数：

```cuda
#include &lt;cuda/barrier&gt;
#include &lt;cooperative_groups.h&gt;

__global__ void init_barrier() {
    __shared__ cuda::barrier&lt;cuda::thread_scope::thread_scope_block&gt; bar;
    auto block = cooperative_groups::this_thread_block();

    if (block.thread_rank() == 0) {
        init(&bar, block.size()); // 单个线程初始化，期望到达数 = 块大小
    }
    block.sync();
}
```

CUDA Programming Guide 强调：

> "Initialization must happen before any thread begins participating in a `cuda::barrier`."

这带来一个"引导挑战"（bootstrapping challenge）：你需要同步来初始化屏障，但你正在创建的正是用来同步的屏障。解决方案是使用传统的 `block.sync()`（或 `__syncthreads()`）作为引导同步。

### 13.4.4 屏障的相位（Phase）

`cuda::barrier` 的工作机制基于相位。CUDA Programming Guide 是这样描述的：

> "A `cuda::barrier` counts down from the expected arrival count to zero as participating threads call `bar.arrive()`. When the countdown reaches zero, a `cuda::barrier` is complete for the current phase. When the last call to `bar.arrive()` causes the countdown to reach zero, the countdown is automatically and atomically reset."

每个 `arrival_token` 与当前相位关联。当 `bar.wait(token)` 被调用时：
- 如果屏障仍在当前相位，线程阻塞直到相位翻转；
- 如果屏障已经翻转到下一相位，线程立即返回（不阻塞）。

PHASE 使用规则：

> 1. A thread's calls to `token=bar.arrive()` and `bar.wait(std::move(token))` must be sequenced such that `token=bar.arrive()` occurs during the barrier's current phase, and `bar.wait(std::move(token))` occurs during the same or next phase.
> 2. A thread's call to `bar.arrive()` must occur when the barrier's counter is non-zero.
> 3. `bar.wait()` must only be called using a token object of the current phase or the immediately preceding phase.

### 13.4.5 空间分割（Warp Specialization）

CUDA Programming Guide 还介绍了 <strong>Warp Specialization</strong>（空间分割）模式：

> "A thread block can be spatially partitioned such that warps are specialized to perform independent computations. Spatial partitioning is used in a producer or consumer pattern, where one subset of threads produces data that is concurrently consumed by the other (disjoint) subset of threads."

在这种模式中：
- <strong>生产者 Warp</strong> 负责从全局内存加载数据、填充共享内存缓冲区；
- <strong>消费者 Warp</strong> 负责在共享内存上执行计算；
- 两者之间通过两个 `cuda::barrier` 实现"缓冲区就绪"和"缓冲区已填充"的信号传递。

```cuda
#include &lt;cuda/barrier&gt;
#include &lt;cooperative_groups.h&gt;

using barrier = cuda::barrier&lt;cuda::thread_scope::thread_scope_block&gt;;

__device__ void producer(barrier ready[], barrier filled[],
                         float* buffer, float* in, int N, int buffer_len)
{
    for (int i = 0; i < (N/buffer_len); ++i) {
        ready[i%2].arrive_and_wait();  /* 等待缓冲区就绪 */
        /* 生产数据，填充 buffer_(i%2) */
        barrier::arrival_token token = filled[i%2].arrive();
        /* buffer_(i%2) 已填满——不等待，继续下一个迭代 */
    }
}

__device__ void consumer(barrier ready[], barrier filled[],
                         float* buffer, float* out, int N, int buffer_len)
{
    barrier::arrival_token token1 = ready[0].arrive(); /* buffer_0 就绪 */
    barrier::arrival_token token2 = ready[1].arrive(); /* buffer_1 就绪 */
    for (int i = 0; i < (N/buffer_len); ++i) {
        filled[i%2].arrive_and_wait(); /* 等待缓冲区被填满 */
        /* 消费 buffer_(i%2) */
        barrier::arrival_token token = ready[i%2].arrive();
        /* buffer_(i%2) 已消费完，可以重新填充 */
    }
}

__global__ void producer_consumer_pattern(int N, int buffer_len,
                                           float* in, float* out) {
    // 双缓冲: buffer_0 = buffer, buffer_1 = buffer + buffer_len
    __shared__ extern float buffer[];

    // bar[0]/bar[1] 跟踪 buffer_0/buffer_1 是否就绪
    // bar[2]/bar[3] 跟踪 buffer_0/buffer_1 是否已填满
    __shared__ barrier bar[4];

    auto block = cooperative_groups::this_thread_block();
    if (block.thread_rank() < 4)
        init(bar + block.thread_rank(), block.size());
    block.sync();

    if (block.thread_rank() < warpSize)
        producer(bar, bar+2, buffer, in, N, buffer_len);
    else
        consumer(bar, bar+2, buffer, out, N, buffer_len);
}
```

### 13.4.6 提前退出（Early Exit）

当参与屏障同步的线程需要提前退出同步序列时，必须显式地"退出参与"（drop out of participation）。剩余线程可以继续正常参与后续的 arrive 和 wait 操作。这是 `cuda::barrier` 相比 `__syncthreads()` 的另一个优势——`__syncthreads()` 要求所有线程都必须到达，无法支持提前退出。

## 13.5 异步数据拷贝：cuda::memcpy_async

### 13.5.1 为什么需要 memcpy_async

在传统 CUDA 编程中，从全局内存拷贝数据到共享内存是通过逐元素的赋值语句完成的：

```cuda
shared[local_idx] = global_in[global_idx];
```

这实际上是：从全局内存读取到寄存器，再从寄存器写入共享内存。它有两个缺点：

1. <strong>占用寄存器</strong>：中间值需要寄存器存储；
2. <strong>同步阻塞</strong>：所有线程必须完成拷贝（通过 `__syncthreads()`）后才能开始计算。

`cuda::memcpy_async` 解决了这两个问题：
1. 数据直接从全局内存拷贝到共享内存，<strong>不经过寄存器</strong>；
2. 拷贝操作是<strong>异步的</strong>——由硬件负责执行，线程在拷贝完成前可以继续其他工作。

CUDA Programming Guide 明确指出：

> "On devices with compute capability 8.0 or higher, `memcpy_async` transfers from global to shared memory can benefit from hardware acceleration, which avoids transferring the data through an intermediate register."

### 13.5.2 基本用法

`memcpy_async` 有三个变体，分别与不同的同步机制配合：

```cuda
// 1. 与 cuda::barrier 配合
cuda::memcpy_async(block, shared, global_in + batch_idx,
                   sizeof(int) * block.size(), barrier);

// 2. 与 cuda::pipeline 配合
cuda::memcpy_async(block, shared, global_in + batch_idx,
                   sizeof(int) * block.size(), pipeline);

// 3. 与 cooperative_groups::wait 配合
cooperative_groups::memcpy_async(block, shared, global_in + batch_idx,
                                 sizeof(int) * block.size());
cooperative_groups::wait(block);
```

### 13.5.3 传统方式 vs 异步拷贝

下面是同一个 "copy and compute" 模式的传统实现和异步实现对比：

<strong>传统方式（无 memcpy_async）：</strong>

```cuda
#include &lt;cooperative_groups.h&gt;

__device__ void compute(int* global_out, int const* shared_in);

__global__ void without_memcpy_async(int* global_out,
                                      int const* global_in,
                                      size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();

  extern __shared__ int shared[]; // block.size() * sizeof(int) 字节

  size_t local_idx = block.thread_rank();

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    size_t block_batch_idx = block.group_index().x * block.size()
                           + grid.size() * batch;
    size_t global_idx = block_batch_idx + threadIdx.x;
    shared[local_idx] = global_in[global_idx];  // 通过寄存器拷贝

    block.sync(); // 等待所有拷贝完成 —— 所有线程被阻塞

    compute(global_out + block_batch_idx, shared); // 计算

    block.sync(); // 等待计算完成
  }
}
```

<strong>使用 memcpy_async + barrier：</strong>

```cuda
#include &lt;cooperative_groups.h&gt;
#include &lt;cuda/barrier&gt;

__device__ void compute(int* global_out, int const* shared_in);

__global__ void with_barrier(int* global_out, int const* global_in,
                              size_t size, size_t batch_sz) {
  auto grid = cooperative_groups::this_grid();
  auto block = cooperative_groups::this_thread_block();

  extern __shared__ int shared[];

  // 创建屏障
  __shared__ cuda::barrier&lt;cuda::thread_scope::thread_scope_block&gt; barrier;
  if (block.thread_rank() == 0) {
    init(&barrier, block.size());
  }
  block.sync();

  for (size_t batch = 0; batch < batch_sz; ++batch) {
    size_t block_batch_idx = block.group_index().x * block.size()
                           + grid.size() * batch;

    // 异步拷贝：数据直接从全局到共享，不经过寄存器
    cuda::memcpy_async(block, shared, global_in + block_batch_idx,
                       sizeof(int) * block.size(), barrier);

    barrier.arrive_and_wait(); // 等待所有拷贝完成

    compute(global_out + block_batch_idx, shared);

    block.sync();
  }
}
```

关键区别：`cuda::memcpy_async` 启动异步拷贝后，它与 barrier 集成——barrier 的相位翻转不仅在所有线程到达时发生，还在<strong>所有异步拷贝完成</strong>时发生。

### 13.5.4 对齐要求

`memcpy_async` 的性能对对齐有严格的要求。CUDA Programming Guide 建议：

> "On devices with compute capability 8.0, the cp.async family of instructions allows copying data from global to shared memory asynchronously. These instructions support copying 4, 8, and 16 bytes at a time."

为了获得最佳性能：
- 共享内存地址和全局内存地址应为 <strong>128 字节对齐</strong>；
- 传输大小应为 4、8 或 16 的倍数；
- 可以使用 `cuda::aligned_size_t&lt;Align&gt;` 向编译器提供对齐证明：

```cuda
cuda::memcpy_async(group, dst, src,
    cuda::aligned_size_t&lt;16&gt;(N * block.size()), pipeline);
```

CUDA Programming Guide 警告：

> "If the proof is incorrect, the behavior is undefined."

### 13.5.5 性能指导

CUDA Programming Guide 提供了关于 `memcpy_async` 的以下性能指导：

1. <strong>TriviallyCopyable 类型</strong>：如果指针类型不指向 TriviallyCopyable 类型，`cp.async` 指令无法被使用。

2. <strong>Warp 纠缠（Warp Entanglement）</strong>：在 Compute Capability 8.x 上，pipeline 机制在 warp 内的 CUDA 线程之间共享。这导致 commit、wait 和 arrive-on 操作在 warp 内产生纠缠效应。建议<strong>保持 commit 和 arrive-on 操作由收敛的线程执行</strong>。

3. <strong>最佳实践</strong>：
> "It is recommended that commit and arrive-on invocations are by converged threads: to not over-wait, by keeping threads' perceived sequence of batches aligned with the actual sequence, and to minimize updates to the barrier object."

## 13.6 Pipeline 异步数据拷贝

### 13.6.1 Pipeline 概念

`cuda::pipeline` 是比 `cuda::barrier` 更高级的同步对象，专门为管理异步数据拷贝设计。CUDA Programming Guide 将其描述为：

> "A pipeline object is a double-ended N stage queue with a head and a tail, and is used to process work in a first-in first-out (FIFO) order."

Pipeline 的核心思想是<strong>多阶段流水线</strong>——通过多个共享内存缓冲区，实现数据拷贝与计算的完全重叠。

### 13.6.2 单阶段 Pipeline

最基本的 Pipeline 使用模式类似于 barrier：

```cuda
#include &lt;cuda/pipeline&gt;

__global__ void single_stage_kernel(int *data, int *result, size_t size) {
    extern __shared__ int shared[];
    auto block = cooperative_groups::this_thread_block();

    // 创建单阶段 pipeline
    __shared__ cuda::pipeline_shared_state&lt;
        cuda::thread_scope::thread_scope_block, 1&gt; shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    for (size_t offset = 0; offset < size; offset += block.size()) {
        // 生产者：获取一个阶段，发起异步拷贝
        pipeline.producer_acquire();
        cuda::memcpy_async(block, shared, data + offset,
                           sizeof(int) * block.size(), pipeline);
        pipeline.producer_commit();

        // 消费者：等待数据就绪
        pipeline.consumer_wait();

    
        // 计算：每个线程处理自己对应的元素
        int i = threadIdx.x;
        result[offset + i] = shared[i] * 2;

        // 释放阶段
        pipeline.consumer_release();
    }
}
```

### 13.6.3 多阶段 Pipeline

多阶段 Pipeline 才是真正展现实力的地方。通过使用多个共享内存缓冲区（"阶段"），你可以让数据拷贝和计算<strong>完全重叠</strong>：

```cuda
#include &lt;cuda/pipeline&gt;

constexpr int stages = 2; // 双缓冲

__global__ void pipeline_kernel(int *data, int *result, size_t size) {
    extern __shared__ int shared[];
    auto block = cooperative_groups::this_thread_block();

    // 分配共享内存缓冲区：stages 个阶段，每个阶段 block.size() 个 int
    int *buffer[stages];
    for (int i = 0; i < stages; i++) {
        buffer[i] = shared + i * block.size();
    }

    // 创建多阶段 pipeline
    __shared__ cuda::pipeline_shared_state&lt;
        cuda::thread_scope::thread_scope_block, stages&gt; shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);

    size_t num_blocks = size / block.size();

    // 预热：先发起前 stages 个拷贝
    for (int stage = 0; stage < stages - 1; stage++) {
        pipeline.producer_acquire();
        cuda::memcpy_async(block, buffer[stage], data + stage * block.size(),
                           sizeof(int) * block.size(), pipeline);
        pipeline.producer_commit();
    }

        // 主循环：消费和生产的流水线重叠
    // 注意：生产者写入的缓冲区索引必须不同于消费者读取的索引
    for (size_t i = 0; i < num_blocks - (stages - 1); i++) {
        int producer_stage = (i + stages - 1) % stages;  // 生产者写入的缓冲区
        int consumer_stage = i % stages;                 // 消费者读取的缓冲区

        // 生产下一批数据（异步拷贝）
        pipeline.producer_acquire();
        size_t producer_offset = (i + stages - 1) * block.size();
        if (producer_offset < size) {
            cuda::memcpy_async(block, buffer[producer_stage],
                               data + producer_offset,
                               sizeof(int) * block.size(), pipeline);
        }
        pipeline.producer_commit();

        // 等待生产者提交完成（确保所有线程的提交操作就绪）
        __syncthreads();

        // 消费当前阶段的数据
        pipeline.consumer_wait();
        int j = threadIdx.x;
        if (j < block.size()) {
            result[i * block.size() + j] = buffer[consumer_stage][j] * 2;
        }
        pipeline.consumer_release();
    }

    // 排空：处理剩余的 stages-1 个阶段
    for (size_t i = num_blocks - (stages - 1); i < num_blocks; i++) {
        int stage = i % stages;
        pipeline.consumer_wait();
        for (int j = threadIdx.x; j < block.size(); j += block.size()) {
            result[i * block.size() + j] = buffer[stage][j] * 2;
        }
        pipeline.consumer_release();
    }
}
```

### 13.6.4 Pipeline 接口总结

| Pipeline 成员函数 | 描述 |
|------------------|------|
| `producer_acquire()` | 获取 pipeline 队列中的一个可用阶段 |
| `producer_commit()` | 提交在 `producer_acquire` 之后发起的异步操作到当前阶段 |
| `consumer_wait()` | 等待下一个阶段的数据就绪（消费者端） |
| `consumer_release()` | 释放当前阶段，使其可被生产者重新获取 |

## 13.7 Pipeline 原语接口详解

CUDA Programming Guide 10.28.4 节详细描述了 Pipeline 的原语接口（pipeline primitives interface）。虽然我们通常使用高级的 `producer_acquire()/consumer_wait()` 接口，但理解底层原语对性能调优和调试至关重要。

### 13.7.1 Pipeline 原语概述

| 原语 | 级别 | 描述 |
|------|------|------|
| `memcpy_async` | 操作 | 发起异步拷贝，绑定到 pipeline 的当前阶段 |
| `commit` | 阶段管理 | 提交当前阶段：递增 pipeline 序列号 |
| `wait` | 阶段管理 | 等待指定阶段完成：阻塞直到数据就绪 |
| `arrive_on` | 同步 | 在 barrier 上到达，将阶段链接到 barrier |

### 13.7.2 原语与高级接口的映射

```cuda
// 高级 API：
pipe.producer_acquire();    // 内部：等待阶段可用
cuda::memcpy_async(...);    // 不变
pipe.producer_commit();     // 内部：pipeline_producer_commit()

// 等价的原语操作：
pipeline_producer_acquire(pipe);   // 获取可用阶段
cuda::memcpy_async(...);           // 发起异步拷贝
pipeline_producer_commit(pipe, shared_mem); // 提交（递增序列号）

// 消费者端：
pipeline_consumer_wait(pipe);       // 等待阶段数据就绪
// ... 使用数据 ...
pipeline_consumer_release(pipe);    // 释放阶段
```

### 13.7.3 commit 原语与 Warp Entanglement

commit 原语的行为受线程收敛性的影响。CUDA Programming Guide 描述了 "Warp Entanglement" 效应：

<strong>收敛 Warp</strong>：如果所有 32 个线程都执行 commit，pipeline 序列号递增 1。

<strong>发散 Warp</strong>：如果只有部分线程执行 commit，pipeline 序列号递增的次数等于执行 commit 的线程数。

<strong>实际序列 vs 感知序列</strong>：

> "Let PB be the warp-shared pipeline's actual sequence of batches. Let TB be a thread's perceived sequence of batches, as if the sequence were only incremented by this thread's invocation of the commit operation."

这意味着在发散 Warp 中，不同线程可能看到不同的序列号，导致意外行为。因此 CUDA Programming Guide 建议：

> "It is recommended that commit and arrive-on invocations are by converged threads: to not over-wait, by keeping threads' perceived sequence of batches aligned with the actual sequence, and to minimize updates to the barrier object."

当代码在 commit 前发散时，应使用 `__syncwarp` 重新收敛：

```cuda
// 坏做法：在发散 warp 中 commit
if (condition) {
    cuda::memcpy_async(...);
    pipe.producer_commit();  // 只有部分线程执行，可能导致 over-wait
}

// 好做法：重新收敛后再 commit
bool any_thread_needs_copy = ...;
if (condition) {
    cuda::memcpy_async(...);
}
__syncwarp();  // 重新收敛 warp
if (any_thread_needs_copy) {
    pipe.producer_commit();  // 所有线程执行，行为可预测
}
```

### 13.7.4 wait 原语与 prior batches

`pipeline_consumer_wait_prior<N>()` 等待"当前实际序列之前 N 个"的阶段完成。`N=0` 等价于 `pipeline_consumer_wait()`。在发散 Warp 中，线程可能等待比预期更多的阶段——这就是 "over-wait" 现象。

### 13.7.5 完成函数（Completion Function）

cuda::barrier 支持一个可选的<strong>完成函数<strong>，它在屏障每次相位翻转时自动执行。完成函数在最后一个 `arrive() `调用触发翻转时运行，且在所有被阻塞的 `wait() `唤醒之前执行。

完成函数通过模板参数指定，而不是运行时设置：

```cuda
// 定义一个完成函数（可以是一个函数对象）
struct MyCompletion {
    __device__ void operator()() {
        // 当屏障翻转时自动执行，例如：递增共享计数器
    }
};

// 使用完成函数作为模板参数
using barrier_with_completion = cuda::barrier<
    cuda::thread_scope::thread_scope_block,
    MyCompletion
>;

__shared__ barrier_with_completion bar;
// ... init(&bar, block.size());
```

完成函数是 CUDA 12.0 及以上版本引入的特性，低版本不支持；完成函数由最后一个到达的线程执行；完成函数中不能调用任何可能阻塞的操作（如` wait()`、`memcpy_async` 等），否则会导致死锁。

## 13.8 性能测量与分析

正确使用异步 SIMT 编程模型的最终目的是性能提升。本节讨论如何测量和分析异步操作的性能。

### 13.8.1 使用 CUDA Events 测量重叠

```cuda
cudaEvent_t copy_start, copy_end, compute_start, compute_end;
cudaEventCreate(&copy_start);
cudaEventCreate(&copy_end);
cudaEventCreate(&compute_start);
cudaEventCreate(&compute_end);

// 在 kernel 内部无法直接使用 CUDA events，
// 可以通过 host 端的事件 + 多个 kernel launch 来测量重叠
for (int iter = 0; iter < num_iters; iter++) {
    cudaEventRecord(copy_start, stream);
    copy_kernel<<<grid, block, smem, stream>>>(d_src, d_buf);
    cudaEventRecord(copy_end, stream);

    cudaEventRecord(compute_start, stream);
    compute_kernel<<<grid, block, smem, stream>>>(d_buf, d_dst);
    cudaEventRecord(compute_end, stream);
}
cudaDeviceSynchronize();

float copy_ms, compute_ms;
cudaEventElapsedTime(&copy_ms, copy_start, copy_end);
cudaEventElapsedTime(&compute_ms, compute_start, compute_end);

// 如果两个操作在同一个流中顺序执行，总时间 = copy_ms + compute_ms
// 如果有重叠，总时间 < copy_ms + compute_ms
```

### 13.8.2 异步操作的性能陷阱

1. <strong>拷贝太小</strong>：如果每次 `memcpy_async` 拷贝的数据量太小（例如，每个线程拷贝 4 字节），异步启动的开销可能超过收益。建议每次拷贝至少 128 字节。

2. <strong>过度同步</strong>：在 pipeline 中过早调用 `consumer_wait()` 可能阻塞消费者线程，失去重叠机会。尽可能延迟 `consumer_wait()` 到数据真正需要之前。

3. <strong>Warp divergence</strong>：在 commit 之前发散的 warp 会导致 over-wait，因为实际序列号可能远大于感知序列号。

4. <strong>寄存器溢出</strong>：pipeline 的多阶段缓冲区增加了共享内存使用。如果同时使用过多共享内存，可能导致线程块占用率下降。

### 13.8.3 使用 Nsight Compute 分析

NVIDIA Nsight Compute 提供了针对异步操作的专门指标：

- `sm__pipe_tensor_op_cycles_active`：Tensor Pipeline 活跃周期
- `l1tex__data_pipe_lsu_wavefronts`：LSU pipeline 负载
- `smsp__warp_cycles_per_issue_stalled`：因等待数据而停滞的周期

通过这些指标可以量化数据搬运与计算的实际重叠程度。

## 13.9 实战进阶：生产-消费者模式的完整实现

下面是一个更完整的 Warp Specialization 实现，展示了生产者和消费者之间通过两个 barrier 实现的双缓冲协调：

```cuda
#include <cuda/barrier>
#include <cooperative_groups.h>

using barrier = cuda::barrier<cuda::thread_scope::thread_scope_block>;

// 生产者：负责加载数据
__device__ void producer_work(barrier &ready, barrier &filled,
                              float *buf, const float *__restrict__ global_in,
                              int block_start, int buffer_len)
{
    // 等待缓冲区就绪
    ready.arrive_and_wait();

    // 加载数据到共享内存缓冲区（异步）
    auto block = cooperative_groups::this_thread_block();
    // 使用 memcpy_async 进行异步加载
    cuda::memcpy_async(block, buf, global_in + block_start,
                       sizeof(float) * buffer_len, filled);

    // 生产者不需要等待加载完成——它在 filled barrier 上 arrive
    // filled.consumer 会等到数据加载完成
}

// 消费者：负责计算
__device__ void consumer_work(barrier &ready, barrier &filled,
                              float *buf, float *global_out,
                              int block_start, int buffer_len)
{
    // 等待数据被填满
    filled.arrive_and_wait();

    // 数据就绪，进行计算
    for (int i = threadIdx.x; i < buffer_len; i += blockDim.x) {
        global_out[block_start + i] = buf[i] * 2.0f + 1.0f;
    }

    // 通知生产者缓冲区可以被重新填充
    ready.arrive();
}

__global__ void full_producer_consumer(
    const float *__restrict__ global_in,
    float *__restrict__ global_out,
    int N, int buffer_len)
{
    // 双缓冲区
    __shared__ extern float buffer[];
    float *buf_a = buffer;                     // buffer_0
    float *buf_b = buffer + buffer_len;        // buffer_1

    // 四个 barrier：
    // ready[0/1]：buf_0/1 就绪（可以被填充）
    // filled[0/1]：buf_0/1 已填满（可以被消费）
    // ready 由消费者释放，filled 由生产者填充
    __shared__ barrier ready[2];
    __shared__ barrier filled[2];

    auto block = cooperative_groups::this_thread_block();
    int tid = block.thread_rank();

    // 初始化 barrier
    if (tid < 2) {
        init(&ready[tid], block.size());
        init(&filled[tid], block.size());
    }
    block.sync();

    // 空间分割：warp 0 是生产者，其余 warp 是消费者
    bool is_producer = (tid / warpSize) == 0;

    if (is_producer) {
        // 预热：消费者需要 ready 信号才知道可以开始填充
        // 但生产者首先需要等待消费者释放 ready
        filled[0].arrive_and_wait();  // 标记 buf_0 已"填充"（初始为空需要此操作来完成初始化）

        int total_blocks = N / buffer_len;
        for (int b = 0; b < total_blocks; b++) {
            int buf_idx = b % 2;
            int block_start = b * buffer_len;

            ready[buf_idx].arrive_and_wait();  // 等待缓冲区就绪

            // 异步加载
            cuda::memcpy_async(block, (buf_idx == 0 ? buf_a : buf_b),
                               global_in + block_start,
                               sizeof(float) * buffer_len,
                               filled[buf_idx]);
        }
    } else {
        // 消费者：先通知生产者所有缓冲区初始可用
        ready[0].arrive();
        ready[1].arrive();

        int total_blocks = N / buffer_len;
        for (int b = 0; b < total_blocks; b++) {
            int buf_idx = b % 2;
            int block_start = b * buffer_len;

            filled[buf_idx].arrive_and_wait();  // 等待缓冲区被填满

            // 消费数据
            float *curr_buf = (buf_idx == 0 ? buf_a : buf_b);
            for (int i = tid; i < buffer_len; i += block.size()) {
                global_out[block_start + i] = curr_buf[i] * 2.0f + 1.0f;
            }

            // 通知生产者缓冲区就绪
            ready[buf_idx].arrive();
        }
    }
}
```

## 13.10 综合实战：软件流水线化的批量数据处理

下面是一个完整的多阶段 Pipeline 程序，模拟了典型的深度学习推理预处理流水线：

```cuda
// 文件: async_pipeline_demo.cu
// 编译: nvcc -arch=sm_80 async_pipeline_demo.cu -o async_pipeline_demo
// 硬件要求: NVIDIA Ampere A100 或更新 (CC 8.0+)

#include &lt;stdio.h&gt;
#include &lt;stdlib.h&gt;
#include &lt;cuda/pipeline&gt;
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
    __shared__ cuda::pipeline_shared_state&lt;
        cuda::thread_scope::thread_scope_block, stages&gt; pipe_state;
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
```

### 流水线效率分析

多阶段 Pipeline 的关键优势在于：当生产者（数据拷贝）在处理阶段 N 时，消费者（计算）同时处理阶段 N-1。通过 `stages` 个缓冲区，实现了以下重叠：

```
时间轴:  T1    T2    T3    T4    T5    T6    ...
阶段0:   [Load] [----] [Load] [----] [Load] ...
阶段1:   [----] [Load] [----] [Load] [----] ...
阶段2:   [----] [----] [Load] [----] [Load] ...
计算:    [Wait] [Comp] [Comp] [Comp] [Comp] ...
```

在稳态下，Load 和 Compute 完全重叠。阶段数越多，越容易在拷贝延迟波动时维持重叠，但也会消耗更多共享内存。

## 13.11 同步机制的对比

| 特性 | `__syncthreads()` | `cuda::barrier` | `cuda::pipeline` |
|------|------------------|-----------------|-----------------|
| arrive/wait 分离 | 否 | 是 | 是（通过 producer/consumer） |
| 多阶段支持 | 否 | 否 | 是 |
| 提前退出 | 否 | 是 | 否（通过 barrier 变通） |
| HW加速（CC 8.0+） | N/A | 是 | 是 |
| 与 memcpy_async 集成 | 否 | 是 | 是 |
| 共享内存开销 | 最小 | 中等 | 较大（每阶段需缓冲区） |
| 适用场景 | 简单同步 | 重叠计算与传输 | 复杂流水线 |

CUDA Programming Guide 也给出了建议：

> "If the intention of the user is to synchronize a full thread block or a full warp we recommend using `__syncthreads()` and `__syncwarp(mask)` respectively for performance reasons."

即：对于不需要异步重叠的简单同步场景，使用传统的 `__syncthreads()` 仍然是最优选择。

## 13.12 异步操作的调试与错误处理

### 13.12.1 常见问题诊断

使用异步操作时，常见的问题包括：

<strong>1. 数据竞争（Data Race）</strong>：在异步拷贝完成前访问共享内存。

```cuda
// 错误：在 barrier 等待之前使用共享内存
cuda::memcpy_async(block, shared, global_in, size, barrier);
int val = shared[threadIdx.x];  // 数据竞态！
barrier.arrive_and_wait();

// 正确：等待 barrier 后再使用
cuda::memcpy_async(block, shared, global_in, size, barrier);
barrier.arrive_and_wait();
int val = shared[threadIdx.x];  // 安全
```

<strong>2. Barrier 未初始化</strong>：在使用 `arrive()` 之前未调用 `init()`。症状：未定义行为。

<strong>3. Phase 混淆</strong>：使用过期的 token 调用 `wait()`。症状：死锁或提前返回。

```cuda
// 错误示例
auto token = bar.arrive();   // phase N
bar.arrive_and_wait();       // 又到达一次
bar.wait(std::move(token));  // 使用 phase N 的过期 token
```

<strong>4. Pipeline 阶段溢出</strong>：`producer_acquire()` 超过 pipeline 的阶段数导致死锁。

### 13.12.2 使用环境变量和工具调试

- `CUDA_LAUNCH_BLOCKING=1`：使所有 kernel 启动变为同步，便于隔离问题；
- `cuda-memcheck` 工具检测共享内存的非法访问；
- NVIDIA Compute Sanitizer 可以检测异步操作中的数据竞争。

## 13.13 高级 Pipeline 模式

### 13.13.1 Pipeline 与 GEMM 的软件流水线

在矩阵乘法（GEMM）中，多阶段 Pipeline 用于实现 Global-to-Shared 内存拷贝与 Tensor Core 计算的重叠：

```cuda
constexpr int STAGES = 3; // 3阶段流水线

__global__ void gemm_pipeline(
    const half *__restrict__ A, const half *__restrict__ B,
    half *__restrict__ C, int M, int N, int K, int tile_size)
{
    extern __shared__ half smem[];
    half *As[STAGES], *Bs[STAGES];
    for (int s = 0; s < STAGES; s++) {
        As[s] = smem + s * 2 * tile_size;
        Bs[s] = As[s] + STAGES * 2 * tile_size;
    }

    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block, STAGES> pipe_state;
    auto pipe = cuda::make_pipeline(block, &pipe_state);

    float accum[16][16] = {0.0f}; // accumulator registers

    int k_tiles = K / tile_size;

    // 预热：加载前 STAGES-1 个 tile
    for (int s = 0; s < STAGES - 1 && s < k_tiles; s++) {
        pipe.producer_acquire();
        cuda::memcpy_async(block, As[s], A + s * tile_size,
                          sizeof(half) * tile_size, pipe);
        cuda::memcpy_async(block, Bs[s], B + s * tile_size,
                          sizeof(half) * tile_size, pipe);
        pipe.producer_commit();
    }

    // 主循环
    for (int k = 0; k < k_tiles - (STAGES - 1); k++) {
        int stage = k % STAGES;

        pipe.producer_acquire();
        int next_k = k + STAGES - 1;
        if (next_k < k_tiles) {
            cuda::memcpy_async(block, As[stage], A + next_k * tile_size,
                              sizeof(half) * tile_size, pipe);
            cuda::memcpy_async(block, Bs[stage], B + next_k * tile_size,
                              sizeof(half) * tile_size, pipe);
        }
        pipe.producer_commit();

        pipe.consumer_wait();
        // Tensor Core 计算（伪代码）
        // mma_sync(accum, As[stage], Bs[stage], accum);
        pipe.consumer_release();
    }

    // 排空
    for (int k = k_tiles - (STAGES - 1); k < k_tiles; k++) {
        int stage = k % STAGES;
        pipe.consumer_wait();
        // mma_sync(accum, As[stage], Bs[stage], accum);
        pipe.consumer_release();
    }

    // 存储结果（省略）
}
```

### 13.13.2 动态阶段数选择

Pipeline 的阶段数涉及权衡：

| 阶段数 | 共享内存 | 延迟隐藏 | 占用率 |
|-------|---------|---------|-------|
| 2 | 2x | 中等 | 高 |
| 3 | 3x | 好 | 中等 |
| 4 | 4x | 很好 | 较低 |
| 5+ | 5x+ | 极好 | 低 |

```cuda
int select_pipeline_stages(size_t tile_bytes, size_t smem_per_block) {
    int max_stages = smem_per_block / tile_bytes;
    return max(2, min(max_stages, 6));
}
```

### 13.13.3 Pipeline 交错模式

```cuda
// 双 Pipeline 交错
__shared__ cuda::pipeline_shared_state<thread_scope_block, 3> pipe_a, pipe_b;
auto pa = cuda::make_pipeline(block, &pipe_a);
auto pb = cuda::make_pipeline(block, &pipe_b);

// 交错生产
pa.producer_acquire(); memcpy_async(..., pa); pa.producer_commit();
pb.producer_acquire(); memcpy_async(..., pb); pb.producer_commit();

// 交错消费
pa.consumer_wait(); compute_a(); pa.consumer_release();
pb.consumer_wait(); compute_b(); pb.consumer_release();
```

## 13.14 与主机端异步操作的对比

| 特性 | 主机端异步 (cudaMemcpyAsync) | 设备端异步 (memcpy_async) |
|------|---------------------------|-------------------------|
| 操作粒度 | 整个 buffer | 块内 tiles |
| 同步机制 | CUDA Streams + Events | barrier / pipeline |
| 重叠范围 | kernel 与拷贝 | 块内计算与拷贝 |
| DMA 使用 | Copy Engine | SM 内部的硬件单元 |
| 编程复杂度 | 低 | 中等 |
| CC 要求 | 所有 | 7.0+ |

两种异步方式可以<strong>同时使用</strong>：设备端 pipeline 负责细粒度的 Global→Shared 重叠，主机端 Stream 负责不同 kernel 间的粗粒度重叠。

## 13.15 常见问题与故障排除

### 13.15.1 `cuda::barrier` 死锁

<strong>症状</strong>：所有线程卡在 `bar.wait()` 调用上。

<strong>常见原因</strong>：
1. Barrier 初始化的 `expected_count` 不正确（大于实际调用 `arrive()` 的线程数）；
2. 有线程在条件分支中未调用 `arrive()`；
3. 使用过期的 token。

<strong>解决方法</strong>：
- 确保 `init(&bar, expected_count)` 中的 `expected_count` 等于实际参与线程数；
- 确保所有参与线程都在同一个代码路径中调用 `arrive()`；
- 每次迭代使用新的 token。

### 13.15.2 Pipeline 死锁

<strong>症状</strong>：`producer_acquire()` 永不返回。

<strong>原因</strong>：pipeline 的所有 `stages` 个阶段都被占用且消费者未释放。

```cuda
// 错误示例：忘记调用 consumer_release
for (int i = 0; i < num_iter; i++) {
    pipe.consumer_wait();
    compute();
    // 忘记 pipe.consumer_release()！
}
// 几个迭代后死锁
```

### 13.15.3 `memcpy_async` 性能差

<strong>常见原因</strong>：
1. 拷贝大小不是 16 字节的倍数（回退到逐字节拷贝）；
2. 指针未对齐（回退到非加速路径）；
3. 每个线程拷贝的数据量太小（启动开销主导）；
4. Warp 发散导致 over-wait。

<strong>检查清单</strong>：
- 使用 `cuda::aligned_size_t<16>()` 声明对齐；
- 确保共享内存 `alignas(16)`；
- 确保全局内存基地址 128 字节对齐；
- 在 commit/wait 前使用 `__syncwarp()` 恢复 warp 收敛。

### 13.15.4 数据完整性错误

<strong>症状</strong>：某些输出值不正确或为零。

<strong>诊断步骤</strong>：
1. 检查 `consumer_wait()` 是否在访问阶段数据之前调用；
2. 检查双缓冲索引计算是否正确；
3. 确认 `producer_commit()` 在所有异步拷贝之后调用；
4. 检查 `fence_proxy_async_shared_cta()` 是否在写回前调用（如果使用 TMA）。

## 13.16 性能基准参考

以下是在 A100 (CC 8.0) 上进行批量数据处理（100 批次，256 个 float 每批次）的性能参考：

| 方案 | 总时间 (us) | 相对性能 | 共享内存 |
|------|----------|---------|---------|
| 串行加载+计算 (`__syncthreads`) | 245 | 1.0x | 1 KB |
| `memcpy_async` + `barrier` (单缓冲) | 198 | 1.2x | 1 KB |
| Pipeline 2-stage | 145 | 1.7x | 2 KB |
| Pipeline 3-stage | 128 | 1.9x | 3 KB |
| Pipeline 4-stage | 125 | 2.0x | 4 KB |

（注：实际性能取决于内存延迟、计算密度、数据大小等因素。上述为示意性参考。）

关键观察：
1. 单缓冲 `memcpy_async` 比传统 `__syncthreads` 稍有提升（约 20%），因为绕过了寄存器；
2. 多阶段 pipeline 带来显著提升（2-stage 比单阶段快 36%）；
3. 从 3-stage 到 4-stage 的增量很小，因为瓶颈从拷贝转向了计算。

## 13.17 迁移指南：从 __syncthreads 到异步模型

如果你的现有代码使用传统的 `__syncthreads()` 模式，迁移到异步模型应该循序渐进：

### 第一步：识别候选 kernel

适合迁移的 kernel 特征：
- 有规律的 "加载数据 → 同步 → 计算 → 同步" 循环；
- 内存操作占用显著时间；
- 计算和加载之间有独立性（计算不依赖于即将加载的数据）。

### 第二步：引入 barrier 替代 __syncthreads

```cuda
// 之前
for (int i = 0; i < n; i++) {
    load_data(shared, global, i);  // 各线程加载数据
    __syncthreads();               // 所有线程等待

    compute(shared, output, i);    // 计算
    __syncthreads();               // 所有线程等待
}

// 之后
for (int i = 0; i < n; i++) {
    load_data_async(shared, global, i, bar);  // 异步加载
    auto token = bar.arrive();                // 到达（不阻塞）
    // 此处可以执行独立计算
    bar.wait(std::move(token));               // 等待数据就绪

    compute(shared, output, i);
    __syncthreads();  // 仍需要——确保计算完成
}
```

### 第三步：引入 pipeline

```cuda
// 使用多阶段 pipeline 实现完全重叠
for (int i = 0; i < n; i++) {
    pipe.producer_acquire();
    load_data_async(stage_buffer[i%stages], global, i, pipe);
    pipe.producer_commit();

    if (i >= stages - 1) {
        pipe.consumer_wait();
        compute(stage_buffer[i%stages], output, i - stages + 1);
        pipe.consumer_release();
    }
}
```

### 第四步：测量和调整

每次迁移后使用 Nsight Compute 测量：
- 检查 `memcpy_async` 是否使用了硬件加速路径；
- 验证计算和数据拷贝是否有实际重叠；
- 调整 pipeline 阶段数以平衡共享内存和吞吐量。

## 13.18 动手体验2：异步归约

在第 12 章我们使用 Thread Block Cluster + DSM 进行分布式归约。这里展示一个使用 `cuda::barrier` 进行分阶段异步归约的替代方案：

```cuda
// async_reduce.cu
// 使用 cuda::barrier 实现异步块内归约
__global__ void async_reduce(const float *__restrict__ input,
                              float *__restrict__ output, int N)
{
    extern __shared__ float smem[];
    auto block = cooperative_groups::this_thread_block();

    // ___shared__ barrier
    __shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> bar;
    if (block.thread_rank() == 0) {
        init(&bar, block.size());
    }
    block.sync();

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // 阶段1：加载数据（异步拷贝）
    float local_sum = 0.0f;
    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        local_sum += input[i];
    }

    // 阶段2：写入共享内存
    smem[tid] = local_sum;

    // 阶段3：arrive（不阻塞，但确保写入可见）
    auto token = bar.arrive();

    // 阶段4：在等待期间可以做独立计算（如果有的话）
    // 这里没有独立计算，直接等待
    bar.wait(std::move(token));

    // 阶段5：归约树（使用共享内存分治）
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();  // 每轮需要同步
    }

    // 写回结果
    if (tid == 0) {
        atomicAdd(output, smem[0]);
    }
}
```

### 为什么这个示例没有完全利用异步模型的优势？

在上述归约示例中，barrier 的 arrive/wait 分离没有带来性能提升，因为：
1. 加载数据后没有可以独立于同步执行的"中间计算"；
2. 归约树本身是严格依赖的——每一轮都依赖于前一轮的结果。

这说明了异步模型的一个重要原则：<strong>arrive/wait 分离只有在存在可以独立于同步结果执行的计算时才有效</strong>。如果所有计算都依赖于同步后的数据，那么分离没有意义。

### 何时使用 barrier 而非 __syncthreads？

```cuda
// 场景A：适合 barrier —— 有独立的中间计算
for (int i = 0; i < n; i++) {
    produce_data(smem);             // 写入共享内存
    auto token = bar.arrive();      // 到达（写入可见）
    independent_compute(registers); // 独立计算（不依赖共享内存）
    bar.wait(std::move(token));     // 等待其他线程
    consume_data(smem);             // 使用共享内存数据
}

// 场景B：不适合 barrier —— 使用 __syncthreads() 即可
for (int i = 0; i < n; i++) {
    produce_data(smem);         // 写入共享内存
    __syncthreads();            // 等待所有线程
    consume_data(smem);         // 立即使用共享内存数据
}
```

## 13.19 本章小结

本章全面介绍了 CUDA 异步 SIMT 编程模型，涵盖以下要点：

1. <strong>异步编程的核心思想</strong>：将同步操作拆分为 arrive 和 wait，在二者之间执行独立计算，实现计算与同步的重叠。

2. <strong>线程作用域</strong>：从 `thread_scope_thread` 到 `thread_scope_system`，定义了同步对象的参与范围。

3. <strong>`cuda::barrier`</strong>：提供 arrive/wait 分离的异步屏障，支持时间分割（Temporal Splitting）、空间分割（Warp Specialization）和提前退出（Early Exit）。

4. <strong>`cuda::memcpy_async`</strong>：实现全局内存到共享内存的异步数据拷贝，不经过寄存器，在 CC 8.0+ 上有硬件加速。使用时注意对齐要求。

5. <strong>`cuda::pipeline`</strong>：多阶段流水线，通过 producer/consumer 模式实现数据拷贝与计算的完全重叠。多阶段是 Pipeline 相比于 Barrier 的核心优势。

6. <strong>选择建议</strong>：简单同步用 `__syncthreads()`；需要计算与同步重叠用 `cuda::barrier`；需要复杂的多阶段数据流水线用 `cuda::pipeline`。

异步 SIMT 编程模型是现代 GPU 编程中不可或缺的工具。随着 GPU 内存带宽与计算能力之间的差距不断拉大，将数据搬运隐藏在计算之后变得愈发重要。掌握这些 API 将显著提升你的 CUDA 程序性能。

## 13.20 习题

1. 解释 `cuda::barrier` 的"时间分割"（Temporal Splitting）五阶段模型。为什么 arrive 和 wait 之间的阶段是实现重叠的关键？

2. 使用 `cuda::barrier` 重写一个简单的向量加法核函数（不使用异步拷贝，仅使用 barrier 进行同步），比较与传统 `__syncthreads()` 版本的区别。

3. 为什么 `cuda::memcpy_async` 要求 16 字节对齐？如果传递了未对齐的指针会发生什么？

4. 在 Pipeline 的多阶段模式中，阶段数越多是否总是越好？讨论阶段数与共享内存消耗之间的权衡。

5. 修改 13.7 节的综合示例，增加一个"预热"核函数在 Pipeline 主循环之前填充 L2 缓存，观察对性能的影响。

6. 说明 Warp Specialization 模式中为什么需要 4 个 barrier（2×2双缓冲），而不是 2 个。

## 13.21 参考文献

1. CUDA C++ Programming Guide 13.0, Section 5.5 "Asynchronous SIMT Programming Model"
2. CUDA C++ Programming Guide 13.0, Section 10.26 "Asynchronous Barrier"
3. CUDA C++ Programming Guide 13.0, Section 10.27 "Asynchronous Data Copies"
4. CUDA C++ Programming Guide 13.0, Section 10.28 "Asynchronous Data Copies using cuda::pipeline"
5. libcu++ API Documentation — barrier and pipeline APIs
6. NVIDIA Ampere GA100 GPU Architecture Whitepaper
