# Lab3 NCU 报告解读：共享内存分块（sgemm_shared_mem_block）

本教程以 `sgemm_shared_mem_block`（共享内存 Tiling 矩阵乘法）内核为例，对照 Lab2（合并访问）的数据，逐步解读 NCU 报告的三个核心视图，帮助你理解共享内存分块优化的实际效果与新出现的瓶颈。

---

## 一、Summary（摘要）视图

<!-- ✅ 在此处插入 Image 1（Summary 标签页截图） -->
![图1-NCU-Summary视图](image1.png)

**1.1 顶部元数据**

| 字段 | 含义 | 本例数值 | Lab2 对比 |
|------|------|---------|----------|
| Result | 内核编号与名称 | 897 - sgemm_shared_mem_block | sgemm_global_mem_coalesce |
| Size | Grid × Block 配置 | (128, 128, 1) × (1024, 1, 1) | (128, 128, 1) × (1024, 1, 1) |
| Time | 单次执行时间 | **28.40 ms** | 46.32 ms |
| Cycles | 经历的时钟周期数 | 36,213,173 | 59,062,641 |
| GPU | GPU 型号 | NVIDIA A800-SXM4-80GB | 同 |
| SM Frequency | SM 实际主频 | 1.27 GHz | 1.27 GHz |

**1.2 表格各列解读**

表格共 2 行（ID=0 和 ID=1），均对应同一内核的两次运行：

| 列名 | 本例数值 | Lab2 数值 | 变化含义 |
|------|---------|----------|---------|
| Estimated Speedup [%] | 0.00% | 0.00% | NCU 未发现可量化的优化空间 |
| Duration [ms] | 28.40 | 46.32 | **↓ 38.7%，执行时间显著缩短** |
| Runtime Improvement [ms] | 0.00 | 0.00 | 无进一步可量化提升 |
| **Compute Throughput [%]** | **75.57** | **67.32** | **↑ 12.3%，计算单元更充分** |
| **Memory Throughput [%]** | **89.76** | **67.33** | **↑ 33.3%，内存子系统利用率回升** |

> **关键结论**：执行时间从 46.32 ms 降至 28.40 ms，缩短了约 **38.7%**。但与此同时，Memory Throughput 重新拉高到 89.76%，NCU 的提示也从 Lab2 的「Balanced Throughput」重新变回「**High Throughput**」，说明共享内存的引入在提升计算效率的同时，**带来了新的内存访问瓶颈**。

---

## 二、Details 视图 —— GPU Speed of Light & Launch Statistics

<!-- ✅ 在此处插入 Image 2（Details 标签页上半部分截图） -->
![图2-NCU-Details-SpeedOfLight](image2.png)

**2.1 GPU Speed of Light Throughput（光速吞吐量）**

| 指标 | 本例数值 | Lab2 数值 | Lab1 数值 | 变化含义 |
|------|---------|----------|----------|---------|
| Compute (SM) Throughput [%] | 75.57 | 67.32 | 5.89 | ↑ 持续提升，计算占比更高 |
| Memory Throughput [%] | 89.76 | 67.33 | 93.92 | ↑ 重新回升，内存再次成为主要瓶颈 |
| **L1/TEX Cache Throughput [%]** | **89.99** | **67.61** | **99.09** | **↑ 33%，L1 再次接近饱和** |
| L2 Cache Throughput [%] | 14.88 | 10.64 | 1.29 | ↑ L2 参与度略有提升 |
| DRAM Throughput [%] | 15.17 | 9.32 | 0.07 | ↑ DRAM 持续被合理利用 |

**核心变化解读**：

Lab3 与 Lab1 的 L1/TEX 都接近 90–99%，但**瓶颈本质完全不同**：

- **Lab1**：L1 高是因为全局内存随机访问导致大量 Cache Miss，数据在 L1 层反复等待，DRAM 几乎不动（0.07%）。
- **Lab3**：L1 高是因为每个线程通过共享内存进行大量数据复用，DRAM 已被合理利用（15.17%），L2 也更活跃（14.88%）。这里 L1 的瓶颈来自**共享内存的 Bank Conflict**，而不是全局内存的低效访问。

**重要对比**：

| 瓶颈来源 | Lab1（Naive） | Lab3（共享内存） |
|---------|-------------|----------------|
| L1/TEX 高的原因 | 全局内存随机访问，Cache Miss | 共享内存 Bank Conflict |
| DRAM 利用率 | 0.07%（几乎不工作） | 15.17%（正常工作） |
| 可优化方向 | 改为合并访问 | 消除 Bank Conflict |

NCU 给出了**High Throughput（高吞吐）**提示：

> 内核已使用超过 80% 的内存性能，要进一步提升，需要将工作从最繁忙的单元转移到其他单元。建议从 **Memory Workload Analysis → L1** 区块开始分析。

**2.2 Launch Statistics（启动统计）**

| 指标 | 本例数值 | Lab2 数值 | 变化说明 |
|------|---------|----------|---------|
| Grid Size | 16,384 | 16,384 | 相同 |
| Block Size | 1,024 | 1,024 | 相同 |
| Registers Per Thread | 32 | 32 | 相同 |
| Threads | 16,777,216 | 16,777,216 | 相同 |
| Waves Per SM | 75.85 | 75.85 | 相同 |
| **Static Shared Memory Per Block** | **8.19 Kbyte** | **0 byte** | **↑ 首次使用共享内存！** |
| Dynamic Shared Memory Per Block | 0 | 0 | 无动态分配 |
| Shared Memory Configuration Size | 32.77 Kbyte | 8.19 Kbyte | SM 共享内存配置扩大 |
| # SMs | 108 | 108 | 相同 |

> **核心变化**：Static Shared Memory Per Block 从 0 变为 **8.19 Kbyte/block**，这是 Lab3 相比 Lab2 最本质的结构性改变。每个 Block 分配了约 8 KB 的共享内存，用于缓存矩阵 A 和 B 的分块数据，从而将对全局内存的重复读取转变为对共享内存的复用读取。

---

## 三、Details 视图 —— Occupancy & 工作负载分布

<!-- ✅ 在此处插入 Image 3（Details 标签页下半部分截图） -->
![图3-NCU-Details-Occupancy](image3.png)

**3.1 Occupancy（占用率）**

| 指标 | 本例数值 | Lab2 数值 | 变化 |
|------|---------|----------|------|
| Theoretical Occupancy [%] | 100 | 100 | 不变 |
| Achieved Occupancy [%] | **99.62** | **99.78** | 基本持平，略微下降 |
| Theoretical Active Warps/SM | 64 | 64 | 不变 |
| Achieved Active Warps/SM | 63.76 | 63.86 | 基本持平 |
| Block Limit Registers [block] | 2 | 2 | 相同（寄存器仍是主限制） |
| **Block Limit Shared Mem [block]** | **3** | **8** | **↓ 共享内存开始参与限制** |
| Block Limit Warps [block] | 2 | 2 | 相同 |
| Block Limit SM [block] | 32 | 32 | 相同 |

解读要点：

- **Block Limit Shared Mem 从 8 降至 3**：每个 SM 有约 32 KB 可用共享内存，每个 Block 使用了 8.19 KB，因此 32 / 8.19 ≈ 3.9，向下取整为 **3**。这意味着共享内存已经成为限制每个 SM 能运行多少 Block 的约束之一（虽然寄存器 Block Limit=2 仍更严格，实际每 SM 同时运行 2 个 Block）。
- **Achieved Occupancy 99.62% ≈ Lab2 的 99.78%**：占用率几乎没有变化，说明共享内存的引入没有显著降低 SM 的调度效率。
- 这意味着 Lab3 的性能提升（时间缩短 38.7%）**完全来自共享内存 Tiling 减少了全局内存访问次数**，而非占用率的变化。

**3.2 GPU and Memory Workload Distribution（工作负载分布）**

| 指标 | 本例数值 | Lab2 数值 | 变化含义 |
|------|---------|----------|---------|
| Average SM Active Cycles | 36,109,085 | 58,851,784 | ↓ 总执行周期大幅减少 |
| Average L1 Active Cycles | 36,109,085 | 58,851,784 | ↓ L1 活跃周期同步减少 |
| **Average L2 Active Cycles** | **34,478,692** | **56,230,860** | **↓ 显著减少，L2 压力降低** |
| **Average DRAM Active Cycles** | **6,865,535** | **6,877,940** | 几乎持平 |

关键发现：

- SM / L1 活跃周期从 58.8M 降至 36.1M（↓39%），与执行时间缩短比例吻合，说明共享内存 Tiling 整体减少了计算和访存的总工作量。
- **DRAM 活跃周期几乎不变**（6,865,535 vs 6,877,940）：这说明每次从 DRAM 取到的数据量本身没有显著变化，但**每块数据被复用的次数大幅增加**——同样的数据，Lab3 让更多线程从共享内存中读取，而不是每个线程都去访问全局内存。
- L2 活跃周期从 56.2M 降至 34.5M（↓39%），进一步验证全局内存流量大幅减少。

---

## 四、综合分析与优化方向

**三个实验的横向对比**

| 维度 | Lab1（Naive） | Lab2（合并访问） | Lab3（共享内存） |
|------|-------------|----------------|----------------|
| 执行时间 | 8.58 ms（小矩阵） | 46.32 ms | **28.40 ms** |
| Compute Throughput | 5.89% | 67.32% | **75.57%** |
| Memory Throughput | 93.92% | 67.33% | **89.76%** |
| L1/TEX Throughput | 99.09% | 67.61% | **89.99%** |
| L2 Throughput | 1.29% | 10.64% | **14.88%** |
| DRAM Throughput | 0.07% | 9.32% | **15.17%** |
| NCU 提示类型 | High（L1 饱和） | **Balanced（均衡）** | High（L1 再次饱和） |
| Achieved Occupancy | 93.80% | 99.78% | **99.62%** |
| 共享内存/Block | 0 byte | 0 byte | **8.19 Kbyte** |
| DRAM 活跃周期 | 9,836 | 6,877,940 | 6,865,535 |

**当前瓶颈：共享内存 Bank Conflict**

DRAM 吞吐仅 15.17%，而 L1/TEX 高达 89.99%，证明主要瓶颈已从全局内存转移到了**共享内存内部访问冲突**。

具体来说，Lab3 的内积计算通常写成：

```cpp
for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
    tmp += As[threadRow][dotIdx] * Bs[dotIdx][threadCol];
}
```

当多个线程在同一 Warp 内同时访问 `As[threadRow][dotIdx]`（不同 `threadRow`，相同 `dotIdx`）时，它们会映射到共享内存的**同一 Bank**，产生 Bank Conflict，导致访问被串行化，拉高 L1/TEX 利用率但降低有效吞吐。

**下一步优化建议**：通过**矩阵转置**（将 `As` 以列优先布局存储）或**调整访问步长**来错开 Bank 访问，消除 Bank Conflict，将 L1/TEX 的高占用率真正转化为有效的数据吞吐。
