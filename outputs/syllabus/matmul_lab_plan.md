# 矩阵乘法（SGEMM）CUDA优化实验计划

## 概述

本实验系列基于 [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) 博客及其配套源代码 [SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)，设计了一套渐进式CUDA性能优化实验。学生将从一个朴素的矩阵乘法实现出发，逐步应用各项CUDA优化技术，最终达到接近cuBLAS的性能水平。

**实验环境**：NVIDIA A6000 (Ampere, Compute Capability 8.6)，峰值计算能力约30 TFLOPS (FP32)，全局内存带宽768 GB/s。

**性能总览**（矩阵尺寸 4096x4096，单精度浮点）：

| 实验 | 优化技术 | GFLOPs/s | 相对cuBLAS |
|------|----------|----------|-----------|
| 实验1 | 朴素实现 | 309.0 | 1.3% |
| 实验2 | 全局内存合并访问 | 1986.5 | 8.5% |
| 实验3 | 共享内存缓存分块 | 2980.3 | 12.8% |
| 实验4 | 一维Block Tile | 8474.7 | 36.5% |
| 实验5 | 二维Block Tile | 15971.7 | 68.7% |
| 实验6 | 向量化内存访问 | 18237.3 | 78.4% |
| 实验7 | 参数自动调优 | 19721.0 | 84.8% |
| 实验8 | Warp Tile + 对比cuBLAS | 21779.3 | 93.7% |
| 参考 | cuBLAS | 23249.6 | 100.0% |

---

## 实验1：朴素矩阵乘法实现

**对应的CUDA概念**：CUDA编程模型核心概念——线程层级（Thread Hierarchy）、网格（Grid）、线程块（Block）、内置变量（blockIdx、threadIdx、blockDim、gridDim）

**参考CUDA Guide章节**：
- CUDA Programming Guide 第2章 "Programming Model"——Thread Hierarchy、Memory Hierarchy
- CUDA Programming Guide 第3章 "Programming Interface"——Kernel Launch

**原始代码**：`SGEMM_CUDA/src/kernels/1_naive.cuh`

**优化目标**：
1. 理解CUDA kernel的启动配置（gridDim、blockDim）
2. 掌握线程索引计算方式（如何将线程映射到结果矩阵的每个元素）
3. 实现基本的SGEMM运算：`C = alpha * A * B + beta * C`
4. 理解矩阵按行主序（row-major）存储的内存布局
5. 理解tile quantization（瓦片量化）问题：当矩阵维度不是block大小的整数倍时需要边界检查

**预期性能**：约309 GFLOPs/s（仅为cuBLAS的1.3%，峰值性能的1%）

**实验步骤**：

1. **环境搭建**：配置CUDA开发环境，编译SGEMM_CUDA项目（CMake + NVCC），运行基准测试脚本确认GPU设备信息。
2. **理解矩阵乘法**：复习矩阵乘法定义，理解 `C[MxN] = A[MxK] * B[KxN]` 的数据依赖关系。对于C中的每个元素，需要A的一整行和B的一整列做内积。
3. **编写朴素Kernel**：实现 `sgemm_naive` kernel，具体包括：
   - 用 `blockIdx` 和 `threadIdx` 计算当前线程对应的C矩阵全局位置 (x, y)
   - 使用2D grid和2D block（例如 blockDim = (32, 32)）
   - 实现内积循环：遍历K维，累加 `A[x*K+i] * B[i*N+y]`
   - 应用 `alpha` 和 `beta` 参数完成GEMM操作：`C[x*N+y] = alpha * tmp + beta * C[x*N+y]`
4. **验证正确性**：与CPU参考实现或cuBLAS结果对比，使用随机矩阵测试多个尺寸。
5. **性能分析**：使用 `cudaEvent` 计时，计算GFLOPs/s。注意：总FLOPS = `2 * M * N * K`。
6. **理论分析**：计算理论所需数据传输量（最少 `3*4092^2*4B + 4092^2*4B = 268MB`），分析为什么实际性能远低于理论峰值。

**关键代码变更**（从无到有）：

```cuda
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
```

启动配置：
```cuda
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
dim3 blockDim(32, 32);
```

**思考题**：
- 朴素实现中每个线程需要从全局内存读取多少数据？总内存流量是多少？（提示：考虑K次循环迭代，每次读A和B各一个元素）
- 为什么2D block的threadIdx计算使用了 `threadIdx.x` 和 `threadIdx.y` 两个维度？如果使用1D block（例如blockDim.x=1024）会有什么区别？
- 当M或N不能被block大小整除时，`if (x < M && y < N)` 的作用是什么？如果没有这个边界检查会发生什么？

---

## 实验2：全局内存合并访问（Global Memory Coalescing）

**对应的CUDA概念**：Warp调度、全局内存合并访问（Coalesced Access）、内存事务（Memory Transaction）、线程ID到Warp的映射规则

**参考CUDA Guide章节**：
- CUDA Programming Guide 第5章 "Performance Guidelines"——"Global Memory"（Global Memory Coalescing）
- CUDA Best Practices Guide——"Coalesced Access to Global Memory"

**原始代码**：`SGEMM_CUDA/src/kernels/2_kernel_global_mem_coalesce.cuh`

**优化目标**：
1. 理解Warp的概念：32个线程为一组，同一Warp内线程的连续内存访问可被合并为单次128字节事务
2. 掌握线程ID到Warp的映射规则：`threadId = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)`
3. 理解连续threadIdx.x的线程属于同一Warp
4. 通过调整线程到C元素的映射关系，使同一Warp内线程访问连续的内存地址

**预期性能提升**：309 GFLOPS -> **1986.5 GFLOPS**（约6.4x提升，达cuBLAS的8.5%）

**内存吞吐量变化**：从15 GB/s提升到110 GB/s（约7.3x提升）

**实验步骤**：

1. **分析朴素实现的内存访问模式**：
   - 画出2D block (32x32) 中线程的内存访问图
   - 分析同一Warp内线程访问A矩阵的模式：`A[x*K+i]`——不同线程的x不同（非连续），只有i相同时访存地址跨度很大
   - 分析同一Warp内线程访问B矩阵的模式：`B[i*N+y]`——不同线程的y不同（非连续）

2. **修改线程到C的映射**：
   - 将blockDim从2D改为1D：`blockDim(32*32)`，即总线程数不变但排列方式改变
   - 使用新的索引映射：`x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE)`, `y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE)`
   - 关键改变：threadIdx.x连续 -> y连续（同一列的行连续），使同一Warp内线程访问连续的B元素

3. **分析优化后的内存访问模式**：
   - 画出优化后的内存访问图
   - 解释为什么 `threadIdx.x % 32` 连续的线程现在访问连续的全局内存地址
   - 解释GPU如何将32个4字节float合并为128字节的单次内存事务

4. **性能对比**：测量GFLOPS和全局内存吞吐量，分析提速原因。

**关键代码变更**：

```cuda
// 变更前（实验1）
const uint x = blockIdx.x * blockDim.x + threadIdx.x;
const uint y = blockIdx.y * blockDim.y + threadIdx.y;

// 变更后（实验2）
const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
```

启动配置变为：
```cuda
dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
dim3 blockDim(32 * 32);  // 1D block，1024线程
```

**思考题**：
- 为什么实验1中使用2D block时无法实现内存合并访问？哪一次内存访问（读A还是读B）是非合并的？
- 如果GPU支持32B、64B和128B的内存事务，32个线程各读取一个float（4B），理想情况下需要几次事务？
- 除了修改线程到元素的映射关系，还有哪些因素可能影响全局内存合并访问的效率？（提示：内存对齐）

---

## 实验3：共享内存缓存分块（Shared Memory Cache-Blocking）

**对应的CUDA概念**：共享内存（Shared Memory）、`__syncthreads()` 同步屏障、内存层级结构（Memory Hierarchy）、SM体系结构（Streaming Multiprocessor）

**参考CUDA Guide章节**：
- CUDA Programming Guide 第3章 "Programming Interface"——"Shared Memory"
- CUDA Best Practices Guide——"Shared Memory"
- CUDA Programming Guide 第5章——"Shared Memory"、Occupancy

**原始代码**：`SGEMM_CUDA/src/kernels/3_kernel_shared_mem_blocking.cuh`

**优化目标**：
1. 理解共享内存的物理位置（on-chip）和性能优势（比全局内存高约16x带宽）
2. 掌握 `__shared__` 关键字声明共享内存
3. 掌握 `__syncthreads()` 的用法和必要性（生产者-消费者同步）
4. 理解缓存分块（Cache-Blocking）的核心思想：将复用数据加载到快速内存中
5. 计算矩阵乘法中数据的复用模式：A的每个元素被N个线程复用，B的每个元素被M个线程复用

**预期性能提升**：1986.5 GFLOPS -> **2980.3 GFLOPS**（约1.5x提升，达cuBLAS的12.8%）

**实验步骤**：

1. **理解数据复用**：
   - 分析矩阵乘法中哪些数据被重复使用
   - 实验2中每个线程内积循环每次迭代都要从全局内存加载A和B各一个元素，大量数据被重复从GMEM读取
   - 例如：A的一块数据可以被block内所有线程共享使用

2. **设计共享内存分块方案**：
   - 定义块大小 `BLOCKSIZE`（例如32）
   - 在共享内存中分配 `As[BLOCKSIZE * BLOCKSIZE]` 和 `Bs[BLOCKSIZE * BLOCKSIZE]`
   - 外层循环沿K维度移动，每次加载一个块到共享内存
   - 内层循环在共享内存上执行分块矩阵乘法

3. **实现共享内存加载**：
   - 每个线程负责从GMEM加载A和B各一个元素到SMEM
   - 注意保持合并访问：让threadIdx.x的连续维度对应于加载的连续地址
   - 使用 `__syncthreads()` 确保所有线程加载完成后再开始计算

4. **实现分块计算**：
   - 在共享内存数据上执行内积
   - 累加部分和到寄存器变量tmp
   - 计算完成后再次 `__syncthreads()`，防止快线程提前加载下一块

5. **Occupancy分析**：
   - 计算当前kernel的occupancy
   - SMEM使用：`2 * 32 * 32 * 4B = 8KB` per block
   - Registers per thread: 37
   - 分析为什么occupancy受限于寄存器和线程数

6. **Roofline模型分析**：使用Nsight Compute获取性能数据，绘制roofline图，分析当前kernel是compute-bound还是memory-bound。

**关键代码变更**：

```cuda
// 声明共享内存
__shared__ float As[BLOCKSIZE * BLOCKSIZE];
__shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

// 指针定位到当前block要计算的C子矩阵位置
A += cRow * BLOCKSIZE * K;
B += cCol * BLOCKSIZE;
C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

float tmp = 0.0;
for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
  // 协作加载数据到共享内存（合并访问）
  As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
  Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
  __syncthreads();  // 确保数据加载完成

  A += BLOCKSIZE;
  B += BLOCKSIZE * N;

  // 在共享内存上执行内积
  for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
    tmp += As[threadRow * BLOCKSIZE + dotIdx] *
           Bs[dotIdx * BLOCKSIZE + threadCol];
  }
  __syncthreads();  // 防止快线程覆盖数据
}
C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
```

**思考题**：
- 为什么需要两个 `__syncthreads()`？少一个会有什么后果？
- 当前kernel的内存访问模式中，每个线程计算结果需要多少次GMEM访问和多少次SMEM访问？（用K表示）
- 如果 `BLOCKSIZE=32`，共享内存使用量为8KB。如果增加到 `BLOCKSIZE=64`，SMEM使用量变为32KB。增大BLOCKSIZE对occupancy和性能分别有什么影响？（提示：每个SM的SMEM总量为100KB，最大48KB per block）

---

## 实验4：一维Block Tile——每线程计算多个结果

**对应的CUDA概念**：算术强度（Arithmetic Intensity）、寄存器缓存（Register File）、计算访存比优化、MIO指令队列、Warp Stall分析

**参考CUDA Guide章节**：
- CUDA Best Practices Guide——"Arithmetic Intensity"
- CUDA Programming Guide 第5章——"Maximize Utilization"（寄存器使用）
- GPU Microarchitecture：MIO pipeline、warp scheduler

**原始代码**：`SGEMM_CUDA/src/kernels/4_kernel_1D_blocktiling.cuh`

**优化目标**：
1. 理解算术强度的定义：FLOPs / Bytes transferred
2. 理解为什么增加每线程的计算量可以提升算术强度：SMEM数据可以被单个线程的多个计算结果复用
3. 掌握寄存器数组的使用：`float threadResults[TM]`
4. 理解编译器循环展开（Loop Unrolling）对性能的影响
5. 学会使用Nsight Compute分析warp stall原因（Stall MIO Throttle等）

**预期性能提升**：2980.3 GFLOPS -> **8474.7 GFLOPS**（约2.8x提升，达cuBLAS的36.5%）

**实验步骤**：

1. **分析实验3的瓶颈**：
   - 使用Nsight Compute分析指令混合：大部分指令是 `LDS`（共享内存加载），而非 `FMA`
   - 分析Warp Stall原因：`Stall MIO Throttle`——等待MIO（内存输入/输出）指令队列排空
   - 结论：kernel被SMEM访问吞吐限制，需要提高算术强度

2. **设计1D Block Tile方案**：
   - 引入新模板参数 `TM`：每个线程计算的输出结果数量
   - `BM`（block M维度）、`BN`（block N维度）、`BK`（block K维度）——设置tile大小
   - 例如：`BM=64, BN=64, BK=8, TM=8`——每个线程计算8个结果（沿M方向排列）

3. **修改内积循环结构**：
   - 将 `dotIdx` 循环放在最外层（而非 `resIdx` 循环）
   - 将Bs的一个元素缓存到寄存器 `Btmp`，避免重复读取SMEM
   - 对于每个dotIdx迭代，将As的TM个元素与Btmp相乘，累加到threadResults

4. **内存访问量对比计算**：
   - 实验3（每线程1个结果）：GMEM = K/16次，SMEM = K*2次
   - 实验4（每线程8个结果）：GMEM = K/32次，SMEM = K*9/8次
   - 每结果内存访问量显著下降

5. **验证和Profiling**：使用NCU分析warp stall状态变化，确认Stall MIO Throttle显著减少。

**关键代码变更**：

```cuda
// 分配线程局部结果缓存在寄存器中
float threadResults[TM] = {0.0};

for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
  // 加载SMEM（与实验3类似）
  As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
  Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
  __syncthreads();

  A += BK;
  B += BK * N;

  // 计算：dotIdx在外层循环，使Bs元素可被缓存复用
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    float Btmp = Bs[dotIdx * BN + threadCol];  // 缓存Bs元素
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
      threadResults[resIdx] +=
          As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
    }
  }
  __syncthreads();
}

// 写回结果
for (uint resIdx = 0; resIdx < TM; ++resIdx) {
  C[(threadRow * TM + resIdx) * N + threadCol] =
      alpha * threadResults[resIdx] + beta * C[(threadRow * TM + resIdx) * N + threadCol];
}
```

**思考题**：
- 为什么将 `dotIdx` 放在外层循环比 `resIdx` 放在外层更好？（提示：分析Bs元素被访问的次数，以及编译器对Bs SMEM加载的向量化优化）
- 如果 `TM=8`，当前配置的SMEM使用量是多少？与实验3相比如何变化？
- 算术强度的提升是否总会带来性能提升？什么时候继续增加TM不再有效？（提示：考虑寄存器压力）

---

## 实验5：二维Block Tile——进一步提高算术强度

**对应的CUDA概念**：2D Thread Tile、Register Blocking、多元素加载、跨步加载（Strided Load）

**参考CUDA Guide章节**：
- CUDA Best Practices Guide——"Arithmetic Intensity"（继续深入）
- CUDA Programming Guide 第5章——"Register Pressure"
- NVIDIA CUTLASS文档——"Efficient GEMM"

**原始代码**：`SGEMM_CUDA/src/kernels/5_kernel_2D_blocktiling.cuh`

**优化目标**：
1. 理解2D tile vs 1D tile的优势：正方形tile能最大程度共享M和N两个维度的数据
2. 理解为何计算正方形tile（TM x TN）比列形tile（TM x 1）更高效
3. 掌握使用加载循环（loadOffset循环）让每个线程负责加载多个元素
4. 掌握寄存器缓存 `regM[TM]` 和 `regN[TN]` 的使用模式——从SMEM批量加载到寄存器再执行外积

**预期性能提升**：8474.7 GFLOPS -> **15971.7 GFLOPS**（约1.88x提升，达cuBLAS的68.7%）

**实验步骤**：

1. **分析1D Tile的局限**：
   - 实验4中只有M维度的数据被复用（复用了TM次），N维度仍每次从SMEM重读
   - 计算正方形tile的算术强度优势：计算TM x TN个结果，只需加载TM + TN个SMEM元素

2. **实现多元素加载**：
   - 每个线程使用跨步循环从GMEM加载多个元素到SMEM
   - A：`for (loadOffset = 0; loadOffset < BM; loadOffset += strideA)`——沿行方向加载
   - B：`for (loadOffset = 0; loadOffset < BK; loadOffset += strideB)`——沿列方向加载
   - strideA = numThreads / BK，strideB = numThreads / BN

3. **实现Register Blocking**：
   - 声明 `float regM[TM]` 和 `float regN[TN]` 作为寄存器缓存
   - 在dotIdx循环内部，先从SMEM加载TM个As元素和TN个Bs元素到寄存器
   - 执行外积：`regM[i] * regN[j]` 累加到 `threadResults[i*TN + j]`
   - 此模式使每个SMEM元素只被加载一次到寄存器，但参与多次FMA运算

4. **编译器行为分析**：
   - 观察编译器自动展开循环（BK=TM=TN=8已知）
   - 分析编译器如何消除重复的SMEM加载（Bs元素向量化为LDS.128）

5. **内存访问计算**（每线程64个结果）：
   - GMEM：K/64次 per result
   - SMEM：K/4次 per result
   - 与实验4对比：GMEM从K/32降至K/64，SMEM从K*9/8降至K/4

**关键代码变更**：

```cuda
float threadResults[TM * TN] = {0.0};
float regM[TM] = {0.0};
float regN[TN] = {0.0};

// 多元素加载（每个线程加载多个元素到SMEM）
for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
  As[(innerRowA + loadOffset) * BK + innerColA] =
      A[(innerRowA + loadOffset) * K + innerColA];
}
for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
  Bs[(innerRowB + loadOffset) * BN + innerColB] =
      B[(innerRowB + loadOffset) * N + innerColB];
}

// Register Blocking 内积循环
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
  // 批量加载到寄存器
  for (uint i = 0; i < TM; ++i) {
    regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
  }
  for (uint i = 0; i < TN; ++i) {
    regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
  }
  // 外积累加
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      threadResults[resIdxM * TN + resIdxN] +=
          regM[resIdxM] * regN[resIdxN];
    }
  }
}
```

**思考题**：
- 为什么2D tile (TM x TN) 的算术强度比1D tile (TM x 1) 更高？计算两者的GMEM访问/FLOPs比和SMEM访问/FLOPs比。
- 如果 `TM = TN = 8`，每个线程的 `threadResults[64]` 占用多少字节？如果寄存器总数为65536 per SM，最多能容纳多少个这样的线程？
- 实验3中编译器自动将Bs的SMEM加载向量化为LDS.128。实验5中As的加载能否自动向量化？为什么？如果不能，如何在实验6中解决？

---

## 实验6：向量化内存访问（Vectorized Memory Access）

**对应的CUDA概念**：向量化数据类型（float4）、128-bit全局内存事务（LDG.E.128 / STG.E.128）、128-bit共享内存加载（LDS.128）、SMEM数据布局转置（Transpose on Load）、内存对齐要求

**参考CUDA Guide章节**：
- CUDA Programming Guide 第5章——"Vectorized Memory Access"
- NVIDIA博客："CUDA Pro Tip: Increase Performance with Vectorized Memory Access"

**原始代码**：`SGEMM_CUDA/src/kernels/6_kernel_vectorize.cuh`

**优化目标**：
1. 理解向量化内存访问的原理：使用float4类型一次加载/存储4个float（128-bit）
2. 掌握 `reinterpret_cast<float4*>()` 的用法和对齐承诺
3. 理解为什么加载A时需要转置（Transpose on Load）：使SMEM中As按列连续排列，便于后续LDS.128向量化加载
4. 向量化C矩阵的写回操作（使用float4存储）

**预期性能提升**：15971.7 GFLOPS -> **18237.3 GFLOPS**（约1.14x提升，达cuBLAS的78.4%）

**实验步骤**：

1. **理解向量化需求的来源**：
   - 分析实验5的SASS汇编：Bs加载已自动向量化为LDS.128（因为Bs按行存储到SMEM），但As加载仍是LDS（32-bit）
   - 原因：As在SMEM中按行存储，SMEM的列维度访问不连续

2. **实现GMEM到SMEM的向量化加载**：
   - 加载A：使用 `float4` 从GMEM一次读取4个元素，边加载边转置到SMEM（行变列）
   - 加载B：直接 `reinterpret_cast<float4*>` 进行128-bit加载
   - 注意：线程索引也需要相应调整（`innerColA` 从 /BK 改为 /(BK/4)）

3. **A矩阵的转置加载详解**：
   - 从GMEM读：`A[innerRowA * K + innerColA * 4]` 开始的4个连续元素（行方向连续）
   - 写入SMEM：分散存储在 `As[(innerColA*4 + 0..3) * BM + innerRowA]`——列方向连续
   - 效果：SMEM中As变为列主序存储，后续从SMEM加载到寄存器时可连续访问

4. **C矩阵写回的向量化**：
   - 使用 `float4` 一次写回4个结果元素
   - 需要 `reinterpret_cast` 承诺对齐

5. **验证向量化效果**：
   - 检查PTX/SASS生成：确认GMEM加载使用 `LDG.E.128`，SMEM加载使用 `LDS.128`
   - 对比 `reinterpret_cast<float4*>` 与手动展开的区别：编译器需要对齐承诺才能生成128-bit指令

**关键代码变更**：

```cuda
// 向量化加载A（带转置）
float4 tmp = reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

// 向量化加载B
reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
    reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];

// 注：As此时已转置为列主序，从SMEM加载到寄存器时：
// regM[i] = As[dotIdx * BM + threadRow * TM + i];  // 连续访问

// 向量化写回C
for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
  float4 tmp = reinterpret_cast<float4 *>(
      &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0];
  tmp.x = alpha * threadResults[resIdxM * TN + resIdxN] + beta * tmp.x;
  tmp.y = alpha * threadResults[resIdxM * TN + resIdxN + 1] + beta * tmp.y;
  tmp.z = alpha * threadResults[resIdxM * TN + resIdxN + 2] + beta * tmp.z;
  tmp.w = alpha * threadResults[resIdxM * TN + resIdxN + 3] + beta * tmp.w;
  reinterpret_cast<float4 *>(
      &C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN])[0] = tmp;
}
```

**思考题**：
- 为什么直接用 `float4` 手动展开加载B也能自动向量化，而加载A不转置就不能向量化？（提示：分析SMEM中As和Bs的布局，以及LDS.128的对齐要求）
- `reinterpret_cast<float4*>()` 的作用是强制生成128-bit加载指令。如果B指针实际未128-bit对齐会发生什么？
- 向量化写回时，为什么需要在 `resIdxN` 循环中以步长4遍历？如果TN=8，需要几次float4写回操作？

---

## 实验7：参数自动调优（Autotuning）

**对应的CUDA概念**：模板元编程、编译时参数空间搜索、Tile Size选择原则、Occupancy与资源限制、不同GPU架构的最优参数差异

**参考CUDA Guide章节**：
- CUDA Best Practices Guide——"Occupancy"
- CUDA Programming Guide 第5章——"Execution Configuration Optimizations"
- NVIDIA Nsight Compute——性能反汇编分析

**原始代码**：`SGEMM_CUDA/src/kernels/9_kernel_autotuned.cuh`

**优化目标**：
1. 理解tile参数（BM, BN, BK, TM, TN）对性能的影响因素
2. 掌握参数空间的约束条件：SMEM大小限制、寄存器数量限制、向量化对齐要求
3. 学会设计参数搜索脚本，自动化benchmark和择优
4. 理解为什么不同GPU架构（A6000 vs A100）的最优参数不同

**预期性能提升**：18237.3 GFLOPS -> **19721.0 GFLOPS**（约1.08x提升，达cuBLAS的84.8%）

**实验步骤**：

1. **分析当前参数**：BM=BN=128, BK=TM=TN=8。计算此配置的SMEM使用量和寄存器使用量。

2. **理解参数约束**：
   - SMEM约束：`(BM*BK + BN*BK) * 4B <= 48KB` (max SMEM per block)
   - 寄存器约束：每线程至少使用 `TM*TN + TM + TN` 个float寄存器
   - 向量化约束：`BM*BK` 必须能被 `4 * NUM_THREADS` 整除（每个线程每次加载4个元素）
   - 线程数约束：`(BM*BN) / (TM*TN)` 必须是合理的block线程数（如128, 256等）

3. **编写参数搜索脚本**：
   - 枚举BM, BN ∈ {64, 128, 256}, BK ∈ {8, 16, 32}, TM, TN ∈ {4, 8}
   - 过滤不满足约束的配置
   - 对每组合法配置编译并benchmark
   - 使用bash/python脚本自动化流程

4. **分析最优参数**：
   - A6000最优：BM=BN=128, BK=16, TM=TN=8（约20 TFLOPS）
   - A100最优：BM=BN=64, BK=16, TM=TN=4（约12.6 TFLOPS）
   - 讨论为什么同一种参数在不同GPU上表现不同（SM数量、SMEM大小、计算能力差异）

5. **添加 `__launch_bounds__` 提示**：
   - `__launch_bounds__(NUM_THREADS)` 告知编译器预期的线程数，帮助编译器优化寄存器分配

**关键代码变更**：

实验7的kernel与实验6结构相似，但模板参数变为可调优，并增加了：
```cuda
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(K9_NUM_THREADS)
    sgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                   float beta, float *C) {
  // ... 使用模板参数BM, BN, BK, TM, TN控制tile大小
}
```

参数搜索脚本示例（概念性）：
```bash
for BM in 64 128; do
  for BN in 64 128; do
    for BK in 8 16; do
      for TM in 4 8; do
        for TN in 4 8; do
          # 检查约束（SMEM、寄存器、向量化）
          # 编译并运行benchmark
          # 记录性能
        done
      done
    done
  done
done
```

**思考题**：
- 为什么BK从8增加到16能提升性能？（提示：BK增大意味着SMEM块更大，外循环迭代次数减半，`__syncthreads()` 开销更少）
- 为什么A100的最优参数（BM=64）比A6000的最优参数（BM=128）小？（提示：A100的SMEM配置和SM数量不同）
- 在实际生产环境中（如cuBLAS），autotuning是如何工作的？参数搜索在编译时还是运行时完成？

---

## 实验8：Warp Tile、Bank Conflict 与性能对比cuBLAS

**对应的CUDA概念**：Warp级并行（Warp-Level Tiling）、共享内存Bank Conflict（32路Bank结构）、双缓冲（Double Buffering）、Warp映射调度、CUTLASS风格GEMM

**参考CUDA Guide章节**：
- CUDA Programming Guide 第5章——"Shared Memory"（Bank Conflicts小节）
- CUDA Best Practices Guide——"Understand the Programming Environment"（Warp概念）
- NVIDIA CUTLASS——Efficient GEMM设计文档

**原始代码**：
- Warptiling: `SGEMM_CUDA/src/kernels/10_kernel_warptiling.cuh`（21.8 TFLOPS, 93.7% of cuBLAS）
- 避免Bank Conflict: `SGEMM_CUDA/src/kernels/7_kernel_resolve_bank_conflicts.cuh` 和 `.../8_kernel_bank_extra_col.cuh`
- 双缓冲: `SGEMM_CUDA/src/kernels/11_kernel_double_buffering.cuh`

**优化目标**：
1. 理解Warp Tiling：在Block Tile和Thread Tile之间增加Warp层级
   - Block Tile由多个Warp Tile组成，每个Warp计算一个连续的子区域
   - 同一Warp内的线程共享更紧密的数据区域，增加寄存器缓存局部性
2. 理解共享内存Bank Conflict的原因和两种解决方案
3. 理解双缓冲（Double Buffering）的基本思想：加载下一次迭代数据的同时计算当前数据
4. 使用Nsight Compute对比自己的实现与cuBLAS的性能差距

**预期性能**：Warp Tiling达到 **21779.3 GFLOPS**（93.7% of cuBLAS）

**实验步骤**：

1. **实现Warp Tiling**：
   - 计算warp索引：`warpIdx = threadIdx.x / WARPSIZE`
   - 将Block Tile按Warp划分：`warpRow = warpIdx / (BN/WN)`, `warpCol = warpIdx % (BN/WN)`
   - 每个Warp计算 `WM x WN` 子区域（例如64x32）
   - 引入Warp Subtile循环 `wSubRowIdx` 和 `wSubColIdx`

2. **Warp Tiling的三层循环结构**：
   ```
   for bkIdx (K维外层)
     for dotIdx (BK维内层)
       for wSubRowIdx (warp subtile行迭代)
         加载 regM[wSubRowIdx*TM + i]
       for wSubColIdx (warp subtile列迭代)
         加载 regN[wSubColIdx*TN + i]
       for wSubRowIdx
         for wSubColIdx
           执行 FMA 累加（利用寄存器缓存局部性）
   ```

3. **理解Bank Conflict**：
   - 共享内存由32个Bank组成，Bank索引 = (address/4) % 32
   - 同一Warp内多个线程访问同一Bank的不同地址时发生Bank Conflict
   - 两种解决方法：
     - **方法一（Linearize）**：重新排列Bs的SMEM布局，使访问模式解耦（kernel 7）
     - **方法二（Extra Col）**：在Bs每行末尾添加额外列（padding），使相邻行的相同列偏移不同的Bank（kernel 8）

4. **理解Double Buffering**：
   - 分配2倍SMEM空间：`As[2 * BM * BK]` 和 `Bs[2 * BK * BN]`
   - 一半线程加载buffer 0，另一半加载buffer 1
   - 交替使用两个buffer进行计算和加载
   - 理想效果：隐藏GMEM加载延迟（加载下一块的同时计算当前块）

5. **性能对比与Gap分析**：
   - 将自己最终实现的性能与cuBLAS对比（本实验系列最终达到93.7%）
   - 分析剩余6.3%的性能差距来自哪里：更多split-K策略、Tensor Core使用（TF32）、更精细的参数选择、更优的数据布局等

**关键代码变更（Warp Tiling核心）**：

```cuda
// Warp级别的索引计算
const uint warpIdx = threadIdx.x / WARPSIZE;
const uint warpCol = warpIdx % (BN / WN);
const uint warpRow = warpIdx / (BN / WN);

// Warp Subtile大小
constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
constexpr uint WSUBM = WM / WMITER;
constexpr uint WSUBN = WN / WNITER;

// 线程在Warp Subtile中的位置
const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

// 加载A子矩阵（以Warp为粒度）
As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
   threadRowInWarp * TM + i]

// 加载B子矩阵（以Warp为粒度）
Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
   threadColInWarp * TN + i]
```

**Bank Conflict解决（Padding方法示例）**：
```cuda
__shared__ float Bs[BK * (BN + extraCols)];  // 添加extraCols列padding
// 访问时使用 (BN + extraCols) 作为行步长
regN[i] = Bs[dotIdx * (BN + extraCols) + threadCol * TN + i];
```

**Double Buffering核心思想**：
```cuda
__shared__ float As[2 * BM * BK];  // 双倍SMEM空间
__shared__ float Bs[2 * BK * BN];

// 交替使用两个buffer
// Buffer 0 计算时，Buffer 1 加载下一次迭代数据
// Buffer 1 计算时，Buffer 0 加载再下一次迭代数据
```

**思考题**：
- 为什么Warp Tiling能带来额外性能提升？（提示：考虑寄存器缓存局部性，以及Warp调度器的特性）
- Bank Conflict中，"Linearize"方法和"Extra Col"方法的原理分别是什么？各自有什么优缺点？
- Double Buffering是否能真正隐藏GMEM加载延迟？什么条件下它的效果不明显？
- cuBLAS在矩阵尺寸256x256时使用了split-K策略（一个matmul kernel + 一个reduce kernel）。split-K的目的是什么？为什么在小矩阵时特别有用？

---

## 实验与课程章节对应关系

本实验系列与NVIDIA CUDA Programming Guide的教学章节和课程体系的对应对应关系如下：

| 实验 | 实验名称 | 前置课程章节 | 建议授课次序 |
|------|----------|-------------|-------------|
| 实验1 | 朴素矩阵乘法实现 | **第2章 Programming Model**（Thread Hierarchy, Kernel Launch） | 学完线程模型后立即安排 |
| 实验2 | 全局内存合并访问 | **第4章 Hardware Implementation**（Warps, SIMT Architecture）+ **第5章 Global Memory**（Coalescing） | 学完Warp和内存合并访问概念后安排 |
| 实验3 | 共享内存缓存分块 | **第3章 Shared Memory** + **第5章 Shared Memory**（Bank Conflicts预备知识） | 学完共享内存概念后安排 |
| 实验4 | 一维Block Tile | **第5章 Performance Guidelines**（Arithmetic Intensity, Occupancy, Roofline Model） | 学完Occupancy和Roofline后安排 |
| 实验5 | 二维Block Tile | **第5章 Performance Guidelines**（Register Usage, Loop Unrolling） | 在实验4基础上深化 |
| 实验6 | 向量化内存访问 | **第5章 Vectorized Access** + **第3章 Memory Types**深入 | 学完向量化内存访问后安排 |
| 实验7 | 参数自动调优 | **第5章 Execution Configuration** + **附录 Compute Capabilities** | 学完所有基础优化后安排 |
| 实验8 | Warp Tile + 对比cuBLAS | **全部章节** + **CUTLASS扩展阅读** | 作为综合实验/期末项目 |

### 课程整体安排建议

**阶段一：基础篇（2-3次课）**
- 课程内容：CUDA编程模型（线程层级、内存模型）
- 配套实验：实验1（朴素实现）
- 教学目标：学生能够编写正确的CUDA kernel，理解线程索引计算

**阶段二：内存优化篇（3-4次课）**
- 课程内容：全局内存访问优化、共享内存使用
- 配套实验：实验2（合并访问）+ 实验3（共享内存缓存）
- 教学目标：掌握GPU最重要的两种内存的优化方法

**阶段三：计算优化篇（3-4次课）**
- 课程内容：算术强度、Occupancy、寄存器使用
- 配套实验：实验4（1D Tile）+ 实验5（2D Tile）+ 实验6（向量化）
- 教学目标：理解计算密集型kernel的优化方法论

**阶段四：进阶篇（2-3次课）**
- 课程内容：Warp级优化、高级优化技术、与cuBLAS对比
- 配套实验：实验7（Autotuning）+ 实验8（Warp Tile + Bank Conflict）
- 教学目标：能够独立分析和优化CUDA kernel，达到生产级性能

### 实验评分建议

每个实验的评分维度：
1. **正确性**（40%）：与cuBLAS结果对比，误差在合理范围内
2. **性能**（40%）：GFLOPs/s达到该实验的预期水平
3. **代码质量**（10%）：注释清晰、索引计算正确、同步使用合理
4. **实验报告**（10%）：包含性能分析、优化思路、Nsight Compute截图

### 参考资料

1. [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) - Simon Boehm, 2022
2. [SGEMM_CUDA 源代码](https://github.com/siboehm/SGEMM_CUDA) - 配套开源代码
3. [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) - NVIDIA官方文档
4. [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) - NVIDIA性能优化指南
5. [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) - 性能分析指南
6. [CUTLASS Efficient GEMM](https://github.com/NVIDIA/cutlass/blob/master/media/docs/efficient_gemm.md) - NVIDIA CUTLASS高效GEMM设计文档
7. [Understanding Latency Hiding on GPUs](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.pdf) - V. Volkov博士论文
