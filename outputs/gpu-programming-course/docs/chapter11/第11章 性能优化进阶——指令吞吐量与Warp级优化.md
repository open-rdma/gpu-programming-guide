# 第11章 性能优化进阶——指令吞吐量与Warp级优化

## 学习目标

通过本章的学习，你将能够：

- 理解各类算术指令的吞吐量差异，学会选择高效的指令
- 掌握单精度快速数学函数（Intrinsics）的使用
- 理解 warp divergence 的产生原因和优化方法
- 熟练使用 warp shuffle 函数进行 warp 内数据交换
- 掌握 warp 级归约（Reduction）的编写方法
- 理解原子操作（Atomic Functions）的性能特性
- 学会使用只读数据缓存（`__ldg()`）优化全局内存读取

## 开篇

在上一章中，我们学习了GPU性能优化的两大支柱：<strong>占用率优化</strong>和<strong>内存访问优化</strong>。你已经掌握了如何让数据高效地流入计算单元。本章我们将深入第三大支柱——<strong>指令吞吐量优化</strong>——以及最令人兴奋的 <strong>Warp级编程技术</strong>。

如果说全局内存合并访问教会了你如何"喂饱"GPU，那么本章将教你如何让GPU"吃得快、消化好"。你会学到：

- 哪些数学运算廉价、哪些昂贵，以及如何"用精度换速度"
- 如何避免 warp divergence 这个"并行杀手"
- <strong>Warp Shuffle 函数</strong>——一种比共享内存更快的线程间通信方式
- 如何使用 warp 级原语编写高性能的归约算法

在这些知识点中，<strong>warp shuffle</strong> 可以说是最强大的技术。它允许一个 warp 内的32个线程直接交换寄存器中的数据，无需经过共享内存。这种寄存器到寄存器的数据交换极其高效——让你的归约、扫描等操作比基于共享内存的实现快数倍。

---

## 11.1 最大化指令吞吐量

为了最大化指令吞吐量，应用程序应该：

1. <strong>最小化低吞吐量算术指令的使用</strong>——在不影响最终结果的情况下用精度换速度
2. <strong>最小化由控制流指令引起的发散 warp</strong>
3. <strong>减少指令数量</strong>——例如通过优化掉同步点或使用受限指针（`__restrict__`）

在讨论具体优化之前，先理解吞吐量的计量方式：本节中，吞吐量以<strong>每个时钟周期每个多处理器的操作数</strong>给出。由于 warp 大小为32，一条指令对应32个操作，如果 N 是每个时钟周期的操作数，指令吞吐量为 N/32 条指令每时钟周期。所有吞吐量都是针对一个多处理器而言的，需要乘以设备中的多处理器数量以获得整个设备的吞吐量。

### 11.1.1 算术指令吞吐量

下表给出了各种算术指令的本机硬件支持吞吐量（每个时钟周期每个多处理器的结果数），数据来源于 CUDA 11.5.1 Programming Guide：

<strong>表 11.1 本机算术指令吞吐量</strong>（结果数/时钟周期/多处理器）

| 指令类型 | 3.5/3.7 | 5.0/5.2 | 5.3 | 6.0 | 6.1/6.2 | 7.x | 8.0 | 8.6 |
|---------|---------|---------|-----|-----|---------|-----|-----|-----|
| 16-bit FP add/mul/fma | N/A | N/A | N/A | 256 | 128 | 256 | 128 | 256 |
| 32-bit FP add/mul/fma | 192 | 128 | 128 | 64 | 128 | 128 | 64 | 128 |
| 64-bit FP add/mul/fma | 64 | 4 | 4 | 32 | 4 | 32 | 32 | 2 |
| 32-bit FP 特殊函数 | 32 | 32 | 32 | 16 | 32 | 16 | 16 | 16 |
| 32-bit INT add/sub | 160 | 128 | 128 | 64 | 128 | 64 | 64 | 64 |
| 32-bit INT mul/mad | 32 | 32 | 32 | 32 | 64 | 64 | 64 | 64 |
| 32-bit INT shift | 64 | 64 | 64 | 32 | 64 | 64 | 64 | 64 |
| 32-bit 位运算 AND/OR/XOR | 160 | 128 | 128 | 64 | 128 | 64 | 64 | 64 |
| warp shuffle | 32 | 32 | 32 | 32 | 32 | 32 | 32 | 32 |
| warp reduce | 多指令 | 多指令 | 多指令 | 多指令 | 多指令 | 多指令 | 多指令 | 16 |

> **注释**：32-bit FP 特殊函数包括 `__fdividef()`、`rsqrtf()`、`__log2f()`、`exp2f()`、`__sinf()`、`__cosf()` 等。`warp reduce` 在 CC 8.6+ 设备上由硬件直接支持。

<strong>从表中得到的关键结论</strong>：

1. <strong>单精度（FP32）加法和乘法非常快</strong>——在大多数架构上每个时钟周期128个操作，意味着每个 SM 每个时钟周期可以完成4个 warp 的 FP32 FMA 指令
2. <strong>双精度（FP64）远慢于单精度</strong>——在消费级 GPU 上（如 RTX 3060），FP64 吞吐量可能只有 FP32 的 1/32
3. <strong>半精度（FP16）吞吐量是单精度的两倍</strong>——在现代 GPU 上使用 FP16 可以大幅提升性能（这就是 Tensor Core 的核心思想）
4. <strong>整数乘法和除法相对昂贵</strong>——整数除法/取模编译为多达20条指令
5. <strong>位运算和移位非常廉价</strong>——能用位运算替代乘除时应该优先使用

### 11.1.2 单精度快速数学函数（Intrinsics）

CUDA 提供了一系列<strong>内建函数（Intrinsic Functions）</strong>，它们比标准的数学函数快得多，但精度略低。这些函数直接映射到硬件指令。

#### 快速除法：`__fdividef()`

```cuda
// 标准除法：完全精度，较慢
float y = x / z;

// 快速除法：更快，但结果在非正规数上略有差异
float y = __fdividef(x, z);
```

`__fdividef()`  提供比除法运算符更快的单精度浮点除法。需要特别注意：当分母的绝对值在 **2^126 < |分母| < 2^128** 且分子为有限值时，该函数会直接返回 `0`，而不是 IEEE 754 标准下的正确结果。因此，在使用该函数加速除法时，必须确保数据范围不会落入这一“精确失效”区间。对于非正规数等其他情况，结果也可能与标准除法存在差异，但上述边界行为是最显著的偏差。

#### 快速倒数平方根：`rsqrtf()`

```cuda
// 标准倒数平方根
float y = 1.0f / sqrtf(x);

// 快速倒数平方根——使用硬件指令
float y = rsqrtf(x);
```

编译器只在使用 `-prec-div=false` 和 `-prec-sqrt=false` 编译选项时才会将 `1.0f/sqrtf()` 优化为 `rsqrtf()`。因此建议需要时直接调用 `rsqrtf()`。

#### 快速正弦和余弦：`__sinf()`、`__cosf()`

```cuda
// 标准正弦/余弦——高精度，但可能极慢（特别是大参数）
float s = sinf(x);
float c = cosf(x);

// 快速版本——性能更高，但参数范围有限
float s = __sinf(x);
float c = __cosf(x);
```

<strong>重要性能注意事项</strong>：标准三角函数 `sinf(x)`、`cosf(x)` 等的性能取决于参数 `x` 的大小：
- <strong>快速路径</strong>：用于幅度小于 `105615.0f` 的参数（单精度），基本上只需几次乘加运算
- <strong>慢速路径</strong>：用于幅度较大的参数，涉及大量计算且可能使用局部内存（28字节用于单精度，44字节用于双精度）
- <strong>慢速路径的吞吐量比快速路径低一个数量级</strong>

因此，在设计算法时应尽量保持参数在快速路径范围内。

#### 编译器标志对性能的影响

```bash
# 更快但精度稍低的编译（推荐用于对精度不敏感的场景）
nvcc -ftz=true -prec-div=false -prec-sqrt=false kernel.cu

# 完全精度编译
nvcc -ftz=false -prec-div=true -prec-sqrt=true kernel.cu
```

- `-ftz=true`：将非正规数刷新为零（Flush To Zero）——往往能提高性能
- `-prec-div=false`：使用不太精确的除法
- `-prec-sqrt=false`：使用不太精确的平方根

### 11.1.3 整数算术的代价

整数除法和取模运算代价高昂——它们编译为多达20条指令。当 `n` 是2的幂时：
- `(i / n)` 等价于 `(i >> log2(n))`
- `(i % n)` 等价于 `(i & (n-1))`

如果 `n` 是字面量（literal），编译器会自动进行这些转换。

```cuda
// 慢：整数除法
int idx = threadIdx.x / 8;  // 编译为多条指令（除非n是2的幂且为字面量）

// 快：位运算
int idx = threadIdx.x >> 3;  // 单条指令
```

其他高效的整数内建函数：
- `__brev(x)` / `__brevll(x)`：位反转（bit reverse）
- `__popc(x)` / `__popcll(x)`：人口计数（population count，统计1的个数）
- `__clz(x)` / `__clzll(x)`：前导零计数

`__[u]mul24` 是遗留的内建函数，不再有任何理由使用。

### 11.1.4 半精度算术（Half-Precision）

要实现16位浮点加法、乘法或乘加的<strong>良好性能</strong>，推荐使用 `half2` 数据类型（而非单独的 `half`），以及 `__nv_bfloat162`（而非单独的 `__nv_bfloat16`）。然后使用向量内建函数（如 `__hadd2`、`__hsub2`、`__hmul2`、`__hfma2`）在<strong>单条指令中执行两个操作</strong>。

```cuda
#include <cuda_fp16.h>

__global__ void half_kernel(half *a, half *b, half *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n / 2) {
        // 使用 half2 进行向量化操作——单条指令处理两个 half
        half2 a2 = ((half2*)a)[idx];
        half2 b2 = ((half2*)b)[idx];
        half2 c2 = __hfma2(a2, b2, __float2half2_rn(0.0f));
        ((half2*)c)[idx] = c2;
    }
}
```

使用 `half2` 或 `__nv_bfloat162` 替代两个单独的 `half` 或 `__nv_bfloat16` 调用，也可能提高其他内建函数（如 warp shuffle）的性能。

### 11.1.5 类型转换的开销

编译器有时必须插入转换指令，引入额外的执行周期。这发生在：

- 对 `char` 或 `short` 类型变量操作的函数，其操作数通常需要转换为 `int`
- <strong>双精度浮点常量</strong>（即没有类型后缀定义的常量）用作单精度浮点计算的输入

```cuda
// 坏：3.141592653589793 是 double 常量，会引入转换开销
float area = 3.141592653589793 * r * r;

// 好：使用 f 后缀明确指定为 float
float area = 3.141592653589793f * r * r;
```

<strong>一般性建议</strong>：在 GPU 代码中优先使用 `int` 和 `float` 类型，避免使用 `char` 和 `short`（除非有特殊原因），并始终为浮点常量添加适当的类型后缀。

---

## 11.2 控制流优化

### 11.2.1 Warp Divergence——并行杀手

任何流控制指令（`if`、`switch`、`do`、`for`、`while`）都可能<strong>显著影响</strong>有效指令吞吐量，因为它会导致同一个 warp 中的线程<strong>发散（diverge）</strong>——即跟随不同的执行路径。

<strong>Warp divergence 的根本原因</strong>：GPU 在 SIMT（Single Instruction, Multiple Thread）模式下执行。一个 warp 中的32个线程共享同一个程序计数器（Program Counter）。当遇到分支时，如果不同线程走了不同路径，硬件必须<strong>串行执行</strong>每个分支路径。

```
// Warp Divergence 示例
if (threadIdx.x < 16) {
    // 线程 0-15 执行这段代码（16个线程活跃）
    do_work_A();  // 线程 16-31 在此处等待
} else {
    // 线程 16-31 执行这段代码（16个线程活跃）
    do_work_B();  // 线程 0-15 在此处等待
}
// 总执行时间 = do_work_A + do_work_B（串行化！）
```

#### 最小化 Warp Divergence 的方法

由于 warp 在 block 内的分布是确定性的，可以<strong>编写使分支条件与 warp 边界对齐的代码</strong>：

```cuda
// 好：分支条件与 warp 边界对齐
// warpSize = 32, 所以 (threadIdx.x / 32) 对所有同一 warp 的线程相同
int warpId = threadIdx.x / 32;
if (warpId % 2 == 0) {
    // 整个 warp 走同一分支，无 divergence！
    do_work_even();
} else {
    do_work_odd();
}

// 不好：分支条件在 warp 内不统一
if (threadIdx.x % 2 == 0) {
    // warp 中一半线程走这里，一半走 else
    do_work_even_lane();
} else {
    do_work_odd_lane();
}
```

#### 实际案例：边界检查

边界检查通常是不可回避的 divergence 来源，但其影响通常很小，因为只是少数线程（warps 末尾的线程）在边界路径上：

```cuda
// 边界检查引起的 divergence 通常可接受
int idx = threadIdx.x + blockIdx.x * blockDim.x;
if (idx < N) {
    // 只有最后一个 warp 可能部分发散，影响有限
    result[idx] = process(input[idx]);
}
```

### 11.2.2 分支预测（Branch Predication）

有时编译器可能展开循环，或者使用<strong>分支预测（Branch Predication）</strong>替代短 `if` 或 `switch` 块。在这种情况下，<strong>不会出现 warp 发散</strong>。

当使用分支预测时，没有指令会被跳过。相反，每条指令都与一个<strong>每线程条件码或谓词（Predicate）</strong>关联：
- 谓词为 `true` 的线程：指令正常执行并写回结果
- 谓词为 `false` 的线程：指令不写回结果，也不计算地址或读取操作数

```cuda
// 编译器可能对短分支使用预测
// 例如这样的短if块：
if (x > 0) {
    y = a + b;
} else {
    y = 0;
}
// 编译器可能使用 predicated 指令代替实际分支
// -> 无 divergence，所有线程执行两种操作但只保留对应的结果
```

程序员也可以使用 `#pragma unroll` 指令来控制循环展开。

### 11.2.3 循环展开与指令级并行

循环展开（Loop Unrolling）有两个好处：
1. 减少循环控制开销
2. 暴露更多指令级并行（Instruction-Level Parallelism，ILP）

```cuda
// 未展开的循环
for (int i = 0; i < 4; i++) {
    acc += data[idx + i];
}

// 手动展开——暴露ILP
acc += data[idx + 0];
acc += data[idx + 1];
acc += data[idx + 2];
acc += data[idx + 3];

// 使用 pragma 指示编译器展开
#pragma unroll
for (int i = 0; i < 4; i++) {
    acc += data[idx + i];
}
```

<strong>ILP 的重要性</strong>：如果单个 warp 在其指令流中有多条独立指令，则可以通过背靠背发射同一 warp 中的多条独立指令来减少隐藏延迟所需的 warp 总数。这意味着具有高 ILP 的内核可以在较低占用率下仍保持良好性能。

---

## 11.3 同步的开销

`__syncthreads()` 的吞吐量因架构而异：

| CC | 吞吐量（操作数/时钟周期/SM） |
|----|---------------------------|
| 3.x | 128 |
| 5.x, 6.1, 6.2 | 64 |
| 6.0 | 32 |
| 7.x, 8.x | 16 |

<strong>关键洞察</strong>：`__syncthreads()` 不仅自身有开销，它还可能通过强制多处理器空闲来影响性能。当越来越多的 warp 在同步点等待同一 block 中的其他 warp 时，SM 可能被迫闲置。

<strong>优化技巧</strong>：
- 如果每个 SM 上有多个常驻 block，来自不同 block 的 warp 不需要在彼此的同步点等待——它们可以继续执行
- 将同步点尽可能推后，让 warp 先完成尽可能多的独立工作
- 避免在 warp 级操作中不必要的块级同步

---

## 11.4 Warp 级原语——寄存器级数据交换

<strong>这是本章最激动人心的部分</strong>。Warp Shuffle 函数允许一个 warp 内的线程<strong>直接在寄存器中交换数据</strong>，无需通过共享内存。这意味着线程间通信可以达到<strong>寄存器级的速度</strong>——比共享内存快得多，更不用说全局内存了。

### 11.4.1 Warp Shuffle 函数

四种 warp shuffle 内建函数（从 CUDA 9.0 起，必须使用 `_sync` 版本）：

```cuda
T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
T __shfl_up_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
T __shfl_down_sync(unsigned mask, T var, unsigned int delta, int width=warpSize);
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);
```

其中 `T` 可以是 `int`、`unsigned int`、`long`、`unsigned long`、`long long`、`unsigned long long`、`float` 或 `double`。包含 `cuda_fp16.h` 后也可以是 `__half` 或 `__half2`。

<strong>Warp 内的线程被称为 lane（通道）</strong>，其索引范围从 0 到 `warpSize-1`（含）。

#### 四种寻址模式

| 函数 | 行为 | 描述 |
|------|------|------|
| `__shfl_sync()` | 从索引的 lane 直接复制 | `srcLane` 指定的 lane 的 `var` |
| `__shfl_up_sync()` | 从 ID 较低的 lane 复制 | caller lane ID 减去 `delta` |
| `__shfl_down_sync()` | 从 ID 较高的 lane 复制 | caller lane ID 加上 `delta` |
| `__shfl_xor_sync()` | 基于位异或复制 | caller lane ID 和 `laneMask` 的 XOR |

#### `mask` 参数

`mask` 是一个32位无符号整数，其中每一位代表对应的 lane 是否参与调用。必须为每个参与线程设置其 lane ID 对应的位。`mask` 中命名的所有未退出线程必须以相同的 mask 执行相同的 intrinsic，否则结果未定义。通常使用 `0xffffffff` 表示所有32个 lane 都参与。

#### `width` 参数

所有 `__shfl_sync()` intrinsic 都接受一个可选的 `width` 参数（必须为 2 的幂，≤ warpSize）。其作用是**限定参与通信的逻辑 lane ID 范围为 `[0, width-1]`**：范围外的线程不参与数据交换，shuffle 返回其自身的 `var` 值。这与“将 warp 分割成多个并行通信的独立子段”有本质区别——需要多子段通信时，需手动处理逻辑 lane ID。

#### Shuffle 函数详解

<strong>`__shfl_sync(mask, var, srcLane, width)`</strong>：返回由 `srcLane` 指定的 lane 持有的 `var` 值。如果 `width < warpSize`，每个子段表现为具有从0开始的逻辑 lane ID 的独立实体。如果 `srcLane` 超出 `[0:width-1]` 范围，返回的值对应于 `srcLane % width` 处的 lane 的值。

```cuda
// 广播：lane 0 的值到整个 warp
unsigned mask = 0xffffffff;
int value;
if (laneId == 0) value = input_data;
value = __shfl_sync(mask, value, 0);  // 所有 lane 现在都有 lane 0 的值
```

<strong>`__shfl_up_sync(mask, var, delta, width)`</strong>：从 caller lane ID 减去 `delta` 得到源 lane ID。源 lane 索引不会环绕 `width` 的值，因此下部的 `delta` 个 lane 保持不变。

<strong>`__shfl_down_sync(mask, var, delta, width)`</strong>：从 caller lane ID 加上 `delta` 得到源 lane ID。源 lane 索引不会环绕 `width` 的值，因此上部的 `delta` 个 lane 保持不变。

<strong>`__shfl_xor_sync(mask, var, laneMask, width)`</strong>：通过对 caller lane ID 与 `laneMask` 进行位异或来计算源 lane ID。该模式实现了<strong>蝴蝶寻址模式</strong>（butterfly addressing pattern），用于树形归约和广播。

### 11.4.2 Warp Shuffle 实战示例

#### 示例一：跨 Warp 广播单个值

```cuda
__global__ void broadcast_example(int *input, int *output) {
    int laneId = threadIdx.x & 0x1f;  // threadIdx.x % 32
    int value;
    
    // 只让 lane 0 读取数据
    if (laneId == 0)
        value = input[blockIdx.x];
    
    // 广播 lane 0 的值给整个 warp
    value = __shfl_sync(0xffffffff, value, 0);
    
    output[threadIdx.x] = value;  // 所有 lane 都有相同的值
}
```

#### 示例二：跨子段的 Inclusive Plus-Scan

```cuda
__global__ void scan_example() {
    int laneId = threadIdx.x & 0x1f;
    int value = 31 - laneId;  // 初始值 = 31 - laneId
    
    // 在8线程的子段内执行 inclusive scan
    for (int i = 1; i <= 4; i *= 2) {
        int n = __shfl_up_sync(0xffffffff, value, i, 8);
        if ((laneId & 7) >= i)
            value += n;
    }
    
    printf("Thread %d final value = %d\n", threadIdx.x, value);
}
```

#### 示例三：跨 Warp 的归约（使用 XOR 模式）

这是 warp shuffle <strong>最强大的应用之一</strong>——高效地实现 warp 级归约：

```cuda
__global__ void warp_reduce_example(float *input, float *output) {
    int laneId = threadIdx.x & 0x1f;
    float value = input[threadIdx.x + blockIdx.x * blockDim.x];
    
    // 使用 XOR 模式进行蝴蝶归约
    // 跨32个线程，需要 log2(32) = 5 步
    for (int offset = 16; offset >= 1; offset /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
    
    // value 现在包含整个 warp 的和（每个 lane 都有完全相同的值）
    if (laneId == 0)
        output[blockIdx.x] = value;
}
```

<strong>为什么 XOR 模式归约更高效？</strong> XOR 模式创建了一个蝴蝶交换模式：第一步 offset=16 时，lane 0 和 lane 16 交换并相加，lane 1 和 lane 17 交换并相加，以此类推。所有加法可以并行完成，没有数据依赖性。经过5步（16, 8, 4, 2, 1）后，每个 lane 都有所有32个值的总和。

对比传统的共享内存归约，warp shuffle 归约：
- <strong>无需共享内存分配</strong>
- <strong>无需 `__syncthreads()`</strong>
- <strong>寄存器级延迟</strong>（通常每个 shuffle 仅几个时钟周期）

### 11.4.3 Warp Vote 函数

CUDA 还提供了 warp 投票函数，用于快速查询 warp 内线程的条件状态：

```cuda
int __all_sync(unsigned mask, int predicate);
int __any_sync(unsigned mask, int predicate);
unsigned __ballot_sync(unsigned mask, int predicate);
```

- <strong>`__all_sync(mask, pred)`</strong>：当 mask 中所有线程的谓词都非零时为真
- <strong>`__any_sync(mask, pred)`</strong>：当 mask 中任何线程的谓词非零时为真
- <strong>`__ballot_sync(mask, pred)`</strong>：返回一个32位 mask，其中谓词非零的每个线程对应的位被设置

```cuda
// Warp Vote 示例：查找 warp 中是否有线程满足条件
__global__ void vote_example(float *data, int n, int *result) {
    int laneId = threadIdx.x & 0x1f;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    bool is_negative = (idx < n) && (data[idx] < 0.0f);
    
    // 检查 warp 中是否有负值
    if (__any_sync(0xffffffff, is_negative)) {
        // 获取所有负值线程的 mask
        unsigned neg_mask = __ballot_sync(0xffffffff, is_negative);
        if (laneId == 0)
            result[blockIdx.x] = __popc(neg_mask);  // 负数计数
    }
}
```

### 11.4.4 Warp Shuffle 与共享内存的比较

让我们做一个公平的比较——用两种方法实现跨 block 的归约：

<strong>共享内存版本</strong>：
```cuda
__global__ void reduce_smem(float *input, float *output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    // 最后32个元素用 warp shuffle
    if (tid < 32) {
        float val = sdata[tid];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_xor_sync(0xffffffff, val, offset);
        if (tid == 0)
            output[blockIdx.x] = val;
    }
}
```

<strong>纯 Warp Shuffle 版本</strong>（仅限 warp 内32个元素）：
```cuda
__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}
```

<strong>性能差距</strong>：纯 warp shuffle 版本免除了共享内存分配和同步的开销，对于 warp 内归约，可以比共享内存版本快 2-3 倍。在实际应用中，最佳做法是结合两者——使用共享内存进行跨 warp 的归约，然后在每个 warp 内使用 shuffle 完成最后的工作。

---

## 11.5 原子操作（Atomic Functions）

原子函数在全局或共享内存中的一个32位或64位字上执行<strong>读-修改-写</strong>原子操作。操作是原子的，意味着它保证在没有其他线程干扰的情况下执行——在操作完成之前，没有其他线程可以访问此地址。

<strong>重要</strong>：原子函数不作内存栅栏，也不对内存操作施加同步或排序约束。

### 11.5.1 原子操作的作用域

- <strong>系统级（System-wide）</strong>：对当前程序中所有线程（包括其他 CPU 和 GPU）原子执行。后缀为 `_system`，如 `atomicAdd_system`
- <strong>设备级（Device-wide）</strong>：对当前计算设备中所有 CUDA 线程原子执行。无特殊后缀，如 `atomicAdd`
- <strong>块级（Block-wide）</strong>：对当前线程块中所有 CUDA 线程原子执行。后缀为 `_block`，如 `atomicAdd_block`

### 11.5.2 算术原子函数

主要算术原子函数包括：

```cuda
int atomicAdd(int* address, int val);
float atomicAdd(float* address, float val);  // CC 2.0+
double atomicAdd(double* address, double val);  // CC 6.0+
int atomicSub(int* address, int val);
int atomicExch(int* address, int val);
int atomicMin(int* address, int val);
int atomicMax(int* address, int val);
unsigned int atomicInc(unsigned int* address, unsigned int val);
unsigned int atomicDec(unsigned int* address, unsigned int val);
int atomicCAS(int* address, int compare, int val);  // Compare-And-Swap
```

位运算原子函数：`atomicAnd()`、`atomicOr()`、`atomicXor()`

### 11.5.3 原子操作的性能特征

原子操作的性能取决于<strong>争用（contention）程度</strong>：

- <strong>低争用</strong>：当不同线程访问不同地址时，原子操作可以接近普通全局内存访问的速度
- <strong>高争用</strong>：当许多线程竞争同一地址时，性能急剧下降——因为操作必须被串行化
- <strong>共享内存中的原子操作通常比全局内存中的更快</strong>——因为共享内存的延迟更低

### 11.5.4 使用 atomicCAS 实现自定义原子操作

任何原子操作都可以基于 `atomicCAS()`（Compare And Swap）实现。例如，在 Compute Capability 低于 6.0 的设备上，`double` 的 `atomicAdd` 不可用，但可以这样实现：

```cuda
#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    // 使用整数比较以避免 NaN 导致死循环（因为 NaN != NaN）
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif
```

### 11.5.5 原子操作优化策略

1. <strong>减少争用</strong>：尽量减少多个线程访问同一地址的频率
2. <strong>使用共享内存进行局部归约后再做全局原子操作</strong>——这是最常见的模式
3. <strong>在支持时使用块级原子</strong>：`atomicAdd_block` 比 `atomicAdd` 快，因为它只需要块内同步
4. <strong>对于 FP16 原子</strong>：`__half2` 的 `atomicAdd` 分别保证两个 `__half` 元素的原子性（但整个 `__half2` 不是原子的）

```cuda
// 优化模式：先在共享内存中做块级归约，再做一次全局原子操作
__global__ void histogram_optimized(int *input, int *histogram, int n) {
    __shared__ int local_hist[256];
    
    // 初始化共享内存直方图
    if (threadIdx.x < 256) local_hist[threadIdx.x] = 0;
    __syncthreads();
    
    // 块级归约到共享内存
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        atomicAdd(&local_hist[input[idx] % 256], 1);
    }
    __syncthreads();
    
    // 将块结果合并到全局直方图（仅每块的第一个 warp）
    if (threadIdx.x < 256) {
        atomicAdd(&histogram[threadIdx.x], local_hist[threadIdx.x]);
    }
}
```

---

## 11.6 只读数据缓存：`__ldg()`

从 Compute Capability 3.5 开始，CUDA 提供了一个<strong>只读数据缓存加载函数</strong> `__ldg()`：

```cuda
T __ldg(const T* address);
```

其中 `T` 可以是 `char`、`short`、`int`、`long long`、`float`、`double` 及相应的向量类型（`float2`、`float4`、`double2` 等），以及 `__half` 和 `__half2`。

`__ldg()` 加载的数据被缓存在<strong>只读数据缓存（Read-Only Data Cache）</strong>中。这个缓存独立于 L1 缓存，对于以读取为主的访问模式特别有用。

<strong>何时使用 `__ldg()`？</strong>

- <strong>数据只读</strong>：内核只从该地址读取，不写入
- <strong>数据可能被多个线程重复读取</strong>：只读缓存可以帮助减少全局内存流量
- <strong>访问模式不规则</strong>：即使访问不完全合并，只读缓存也可能比全局加载提供更好的性能

```cuda
__global__ void read_only_kernel(const float *input, float *output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // 使用 __ldg() 通过只读数据缓存加载
    float val = __ldg(input + idx);
    
    output[idx] = val * 2.0f;
}
```

使用 `__ldg()` 与使用 `const T* __restrict__` 指针声明是互补的优化——`__restrict__` 告诉编译器不会有别名（aliasing），而 `__ldg()` 显式指定了缓存行为。

---

## 动手体验

### 实验 11-1：Warp Divergence 的代价

在这个实验中，你将测量不同分支模式对性能的影响。完整代码保存在 `code/chapter11/divergence_benchmark.cu`。

<strong>实验目标</strong>：测量并比较以下分支模式：
1. 无分支（基线）
2. warp 对齐的分支（`if (threadIdx.x / 32) % 2 == 0`）
3. warp 内发散的分支（`if (threadIdx.x % 2 == 0`）
4. 随机分支（`if (rand() % 2 == 0`）

<strong>预期结果</strong>：warp 对齐的分支与无分支性能接近；warp 内发散导致性能显著下降。

### 实验 11-2：Warp Shuffle 归约 vs 共享内存归约

这是本章的<strong>核心实验</strong>。完整代码保存在 `code/chapter11/warp_reduce_benchmark.cu`。

<strong>实验目标</strong>：实现对 N 个浮点数的求和归约，比较以下版本：
1. 纯全局内存原子累加（最慢，作为基线）
2. 共享内存归约（块内归约 + 块间原子操作）
3. Warp Shuffle 归约（warp 内 shuffle，跨 warp 用共享内存 + 块间原子操作）
4. 完全优化的归约（向量化加载 + warp shuffle + 展开循环）

<strong>预期结果</strong>（N = 16M，RTX 3060）：
- 版本1：约 5 GB/s（原子操作瓶颈）
- 版本2：约 150 GB/s
- 版本3：约 250 GB/s
- 版本4：约 320+ GB/s

### 实验 11-3：快速数学函数性能对比

完整代码保存在 `code/chapter11/math_benchmark.cu`。

<strong>实验目标</strong>：测量标准函数与 intrinsic 函数的吞吐量对比：
1. `1.0f / sqrtf(x)` vs `rsqrtf(x)`
2. `x / y` vs `__fdividef(x, y)`
3. `sinf(x)` vs `__sinf(x)`（小参数 vs 大参数）

---

## 本章小结

本章深入讲解了 CUDA 性能优化的进阶主题——指令吞吐量优化和 Warp 级编程技术：

1. <strong>指令吞吐量因指令类型差异巨大</strong>：FP32 FMA 最快（128 ops/clock/SM），FP64 慢很多（2-32 ops），整数除法和三角函数最慢
2. <strong>使用 Intrinsic 函数以精度换速度</strong>：`__fdividef()`、`rsqrtf()`、`__sinf()`、`__cosf()` 等
3. <strong>Warp Divergence 是性能杀手</strong>：尽量让分支条件与 warp 边界对齐，或使用分支预测
4. <strong>Warp Shuffle 是终极优化武器</strong>：寄存器级数据交换，比共享内存快数倍
5. <strong>原子操作性能取决于争用程度</strong>：使用共享内存局部归约来减少全局原子操作
6. <strong>`__ldg()` 通过只读数据缓存优化全局内存读取</strong>
7. <strong>组合使用多种技术才能达到最佳性能</strong>：最优化的内核通常结合了合并访问、向量化加载、warp shuffle、循环展开等技术

---

## 习题

### 习题 11-1：Warp Divergence 分析

(a) 编写一个内核，使用以下模式处理数组中的偶数和奇数元素：`if (idx % 2 == 0) { /* 偶数处理 */ } else { /* 奇数处理 */ }`。测量性能。

(b) 重构内核，改为先处理所有偶数元素，再处理所有奇数元素（使用两个独立的 pass，每个 pass 无 divergence）。测量性能并比较。

(c) 对于大数组（N >> 线程数），方案 (b) 的总工作量翻倍（每个元素被检查两次）。设计一个折中方案，使用 `__ballot_sync()` 或协作组（Cooperative Groups）来最小化 divergence 而不增加总工作量。

### 习题 11-2：Warp Shuffle 深度练习

(a) 使用 `__shfl_xor_sync()` 实现 warp 内求最大值（而非求和）的归约。验证正确性。

(b) 使用 `__shfl_down_sync()` 实现 warp 级的包容性前缀和（Inclusive Prefix Sum）。输出每个 lane 的前缀和。

(c) 将 warp 级归约扩展为完整的跨 block 归约：实现向量求和，其中第一阶段在每个 warp 内用 shuffle 归约，第二阶段在每个 block 内收集 warp 结果，第三阶段在多个 block 之间用原子操作合并。测量性能并与仅使用共享内存的版本比较。

(d) 挑战题：使用 warp shuffle 实现 warp 内排序（bitonic sort within a warp）。测量并讨论 warp shuffle 在排序算法中的优势。

### 习题 11-3：指令吞吐量优化

(a) 编写一个内核，使用双精度浮点数进行向量运算（加法、乘法、除法）。测量其吞吐量（GFLOPS）。

(b) 将内核中的双精度运算改为单精度运算。测量吞吐量并计算加速比。在你的 GPU 上，FP32 和 FP64 的理论吞吐量比是多少？

(c) 将除法替换为 `__fdividef()`，将平方根替换为 `rsqrtf()` 后接倒数。测量吞吐量变化。

(d) 使用 `half2` 类型和内建向量函数（`__hadd2`、`__hmul2`）将相同的内核改写为半精度版本。测量吞吐量并与单精度版本比较。

### 习题 11-4：综合优化——直方图

直方图计算是许多图像处理和数据分析算法的核心。实现一个高性能直方图内核：

(a) 版本1：朴素实现——每个线程使用全局原子操作更新直方图。测量性能。

(b) 版本2：每个 block 使用共享内存局部直方图，然后在写入全局内存之前使用一个全局原子操作合并。测量性能提升。

(c) 版本3：使用 warp shuffle 在 warp 内合并部分结果，减少共享内存原子操作。测量额外性能提升。

(d) 分析各版本的瓶颈（使用 Nsight Compute 或理论分析）。你的实现达到了多少百分比的峰值全局内存原子吞吐量？

---

## 参考文献

1. NVIDIA CUDA C++ Programming Guide v11.5.1, Chapter 5: Performance Guidelines, Sections 5.4 (Maximize Instruction Throughput) — [https://docs.nvidia.com/cuda/archive/11.5.1/cuda-c-programming-guide/index.html#maximize-instruction-throughput](https://docs.nvidia.com/cuda/archive/11.5.1/cuda-c-programming-guide/index.html#maximize-instruction-throughput)
2. NVIDIA CUDA C++ Programming Guide v11.5.1, Appendix B.22: Warp Shuffle Functions — [https://docs.nvidia.com/cuda/archive/11.5.1/cuda-c-programming-guide/index.html#warp-shuffle-functions](https://docs.nvidia.com/cuda/archive/11.5.1/cuda-c-programming-guide/index.html#warp-shuffle-functions)
3. NVIDIA CUDA C++ Programming Guide v11.5.1, Appendix B.14: Atomic Functions — [https://docs.nvidia.com/cuda/archive/11.5.1/cuda-c-programming-guide/index.html#atomic-functions](https://docs.nvidia.com/cuda/archive/11.5.1/cuda-c-programming-guide/index.html#atomic-functions)
4. NVIDIA CUDA C++ Programming Guide v11.5.1, Appendix B.10: Read-Only Data Cache Load Function — [https://docs.nvidia.com/cuda/archive/11.5.1/cuda-c-programming-guide/index.html#ldg-function](https://docs.nvidia.com/cuda/archive/11.5.1/cuda-c-programming-guide/index.html#ldg-function)
5. NVIDIA CUDA C++ Programming Guide v13.0, Chapter 8: Performance Guidelines — [https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-programming-guide/index.html#performance-guidelines](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-programming-guide/index.html#performance-guidelines)
6. NVIDIA CUDA C++ Best Practices Guide — [https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
