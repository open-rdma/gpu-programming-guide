# 第4章 NVCC编译模型

欢迎来到 CUDA 编译技术的世界！在前面三章中，我们已经学习了 CUDA 编程模型的核心概念——线程层次结构、内存层次以及异构编程的基本模式。现在是时候深入幕后，了解驱动这一切的编译工具链了。在本章中，我们将系统学习 <strong>NVCC（NVIDIA CUDA Compiler）</strong>——这个编译器驱动程序如何将我们编写的 CUDA C++ 代码转化为 GPU 可执行的二进制指令。理解编译模型不仅有助于我们写出更高效的代码，还能帮助我们在面对不同 GPU 架构时做出正确的编译选项选择。让我们开始吧！

## 4.1 NVCC 编译器概述

<strong>NVCC（NVIDIA CUDA Compiler Driver）</strong> 是 NVIDIA 提供的 CUDA 编译器驱动程序。内核可以使用 CUDA 指令集架构（称为 <strong>PTX, Parallel Thread Execution</strong>）来编写，PTX 在 PTX 参考手册中有详细说明。然而，使用 C++ 这样的高级编程语言通常更高效。无论哪种情况，内核都必须由 NVCC 编译为二进制代码才能在设备上执行。

NVCC 简化了 C++ 或 PTX 代码的编译流程：它提供简洁、熟悉的命令行选项，并通过调用实现不同编译阶段的工具集合来执行相应的编译步骤。

### 4.1.1 NVCC 的核心角色

NVCC 的定位可以从以下几个角度理解：

1. <strong>编译器驱动程序（Compiler Driver）</strong>：NVCC 本身不是一个"编译器"，而是一个编译器驱动程序。它负责协调多个底层工具的调用，包括 CUDA 专用编译器（cicc —— CUDA C/C++ 前端编译器）和标准 C++ 主机编译器（如 Linux 上的 `gcc`、Windows 上的 `cl.exe`）。

2. <strong>代码分离器</strong>：NVCC 能够识别 `.cu` 文件中的主机代码和设备代码，将它们分离开来，分别交由不同的编译器处理。这涉及到对 CUDA 语法（如 `__global__`、`__device__`、`<<<...>>>`）的解析。

3. <strong>多目标代码生成器</strong>：NVCC 可以为同一个 CUDA 程序生成多种目标格式——主机目标代码（`.o`/`.obj`）、设备 PTX 汇编代码、设备 cubin 二进制代码，或者直接生成最终的可执行文件。

4. <strong>链接协调器</strong>：NVCC 协调主机端目标文件与设备端目标文件的链接过程，确保最终可执行文件中正确嵌入了设备代码。

> <strong>提示</strong>：NVCC 的完整说明参见《NVCC 用户手册》。本章重点介绍与 CUDA 编程直接相关的核心概念和常用选项。

### 4.1.2 NVCC 与标准 C++ 编译器的关系

NVCC 的工作方式与传统 C++ 编译器有一个关键区别：它需要分别处理两种不同的代码——<strong>主机代码（Host Code）</strong>和<strong>设备代码（Device Code）</strong>。主机代码运行在 CPU 上，需要由标准 C++ 编译器（如 gcc、clang 或 MSVC）编译；设备代码运行在 GPU 上，需要由 NVIDIA 的 GPU 编译器编译。

NVCC 的设计哲学是：对程序员而言，这两种代码的编译看起来像一个统一的编译过程。你只需要一个命令：

```bash
nvcc my_program.cu -o my_program
```

NVCC 会在后台自动完成：
1. 代码分离（separate host and device code）
2. 设备代码编译（compile device code to PTX/cubin）
3. 主机代码转换（transform `<<<...>>>` to runtime calls）
4. 主机代码编译（compile host code via host compiler）
5. 最终链接（link everything together）

整个过程的起点是一个 `.cu` 文件，终点是一个可以在目标系统上运行的可执行文件。

### 4.1.3 NVCC 的常见命令行选项概览

在深入细节之前，让我们先熟悉一下 NVCC 最常用的几类命令行选项：

| 类别 | 常用选项 | 说明 |
| :--- | :--- | :--- |
| <strong>输出控制</strong> | `-o <file>` | 输出文件名 |
| | `-c` | 只编译不链接（生成 `.o` 文件） |
| | `-ptx` | 生成 PTX 汇编文件 |
| | `-cubin` | 生成 cubin 二进制文件 |
| <strong>架构控制</strong> | `-arch=<arch>` | 指定虚拟/真实架构 |
| | `-code=<code>` | 指定目标代码类型 |
| | `-gencode <spec>` | 生成多架构胖二进制 |
| <strong>优化控制</strong> | `-O<n>` | 优化级别（0-3） |
| | `-G` | 生成设备调试信息 |
| | `-lineinfo` | 生成行号信息 |
| <strong>平台控制</strong> | `-m32` / `-m64` | 32 位或 64 位模式 |
| | `--compiler-options` | 传递选项给主机编译器 |

## 4.2 编译工作流

### 4.2.1 离线编译（Offline Compilation）—— 编译流程详解

NVCC 的<strong>基本工作流程</strong>是将设备代码与主机代码分离，然后分别处理。让我们逐步详细展开这个流程。

<strong>第 1 步：预处理与代码分离</strong>

NVCC 首先对 `.cu` 源文件进行预处理（展开宏、处理 `#include` 等）。然后，CUDA 前端（cicc）扫描源文件，识别设备代码部分——即那些由 `__global__`、`__device__`、`__host__` 等声明修饰符标记的函数和变量。设备代码被提取出来，主机代码的其余部分保留等待后续处理。

<strong>第 2 步：设备代码编译</strong>

提取出的设备代码被编译为以下一种或两种形式：

- <strong>PTX 代码（Parallel Thread Execution）</strong>：一种类似汇编的中间表示（Intermediate Representation, IR），是 CUDA 指令集架构的文本形式。PTX 是稳定、文档化的 ISA，由 NVIDIA 公开发布。
- <strong>cubin 目标代码（CUDA Binary）</strong>：针对特定 GPU 架构（如 sm_70、sm_80）的二进制代码，可直接加载并在目标 GPU 上执行。cubin 遵循 ELF（Executable and Linkable Format）格式。

<strong>第 3 步：主机代码转换</strong>

这是 NVCC 最精巧的部分。主机代码中的 `<<<...>>>` 内核启动语法（如 `VecAdd<<<blocksPerGrid, threadsPerBlock>>>(...)`）被替换为必要的 CUDA 运行时（CUDA Runtime）函数调用。这些调用负责：

1. 从嵌入的 PTX 代码和/或 cubin 二进制中加载编译好的内核。
2. 设置执行配置（grid 维度、block 维度、共享内存大小、流）。
3. 在设备上启动内核执行。

替换后的主机代码可被标准 C++ 编译器理解。修改后有两套输出方案：

- <strong>输出为 C++ 代码</strong>：NVCC 输出转换后的中间 C++ 文件（`.cpp`），留待后续由另一个工具（如 gcc）编译。
- <strong>直接调用主机编译器</strong>：NVCC 在最后编译阶段自动调用主机编译器，直接输出为目标代码（`.o`/`.obj`）。这是默认行为，对程序员完全透明。

<strong>第 4 步：链接</strong>

应用程序可以：
- <strong>链接到编译后的主机代码</strong>：这是最常见的情况。NVCC 调用主机链接器，将主机端目标文件和嵌入的设备代码链接为最终可执行文件。
- <strong>忽略主机代码，使用 Driver API</strong>：将设备代码编译为独立的 cubin 或 PTX 文件，在运行时通过 <strong>CUDA Driver API</strong>（而不是 Runtime API）自行加载和执行。这为需要精细控制设备代码加载的应用程序（如框架类库）提供了最大的灵活性。

下面用一个简明的图示来总结这个编译流程：

```
                   .cu 源文件
                       |
                   NVCC 驱动
                       |
            +----------+-----------+
            |                      |
       设备代码提取           主机代码保留
            |                      |
      cicc (CUDA 前端)       <<<...>>> 替换
            |                为 Runtime 调用
       +----+----+                 |
       |         |           调用主机编译器
     PTX       cubin              |
       |         |            host_obj.o
       +----+----+                 |
            |                      |
      嵌入到可执行文件 ------------+
            |
        最终可执行文件
```

### 4.2.2 离线编译的完整示例

让我们通过一个完整的编译和观察过程来理解离线编译。假设我们有一个简单的 CUDA 文件 `hello.cu`：

```cuda
#include <stdio.h>

__global__ void hello_kernel() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    hello_kernel<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

<strong>阶段观察法 1：只生成 PTX</strong>

```bash
nvcc hello.cu -ptx -arch=compute_80 -o hello.ptx
```

生成的 `hello.ptx` 文件片段如下：

```text
.version 7.5
.target sm_80
.address_size 64

.visible .entry _Z12hello_kernelv()
{
    .reg .b32   %r<2>;
    .reg .b64   %rd<3>;

    // 获取 threadIdx.x
    mov.u32     %r1, %tid.x;
    // 准备 printf 参数
    ...
    // 调用 vprintf
    ...
    ret;
}
```

观察要点：
- `.version 7.5` 表示 PTX ISA 版本。
- `.target sm_80` 表示目标架构。
- `_Z12hello_kernelv` 是 `hello_kernel()` 的名称修饰（name mangling）后的符号名。
- 可以看到实际的 PTX 指令（`mov.u32`, `ld.param`, `ret` 等）。

<strong>阶段观察法 2：生成编译中间文件</strong>

```bash
# 保留中间文件
nvcc hello.cu -keep -o hello
ls hello*
# hello.cpp4.ii   <-- 预处理后的文件
# hello.ptx       <-- 设备代码的 PTX 版本
```

`-keep` 选项让 NVCC 保留所有中间文件，非常适合学习和调试。

<strong>阶段观察法 3：只编译不链接</strong>

```bash
nvcc hello.cu -c -o hello.o
```

生成目标文件 `hello.o`，可以稍后与其它目标文件链接。设备代码信息（PTX/cubin）已嵌入目标文件中。

### 4.2.3 即时编译（Just-in-Time Compilation, JIT）

在运行时，应用程序加载的任何 PTX 代码都会被<strong>设备驱动程序（device driver）</strong>进一步编译为目标 GPU 的二进制代码。这个过程称为<strong>即时编译（JIT Compilation）</strong>。

<strong>JIT 编译的工作时机</strong>：

JIT 编译发生在 CUDA 运行时（或 Driver API）首次需要加载并启动一个内核时。当运行时找不到与目标 GPU 完全匹配的 cubin 二进制时，它会查找可用的 PTX 代码，并调用驱动程序中的 JIT 编译器来生成针对该特定 GPU 的优化二进制。

<strong>JIT 编译的优点</strong>：
- <strong>驱动程序级别的优化</strong>：允许应用受益于每个新设备驱动程序中的编译器改进。即使你的应用是两年前编译的，新驱动程序中的 JIT 编译器可以生成更优的代码。
- <strong>未来硬件支持</strong>：是使应用能够在编译时尚不存在的 GPU 上运行的唯一方式（详见 4.4 节"PTX 兼容性"）。
- <strong>减少分发体积</strong>：可以只分发 PTX 代码而非为每个架构都嵌入独立的 cubin，从而减小可执行文件体积。

<strong>JIT 编译的代价</strong>：
- <strong>首次加载时间增加</strong>：应用的首次启动会将 PTX 编译为目标 GPU 的二进制代码，这可能耗时几百毫秒到几秒，取决于 PTX 代码的复杂度。
- <strong>编译结果的不确定性</strong>：不同驱动程序版本的 JIT 编译器可能生成不同的二进制，导致性能在不同用户环境中有细微差异。

<strong>计算缓存（Compute Cache）</strong>：

为了缓解首次加载时间的问题，设备驱动程序会自动缓存 JIT 编译的结果：

- 当驱动程序 JIT 编译某个 PTX 代码后，会将生成的二进制代码副本存入<strong>计算缓存（compute cache）</strong>。
- 缓存是<strong>基于磁盘的</strong>，因此即使重启系统也能保留。
- 缓存 key 由 PTX 代码的内容和 GPU 架构信息共同决定。
- 缓存会在设备驱动程序升级时<strong>自动失效</strong>——这确保了新驱动可以重新编译生成更优的代码。
- 在 Linux 上，默认缓存目录为 `~/.nv/ComputeCache/`。在 Windows 上，为 `%APPDATA%\NVIDIA\ComputeCache\`。

<strong>JIT 缓存的环境变量控制</strong>：

可以通过以下环境变量精细控制 JIT 缓存行为：

| 环境变量 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `CUDA_CACHE_DISABLE` | 0 或 1 | 0 | 设为 1 禁用 JIT 缓存（每次重新编译） |
| `CUDA_CACHE_PATH` | 路径 | OS 默认 | 指定 JIT 缓存目录路径 |
| `CUDA_CACHE_MAXSIZE` | 整数（字节） | 1073741824 (1GB) | 指定 JIT 缓存最大大小 |

调试示例：观察 JIT 缓存行为

```bash
# 禁用缓存，观察首次加载时间
CUDA_CACHE_DISABLE=1 ./my_cuda_app

# 第二次运行（缓存命中，应该更快）
./my_cuda_app

# 查看缓存内容
ls -la ~/.nv/ComputeCache/
```

> <strong>提示</strong>：除了使用 NVCC 进行离线编译外，NVIDIA 还提供了 <strong>NVRTC（CUDA Runtime Compilation）</strong> 库。NVRTC 可以在运行时将 CUDA C++ 设备代码字符串动态编译为 PTX。这在需要在运行时生成或定制内核的应用场景中非常有用（如自动调优框架、表达式 JIT 引擎等）。更多信息见 NVRTC User Guide。

### 4.2.4 离线编译 vs JIT 编译的决策指南

| 特性 | 离线编译 | 即时编译（JIT） |
| :--- | :--- | :--- |
| <strong>编译时机</strong> | 开发时，在开发者机器上 | 运行时，在用户机器上 |
| <strong>输出格式</strong> | cubin（二进制）或 PTX | 从 PTX 生成 cubin |
| <strong>加载时间</strong> | 快（无需编译） | 首次较慢，缓存命中后快 |
| <strong>未来硬件支持</strong> | 仅限编译时已支持的架构 | 可支持编译时尚不存在的硬件 |
| <strong>编译器优化</strong> | 固定在编译时 | 随驱动程序更新而改进 |
| <strong>二进制体积</strong> | 大（嵌入多份 cubin） | 小（只嵌入 PTX） |
| <strong>性能确定性</strong> | 高（同一 cubin 同一性能） | 中等（依赖驱动版本） |
| <strong>推荐场景</strong> | 生产环境，已知目标硬件 | 需要跨代兼容的应用 |

## 4.3 二进制兼容性

### 4.3.1 二进制兼容性规则

<strong>二进制代码是体系结构相关的</strong>。一个 <strong>cubin</strong> 对象通过编译器选项 `-code` 指定目标架构来生成。例如，使用 `-code=sm_35` 编译会生成针对<strong>计算能力（Compute Capability）</strong> 3.5 的设备的二进制代码。

二进制兼容性规则：
cubin遵循<strong>单主版本内、向前兼容</strong>的规则：

> 为计算能力 <em>X.y</em> 生成的 cubin 对象仅能在计算能力 <em>X.z</em>（其中 <em>z >= y</em>）的设备上执行。

PTX + JIT 兼容性规则：
PTX 虚拟指令集遵循跨主版本向前兼容规则：
为计算能力 <em>X.y </em>生成的 PTX 代码，可以在所有计算能力≥<em>X.y </em>的设备上通过 JIT 编译执行。

具体来说：
- <strong>向前兼容</strong>（同一个主版本 X 内）：`compute_35` 可以在 `sm_37` 上运行。`compute_50` 可以在 `sm_52`、`sm_53` 上运行。`compute_80` 可以在 `sm_86`、`sm_89` 上运行。
- <strong>不支持向后兼容</strong>：`compute_80` 不能在 `sm_75` 上运行。
- <strong>不支持跨主版本兼容</strong>：`compute_35` 不能在 `sm_50` 上运行（因为主版本 3 ≠ 5）。`compute_60` 不能在 `sm_70` 上运行（因为主版本 6 ≠ 7）。

> <strong>注意</strong>：二进制兼容性仅支持桌面平台（Desktop）。Tegra 平台不支持二进制兼容性。桌面与 Tegra 之间的二进制兼容性也不支持。

这个规则可以用下图直观表示：

```
  主版本 X (如 Kepler = 3)          主版本 Y (如 Maxwell = 5)
 ┌─────────────────────┐         ┌─────────────────────┐
 │ 3.0 → 3.2 → 3.5 → 3.7│         │ 5.0 → 5.2 → 5.3    │
 │    ← z >= y 兼容 →   │         │    ← z >= y 兼容 →   │
 └─────────────────────┘         └─────────────────────┘
       ↓ 不可兼容 ↓                     ↓ 不可兼容 ↓
 ┌─────────────────────┐         ┌─────────────────────┐
 │ 主版本 Z (如 Ampere = 8) │         │ ...                     │
 └─────────────────────┘         └─────────────────────┘
```

### 4.3.2 compute_XY 与 sm_XY 的深入理解

这是 CUDA 编译模型中最基础也最容易混淆的概念：

- <strong>`compute_XY`（虚拟架构）</strong>：指定 PTX 代码编译时<strong>假定的计算能力级别</strong>。它定义了一个"功能基线"——编译器在生成 PTX 时，会假设硬件具备这个级别所定义的所有特性（指令、寻址模式等）。"虚拟"意味着它是一组抽象的硬件特性集合，不直接对应某个具体芯片。

- <strong>`sm_XY`（真实架构）</strong>：指定 cubin 二进制代码的<strong>具体目标硬件</strong>。`sm_35` 表示生成可直接在计算能力 3.5 硬件上执行的指令。

<strong>关键区别</strong>：
- `compute_XY` 控制<strong>什么样的 PTX 指令可以被生成</strong>。如果你的代码使用了 warp shuffle 指令，你必须至少指定 `compute_30`（因为 shuffle 从 CC 3.0 开始支持）。
- `sm_XY` 控制<strong>二进制代码针对哪个具体芯片生成</strong>。编译到 `sm_80` 的 cubin 可以在 A100（sm_80）上直接运行，也可以在 A10（sm_86）上通过二进制兼容性运行。

### 4.3.3 常见计算能力的对应关系

| 计算能力 | 架构代号 | 年份 | 代表产品 |
| :--- | :--- | :---: | :--- |
| 2.0, 2.1 | Fermi | 2010 | C2050, GTX 480 |
| 3.0, 3.5, 3.7 | Kepler | 2012 | K20, K40, K80 |
| 5.0, 5.2, 5.3 | Maxwell | 2014 | M40, M60, Tegra X1 |
| 6.0, 6.1, 6.2 | Pascal | 2016 | P100, GTX 1080, Tegra X2 |
| 7.0, 7.5 | Volta | 2017 | V100, T4, Xavier |
| 8.0, 8.6, 8.7, 8.9 | Ampere | 2020 | A100, RTX 3060/3070/3080/3090 |
| 9.0 | Hopper | 2022 | H100 |
| 10.0 | Blackwell | 2024 | B100, B200 |

## 4.4 PTX 兼容性

### 4.4.1 PTX 的向前兼容保证

某些 PTX 指令仅在较高计算能力的设备上受支持。例如，<strong>Warp Shuffle 函数</strong>（`__shfl_sync` 等）仅在计算能力 3.0 及以上的设备上受支持。`-arch` 编译器选项指定在将 C++ 编译为 PTX 代码时所假定的计算能力。因此，包含 warp shuffle 的代码必须使用 `-arch=compute_30`（或更高）进行编译。

PTX 兼容性的核心规则是：

> 为某个特定计算能力生成的 PTX 代码，<strong>总是可以编译为大于或等于该计算能力的二进制代码</strong>。

这意味着 PTX 具有极强的<strong>向前兼容（forward compatibility）</strong>保证。例如，为计算能力 5.0 生成的 PTX 代码可以被 JIT 编译并运行在：
- 计算能力 6.0 的 Pascal 设备上
- 计算能力 7.0 的 Volta 设备上
- 计算能力 8.0 的 Ampere 设备上
- 计算能力 9.0 的 Hopper 设备上
- 甚至尚未发布的未来架构上

<strong>PTX 兼容性与二进制兼容性的关键区别</strong>：

二进制兼容性只在<strong>同一个主版本内</strong>有效（如 8.0 → 8.9），但 PTX 的向前兼容是<strong>跨主版本的</strong>（如 compute_50 PTX → 可 JIT 到 sm_80 二进制）。这是为何 PTX 是"未来兼容"的基石。

### 4.4.2 旧版本 PTX 的性能限制

一个重要的注意事项是：<strong>从较早版本的 PTX 编译出的二进制可能无法利用某些新硬件特性</strong>。

考虑这个场景：你从为计算能力 6.0（Pascal）生成的 PTX 编译出一个针对计算能力 7.0（Volta）设备的二进制。这个二进制能运行吗？<strong>能</strong>。但它能使用 Tensor Core 指令来加速矩阵乘法吗？<strong>不能</strong>。原因是 Tensor Core 在 Pascal 架构上不存在，因此为 Pascal 生成的 PTX 代码中绝不可能包含 Tensor Core 指令。JIT 编译器只能将 PTX 中已有的指令翻译为机器码，不能"凭空创造出"PTX 中没有的 Tensor Core 指令。

最终的结果是：二进制性能可能<strong>远低于</strong>使用最新版本 PTX 生成的二进制。

> <strong>提示</strong>：出于最佳性能考虑，建议在编译时包含<strong>最高计算能力</strong>的 PTX 代码（使用 `-gencode arch=compute_XX,code=compute_XX`，其中 `XX` 取你能使用的最高值）。这样 JIT 编译器可以在最新硬件上充分使用所有最新的硬件特性。

### 4.4.3 CUDA 13.0 新增的 PTX 兼容性类型

从 CUDA 13.0（Compute Capability 10.0+）开始，PTX 兼容性引入了两种新类别，以适应未来 GPU 架构日益增加的多样性：

<strong>1. 架构特定特性（Architecture-Specific Features）</strong>

使用 `sm_90a` 或 `compute_90a` 编译的 PTX 代码<strong>仅</strong>在完全相同的物理架构上运行，不支持向前或向后兼容。这对使用特定芯片独有特性（如特定的硬件加速器）的场景有用。

编译时还会定义一个额外的宏 `__CUDA_ARCH_SPECIFIC__`（例如，在 `sm_90a` 编译时等于 `900`）。

<strong>2. 系列特定特性（Family-Specific Features）</strong>

使用 `sm_100f` 或 `compute_100f` 编译的 PTX 代码仅在同一系列的设备上运行（例如 10.0 和 10.3 共享同一系列），在同一系列内支持向前兼容，但不跨系列兼容。

编译时还会定义一个额外的宏 `__CUDA_ARCH_FAMILY_SPECIFIC__`（例如，在 `sm_100f` 编译时等于 `1000`）。

> <strong>注意</strong>：作为基础课程，我们主要关注标准 PTX 编译方式（`compute_XY` / `sm_XY`）。架构特定（后缀 `a`）和系列特定（后缀 `f`）的编译属于高级主题，在进阶章节中会详细涉及。

## 4.5 应用程序兼容性

### 4.5.1 多架构 Fat Binary —— 解决"一次编译，到处运行"

为了在特定计算能力的设备上执行代码，应用程序必须加载与该计算能力兼容的二进制代码或 PTX 代码。特别是，<strong>要能在尚未发布、计算能力更高的未来架构上执行，应用程序必须加载 PTX 代码</strong>，让其在运行时进行 JIT 编译。

CUDA 通过<strong>胖二进制（Fat Binary）</strong>机制解决多架构兼容问题。应用程序中嵌入哪些 PTX 和二进制代码，由 `-arch` 和 `-code` 编译器选项或者更灵活的 `-gencode` 编译器选项来控制。

以下是一个典型的 Fat Binary 编译命令：

```bash
nvcc x.cu \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_70,code="compute_70,sm_70"
```

让我们<strong>逐行解读</strong>这个命令：

- <strong>`-gencode arch=compute_50,code=sm_50`</strong>：为计算能力 5.0 生成 PTX（`arch=compute_50`），再从这个 PTX 生成 sm_50 的 cubin 二进制（`code=sm_50`）。最终嵌入：一份 sm_50 cubin。由于 code 部分是 `sm_50` 而非 `compute_50,sm_50`，<strong>不嵌入额外的 PTX</strong>。这意味着 CC 5.2 的设备可以运行（二进制兼容），但 CC 8.0 的设备不行（因为主版本不同且没有 PTX）。

- <strong>`-gencode arch=compute_60,code=sm_60`</strong>：同上，为 CC 6.0-6.2 嵌入 sm_60 cubin。

- <strong>`-gencode arch=compute_70,code="compute_70,sm_70"`</strong>：为 CC 7.0 生成 PTX，并从这个 PTX 生成：
  - 一份 sm_70 cubin（`sm_70`）
  - 一份可 JIT 编译的 PTX（`compute_70`）
  - 这意味着 CC 7.0-7.5 可直接运行 cubin，CC 8.0+ 设备通过 JIT 编译 PTX 来运行。

运行时，主机代码会自动选择最合适的代码来加载和执行：

- CC 5.0 和 5.2 设备：加载 5.0 二进制代码
- CC 6.0 和 6.1 设备：加载 6.0 二进制代码
- CC 7.0 和 7.5 设备：加载 7.0 二进制代码
- CC 8.0 及更高的设备：JIT 编译 PTX 代码

<strong>Fat Binary 选择逻辑</strong>（运行时自动执行）：
1. 查找与当前设备计算能力完全匹配的 cubin。
2. 如果没找到，查找二进制兼容的 cubin（同一主版本，z >= y）。
3. 如果还没找到，寻找可用的 PTX 代码进行 JIT 编译。
4. 如果以上都失败，程序报错退出。

下面是一个更全面的 Fat Binary 编译策略示例：

```bash
nvcc my_kernel.cu -o my_kernel \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_90,code="compute_90,sm_90"
```

这个命令：
- 为 CC 5.0、6.0、7.0、8.0 各嵌入一份直接的 cubin
- 为 CC 9.0 嵌入 cubin <strong>加</strong> PTX
- 结果：CC 5.0-9.0 都有直接 cubin 可用，CC 10.0+ 可以通过 JIT 编译运行

### 4.5.2 利用 `__CUDA_ARCH__` 宏进行条件编译

在同一个源文件 `x.cu` 中，可以有针对不同计算能力的<strong>优化代码路径</strong>。`__CUDA_ARCH__` 宏允许你基于目标计算能力有条件地编译不同的设备代码：

```cuda
// 设备端代码中根据架构选择不同实现
__global__ void myKernel(float *data, int N)
{
    // ... 通用代码 ...

#if __CUDA_ARCH__ >= 800
    // 计算能力 8.0+ 的优化路径：使用异步拷贝
    // 这些操作在 CC < 8.0 的设备上不被支持
    asm volatile("cp.async.ca.shared.global ...");
#elif __CUDA_ARCH__ >= 700
    // 计算能力 7.0+ 的优化路径：使用 warp shuffle
    float val = __shfl_down_sync(0xffffffff, data[threadIdx.x], 1);
#else
    // 计算能力 < 7.0 的兼容路径：使用共享内存
    __shared__ float sdata[256];
    sdata[threadIdx.x] = data[threadIdx.x];
    __syncthreads();
    // ...
#endif
}
```

关键要点：
- `__CUDA_ARCH__` 宏<strong>仅在设备代码中定义</strong>。在主机代码中引用它是未定义行为。
- 当使用 `-arch=compute_80` 编译时，`__CUDA_ARCH__` 等于 `800`（= 8.0 × 100）。
- 当使用 `-arch=compute_35` 编译时，`__CUDA_ARCH__` 等于 `350`。
- 条件检查必须使用 `#if`（预处理指令），而非 `if`（运行时语句），因为不同架构支持的语法可能不同。
- 在 Fat Binary 编译下，每个 `-gencode` 选项都会导致源文件被重新编译一次（使用该选项对应的 `__CUDA_ARCH__` 值）。因此，当你指定了 3 个 `-gencode`，设备代码实际上被编译了 3 次，生成了 3 份不同架构的代码。

### 4.5.3 `-arch` 和 `-code` 的简写形式和组合规则

NVCC 用户手册中列出了 `-arch`、`-code` 和 `-gencode` 的各种简写形式。理解这些简写有助于你快速编写编译命令：

<strong>简写等价关系表</strong>：

| 简写命令 | 等价完整展开 | 最终嵌入内容 |
| :--- | :--- | :--- |
| `-arch=sm_50` | `-arch=compute_50 -code=compute_50,sm_50` | sm_50 cubin + PTX |
| `-arch=sm_70` | `-arch=compute_70 -code=compute_70,sm_70` | sm_70 cubin + PTX |
| `-arch=sm_80` | `-arch=compute_80 -code=compute_80,sm_80` | sm_80 cubin + PTX |
| `-arch=compute_80` | `-arch=compute_80`（只指定虚拟架构） | 无 cubin，需配合 `-code` |
| `-code=sm_80` | `-code=sm_80`（只指定真实架构） | sm_80 cubin，但无 PTX 后备 |

<strong>重要提示</strong>：

- `-arch=sm_XY` 比 `-arch=compute_XY -code=sm_XY` 多嵌入了 `compute_XY`（即 PTX 代码），因此它除了生成 cubin 外，还提供了 JIT 的 PTX 后备。
- 多个 `-gencode` 可以共存来实现多架构支持。`-arch` 和 `-code` 只能指定一次。

<strong>实际使用建议</strong>：

```bash
# 最简单：只支持当前架构（例如用于开发的机器）
nvcc file.cu -o file -arch=native
# native 自动检测当前 GPU 的架构

# 最佳实践：支持你关心的所有架构
nvcc file.cu -o file \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_90,code=compute_90
# 最后一行 code=compute_90 嵌入了 PTX 作为未来兼容后备
```

### 4.5.4 Volta 架构的特殊兼容性选项

Volta 架构引入了<strong>独立线程调度（Independent Thread Scheduling）</strong>，从根本上改变了 GPU 上 warp 内线程的调度方式。在 Volta 之前的架构中（Kepler、Maxwell、Pascal），一个 warp 中 32 个线程共享一个程序计数器（Program Counter, PC），这意味着它们必须以锁步（lockstep）方式沿相同执行路径前进。Volta 开始，每个线程拥有独立的 PC 和调用栈，允许 warp 内的线程真正独立发散和重汇聚。

对于依赖前代架构中 <strong>SIMT 调度</strong>特定行为的代码（特别是那些在 warp 内部不使用显式同步的协作操作），独立线程调度可能导致错误的执行结果。

为了帮助开发者迁移，Volta 开发者可以使用以下编译器选项组合，选择使用 Pascal 的线程调度方式：

```bash
-arch=compute_60 -code=sm_70
```

这个命令的含义是：
- `arch=compute_60`：使用 Pascal（compute_60）的调度语义和行为模型生成 PTX。
- `code=sm_70`：生成针对 Volta（sm_70）硬件的二进制代码。

这样可以在新的 Volta 硬件上运行，同时保持 Pascal 的 warp 同步行为。当然，这也意味着不能使用 Volta 独有的新特性（如独立线程调度带来的性能优化）。

### 4.5.5 Driver API 的兼容性处理

使用 <strong>CUDA Driver API</strong> 的应用程序与使用 Runtime API 的应用在处理兼容性时有不同的要求：

<strong>Runtime API 的自动处理</strong>：Fat Binary 机制完全透明——主机代码被 NVCC 自动修改后，在运行时自动选择最佳代码。开发者无需编写任何选择逻辑。

<strong>Driver API 需要手动处理</strong>：使用 Driver API 的应用必须：
1. 将不同架构的代码编译为单独的文件（例如 `kernel_sm70.cubin`、`kernel_sm80.cubin`、`kernel.ptx`）。
2. 在运行时，通过 `cuModuleLoad()` 或 `cuModuleLoadData()` 显式加载最合适的文件。
3. 自行编写兼容性检查逻辑（通过 `cuDeviceGetAttribute()` 查询设备计算能力）。

这就是为什么大多数应用使用 Runtime API——它大大简化了兼容性管理。

## 4.6 常见编译错误与排查

在实际开发中，NVCC 编译错误可能比标准 C++ 编译器更难以理解。本节整理了一些最常见的编译与运行时错误，帮助你在遇到问题时快速定位。

### 4.6.1 "No kernel image is available for execution on the device" 错误

这是 CUDA 程序最经典的运行时错误之一。完整错误信息通常为：

```text
no kernel image is available for execution on the device
```

<strong>原因</strong>：可执行文件中没有适配当前 GPU 计算能力的 cubin 二进制代码。

<strong>常见触发场景</strong>：
- 编译时只指定了一个 sm 目标（如 `-arch=sm_80`），但运行的 GPU 是 CC 7.0 的 V100。
- 只嵌入了 cubin 而没有嵌入 PTX（如 `-code=sm_80` 而非 `-code=compute_80,sm_80`）。

<strong>解决方案</strong>：
```bash
# 1. 确保 arch/code 覆盖目标 GPU 的计算能力
nvcc file.cu -o file -arch=sm_70  # 如果目标是 V100

# 2. 使用 Fat Binary 覆盖多个架构 + PTX 后备
nvcc file.cu -o file \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_90,code=compute_90

# 3. 使用 native 自动适配当前 GPU（仅用于开发机）
nvcc file.cu -o file -arch=native
```

### 4.6.2 "identifier not found in device code" 错误

```text
error: identifier "some_function" is undefined in device code
```

<strong>原因</strong>：在设备代码中调用了一个仅在主机端定义或链接的函数。

<strong>解决方案</strong>：
- 确认函数是否被正确声明为 `__device__` 或 `__host__ __device__`。
- 如果函数定义在另一个 `.cu` 文件中，需要启用 `-rdc=true`（Relocatable Device Code）。

### 4.6.3 链接阶段的 "undefined reference" 错误

```text
undefined reference to `cudaMalloc'
```

<strong>原因</strong>：链接器找不到 CUDA 运行时库。

<strong>解决方案</strong>：
```bash
# 使用 nvcc 做链接（推荐——自动处理 cudart 库路径）
nvcc file1.o file2.o -o myapp

# 如果使用 gcc 直接链接，需要手动添加 cudart 库路径
gcc file1.o file2.o -o myapp -lcudart -L/usr/local/cuda/lib64
```

### 4.6.4 "extern shared memory" 相关错误

```text
error: declaration is incompatible with previous "extern __shared__"
```

<strong>原因</strong>：在内核中声明了多个 `extern __shared__` 变量（每个内核只能有一个）。

<strong>解决方案</strong>：
```cuda
// 错误：两个 extern __shared__ 声明
extern __shared__ float array1[];
extern __shared__ int array2[];  // 不允许！

// 正确：只声明一个，手动管理分区
extern __shared__ char sMem[];
float *array1 = (float *)sMem;
int *array2 = (int *)(sMem + FLOAT_SIZE);
```

### 4.6.5 编译超时或内存不足

对于非常大的内核（特别是在 Fat Binary 编译中），编译可能消耗大量内存和时间：

```bash
# 减少 Fat Binary 架构数（编译时间与 -gencode 数量成正比）
# 增加编译超时时间
nvcc file.cu -o file --compiler-options="-Wno-deprecated" ...

# 分离编译大型项目（逐个文件编译）
nvcc -c large_kernel.cu -o large_kernel.o -rdc=true -arch=sm_80
```

### 4.6.6 调试编译错误的高级技巧

```bash
# 1. 保留中间文件，查看预处理输出
nvcc file.cu -keep -arch=sm_80
# 查看 file.cpp4.ii（预处理输出）

# 2. 增加编译器信息输出
nvcc file.cu --verbose -arch=sm_80

# 3. 仅做语法检查，不生成代码
nvcc file.cu -arch=sm_80 --dryrun

# 4. 仅编译设备代码（检查设备代码语法）
nvcc file.cu -ptx -arch=compute_80
```

## 4.8 C++ 兼容性与 64 位支持

### 4.8.1 C++ 兼容性

NVCC 编译器的前端按照 C++ 语法规则处理 CUDA 源文件。但主机代码和设备代码的 C++ 支持程度不同：

<strong>主机代码（Host Code）</strong>：
- <strong>完整支持 C++</strong>。主机代码最终由标准 C++ 编译器（如 gcc、clang、MSVC）编译，因此支持该编译器提供的完整 C++ 标准（取决于编译器版本和 `-std` 选项）。
- NVCC 支持通过 `-std=c++11`、`-std=c++14`、`-std=c++17` 等选项指定 C++ 标准版本。

<strong>设备代码（Device Code）</strong>：
- 仅<strong>完整支持 C++ 的一个子集</strong>。GPU 设备没有完整的 C++ 运行时库，因此很多 C++ 特性在设备代码中不可用或受限。
- <strong>不支持的特性包括</strong>：
  - C++ 标准库（`std::vector`、`std::string`、`std::map` 等）——GPU 设备上没有对应的运行时实现。
  - 异常处理（`try`/`catch`/`throw`）——GPU 硬件不能支持 C++ 异常。
  - 运行时类型识别（RTTI, `typeid`、`dynamic_cast`）——设备代码中不支持。
  - `new`/`delete` 操作符（某些架构的特定版本除外）。
  - 虚函数的多态调用（从 CC 3.5 开始有限支持）。
- <strong>支持的特性包括</strong>：
  - 模板（Templates）
  - Lambda 表达式（从 CC 5.0+，CUDA 8.0+）
  - `auto` 关键字
  - 操作符重载
  - 函数重载
  - 类和结构体
  - 静态成员变量（从 CC 2.0+）
  - `constexpr`

设备代码支持的 C++ 子集的完整列表在 CUDA Programming Guide 的"C++ Language Support"章节中详细描述。

### 4.8.2 64 位兼容性

NVCC 对 64 位和 32 位编译模式有明确的规则：

- <strong>64 位模式</strong>：64 位版本的 NVCC 默认在 64 位模式下编译设备代码（即指针为 64 位）。在 64 位模式下编译的设备代码<strong>仅支持</strong>与 64 位模式编译的主机代码一起使用。
- <strong>32 位模式</strong>：32 位版本的 NVCC 默认在 32 位模式下编译设备代码。在 32 位模式下编译的设备代码<strong>仅支持</strong>与 32 位模式编译的主机代码一起使用。

使用 `-m64` 和 `-m32` 选项可以覆盖默认行为：
- 32 位版本的 NVCC 可以使用 `-m64` 选项在 64 位模式下编译设备代码。
- 64 位版本的 NVCC 可以使用 `-m32` 选项在 32 位模式下编译设备代码。

> <strong>提示</strong>：在现代 CUDA 开发中，NVIDIA 已在 CUDA 11.0 开始逐步废弃 32 位主机应用支持，CUDA 12.x 已完全移除 32 位编译功能。所有开发都应在 64 位模式下进行。除非你在维护非常老旧的遗留系统，否则不需要关心 32 位模式。

### 4.8.3 独立编译与分离编译

CUDA 程序可以在多个 `.cu` 文件中组织代码，这涉及到<strong>独立编译（Separate Compilation）</strong>：

```bash
# 独立编译各文件为目标文件
nvcc -c kernel1.cu -o kernel1.o -arch=sm_80
nvcc -c kernel2.cu -o kernel2.o -arch=sm_80
nvcc -c main.cpp -o main.o

# 链接为可执行文件
nvcc kernel1.o kernel2.o main.o -o myapp
```

使用独立编译时，`-rdc=true`（Relocatable Device Code）选项可以让设备代码支持跨文件的函数调用和全局变量引用：

```bash
# 生成可重定位设备代码（允许跨文件设备端调用）
nvcc -c kernel.cu -o kernel.o -arch=sm_80 -rdc=true
nvcc -c util.cu -o util.o -arch=sm_80 -rdc=true

# 链接时需要额外的设备链接步骤
nvcc kernel.o util.o -o myapp -rdc=true
```

独立编译对大型项目特别重要——它允许增量编译、并行编译和代码模块化。

### 4.8.4 实用编译场景速查表

以下是根据常见需求整理的 NVCC 编译命令速查表：

<strong>场景 1：快速开发测试</strong>

```bash
nvcc file.cu -o file -arch=native -O0 -G -lineinfo
# -arch=native: 自动适配当前 GPU
# -O0: 无优化，编译快
# -G: 调试信息
# -lineinfo: 保留行号
```

<strong>场景 2：性能测试</strong>

```bash
nvcc file.cu -o file -arch=native -O3 --use_fast_math
# -O3: 最高优化
# --use_fast_math: 启用快速但精度略低的数学函数
```

<strong>场景 3：面向公众发布（最大化兼容性）</strong>

```bash
nvcc file.cu -o file -O2 \
    -gencode arch=compute_50,code=sm_50 \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_90,code=compute_90
# 最后一行嵌入 PTX 确保未来硬件兼容
```

<strong>场景 4：HPC 集群（已知硬件，极致性能）</strong>

```bash
nvcc file.cu -o file -arch=sm_80 -O3 --use_fast_math \
    --ftz=true --prec-div=false -maxrregcount=64
# 针对特定架构（A100）做极致优化
```

<strong>场景 5：生成独立的 PTX 文件（用于 JIT 框架）</strong>

```bash
nvcc kernel.cu -ptx -arch=compute_80 -o kernel.ptx
# 生成可被 Driver API 加载的 PTX 文件
```

<strong>场景 6：跨平台编译（在 x86 上为 ARM64 编译）</strong>

```bash
nvcc file.cu -o file --target-os-variant=Linux --target-arch=arm64 \
    --cross-compile --compiler-bindir=/usr/bin/aarch64-linux-gnu-g++
```

### 4.8.5 NVCC 优化级别与控制选项

NVCC 提供多级优化控制，从完全不优化到激进优化：

```bash
# -O0: 无优化（调试友好，但性能最差）
nvcc file.cu -O0 -G -o file_debug

# -O1: 基本优化
nvcc file.cu -O1 -o file

# -O2: 标准优化（默认，推荐的发布级别）
nvcc file.cu -O2 -o file

# -O3: 激进优化（包含循环展开、内联等，可能增加代码体积）
nvcc file.cu -O3 -o file
```

<strong>其他重要的代码生成选项</strong>：

| 选项 | 作用 | 说明 |
| :--- | :--- | :--- |
| `-G` | 生成设备调试信息 | 禁用优化，包含调试符号 |
| `-lineinfo` | 生成行号信息 | 保持优化但添加行号映射 |
| `--use_fast_math` | 快速数学 | 使用精度较低的 `__fdividef`、`__sinf` 等 |
| `--ftz=true` | 非规格化数刷新为零 (FTZ) | 提升浮点运算性能 |
| `--prec-div=false` | 低精度除法 | 使用倒数近似替代除法 |
| `--fmad=true` | 融合乘加 (FMA) | 默认启用，`a*b+c` 单条指令完成 |
| `-maxrregcount=N` | 限制每线程寄存器数 | 溢出到局部内存以提升占用率 |
| `--ptxas-options=-v` | 显示寄存器/共享内存使用 | 极其有用：`--ptxas-options=-v` |

<strong>查看编译器统计信息</strong>：

```bash
# 显示寄存器和共享内存使用量（性能优化的关键信息）
nvcc file.cu -o file --ptxas-options=-v
```

输出示例：

```text
ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z6KernelPfS_S_i' for 'sm_80'
ptxas info    : Function properties for _Z6KernelPfS_S_i
ptxas         :    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 32 registers, 4096 bytes smem, 352 bytes cmem[0]
```

关键指标解读：
- <strong>registers</strong>：每个线程使用的寄存器数。越少越好（允许更多 warp 驻留），但太少会导致溢出（spill）。
- <strong>smem</strong>：静态 + 动态共享内存每块使用量（字节）。
- <strong>spill stores/loads</strong>：寄存器溢出到局部内存的次数。0 是最理想的——所有变量都装在寄存器中。
- <strong>cmem[0]</strong>：常量内存使用量（字节）。用于存储内核参数等。

<strong>NVCC 编译阶段及对应工具链</strong>：

```
.cu 源文件
  │
  ├── [预处理]     cpp / cl.exe    → .cpp4.ii (预处理后的 C++ 文件)
  ├── [CUDA 前端]   cicc            → .ptx (PTX 汇编)
  ├── [PTX 汇编]    ptxas           → .cubin (设备二进制)
  ├── [主机编译]    gcc / cl.exe    → .o (主机目标文件)
  └── [Fatbinary]   fatbinary       → 将 .cubin 和/或 .ptx 嵌入 .o
```

了解这个工具链能帮助你更好地理解 NVCC 的输出和错误信息。例如，`ptxas` 的错误表明 PTX 汇编阶段出了问题（通常是代码使用了当前架构不支持的指令）；`cicc` 的错误表明 CUDA C++ 前端解析出了问题。

### 4.8.5 编译时间优化技巧

大型 CUDA 项目（特别是使用大量模板或自动生成代码的）可能遇到较长的编译时间。以下是一些优化技巧：

<strong>1. 使用 -dc 进行并行编译</strong>：

```bash
# 编译各 .cu 文件为独立目标文件（可并行！）
nvcc -dc kernel1.cu -o kernel1.o -arch=sm_80 &
nvcc -dc kernel2.cu -o kernel2.o -arch=sm_80 &
nvcc -dc kernel3.cu -o kernel3.o -arch=sm_80 &
wait

# 设备链接 + 主机链接
nvcc -dlink kernel1.o kernel2.o kernel3.o -o device_link.o
nvcc kernel1.o kernel2.o kernel3.o device_link.o main.o -o myapp
```

<strong>2. 使用 ccache 或 sccache</strong>：

```bash
# 安装 ccache
sudo apt install ccache  # Linux

# 设置 NVCC 使用 ccache
export NVCC="ccache nvcc"
nvcc file.cu -o file
```

<strong>3. 减少 Fat Binary 架构数</strong>：

开发和测试阶段可以只编译当前设备架构：

```bash
# 开发阶段：只编译当前架构
nvcc file.cu -o file -arch=native

# 发布阶段：编译所有目标架构（可能较慢）
nvcc file.cu -o file -gencode ... -gencode ... -gencode ...
```

<strong>4. 合理使用头文件</strong>：

CUDA 编译器需要为每个 `-gencode` 选项重新编译设备代码。将大型设备函数放在 `.cuh` 头文件中会导致它们在每个翻译单元中被重复编译。尽可能将设备代码放在 `.cu` 文件中，只将简洁的接口声明放在头文件中。

### 4.8.6 CUDA Driver API 编译与运行时加载

虽然大多数 CUDA 应用使用 Runtime API（`cuda*` 函数和自动 Fat Binary 管理），但了解 Driver API（`cu*` 函数）的编译模型也很重要。Driver API 提供了更精细的控制：

<strong>编译为独立的 cubin 文件</strong>：

```bash
# 编译设备代码为独立 cubin（不包含主机代码）
nvcc kernel.cu -cubin -arch=sm_80 -o kernel.cubin

# 编译设备代码为 PTX（用于 JIT）
nvcc kernel.cu -ptx -arch=compute_80 -o kernel.ptx

# 或使用 Fat Binary 但保存为独立文件
nvcc kernel.cu -fatbin -arch=sm_80 -o kernel.fatbin
```

<strong>在 Driver API 中加载 cubin</strong>：

```cuda
#include <cuda.h>  // Driver API 头文件

// 初始化 Driver API（必须！）
cuInit(0);

// 获取设备
CUdevice device;
cuDeviceGet(&device, 0);

// 创建上下文
CUcontext context;
cuCtxCreate(&context, 0, device);

// 从文件加载 cubin 或 PTX 模块
CUmodule module;
cuModuleLoad(&module, "kernel.cubin");
// 或加载 PTX：cuModuleLoad(&module, "kernel.ptx");

// 获取内核函数句柄
CUfunction kernel;
cuModuleGetFunction(&kernel, module, "_Z8myKernelPf");

// 设置参数并启动
void *args[] = { &d_data, &N };
cuLaunchKernel(kernel,
               gridDimX, gridDimY, 1,    // grid 维度
               blockDimX, 1, 1,           // block 维度
               0,                          // 共享内存
               NULL,                       // 流
               args, NULL);                // 参数

// 清理
cuCtxDestroy(context);
```

Driver API 的优势是精确控制——你可以：
- 在运行时选择加载哪个 .cubin 文件（基于设备检测）。
- 看 JIT 编译确切的 PTX 代码。
- 管理多个 CUDA 上下文。
- 实现自定义的内核缓存和加载策略。

> <strong>提示</strong>：Runtime API 在内部就是通过 Driver API 实现的。使用 Runtime API 时，NVCC 自动生成的代码本质上就是在执行类似上面 `cuModuleLoad` → `cuModuleGetFunction` → `cuLaunchKernel` 的操作。

## 4.9 构建系统集成与实战

在真实项目中，CUDA 代码很少单独使用 `nvcc` 命令行编译，而是集成在构建系统中。本节介绍如何将 CUDA 编译集成到 CMake 和 Makefile 中。

### 4.9.1 CMake 项目配置

CMake 3.8+ 提供了对 CUDA 作为一等语言（first-class language）的支持：

```cmake
cmake_minimum_required(VERSION 3.18)
project(MyCUDAProject LANGUAGES CXX CUDA)

# 方式1：使用 enable_language
enable_language(CUDA)

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
# 或者
set_target_properties(my_target PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86"
)

# 方式2：使用 FindCUDAToolkit
find_package(CUDAToolkit REQUIRED)

# 设置 CUDA 编译选项
target_compile_options(my_target PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_80>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_80,code=sm_80>
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_86,code=sm_86>
)

# 添加 NVCC 特定的标志
set_target_properties(my_target PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON      # 等价于 -rdc=true
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
```

<strong>完整的 CMakeLists.txt 示例</strong>：

```cmake
cmake_minimum_required(VERSION 3.18)
project(GPUProgrammingCourse LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 设置目标架构
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89")

# 可执行文件
add_executable(vector_add vector_add.cu)

# 设置属性
set_target_properties(vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# 链接 CUDA 运行时
target_link_libraries(vector_add CUDA::cudart)

# 可选：添加优化标志
target_compile_options(vector_add PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
```

### 4.9.2 Makefile 示例

```makefile
# Makefile for CUDA project
NVCC = nvcc
ARCH_FLAGS = -arch=sm_60 -arch=sm_70 -arch=sm_80

# 或者使用 gencode 多架构支持
FATBIN_FLAGS = -gencode arch=compute_60,code=sm_60 \
               -gencode arch=compute_70,code=sm_70 \
               -gencode arch=compute_80,code=compute_80,sm_80

CFLAGS = -O2 -lineinfo $(ARCH_FLAGS)
LDFLAGS = -lcudart

.PHONY: all clean

all: vector_add device_query

vector_add: vector_add.cu
	$(NVCC) $(CFLAGS) $< -o $@

device_query: device_query.cu
	$(NVCC) $(CFLAGS) $< -o $@

# PTX 目标：生成 PTX 文件
%.ptx: %.cu
	$(NVCC) -ptx -arch=compute_80 $< -o $@

clean:
	rm -f vector_add device_query *.o *.ptx *.cubin
```

### 4.9.3 CUDA 相关环境变量

CUDA Toolkit 在运行时受多种环境变量的影响。以下是开发中常用的：

| 环境变量 | 作用 | 示例 |
| :--- | :--- | :--- |
| `CUDA_VISIBLE_DEVICES` | 限制程序可见的 GPU 列表 | `CUDA_VISIBLE_DEVICES=0,2` |
| `CUDA_LAUNCH_BLOCKING` | 设为 1 使内核启动同步（调试用） | `CUDA_LAUNCH_BLOCKING=1` |
| `CUDA_CACHE_DISABLE` | 设为 1 禁用 JIT 缓存 | `CUDA_CACHE_DISABLE=1` |
| `CUDA_CACHE_PATH` | JIT 缓存目录 | `CUDA_CACHE_PATH=/tmp/cuda_cache` |
| `CUDA_DEVICE_MAX_CONNECTIONS` | 每个设备的 CUDA 流多路复用能力 | 默认 8 |
| `CUDA_MANAGED_FORCE_DEVICE_ALLOC` | 强制统一内存分配在设备端 | |

<strong>在开发中常用环境变量组合</strong>：

```bash
# 开发/调试：同步内核启动，方便定位崩溃点
CUDA_LAUNCH_BLOCKING=1 ./my_app

# 测试：仅使用特定 GPU
CUDA_VISIBLE_DEVICES=1 ./my_app

# 性能测试：禁用 JIT 缓存，确保测量首次加载时间
CUDA_CACHE_DISABLE=1 ./my_app
```

### 4.9.4 多架构部署策略小结

根据你的目标用户群体，推荐以下几种 Fat Binary 策略：

| 策略 | NVCC 命令 | 适用场景 |
| :--- | :--- | :--- |
| <strong>最小体积</strong> | `-arch=sm_70` | 内部分发，已知硬件 |
| <strong>广泛兼容</strong> | 多 `-gencode` 含 PTX 后备 | 面向公众的桌面应用 |
| <strong>极致性能</strong> | 每架构单独编译分发 | 超算/HPC 环境 |
| <strong>动态选择</strong> | Driver API 运行时加载 | 框架类库 |


## 4.10 动手体验：编译和观察 Fat Binary

在本节中，我们将通过多个实际动手实验来直观感受 NVCC 编译模型的每个方面。

### 4.10.1 实验环境准备

首先，准备一个完整的 CUDA 程序 `device_query.cu`。这个程序将：
1. 查询并打印当前 GPU 的计算能力和属性。
2. 展示 `__CUDA_ARCH__` 宏在不同编译选项下的值。
3. 演示条件编译的不同代码路径选择。

本节所用的完整可编译代码参见 `code/chapter4/device_query.cu`。

### 4.10.2 实验一：单架构编译

<strong>实验一：针对特定架构编译</strong>

```bash
# 针对 sm_70 (Volta) 编译
nvcc device_query.cu -o device_query_sm70 -arch=sm_70
./device_query_sm70

# 针对 sm_80 (Ampere) 编译
nvcc device_query.cu -o device_query_sm80 -arch=sm_80
./device_query_sm80
```

预期观察：
- 在 RTX 3060（CC 8.6）上运行 `device_query_sm70`：程序正常运行（二进制兼容：8.6 >= 7.0 且同一主版本？不！8 ≠ 7！但是注意：这里用的是 `-arch=sm_70`，它等价于 `-arch=compute_70 -code=compute_70,sm_70`，所以<strong>同时嵌入了 PTX</strong>。所以在 CC 8.6 上实际上是通过 JIT 编译 PTX 来运行的）。
- 在 V100（CC 7.0）上运行 `device_query_sm80`：如果只用了 `-arch=sm_80`，程序<strong>可能无法运行</strong>（8.0 cubin 在 7.0 上不兼容，且如果编译时确实只有 cubin 而没有 PTX 后缀，则 JIT 后备也没有）。
- `__CUDA_ARCH__` 的值在 PTX 中为 700 或 800。

让我们验证一下 cubin 的存在性：

```bash
# 查看可执行文件中的架构信息
cuobjdump device_query_sm80 | head -20
```

### 4.10.3 实验二：Fat Binary 编译与检查

<strong>实验二：生成 Fat Binary</strong>

```bash
# Fat Binary: 同时支持多个架构
nvcc device_query.cu -o device_query_fat \
    -gencode arch=compute_60,code=sm_60 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code="compute_80,sm_80"
```

这个命令生成了一个包含三个架构的 Fat Binary：
- 直接可运行在 CC 6.0-6.2 的设备上（sm_60 cubin）
- 直接可运行在 CC 7.0-7.5 的设备上（sm_70 cubin）
- 直接可运行在 CC 8.0-8.9 的设备上（sm_80 cubin）
- CC 9.0+ 设备则通过嵌入的 compute_80 PTX 进行 JIT 编译

<strong>实验三：使用 cuobjdump 检查 Fat Binary</strong>

CUDA Toolkit 提供了 `cuobjdump` 工具来检查 cubin 和 Fat Binary 文件的内容：

```bash
# 查看可执行文件中包含的架构信息
cuobjdump device_query_fat

# 列出所有的 cubin 和 PTX 嵌入
cuobjdump -list-ptx device_query_fat
cuobjdump -list-sass device_query_fat
```

典型输出示例：

```text
Fatbin elf code:
================
arch = sm_60
code version = [6,1]
host = linux
compile_size = 64bit

arch = sm_70
code version = [7,1]
host = linux
compile_size = 64bit

arch = sm_80
code version = [8,1]
host = linux
compile_size = 64bit
```

这表明生成的 Fat Binary 确实包含了 sm_60、sm_70 和 sm_80 三个架构的二进制代码。

更详细地，使用 `-sass` 选项可以反汇编出 SASS（原生机器码）指令：

```bash
# 反汇编 sm_80 部分的 SASS
cuobjdump -sass device_query_fat
```

### 4.10.4 实验三：PTX 与 cubin 分步生成

<strong>实验四：PTX 代码的分步生成</strong>

```bash
# Step 1: 只生成 PTX 代码（不生成二进制，不链接）
nvcc device_query.cu -ptx -arch=compute_80 -o device_query.ptx

# Step 2: 查看生成的 PTX 汇编代码
head -50 device_query.ptx
```

典型的 PTX 输出片段：

```ptx
//
// Generated by NVIDIA NVVM Compiler
//
// Compiler Build ID: CL-...
// Cuda compilation tools, release 11.5, V11.5....
//

.version 7.5
.target sm_80
.address_size 64

	// .globl	_Z15archCheckKernelv

.visible .entry _Z15archCheckKernelv()
{
	.reg .b32 	%r<2>;
	.reg .b64 	%rd<3>;


	ld.param.u64 	%rd1, [__unnamed_1];
	mov.u32 	%r1, 800;
	st.global.u32 	[%rd1], %r1;
	ret;
}
```

在 PTX 代码中，你可以看到：
- `.version 7.5`：PTX ISA 版本
- `.target sm_80`：目标架构
- `_Z15archCheckKernelv`：函数名修饰（Name Mangling）后的符号
- `ld.param.u64`、`mov.u32`、`st.global.u32`：实际的 PTX 指令

<strong>Step 3: 从 PTX 生成 cubin</strong>

```bash
# 将 PTX 编译为 cubin
nvcc device_query.ptx -cubin -o device_query.cubin -arch=sm_80

# 查看 cubin 文件
cuobjdump -sass device_query.cubin
```

### 4.10.5 实验四：保留所有中间文件

```bash
# 使用 -keep 选项保留所有编译中间文件
nvcc device_query.cu -keep -arch=sm_80 -o device_query

# 查看生成的所有中间文件
ls -la device_query*
# device_query.cpp4.ii  ← 预处理后的输出
# device_query.ptx       ← PTX 汇编代码
# device_query.o         ← 主机端目标文件
# ... (可能还有更多)
```

### 4.10.6 实验五：编译器选项演练

以下是本实验中用到的所有相关 NVCC 选项的总结：

<div align="center">

| 选项 | 含义 | 示例 |
| :--- | :--- | :--- |
| `-arch=compute_XY` | 指定虚拟架构（PTX 目标功能级别） | `-arch=compute_80` |
| `-code=sm_XY` | 指定真实架构（cubin 二进制目标） | `-code=sm_80` |
| `-code=compute_XY,sm_XY` | 同时嵌入 PTX 和 cubin | `-code=compute_80,sm_80` |
| `-arch=sm_XY` | 简写：compute_XY + compute_XY,sm_XY | `-arch=sm_80` |
| `-gencode arch=...,code=...` | 指定多组架构（Fat Binary） | 见实验二 |
| `-ptx` | 仅生成 PTX 代码 | `nvcc file.cu -ptx` |
| `-cubin` | 仅生成 cubin 二进制 | `nvcc file.cu -cubin` |
| `-c` | 只编译不链接 | `nvcc -c file.cu` |
| `-keep` | 保留所有中间文件 | `nvcc -keep file.cu` |
| `-m64` / `-m32` | 指定 64 位或 32 位模式 | `-m64` |
| `-O<n>` | 优化级别（0/1/2/3） | `-O2` |
| `-G` | 生成设备调试信息 | `-G` |
| `-lineinfo` | 生成行号信息 | `-lineinfo` |
| `-std=<standard>` | 指定 C++ 标准 | `-std=c++14` |
| `-rdc=true` | 启用可重定位设备代码 | `-rdc=true` |
| `-arch=native` | 自动检测当前 GPU 架构 | `-arch=native` |

</div>

## 4.11 本章小结

在本章中，我们深入探索了 CUDA 程序的编译世界。我们的旅程从 NVCC 编译器驱动程序的角色开始，一直延伸到如何构建跨多代 GPU 架构都兼容的 Fat Binary：

- <strong>NVCC 是什么？</strong> NVCC 是一个编译器驱动程序，简化了 C++ 或 PTX 代码的编译流程。它将 CUDA 源文件中的设备代码和主机代码分离，分别交由 GPU 专用编译器和标准 C++ 编译器处理。对程序员来说，整个过程通过一条简单的 `nvcc` 命令完成。

- <strong>离线编译的四个阶段？</strong> （1）<strong>代码分离</strong>：识别并提取设备代码。（2）<strong>设备代码编译</strong>：将设备代码编译为 PTX 和/或 cubin。（3）<strong>主机代码转换</strong>：将 `<<<...>>>` 语法替换为 CUDA Runtime 函数调用。（4）<strong>链接</strong>：将主机目标代码和设备二进制代码链接为可执行文件。

- <strong>JIT 编译的作用和缓存？</strong> 即时编译在运行时将 PTX 代码翻译为目标硬件的二进制。它允许应用在未来硬件上运行并从新驱动程序的编译器优化中受益。计算缓存（存储在磁盘上）避免了重复编译的开销，并在驱动升级时自动刷新。

- <strong>二进制兼容性规则？</strong> 为计算能力 X.y 生成的 cubin 仅能在计算能力 X.z（其中 z >= y）的设备上运行。兼容性仅限于同一个主版本号内。不支持跨主版本兼容。

- <strong>PTX 兼容性规则？</strong> 为较低计算能力生成的 PTX 代码可以 JIT 编译到任意更高计算能力的设备上。这是应用"在未来硬件上也能运行"的基础机制。但旧版本 PTX 无法利用新硬件的全部特性（如 Tensor Core），因此最佳实践是嵌入尽可能新版本的 PTX。

- <strong>Fat Binary 如何保证多架构兼容？</strong> 通过 `-gencode` 选项在同一个可执行文件中嵌入多个架构的 cubin 和 PTX 代码。运行时自动选择最佳匹配：精确匹配 cubin > 二进制兼容 cubin > JIT 编译 PTX。`__CUDA_ARCH__` 宏允许在源码中为不同架构编写条件编译的代码路径。

- <strong>`-arch` 和 `-code` 的关系？</strong> `-arch=compute_80` 指定 PTX 功能级别。`-code=sm_80` 指定 cubin 目标。简写 `-arch=sm_80` 等价于 `-arch=compute_80 -code=compute_80,sm_80`，同时嵌入了 cubin 和 PTX。

- <strong>C++ 兼容性和 64 位支持？</strong> 主机代码完全支持 C++。设备代码仅支持 C++ 的一个子集（不支持标准 C++ 异常和 RTTI，仅支持有限的 STL 子集）。现代 CUDA 开发均应在 64 位模式下进行（32 位主机应用支持已在 CUDA 11.0 开始逐步废弃，CUDA 12.x 已完全移除 32 位支持）。

通过本章的学习，我们建立了对 CUDA 编译模型的系统理解。在下一章中，我们将进入 CUDA 运行时（CUDA Runtime）的世界，学习设备内存管理——如何在代码中分配和释放 GPU 内存，以及如何在主机和设备之间传输数据。这是编写任何 CUDA 程序都必不可少的基础技能！

## 习题

> <strong>提示</strong>：以下的部分习题没有标准答案，重点在于培养学习者对 CUDA 编译模型和架构兼容性批判性的深入思考和动手实践能力。

1. <strong>NVCC 工作流分析</strong>：
   a. 请简述 NVCC 将 `.cu` 源文件编译为可执行文件的基本工作流程（至少包含四个主要阶段）。
   b. NVCC 如何处理主机代码中的 `<<<...>>>` 语法？请解释转换前后发生了什么。
   c. 在什么情况下，应用程序加载 PTX 代码而非直接使用编译好的 cubin？请列出至少三种场景。

2. <strong>即时编译（JIT）分析</strong>：
   a. JIT 编译的代价和优势分别是什么？
   b. 计算缓存（compute cache）是什么？它存储在磁盘的什么位置？在什么情况下会失效？
   c. 如果用户反馈你的 CUDA 程序首次运行很慢、后续运行很快，最可能的原因是什么？如何验证你的假设？

3. <strong>二进制兼容性推理</strong>：
   a. 为计算能力 6.0 编译的 cubin 能否在计算能力 7.5 的设备上运行？为什么？
   b. 为计算能力 7.5 编译的 cubin 能否在计算能力 6.1 的设备上运行？为什么？
   c. 为计算能力 5.0 编译的 cubin 能否在计算能力 5.3 的设备上运行？为什么？
   d. 如果想支持从 CC 5.0 到 CC 8.9 的所有设备，至少需要几个不同的 cubin？请列出你的 cubin 集合并说明理由。

4. <strong>PTX 兼容性推理</strong>：
   a. 为计算能力 5.0 生成的 PTX 能 JIT 编译到计算能力 8.0 的设备上运行吗？这个方案有什么潜在的性能问题？
   b. 假设你的应用只包含 sm_80 的 cubin，用户能在 RTX 2060（CC 7.5）上运行吗？你需要在编译选项中添加什么来支持这个设备？
   c. "PTX 的向前兼容是跨主版本的，而 cubin 的二进制兼容不是。"这句话对吗？请用具体例子说明。
   d. 如果为计算能力 6.0 生成的 PTX JIT 编译到计算能力 9.0，能使用 Hopper 的 TMA（Tensor Memory Accelerator）特性吗？为什么？

5. <strong>Fat Binary 设计与分析</strong>：
   你正在发布一个面向广泛用户群的 CUDA 应用。你的用户可能使用从 Maxwell（CC 5.x）到 Hopper（CC 9.0）的各种 GPU。
   a. 请写出一条 `nvcc` 命令，生成支持所有这五代架构的 Fat Binary，同时确保 Hopper 之后的未来架构也能通过 JIT 运行。
   b. 如果你的代码中使用了一个仅在 CC 8.0+ 上可用的 `cp.async` 操作，如何使用 `__CUDA_ARCH__` 宏来保证代码在旧设备上也能编译通过（使用替代实现）？
   c. 只使用 `-arch=sm_50` 和同时使用 `-gencode arch=compute_50,code=sm_50 -gencode arch=compute_80,code=compute_80` 有什么区别？在 V100（CC 7.0）和 A100（CC 8.0）上这两种方案的行为分别是什么？

6. <strong>动手实践题</strong>：
   > <strong>提示</strong>：这是一道动手实践题，建议实际编写代码和编译。

   基于 `code/chapter4/device_query.cu` 程序：
   a. 使用 `-arch=sm_60`、`-arch=sm_70`、`-arch=sm_80` 分别编译程序，使用 `cuobjdump` 检查各自生成的架构信息有何不同。说明每个可执行文件能在哪些 CC 的设备上运行。
   b. 编译一个 Fat Binary，包含 sm_60、sm_70 和 sm_80 三个架构的代码，使用 `cuobjdump` 确认确实列出了三个架构。
   c. 使用 `-keep` 选项保留中间文件，查看 `.ptx` 文件，找出 `__CUDA_ARCH__` 宏在 PTX 中的体现。
   d. （选做）在不同 CC 的设备上（如果有条件）运行你编译的 Fat Binary，观察内核中打印的 `__CUDA_ARCH__` 值是否符合预期。

## 参考文献

[1] NVIDIA Corporation. CUDA C++ Programming Guide (Version 11.5.1)[Z]. https://docs.nvidia.com/cuda/archive/11.5.1/cuda-c-programming-guide/index.html

[2] NVIDIA Corporation. CUDA C++ Programming Guide (Version 13.0)[Z]. https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-programming-guide/index.html

[3] NVIDIA Corporation. NVCC User Manual[Z]. https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/

[4] NVIDIA Corporation. PTX ISA Reference Guide[Z]. https://docs.nvidia.com/cuda/parallel-thread-execution/

[5] NVIDIA Corporation. CUDA Binary Utilities (cuobjdump)[Z]. https://docs.nvidia.com/cuda/cuda-binary-utilities/

---

## 讨论与交流

本章学习过程中遇到问题?想与其他学习者交流心得?

**前往 GitHub Discussions 讨论区:**
- [习题讨论与问答](https://github.com/open-rdma/gpu-programming-guide/discussions)
- 在这里你可以:
  - 提问习题相关问题
  - 分享你的解题思路
  - 与其他学习者交流经验
  - 获得社区的帮助和反馈

**提示:** 每个页面底部也有评论区,可以直接在页面内讨论!

---
