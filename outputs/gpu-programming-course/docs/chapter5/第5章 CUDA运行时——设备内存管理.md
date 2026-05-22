# 第5章 CUDA运行时——设备内存管理

欢迎来到 CUDA 设备内存管理的世界！在第 4 章中，我们学习了如何将 CUDA 代码编译为 GPU 可执行的指令。现在，是时候动手编写一个完整的 CUDA 应用程序了。在本章中，我们将深入学习 <strong>CUDA 运行时（CUDA Runtime）</strong> 的核心功能——设备内存管理。任何 CUDA 程序都离不开一个基本模式：将数据从主机内存传输到设备内存、在设备上执行计算、再将结果传回主机。这个看似简单的流程背后，涉及内存分配、数据拷贝、地址空间管理、对齐要求、错误检查等多个关键概念。让我们从理解 CUDA 运行时开始，逐步掌握这一基础技能！

## 5.1 CUDA 运行时概述

### 5.1.1 什么是 CUDA 运行时？

<strong>CUDA 运行时（CUDA Runtime）</strong> 实现在 `cudart` 库中，是 CUDA 编程中最常用的 API 层。它为开发者提供了管理设备内存、数据传输、内核启动、设备管理等核心功能的高级抽象。

应用程序可以通过两种方式链接 CUDA 运行时：

<div align="center">

| 链接方式 | Linux 库文件 | Windows 库文件 | 特点 |
| :--- | :--- | :--- | :--- |
| <strong>静态链接</strong> | `libcudart_static.a` | `cudart_static.lib` | 运行时代码嵌入可执行文件，无需附带 DLL |
| <strong>动态链接</strong> | `libcudart.so` | `cudart.dll` | 可执行文件更小，运行时需要 DLL/SO 可用 |

</div>

动态链接时需要将 `cudart.dll` 和/或 `cudart.so` 包含在应用程序安装包中。一个重要的安全规则是：<strong>只有在链接到同一个 CUDA 运行时实例的组件之间，传递 CUDA 运行时符号的地址才是安全的</strong>。

编译时的链接方式选择：

```bash
# 默认：静态链接 libcudart
nvcc program.cu -o program

# 显式指定动态链接
nvcc program.cu -o program -lcudart

# 对于 CMake 项目
find_package(CUDAToolkit REQUIRED)
target_link_libraries(my_target CUDA::cudart)
```

### 5.1.2 API 命名约定

所有 CUDA 运行时入口函数都以 `cuda` 为前缀。这是一个重要的识别标志——在阅读或搜索 CUDA 代码时，看到 `cuda` 前缀就知道这是运行时 API 调用：

```
cudaMalloc()            — 设备内存分配
cudaFree()              — 设备内存释放
cudaMemcpy()            — 内存数据传输
cudaGetDeviceCount()    — 获取系统中 CUDA 设备数量
cudaGetDeviceProperties() — 获取设备属性
cudaSetDevice()         — 设置当前设备
cudaDeviceSynchronize() — 同步设备
```

这种命名一致性使得 CUDA Runtime API 非常容易识别和记忆。

### 5.1.3 CUDA Runtime 与 Driver API 的关系

CUDA 编程有两个主要的 API 层：

- <strong>CUDA Runtime API</strong>（`cuda*` 前缀）：高级 API，封装了设备初始化、上下文管理、模块加载等底层细节。大多数 CUDA 应用使用 Runtime API。
- <strong>CUDA Driver API</strong>（`cu*` 前缀）：低级 API，提供对 GPU 的精细控制。需要显式初始化（`cuInit()`），管理上下文（`cuCtxCreate()`），加载模块（`cuModuleLoad()`）等。

Runtime API 建立在 Driver API 之上——实际上，Runtime API 在内部会自动处理所有 Driver API 的繁琐细节，对开发者完全透明。除非你需要 Driver API 特有的功能（如运行时 PTX 编译、上下文迁移等），否则 Runtime API 是更好的选择。

## 5.2 运行时初始化

### 5.2.1 隐式初始化机制

CUDA 运行时的一个关键设计决策是：<strong>没有显式的初始化函数</strong>。运行时在<strong>第一次调用运行时函数</strong>时自动初始化（更具体地说，是除错误处理（Error Handling）和版本管理（Version Management）部分之外的任何运行时函数）。

这意味着：
- 你的第一个 CUDA 运行时调用（如 `cudaMalloc`、`cudaGetDeviceCount` 等）会自动触发整个运行时的初始化。
- 第一个调用的执行时间会比后续调用显著更长，因为它包含了初始化开销。
- 在测量运行时函数调用的计时时，需要将初始化的因素考虑在内。
- 从第一个调用返回的错误码需要被理解为其包含了初始化可能的失败。

> <strong>注意</strong>：CUDA 接口使用在主机程序启动时初始化、在主机程序终止时销毁的全局状态。CUDA 运行时和驱动程序无法检测此状态是否无效，因此在程序启动或终止期间（包括在 `main()` 执行完毕后，如全局对象的析构函数中）使用这些接口（隐式或显式）将导致<strong>未定义行为（undefined behavior）</strong>。

### 5.2.2 CUDA 主上下文（Primary Context）

初始化时，运行时为系统中的<strong>每个设备</strong>创建一个 CUDA 上下文（Context）。这个上下文被称为该设备的<strong>主上下文（Primary Context）</strong>。

主上下文的关键特性：
- 在每个设备上，在第一个<strong>需要该设备上活跃上下文</strong>的运行时函数调用时初始化。
- 应用程序的<strong>所有主机线程共享</strong>这些主上下文。
- 作为上下文创建的一部分，设备代码在必要时被 JIT 编译（见第 4 章）并加载到设备内存中。
- 这一切都是<strong>透明地</strong>发生的——开发者无需手动创建上下文。

如果需要通过 Driver API 访问主上下文（例如进行 Runtime/Driver 混合编程），可以通过 Driver API 的 `cuCtxGetCurrent()` 或 `cuDevicePrimaryCtxRetain()` 获取主上下文的句柄。

<div align="center">
  <img src="../images/chapter5-figures/heterogeneous-programming.png" alt="异构编程模型" width="85%"/>
  <p>图 5.1 异构编程模型：主机（CPU）和设备（GPU）各自拥有独立的内存空间，CUDA 运行时提供在两者之间传输数据的 API</p>
</div>

### 5.2.3 设备重置（cudaDeviceReset）

当一个主机线程调用 `cudaDeviceReset()` 时：
- 这会销毁该主机线程<strong>当前操作的设备</strong>（即 `cudaSetDevice()` 选择的设备）的主上下文。
- 任何主机线程对该设备的下一次运行时函数调用，都会触发创建<strong>新的主上下文</strong>。
- 在重置前分配的所有设备内存和创建的资源（流、事件等）都会失效。

```cuda
// 一个使用 cudaDeviceReset 的场景（通常用于测试）
cudaSetDevice(0);
float *d_data;
cudaMalloc(&d_data, 1024);
// ... 使用 d_data ...

cudaDeviceReset();  // 重置设备 0 的上下文，d_data 失效
// 下一个 CUDA 调用将创建新上下文
```

> <strong>注意</strong>：`cudaDeviceReset()` 是一个比较"重"的操作，通常不在生产代码中使用，更常用于测试框架和调试场景。

### 5.2.4 CUDA 初始化的实际表现

让我们通过一个简单的程序观察隐式初始化的实际表现：

```cuda
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

int main() {
    clock_t start, end;

    // 第一个 CUDA 调用——触发初始化
    start = clock();
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    end = clock();
    printf("First call (with init):  %.3f ms\n",
           1000.0 * (end - start) / CLOCKS_PER_SEC);

    // 第二个 CUDA 调用——无需初始化
    start = clock();
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    end = clock();
    printf("Second call (no init):  %.3f ms\n",
           1000.0 * (end - start) / CLOCKS_PER_SEC);

    return 0;
}
```

典型输出：
```text
First call (with init):  125.430 ms    ← 包含初始化
Second call (no init):   0.015 ms      ← 仅函数调用本身
```

第一次调用的时间中，绝大部分是 CUDA 运行时的初始化（包括驱动通信、上下文创建、可能的 JIT 编译等）。

## 5.3 设备内存基础

### 5.3.1 线性内存与 CUDA 数组

CUDA 编程模型假设系统由主机和设备组成，各自拥有独立的内存空间。内核在设备内存上进行操作，因此运行时提供了分配、释放设备内存以及在主机与设备间传输数据的函数。

设备内存可以以两种形式存在：

1. <strong>线性内存（Linear Memory）</strong>：分配在统一的设备地址空间中。不同分配之间的指针可以互相引用（例如构建二叉树或链表）。这是 CUDA 程序中最常使用的内存形式。

2. <strong>CUDA 数组（CUDA Arrays）</strong>：不透明（opaque）的内存布局，针对纹理获取（texture fetching）进行了优化。CUDA 数组对程序员隐藏了具体的内存布局，更适合图像处理和纹理采样等场景。

本章重点介绍线性内存的完整管理。

### 5.3.2 设备地址空间

线性内存在统一的地址空间中分配。地址空间的大小取决于主机系统（CPU 架构）和 GPU 的计算能力：

<div align="center">
  <p>表 5.1 线性内存地址空间</p>

| | x86_64 (AMD64) | POWER (ppc64le) | ARM64 |
| :--- | :---: | :---: | :---: |
| CC 5.3 (Maxwell) 及以下 | 40 bit | 40 bit | 40 bit |
| CC 6.0 (Pascal) 及以上 | 最高 47 bit | 最高 49 bit | 最高 48 bit |

</div>

> <strong>注意</strong>：在计算能力 5.3 (Maxwell) 及更早的设备上，CUDA 驱动程序会创建一个未提交的 40 位虚拟地址预留，以确保内存分配（指针）落在支持的范围内。此预留显示为保留虚拟内存，但实际在程序分配内存之前不占用任何物理内存。

## 5.4 基本内存操作

### 5.4.1 cudaMalloc —— 设备内存分配

线性内存使用 `cudaMalloc()` 进行分配。函数签名如下：

```cuda
cudaError_t cudaMalloc(void **devPtr, size_t size);
```

<strong>参数说明</strong>：
- `devPtr`：<strong>输出参数</strong>，指向一个指针。函数将分配的设备内存地址写入这个指针（因此参数类型是 `void**`）。
- `size`：要分配的字节数。
- 返回值：`cudaError_t` 类型，`cudaSuccess` 表示成功。

<strong>与标准 C 的 `malloc()` 的对比</strong>：

```c
// C 标准库
void *ptr = malloc(size);

// CUDA：注意取地址的间接层
float *d_A;
cudaMalloc((void**)&d_A, N * sizeof(float));
```

<strong>关键特性</strong>：
- `cudaMalloc()` 返回的指针始终对齐到至少 256 字节。这确保了全局内存对齐要求得到满足。
- 设备指针 `d_A` <strong>只能在设备端代码中解引用（dereference）</strong>。在主机代码中解引用设备指针会导致段错误（segmentation fault）或未定义行为。
- 在主机代码中，设备指针只是一个"不透明句柄"——你可以用 `cudaMemcpy()` 将其作为源或目标地址，但不能直接用 `*d_A` 来读取其内容。

### 5.4.2 cudaFree —— 设备内存释放

已分配的设备内存通过 `cudaFree()` 释放：

```cuda
cudaError_t cudaFree(void *devPtr);
```

- `cudaFree(NULL)` 是安全的（不会产生错误，与 `free(NULL)` 一致）。
- 释放后不应再访问该内存，且不能重复释放（double free 行为未定义）。
- 如果 `devPtr` 指向的内存不是由 `cudaMalloc()` 等设备内存分配函数分配的，行为未定义。

### 5.4.3 cudaMemcpy —— 内存数据传输

主机内存与设备内存之间的数据传输使用 `cudaMemcpy()` 完成：

```cuda
cudaError_t cudaMemcpy(void *dst, const void *src,
                       size_t count, cudaMemcpyKind kind);
```

<strong>参数说明</strong>：
- `dst`：目标内存地址。
- `src`：源内存地址。
- `count`：要拷贝的字节数。
- `kind`：拷贝方向，指定为以下枚举值之一：
  - `cudaMemcpyHostToDevice` —— 主机 → 设备
  - `cudaMemcpyDeviceToHost` —— 设备 → 主机
  - `cudaMemcpyDeviceToDevice` —— 设备 → 设备
  - `cudaMemcpyHostToHost` —— 主机 → 主机

<strong>默认同步行为</strong>：

`cudaMemcpy()` 是<strong>同步函数</strong>——它在数据传输完成之后才返回。这意味着：
- 主机线程在 `cudaMemcpy()` 返回之前会阻塞。
- 返回后，可以安全地释放源缓冲区或使用目标缓冲区的数据。
- 对于主机到设备和设备到主机的拷贝，它在传输完成前会阻塞主机。

<strong>UVA 下的自动方向推断</strong>：

如果设备支持<strong>统一虚拟地址空间（UVA, Unified Virtual Address Space）</strong>（CC 2.0+），可以使用 `cudaMemcpyDefault` 作为 `kind` 参数：

```cuda
cudaMemcpy(dst, src, count, cudaMemcpyDefault);
```

运行时将从指针值（地址在 UVA 中的位置）自动推断拷贝方向，无需手动指定。这简化了代码，特别是在处理通用指针时。

### 5.4.4 完整的向量加法程序（五步模式）

让我们结合上述三个函数，实现第 2 章中引入的向量加法程序的完整版本。这个程序展示了 CUDA 内存管理的标准<strong>"五步模式"</strong>：

```
 1. cudaMalloc()    分配设备内存
        ↓
 2. cudaMemcpy(H2D)  拷贝输入数据到设备
        ↓
 3. kernel<<<>>>()   启动内核计算
        ↓
 4. cudaMemcpy(D2H)  拷贝结果回主机
        ↓
 5. cudaFree()      释放设备内存
```

完整的向量加法实现：

```cuda
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// 设备端内核：向量加法 C[i] = A[i] + B[i]
__global__ void VecAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main()
{
    int N = 1 << 20;        // 1M 元素
    size_t size = N * sizeof(float);

    // === 准备主机端数据 ===
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    // === 步骤 1: 分配设备内存 ===
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // === 步骤 2: 主机→设备 数据拷贝 ===
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // === 步骤 3: 启动内核 ===
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // === 步骤 4: 设备→主机 结果拷贝 ===
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // === 步骤 5: 释放设备内存 ===
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // === 验证结果 ===
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            errors++;
        }
    }
    printf("Verification: %s (%d errors)\n",
           errors == 0 ? "PASSED" : "FAILED", errors);

    free(h_A); free(h_B); free(h_C);
    return 0;
}
```

完整可编译版本（含错误检查和性能计时）参见 `code/chapter5/vector_add.cu`。

### 5.4.5 符号（Symbol）内存操作

除了通过指针直接操作内存外，CUDA 运行时还提供了直接操作<strong>在设备代码中声明的变量（符号）</strong>的函数。这些函数允许你在主机端直接读写声明在全局或常量内存空间的变量。

<strong>cudaMemcpyToSymbol / cudaMemcpyFromSymbol</strong>：

```cuda
// 声明在常量内存中的数组
__constant__ float constData[256];

// 主机端将数据拷贝到常量内存符号
float data[256];
// 初始化 data...
cudaMemcpyToSymbol(constData, data, sizeof(data));

// 主机端从常量内存符号读取数据
cudaMemcpyFromSymbol(data, constData, sizeof(data));
```

<strong>处理设备端指针变量</strong>：

```cuda
__device__ float *devPointer;  // 设备端声明的指针

// 主机端分配设备内存
float *ptr;
cudaMalloc(&ptr, 256 * sizeof(float));

// 将设备端指针变量的值设置为 ptr（主机端操作）
cudaMemcpyToSymbol(devPointer, &ptr, sizeof(ptr));
// 现在内核中 devPointer 指向 ptr 分配的内存
```

<strong>获取符号地址和大小</strong>：

```cuda
__device__ int devArray[1024];

size_t symbolSize;
void *symbolAddress;

// 获取符号的地址（设备端地址）
cudaGetSymbolAddress(&symbolAddress, devArray);

// 获取符号的大小
cudaGetSymbolSize(&symbolSize, devArray);
printf("devArray size: %zu bytes\n", symbolSize);
```

这些函数在使用动态加载的设备代码（Driver API）或需要精细控制设备端变量时非常有用。

## 5.5 多维内存分配

### 5.5.1 为什么需要多维内存分配？

在标准 C 中，二维数组通常通过 `malloc` 分配为一维数组，然后用 `row * width + col` 索引访问。然而在 GPU 上，这种简单的分配方式可能导致严重的性能问题，原因在于<strong>全局内存合并访问（Memory Coalescing）</strong>的要求。

<strong>合并访问的核心原理</strong>：当一个 warp（32 个线程）中的线程访问连续的全局内存地址时，这些访问可以被硬件合并为一次或少数几次内存事务（memory transaction）。反之，如果线程访问不连续或未对齐的地址，就会产生大量独立的内存事务，带宽利用率大幅下降。

<strong>问题场景</strong>：假设你有一个 $4096 \times 4096$ 的浮点矩阵，用标准方式分配。如果 warp 中的线程按行遍历（线程 0 访问 `row[0]`，线程 1 访问 `row[1]`...），这是合并的。但如果按列遍历（线程 0 访问 `col[0]`，线程 1 访问 `row[4096]`，线程 2 访问 `row[8192]`...），这是完全非合并的——stride 为 4096 个元素，每个线程的访问都在不同的 128 字节段中，导致 32 次独立的内存事务。

`cudaMallocPitch()` 和 `cudaMalloc3D()` 通过确保分配满足特定的对齐要求来解决这一问题。

### 5.5.2 cudaMallocPitch —— 二维内存分配

`cudaMallocPitch()` 为二维数组分配线性内存，确保每行的起始地址满足合并访问的对齐要求：

```cuda
cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch,
                            size_t widthInBytes, size_t height);
```

<strong>参数说明</strong>：
- `devPtr`：输出参数，返回已分配内存的指针。
- `pitch`：输出参数，返回<strong>实际行宽</strong>（以字节为单位）。这个值可能大于 `widthInBytes`。
- `widthInBytes`：请求的逻辑行宽（以字节为单位）。
- `height`：行数。

<strong>Pitch（步长）的本质</strong>：`pitch` 是每行在内存中的实际字节数。由于 GPU 硬件要求某些对齐（通常对齐到 128 或 256 字节），`pitch` >= `widthInBytes`。两个值之间的差值就是 padding 字节。

<strong>完整使用示例</strong>：

```cuda
// ========== 主机代码 ==========
int width = 64, height = 64;
float *devPtr;
size_t pitch;

// 分配二维数组
cudaMallocPitch(&devPtr, &pitch,
                width * sizeof(float), height);

printf("Logical width: %zu bytes\n", width * sizeof(float));
printf("Actual pitch:  %zu bytes\n", pitch);
printf("Padding:       %zu bytes\n",
       pitch - width * sizeof(float));

// 启动内核
dim3 block(16, 16);
dim3 grid((width + 15) / 16, (height + 15) / 16);
MyKernel<<<grid, block>>>(devPtr, pitch, width, height);

// ========== 设备代码 ==========
__global__ void MyKernel(float *devPtr, size_t pitch,
                         int width, int height)
{
    for (int r = 0; r < height; ++r) {
        // 关键：使用 pitch（而非 width*sizeof(float)）计算行偏移
        float *row = (float *)((char *)devPtr + r * pitch);
        for (int c = 0; c < width; ++c) {
            float element = row[c];
            // ... 处理 element ...
        }
    }
}
```

> <strong>关键规则</strong>：在访问 `cudaMallocPitch()` 分配的内存时，<strong>必须使用 pitch 而非 `width * sizeof(type)`</strong> 来计算行地址。使用错误的步长会导致访问到错误的 row（通常导致结果错误，而不会触发段错误，因为访问的仍然是已分配内存范围内的地址）。

<strong>配合 cudaMemcpy2D 使用</strong>：

```cuda
// 二维数据拷贝（主机 ↔ 设备）
cudaMemcpy2D(dstPtr, dpitch,     // 设备端：目标指针 + 目标 pitch
             srcPtr, spitch,     // 主机端：源指针 + 源 pitch
             width,              // 逻辑行宽（字节数）
             height,             // 行数
             cudaMemcpyHostToDevice);
```

`cudaMemcpy2D()` 知道如何正确处理 pitch——它只拷贝每行的逻辑宽度部分，跳过 padding 区域。

### 5.5.3 cudaMalloc3D —— 三维内存分配

对于三维数据（如体积数据、3D 卷积、物理模拟），使用 `cudaMalloc3D()`：

```cuda
cudaError_t cudaMalloc3D(cudaPitchedPtr *pitchedDevPtr,
                         cudaExtent extent);
```

<strong>cudaPitchedPtr 结构体</strong>：

```cuda
struct cudaPitchedPtr {
    void   *ptr;       // 指向已分配内存的指针
    size_t  pitch;     // 二维 pitch（行宽，字节）
    size_t  xsize;     // 逻辑宽度（字节），即 extent.width
    size_t  ysize;     // 逻辑高度，即 extent.height
};
```

<strong>cudaExtent 的创建</strong>：

```cuda
// make_cudaExtent 创建一个三维范围描述符
cudaExtent extent = make_cudaExtent(
    width * sizeof(float),  // 宽度（字节）
    height,                 // 高度（行数）
    depth                   // 深度（层数）
);
```

<strong>三维内存中的概念</strong>：

- <strong>pitch（行步长）</strong>：从一行开头到下一行开头的字节数。
- <strong>slicePitch（层步长）</strong>：从一层开头到下一层开头的字节数。计算公式：`slicePitch = pitch * height`。

<strong>完整的三维分配和遍历示例</strong>：

```cuda
// ========== 主机代码 ==========
int width = 64, height = 64, depth = 64;
cudaExtent extent = make_cudaExtent(
    width * sizeof(float), height, depth);
cudaPitchedPtr devPitchedPtr;
cudaMalloc3D(&devPitchedPtr, extent);

MyKernel<<<100, 512>>>(devPitchedPtr, width, height, depth);

// ========== 设备代码 ==========
__global__ void MyKernel(cudaPitchedPtr devPitchedPtr,
                         int width, int height, int depth)
{
    char *devPtr = (char *)devPitchedPtr.ptr;
    size_t pitch = devPitchedPtr.pitch;
    size_t slicePitch = pitch * height;  // 从一层到下一层的字节偏移

    // 三维遍历
    for (int z = 0; z < depth; ++z) {
        char *slice = devPtr + z * slicePitch;      // 第 z 层
        for (int y = 0; y < height; ++y) {
            float *row = (float *)(slice + y * pitch); // 第 y 行
            for (int x = 0; x < width; ++x) {
                float element = row[x];              // 第 x 列
            }
        }
    }
}
```

<strong>配合 cudaMemcpy3D 使用</strong>：

```cuda
// 使用 cudaMemcpy3DParms 结构体描述三维拷贝参数
cudaMemcpy3DParms params = {0};
params.srcPtr = make_cudaPitchedPtr(srcHostPtr, pitch,
                                    width * sizeof(float), height);
params.dstPtr = devPitchedPtr;
params.extent = extent;    // width, height, depth
params.kind   = cudaMemcpyHostToDevice;
cudaMemcpy3D(&params);
```

### 5.5.4 对齐与 Padding 的深入理解

你可能好奇：为什么 GPU 需要 padding？让我们用具体的数字来解释。

<strong>全局内存事务</strong>：GPU 访问全局内存的最小单位是<strong>内存事务（memory transaction）</strong>，大小通常为 32 字节、64 字节或 128 字节。一个 warp 的 32 个线程的访问如果能被合并到尽可能少的事务中，就能获得最高带宽。

<strong>对齐要求</strong>：在大多数 GPU 上，合并访问要求同一 warp 的线程访问的地址落在对齐的 128 字节段内。如果行宽恰好不是 128 字节的倍数，下一行的第一个元素可能与上一行的最后一个元素不在同一个 128 字节段内。

<strong>示例计算</strong>：

```text
假设：width = 30 floats, 每个 float = 4 bytes
逻辑行宽 = 30 × 4 = 120 bytes

如果直接分配 120 bytes 的行：
  第 0 行：地址 0-119
  第 1 行：地址 120-239
  ...

问题：第 0 行的最后一个 128 字节段包含了地址 0-127，即包括了第 1 行
的前 8 个字节。这使得不同行之间的 warp 访问可能跨越段边界。

cudaMallocPitch 的解决方案：
  添加 8 bytes padding，使 pitch = 128 bytes
  第 0 行：地址 0-119 (data) + 120-127 (padding)
  第 1 行：地址 128-247 (data) + 248-255 (padding)
  ...

现在每行都对齐到 128 字节边界，不同 warp 的访问不会相互干扰。
```

<div align="center">
  <img src="../images/chapter5-figures/memory-hierarchy.png" alt="CUDA内存层次结构" width="80%"/>
  <p>图 5.2 CUDA 内存层次结构：每个线程有私有的局部内存，每个线程块有共享内存可以被块内所有线程访问，所有线程都可以访问全局内存</p>
</div>

### 5.5.5 内存分配失败的优雅处理

> <strong>注意</strong>：为避免分配过多内存从而影响系统范围性能，应根据问题规模向用户请求分配参数。如果分配失败，可以采取以下策略：

1. <strong>降级策略</strong>：回退到其他较慢的内存类型。例如，如果 `cudaMalloc()` 失败，可以尝试使用 `cudaMallocHost()`（页锁定主机内存）或 `cudaHostRegister()`。

2. <strong>分级分配</strong>：先尝试分配全量设备内存。如果失败，分配部分设备内存 + 部分主机端缓冲区，分批处理数据。

3. <strong>统一内存后备</strong>：在支持统一内存（Unified Memory）的平台上，使用 `cudaMallocManaged()` 作为后备方案——它可以在物理显存不足时利用系统内存。

4. <strong>报告给用户</strong>：如果应用无法自行决定分配参数，返回错误并告知用户需要多少内存。

```cuda
float *d_data;
cudaError_t err = cudaMalloc(&d_data, largeSize);
if (err == cudaErrorMemoryAllocation) {
    // 策略 1：尝试更小的分配
    err = cudaMalloc(&d_data, smallerSize);
    if (err != cudaSuccess) {
        // 策略 2：回退到页锁定主机内存
        err = cudaMallocHost(&h_data, size);
        if (err != cudaSuccess) {
            printf("Error: Cannot allocate memory. "
                   "Requested %zu MB.\n", size / (1024*1024));
            return -1;
        }
        useHostMemory = true;  // 标记使用主机内存路径
    }
}
```

## 5.6 错误检查

### 5.6.1 CUDA API 错误检查的必要性

CUDA 程序的一个常见"新手陷阱"是<strong>忽略错误检查</strong>。CUDA API 调用和内核启动可能会失败，但如果不主动检查错误，程序可能继续执行并产生难以调试的错误结果。

所有 CUDA Runtime API 函数都返回 `cudaError_t` 类型的错误码。`cudaSuccess` 表示成功。常见的错误码包括：

| 错误码 | 含义 |
| :--- | :--- |
| `cudaSuccess` | 操作成功 |
| `cudaErrorMemoryAllocation` | 内存分配失败 |
| `cudaErrorInvalidDevicePointer` | 无效的设备指针 |
| `cudaErrorInvalidValue` | 无效的参数值 |
| `cudaErrorLaunchFailure` | 内核启动失败 |
| `cudaErrorNoDevice` | 无 CUDA-capable 设备 |
| `cudaErrorUnknown` | 未知错误 |

### 5.6.2 错误检查宏

推荐的做法是定义错误检查宏，在每个 CUDA API 调用后进行自动检查：

```cuda
#define CUDA_CHECK(err)                                               \
    do {                                                              \
        cudaError_t err_ = (err);                                     \
        if (err_ != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s (code %d)\n",   \
                    __FILE__, __LINE__,                               \
                    cudaGetErrorString(err_), err_);                  \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)
```

使用方式：

```cuda
CUDA_CHECK(cudaMalloc(&d_data, size));
CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
CUDA_CHECK(cudaGetLastError());  // 检查内核启动错误
```

### 5.6.3 内核启动错误检查

内核启动（`<<<...>>>`）不返回错误码。要检查内核启动是否成功，必须在启动后调用 `cudaGetLastError()`：

```cuda
kernel<<<grid, block>>>(args);
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
}
```

由于内核启动是异步的，`cudaGetLastError()` 返回的是<strong>启动阶段的错误</strong>（如无效的 grid/block 维度、不存在的内核等）。内核<strong>执行过程中的错误</strong>（如越界访问、除以零等）需要通过同步操作（`cudaDeviceSynchronize()` 或 `cudaStreamSynchronize()`）来捕获。

```cuda
kernel<<<grid, block>>>(args);
cudaError_t err = cudaDeviceSynchronize();  // 等待执行完成 + 获取执行错误
if (err != cudaSuccess) {
    printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
}
```

## 5.7 内存传输进阶

### 5.7.1 内存传输方向详解

| 传输方向 | cudaMemcpyKind | 典型用途 |
| :--- | :--- | :--- |
| <strong>主机 → 设备</strong> | `cudaMemcpyHostToDevice` | 将输入数据发送到 GPU 进行并行处理 |
| <strong>设备 → 主机</strong> | `cudaMemcpyDeviceToHost` | 取回 GPU 计算结果供 CPU 后续使用 |
| <strong>设备 → 设备</strong> | `cudaMemcpyDeviceToDevice` | GPU 内存内部的数据迁移或重排 |
| <strong>主机 → 主机</strong> | `cudaMemcpyHostToHost` | 配合页锁定内存的 CPU 端快速拷贝 |

### 5.7.2 设备间数据传输

在多 GPU 系统中，有时需要将数据从一个 GPU 的内存传输到另一个 GPU。有几种方式：

<strong>1. 显式设备间拷贝（UVA 下自动路由）</strong>：

在支持 UVA 的系统上，`cudaMemcpy()` 可以自动路由设备间的拷贝：

```cuda
// d_src 在 Device 0，d_dst 在 Device 1
cudaMemcpy(d_dst, d_src, size, cudaMemcpyDefault);
```

<strong>2. Peer-to-Peer（P2P）直接拷贝</strong>：

```cuda
// 启用 P2P 访问
cudaDeviceEnablePeerAccess(peerDevice, 0);

// P2P 拷贝（不经过主机内存）
cudaMemcpyPeer(d_dst, dstDevice, d_src, srcDevice, size);
```

P2P 拷贝的性能通常远优于通过主机内存中转的方案，因为数据直接在 GPU 之间通过 PCIe 总线或 NVLink 桥传输。

<strong>3. 通过主机内存中转</strong>：

```cuda
// Device 0 → Host → Device 1
cudaMemcpy(h_buf, d_src, size, cudaMemcpyDeviceToHost);   // 第1步
cudaMemcpy(d_dst, h_buf, size, cudaMemcpyHostToDevice);   // 第2步
```

这种方案虽然简单，但性能最低（两次 PCIe 传输）。

### 5.7.3 异步传输

默认情况下，`cudaMemcpy()` 是同步的。异步传输可以通过以下方式实现：

- 使用 `cudaMemcpyAsync()` —— 异步版本的内存拷贝函数。
- 配合页锁定内存和 CUDA 流（CUDA Streams）使用，实现数据传输和内核执行的并发重叠。

异步传输是第 7 章"流（Streams）、事件（Events）与异步并发执行"的核心主题，届时我们将详细展开。

### 5.7.4 写入设备内存的其他方法

除了 `cudaMemcpy()`，还有几种向设备内存写入数据的方法：

- <strong>cudaMemset()</strong>：将设备内存的每个字节设置为指定的值（类似于 `memset`）。
- <strong>内核直接写入</strong>：内核可以直接向设备内存写入计算结果。
- <strong>映射内存（Zero-Copy）</strong>：将主机内存映射到设备地址空间，内核直接访问。

## 5.8 CUDA 事件与精确计时

CUDA 提供了 <strong>事件（Events）</strong>机制来进行精确的 GPU 计时和流间同步。对于性能分析、衡量内核和数据传输的时间，事件是不可或缺的工具。

### 5.8.1 事件的基本使用

```cuda
// 创建事件
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// 在默认流中记录"开始"事件
cudaEventRecord(start);   // 或者在特定流中：cudaEventRecord(start, stream)

// ... 执行要计时的 GPU 操作（内核启动、数据传输等）...

// 记录"结束"事件
cudaEventRecord(stop);

// 等待结束事件完成
cudaEventSynchronize(stop);

// 计算经过的时间（毫秒）
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("GPU operation took: %.3f ms\n", milliseconds);

// 销毁事件
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### 5.8.2 事件的几个重要特性

<strong>1. 事件记录在 GPU 时间线上</strong>：`cudaEventRecord()` 将一个事件"标记"在 GPU 的执行流中。当 GPU 执行到这个点时，事件被标记为"已完成"。这允许你测量特定 GPU 操作序列的执行时间，而不受主机端 CPU 时间测量精度的影响。

<strong>2. 事件基于 GPU 时钟</strong>：`cudaEventElapsedTime()` 返回的时间基于 GPU 的高精度时钟，分辨率通常为微秒级甚至纳秒级（取决于 GPU 型号）。

<strong>3. 事件可用于流间同步</strong>：

```cuda
// 让 stream1 等待 stream2 中的事件完成
cudaStreamWaitEvent(stream1, event, 0);
```

<strong>4. 事件的阻塞与查询</strong>：

```cuda
// 阻塞等待事件完成
cudaEventSynchronize(event);

// 非阻塞查询事件状态
cudaError_t err = cudaEventQuery(event);
if (err == cudaSuccess) {
    // 事件已完成
} else if (err == cudaErrorNotReady) {
    // 事件尚未完成
}
```

### 5.8.3 使用事件进行分步计时

以下是一个使用事件来分解测量向量加法各部分时间的完整示例：

```cuda
// 分步计时向量加法
cudaEvent_t e_alloc, e_h2d, e_kernel_start, e_kernel_end, e_d2h, e_free;
cudaEventCreate(&e_alloc); cudaEventCreate(&e_h2d);
cudaEventCreate(&e_kernel_start); cudaEventCreate(&e_kernel_end);
cudaEventCreate(&e_d2h); cudaEventCreate(&e_free);

// 记录分配开始
cudaEventRecord(e_alloc);
cudaMalloc(&d_A, size);
cudaMalloc(&d_B, size);
cudaMalloc(&d_C, size);
cudaEventRecord(e_h2d);  // 分配结束 = 拷贝开始

cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
cudaEventRecord(e_kernel_start);  // 拷贝结束 = 内核开始

VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
cudaEventRecord(e_kernel_end);  // 内核结束

cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
cudaEventRecord(e_d2h);  // 拷回结束

cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
cudaEventRecord(e_free);  // 释放结束

cudaEventSynchronize(e_free);

// 输出各阶段耗时
float t_alloc, t_h2d, t_kernel, t_d2h, t_free;
cudaEventElapsedTime(&t_alloc, e_alloc, e_h2d);
cudaEventElapsedTime(&t_h2d, e_h2d, e_kernel_start);
cudaEventElapsedTime(&t_kernel, e_kernel_start, e_kernel_end);
cudaEventElapsedTime(&t_d2h, e_kernel_end, e_d2h);
cudaEventElapsedTime(&t_free, e_d2h, e_free);

printf("Time breakdown:\n");
printf("  Allocation:  %.3f ms\n", t_alloc);
printf("  H->D copy:   %.3f ms\n", t_h2d);
printf("  Kernel:      %.3f ms\n", t_kernel);
printf("  D->H copy:   %.3f ms\n", t_d2h);
printf("  Free:        %.3f ms\n", t_free);

// 销毁所有事件
cudaEventDestroy(e_alloc); cudaEventDestroy(e_h2d);
cudaEventDestroy(e_kernel_start); cudaEventDestroy(e_kernel_end);
cudaEventDestroy(e_d2h); cudaEventDestroy(e_free);
```

这种分步计时是性能分析的起点。通过测量每个阶段的耗时，你可以快速定位性能瓶颈——是数据传输太慢？还是内核计算本身需要优化？

## 5.9 内存管理策略与最佳实践

### 5.9.1 设备内存分配的最佳实践

在实际开发中，以下策略可以帮你更好地管理设备内存：

<strong>1. 预分配与内存复用</strong>：

```cuda
// 初始化阶段：一次性分配工作缓冲区
float *d_workBuf1, *d_workBuf2, *d_result;
cudaMalloc(&d_workBuf1, MAX_SIZE);
cudaMalloc(&d_workBuf2, MAX_SIZE);
cudaMalloc(&d_result, MAX_SIZE);

// 工作循环：重复使用已分配的内存
for (int iter = 0; iter < NUM_ITERS; iter++) {
    // 拷贝本轮输入数据到设备
    cudaMemcpy(d_workBuf1, h_input + iter * CHUNK, chunkSize,
               cudaMemcpyHostToDevice);

    // 处理
    kernel<<<grid, block>>>(d_workBuf1, d_workBuf2, d_result, ...);
    cudaDeviceSynchronize();

    // 拷贝结果
    cudaMemcpy(h_output + iter * CHUNK, d_result, chunkSize,
               cudaMemcpyDeviceToHost);
}

// 清理阶段：释放所有内存
cudaFree(d_workBuf1); cudaFree(d_workBuf2); cudaFree(d_result);
```

<strong>2. 分级内存分配策略</strong>：

当全量数据无法装入设备内存时：

```cuda
size_t totalSize = ...;  // 超过设备内存的数据总量
size_t deviceFree, deviceTotal;
cudaMemGetInfo(&deviceFree, &deviceTotal);

// 计算安全的分配量（留 10% 余量）
size_t chunkSize = (size_t)(deviceFree * 0.9);
int numChunks = (totalSize + chunkSize - 1) / chunkSize;

printf("Processing in %d chunks of %.2f MB each.\n",
       numChunks, (float)chunkSize / (1024*1024));

float *d_chunk;
cudaMalloc(&d_chunk, chunkSize);

for (int c = 0; c < numChunks; c++) {
    size_t offset = (size_t)c * chunkSize;
    size_t thisChunk = min(chunkSize, totalSize - offset);

    cudaMemcpy(d_chunk, h_data + offset / sizeof(float),
               thisChunk, cudaMemcpyHostToDevice);
    processKernel<<<grid, block>>>(d_chunk, thisChunk / sizeof(float));
    cudaMemcpy(h_result + offset / sizeof(float),
               d_chunk, thisChunk, cudaMemcpyDeviceToHost);
}

cudaFree(d_chunk);
```

<strong>3. 查询可用内存</strong>：

```cuda
size_t freeMem, totalMem;
cudaMemGetInfo(&freeMem, &totalMem);
printf("GPU memory: %.2f GB used / %.2f GB total\n",
       (float)(totalMem - freeMem) / (1024*1024*1024),
       (float)totalMem / (1024*1024*1024));
```

### 5.9.2 避免常见内存陷阱

<strong>陷阱 1：主机端解引用设备指针</strong>

```cuda
float *d_data;
cudaMalloc(&d_data, size);
// printf("%f\n", *d_data);  // 错误！段错误或垃圾值
// 正确：通过 cudaMemcpy 传递数据
cudaMemcpy(&h_val, d_data, sizeof(float), cudaMemcpyDeviceToHost);
printf("%f\n", h_val);
```

<strong>陷阱 2：忘记检查内存分配失败</strong>

```cuda
float *d_data;
cudaError_t err = cudaMalloc(&d_data, VERY_LARGE_SIZE);
if (err != cudaSuccess) {
    printf("Memory allocation failed: %s\n",
           cudaGetErrorString(err));
    // 清理已分配的资源，降级处理
    return -1;
}
```

<strong>陷阱 3：释放后使用（Use-after-free）</strong>

```cuda
cudaFree(d_data);
// kernel<<<...>>>(d_data);  // 错误！d_data 已释放
```

<strong>陷阱 4：同步与异步的混淆</strong>

```cuda
// cudaMemcpy 是同步的——拷贝完成才返回
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
// 这里 d_data 已经被完整填充，可以安全使用

// 内核启动是异步的——启动后立即返回，不等待执行完毕
kernel<<<grid, block>>>(d_data);
// 这里内核可能还没开始执行！
printf("Kernel launched, but may not have finished.\n");

// 确保内核执行完毕后再访问结果
cudaDeviceSynchronize();
cudaMemcpy(h_result, d_data, size, cudaMemcpyDeviceToHost);
// 现在 h_result 中的数据是完整的
```

### 5.9.3 查询设备内存信息

CUDA 运行时提供了丰富的 API 来查询设备内存信息：

```cuda
int deviceCount;
cudaGetDeviceCount(&deviceCount);

for (int dev = 0; dev < deviceCount; dev++) {
    cudaSetDevice(dev);

    // 获取设备属性（包含内存总量等）
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("Device %d: %s\n", dev, prop.name);
    printf("  Total global memory:   %zu MB\n",
           prop.totalGlobalMem / (1024 * 1024));
    printf("  Shared memory per block: %zu KB\n",
           prop.sharedMemPerBlock / 1024);
    printf("  Max shared memory per block (opt-in): %zu KB\n",
           prop.sharedMemPerBlockOptin / 1024);
    printf("  Constant memory:        %zu KB\n",
           prop.totalConstMem / 1024);
    printf("  Registers per block:    %d\n", prop.regsPerBlock);
    printf("  Registers per SM:       %d\n",
           prop.regsPerMultiprocessor);
    printf("  L2 cache size:          %d KB\n",
           prop.l2CacheSize / 1024);
    printf("  Max threads per SM:     %d\n",
           prop.maxThreadsPerMultiProcessor);
    printf("  Warp size:              %d\n", prop.warpSize);

    // 获取当前空闲内存（运行时动态查询）
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("  Free memory:            %zu MB\n",
           freeMem / (1024 * 1024));
    printf("  Total memory:           %zu MB\n",
           totalMem / (1024 * 1024));
    printf("\n");
}
```

### 5.9.4 cudaMemcpy2D 和 cudaMemcpy3D 详解

与 `cudaMallocPitch()` 和 `cudaMalloc3D()` 对应，CUDA 提供了二维和三维的拷贝函数：

<strong>cudaMemcpy2D</strong>：

```cuda
cudaError_t cudaMemcpy2D(
    void *dst,         size_t dpitch,  // 目标基址 + 目标行距
    const void *src,   size_t spitch,  // 源基址 + 源行距
    size_t width,                       // 要拷贝的逻辑行宽（字节）
    size_t height,                      // 要拷贝的行数
    cudaMemcpyKind kind                 // 拷贝方向
);
```

<strong>cudaMemcpy3D</strong>：

```cuda
cudaError_t cudaMemcpy3D(
    const cudaMemcpy3DParms *p         // 三维拷贝参数结构体
);

// cudaMemcpy3DParms 结构体
struct cudaMemcpy3DParms {
    cudaArray_t            srcArray;  // 源 CUDA 数组（与 srcPtr 二选一）
    struct cudaPos         srcPos;    // 源偏移
    struct cudaPitchedPtr  srcPtr;    // 源线性内存（与 srcArray 二选一）

    cudaArray_t            dstArray;  // 目标 CUDA 数组（与 dstPtr 二选一）
    struct cudaPos         dstPos;    // 目标偏移
    struct cudaPitchedPtr  dstPtr;    // 目标线性内存（与 dstArray 二选一）

    struct cudaExtent      extent;    // 拷贝范围 (width, height, depth)
    enum cudaMemcpyKind    kind;      // 拷贝方向
};
```

二维拷贝的典型使用示例：

```cuda
// 主机端构造二维数据
float *h_data = (float *)malloc(height * hostPitch);
// 填充 h_data ...

// 设备端分配
float *d_data;
size_t devPitch;
cudaMallocPitch(&d_data, &devPitch, width * sizeof(float), height);

// 二维拷贝（只拷贝每行的有效数据部分，跳过 padding）
cudaMemcpy2D(d_data, devPitch,          // 目标
             h_data, hostPitch,         // 源
             width * sizeof(float),      // 行宽（有效数据）
             height,                     // 行数
             cudaMemcpyHostToDevice);
```

这种拷贝方式意味着数据的物理行宽（pitch/stride）可以不同于逻辑行宽，非常适合处理 padding。

### 5.9.5 编写一个 RAII 风格的内存管理器

在大型 CUDA 项目中手动管理设备内存的分配和释放容易出错（遗漏释放、异常路径下的泄漏等）。以下是一个使用 RAII（Resource Acquisition Is Initialization）模式的简单设备内存管理器：

```cuda
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

/**
 * RAII 风格的 CUDA 设备内存包装器
 *
 * 在构造时分配，在析构时自动释放
 * 不可拷贝（防止 double-free），但支持移动语义
 */
template <typename T>
class DeviceMemory {
private:
    T *ptr_;
    size_t size_;  // 元素个数（不是字节数）
    bool owns_;    // 是否拥有内存的所有权

public:
    // 构造函数：分配设备内存
    DeviceMemory(size_t count)
        : ptr_(nullptr), size_(count), owns_(true)
    {
        cudaError_t err = cudaMalloc(&ptr_, count * sizeof(T));
        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                "cudaMalloc failed: " +
                std::string(cudaGetErrorString(err)));
        }
    }

    // 析构函数：自动释放内存
    ~DeviceMemory()
    {
        if (owns_ && ptr_)
        {
            cudaFree(ptr_);
        }
    }

    // 禁止拷贝
    DeviceMemory(const DeviceMemory &) = delete;
    DeviceMemory &operator=(const DeviceMemory &) = delete;

    // 允许移动
    DeviceMemory(DeviceMemory &&other) noexcept
        : ptr_(other.ptr_), size_(other.size_), owns_(other.owns_)
    {
        other.ptr_ = nullptr;
        other.owns_ = false;
    }

    DeviceMemory &operator=(DeviceMemory &&other) noexcept
    {
        if (this != &other)
        {
            if (owns_ && ptr_) cudaFree(ptr_);
            ptr_ = other.ptr_;
            size_ = other.size_;
            owns_ = other.owns_;
            other.ptr_ = nullptr;
            other.owns_ = false;
        }
        return *this;
    }

    // 获取裸指针（用于 CUDA API 调用）
    T *get() const { return ptr_; }
    T *data() const { return ptr_; }

    // 获取元素个数
    size_t size() const { return size_; }

    // 将数据拷贝到主机
    void copyToHost(T *hostPtr) const
    {
        cudaMemcpy(hostPtr, ptr_, size_ * sizeof(T),
                   cudaMemcpyDeviceToHost);
    }

    // 将数据从主机拷贝到设备
    void copyFromHost(const T *hostPtr)
    {
        cudaMemcpy(ptr_, hostPtr, size_ * sizeof(T),
                   cudaMemcpyHostToDevice);
    }
};

// 使用示例
int main()
{
    try
    {
        // RAII：分配自动管理，函数返回/异常时自动释放
        DeviceMemory<float> d_A(1024 * 1024);
        DeviceMemory<float> d_B(1024 * 1024);
        DeviceMemory<float> d_C(1024 * 1024);

        // 分配主机内存并初始化
        std::vector<float> h_A(1024 * 1024);
        std::vector<float> h_B(1024 * 1024);
        for (size_t i = 0; i < h_A.size(); i++)
        {
            h_A[i] = (float)i;
            h_B[i] = (float)(i * 2);
        }

        // 拷贝数据
        d_A.copyFromHost(h_A.data());
        d_B.copyFromHost(h_B.data());

        // 启动内核
        int threads = 256;
        int blocks = (d_A.size() + threads - 1) / threads;
        VecAdd<<<blocks, threads>>>(d_A.get(), d_B.get(),
                                     d_C.get(), d_A.size());

        // 取回结果
        std::vector<float> h_C(1024 * 1024);
        d_C.copyToHost(h_C.data());

        // d_A, d_B, d_C 自动释放（ RAII 析构函数）
    }
    catch (const std::exception &e)
    {
        printf("Error: %s\n", e.what());
        return -1;
    }
    return 0;
}
```

这种 RAII 包装器在大型项目中能显著减少内存泄漏和 use-after-free 的风险。许多工业级 CUDA 库（如 Thrust 的 `thrust::device_vector`）就是基于类似原理设计的。

如果要使用更成熟的解决方案，可以参考 <strong>thrust::device_vector</strong>：

```cuda
#include <thrust/device_vector.h>

// Thrust 风格（自动管理设备内存）
thrust::device_vector<float> d_vec(1000000);
// d_vec 会自动在析构时释放设备内存

// 获取裸指针用于 CUDA kernel 调用
float *raw_ptr = thrust::raw_pointer_cast(d_vec.data());
kernel<<<grid, block>>>(raw_ptr, ...);
```

## 5.10 动手体验：完整的向量加法与二维数组处理

### 5.10.1 实验一：向量加法（带错误检查和计时）

完整可编译代码参见 `code/chapter5/vector_add.cu`。

编译和运行：

```bash
nvcc vector_add.cu -o vector_add -arch=sm_60
./vector_add
```

预期输出：

```text
========================================
Vector Addition with CUDA
========================================
Vector length: 1048576 elements
Data size: 4.00 MB per vector

[Step 1] Allocating host memory...
[Step 2] Initializing input vectors...
  h_A[0]=0.00, h_A[N-1]=1048575.00
  h_B[0]=0.00, h_B[N-1]=2097150.00
[Step 3] Allocating device memory...
[Step 4] Copying data Host -> Device...
[Step 5] Launching kernel...
  Grid: 4096 blocks, Block: 256 threads
  Kernel execution time: 0.123 ms
[Step 6] Copying result Device -> Host...
[Step 7] Verifying results...
  All 1048576 elements verified successfully!
[Step 8] Freeing device memory...
========================================
Done!
========================================
```

### 5.10.2 实验二：二维卷积操作（cudaMallocPitch 实践）

完整可编译代码参见 `code/chapter5/conv2d_pitch.cu`。

编译和运行：

```bash
nvcc conv2d_pitch.cu -o conv2d_pitch -arch=sm_60
./conv2d_pitch
```

预期输出：

```text
========================================
2D Convolution using cudaMallocPitch
========================================
Image size: 1024 x 1024
Requested line width: 4096 bytes
Actual pitch:         4096 bytes
Padding per row:      0 bytes
Input initialized (value = row + col).
  h_input[0][0] = 0.000000
  h_input[1023][1023] = 2046.000000
Copying data Host -> Device (cudaMemcpy2D)...
Launching kernel: grid(64,64) x block(16,16)...
Kernel time: 2.145 ms
Verifying results...
  All 1048576 elements verified successfully!
========================================
Done!
========================================
```

注意在这个例子中，`pitch` 恰好等于 `width * sizeof(float)`（4096 = 1024 * 4），因为 4096 是 128 的整数倍，满足对齐要求。如果你将 width 改为 1023（4092 bytes，不是 128 的倍数），你会看到 padding 的出现。

## 5.11 本章小结

在本章中，我们系统学习了 CUDA 运行时（CUDA Runtime）的设备内存管理机制。从运行时的架构到具体的 API 调用，再到错误检查和多维内存分配，我们建立了完整的知识体系：

- <strong>CUDA 运行时是什么？</strong> CUDA 运行时实现在 `cudart` 库中，提供以 `cuda` 为前缀的 API。它封装了 Driver API 的底层细节，为开发者提供高级、易用的内存管理、数据传输和设备管理接口。运行时没有显式初始化函数——在第一次 API 调用时自动完成初始化。

- <strong>运行时初始化是如何工作的？</strong> 第一次调用任何运行时函数时自动触发初始化，为系统中的每个设备创建主上下文（Primary Context）。所有主机线程共享这些上下文。初始化开销较大（100ms+），之后调用极快。

- <strong>标准五步内存管理模式？</strong> `cudaMalloc()` 分配 → `cudaMemcpy(HtoD)` 拷贝输入 → `kernel<<<>>>()` 启动计算 → `cudaMemcpy(DtoH)` 拷贝结果 → `cudaFree()` 释放。这是任何 CUDA 程序的骨架模式。

- <strong>多维内存分配有什么用？</strong> `cudaMallocPitch()` 和 `cudaMalloc3D()` 为二维和三维数组提供对齐的分配。它们通过自动添加 padding 确保每行对齐到合适的边界，支持 warp 级合并访问以最大化全局内存带宽。访问时必须使用返回的 `pitch`（而非逻辑行宽）来计算行偏移。

- <strong>符号内存操作是什么？</strong> `cudaMemcpyToSymbol()` 和 `cudaMemcpyFromSymbol()` 允许在主机端直接读写声明在设备端全局或常量内存空间的变量。这在配置常量参数和操作设备端全局状态时非常有用。

- <strong>为什么错误检查很重要？</strong> CUDA API 的错误是无声的——不检查错误码就可能导致难以调试的错误结果。建议使用 `CUDA_CHECK` 宏在所有 API 调用后进行错误检查。内核启动错误需要通过 `cudaGetLastError()` 检查，内核执行错误需要通过同步操作捕获。

- <strong>内存传输有哪些模式？</strong> HtoD / DtoH / DtoD / HtoH 四种基本方向，以及设备间的 P2P 拷贝。异步传输（Async）配合页锁定内存和流可以实现传输与计算的并发重叠。

通过本章的学习，我们掌握了 CUDA 程序中最基础、最核心的内存管理模式。在下一章中，我们将学习共享内存（Shared Memory）和页锁定主机内存（Page-Locked Host Memory）——两种对提升 CUDA 程序性能至关重要的技术！

## 习题

> <strong>提示</strong>：以下的部分习题没有标准答案，重点在于培养学习者对 CUDA 设备内存管理批判性的深入思考和动手实践能力。

1. <strong>CUDA 运行时初始化分析</strong>：
   a. 为什么 CUDA 运行时选择隐式初始化而非要求显式调用初始化函数？这种设计有什么优缺点？
   b. 如果你在全局构造函数中（`main()` 之前）调用了一个 CUDA 运行时函数，会发生什么？请解释原因。
   c. `cudaDeviceReset()` 调用后，之前分配的设备内存 `d_ptr` 还有效吗？如果尝试使用它会发生什么？

2. <strong>内存分配 API 深入分析</strong>：
   a. 一台 GPU 拥有 8 GB 显存。如果程序尝试用 `cudaMalloc()` 分配 9 GB，会发生什么？如何优雅地在代码中处理这种情况？
   b. `cudaMalloc()` 返回的指针保证对齐到多少字节？为什么这个对齐值对性能重要？
   c. 解释 `cudaMalloc((void**)&ptr, size)` 中为什么要用 `(void**)` 转型和取地址 `&ptr`。

3. <strong>Pitch 与二维分配</strong>：
   a. 请解释 pitch 的概念。为什么 GPU 需要在一维线性内存中加入 pitch，而不是直接用 `width * sizeof(float)` 作为行宽？
   b. 在以下两种情况下，`cudaMallocPitch()` 返回的 pitch 值分别是多少（假设 GPU 要求 128 字节对齐）？
      情况 A：`width = 63`，元素类型 `float`（4 bytes）
      情况 B：`width = 32`，元素类型 `double`（8 bytes）
   c. 如果内核代码错误地使用了 `width * sizeof(float)` 来遍历 `cudaMallocPitch()` 分配的内存（而实际上 `pitch > width * sizeof(float)`），程序的执行结果会怎样？请画出内存布局来辅助解释。

4. <strong>内存传输与生命周期分析</strong>：
   分析以下代码段是否正确。如果不正确，指出问题并修正。
   ```cuda
   float *d_A;
   cudaMalloc(&d_A, size);
   cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
   kernel<<<grid, block>>>(d_A);
   cudaFree(d_A);  // 释放 d_A
   float *h_result = (float *)malloc(size);
   cudaMemcpy(h_result, d_A, size, cudaMemcpyDeviceToHost);  // 使用已释放的 d_A!
   ```

5. <strong>向量加法扩展设计</strong>：
   基于 5.4.4 节的向量加法示例：
   a. 修改程序使其能处理向量长度 N 不是 threadsPerBlock 整数倍的情况。
   b. 添加计时功能：分别测量数据拷贝时间（HtoD + DtoH）和内核执行时间，分析各自占比。
   c. 使用不同的向量大小（1K、1M、10M、100M 个元素）进行测试，绘制时间占比的变化曲线，解释为什么小数据量和大数据量下的瓶颈不同。

6. <strong>错误处理综合设计</strong>：
   a. 设计一个健壮的 `CUDA_CHECK` 宏：除了打印错误信息外，还应该记录错误发生的上下文（设备 ID、API 函数名）。
   b. 编写一个函数 `safe_cudaMalloc(void **ptr, size_t size)`，在 `cudaMalloc` 失败时自动重试（最多 3 次，每次等待 100ms），如果最终仍失败则释放所有已分配资源并退出。
   c. 在一个包含多次分配（5 次以上）和多次内核启动的程序中，实现完整的错误处理：任何一步失败都触发已分配资源的清理。

7. <strong>动手实践题</strong>：
   > <strong>提示</strong>：这是一道动手实践题，建议实际编写代码。

   a. 使用 `cudaMallocPitch()` 和 `cudaMemcpy2D()` 编写一个 2D 矩阵转置（matrix transpose）程序。测试不同 width 值（如 1023 vs 1024）下的 pitch 差异。
   b. 在代码中实现完整的错误检查，确保每个 CUDA API 调用都经过了验证。编写一个测试用例故意触发 `cudaErrorMemoryAllocation`（请求超过显存大小的内存），验证你的错误处理是否正确工作。
   c. 使用 `cudaPointerGetAttributes()` 编写一个函数 `printPointerInfo(void *ptr)`，它可以打印任意指针的内存类型（设备/主机/统一/未知）和设备 ID。

## 参考文献

[1] NVIDIA Corporation. CUDA C++ Programming Guide (Version 11.5.1)[Z]. https://docs.nvidia.com/cuda/archive/11.5.1/cuda-c-programming-guide/index.html

[2] NVIDIA Corporation. CUDA C++ Programming Guide (Version 13.0)[Z]. https://docs.nvidia.com/cuda/archive/13.0.3/cuda-c-programming-guide/index.html

[3] NVIDIA Corporation. CUDA Runtime API Reference[Z]. https://docs.nvidia.com/cuda/cuda-runtime-api/

[4] NVIDIA Corporation. CUDA C Best Practices Guide[Z]. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

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
