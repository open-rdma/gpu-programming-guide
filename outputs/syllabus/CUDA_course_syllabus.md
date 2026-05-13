# CUDA 编程课程教学大纲（核心/基础部分）

> 本大纲基于 NVIDIA CUDA Programming Guide 的 11.5.1 版本与 13.0 版本中**共同存在的核心内容**编写。  
> 课程目标：帮助初学者系统掌握 CUDA 编程的基石概念，从零开始理解 GPU 并行编程模型。  
> 进阶章节（基于 CUDA 13.0 新增内容）将在最后列出概要，由后续部分单独详述。

---

## 第1章：GPU计算与CUDA入门

### 本章目标
- 理解 GPU 与 CPU 在架构设计上的根本差异
- 了解 CUDA 作为通用并行计算平台和编程模型的定位
- 掌握 CUDA 的可扩展编程模型的核心抽象
- 建立对课程整体学习路径的宏观认识

### 核心知识点
1. **GPU 的设计哲学**：GPU 将更多晶体管用于数据处理（浮点运算单元），而非数据缓存和流控制。这使得 GPU 能够通过大规模并行计算来隐藏内存访问延迟，而不依赖大容量的缓存。
2. **异构计算**：现代系统将 CPU 与 GPU 配合使用——CPU 负责串行逻辑（延迟优化），GPU 负责高度并行的计算任务（吞吐量优化）。
3. **CUDA 的三个核心抽象**：线程组层次结构（a hierarchy of thread groups）、共享内存（shared memories）、以及屏障同步（barrier synchronization）。这三个抽象通过一组最简的语言扩展暴露给程序员。
4. **自动可扩展性**：每个线程块可以在任意可用的多处理器（SM）上独立调度，因此编译好的 CUDA 程序可以在具有不同数量 SM 的 GPU 上自动扩展运行，无需修改代码。

### 章节内容大纲
1. **GPU 的优势与局限性** — CPU 与 GPU 的架构差异对比（晶体管用途分布图），GPU 适用于高度并行任务的原理。
2. **CUDA 平台概述** — NVIDIA 在 2006 年推出 CUDA 的历史背景，支持的编程语言和应用接口（C++、FORTRAN、DirectCompute、OpenACC）。
3. **可扩展编程模型** — 核心抽象的详细介绍：线程组、共享内存、屏障同步；粗粒度与细粒度并行的分解方式；自动可扩展性的工作原理。

### 参考来源
- 旧版 Guide: 1.1 (The Benefits of Using GPUs), 1.2 (CUDA: A General-Purpose Parallel Computing Platform), 1.3 (A Scalable Programming Model), 1.4 (Document Structure)
- 新版 Guide: 3.1 (The Benefits of Using GPUs), 3.2 (CUDA: A General-Purpose Parallel Computing Platform), 3.3 (A Scalable Programming Model)

---

## 第2章：CUDA编程模型——内核函数与线程层次结构

### 本章目标
- 掌握 CUDA 内核函数（kernel）的定义和启动语法
- 理解线程层次结构：线程（thread）→ 线程块（thread block）→ 网格（grid）
- 学会使用内置变量 `threadIdx`、`blockIdx`、`blockDim` 和 `gridDim`
- 能够写出正确的向量加法和矩阵加法的 CUDA 内核

### 核心知识点
1. **内核函数定义**：使用 `__global__` 声明修饰符定义，通过 `<<<...>>>` 执行配置语法指定启动的线程数量。每个执行内核的线程获得一个唯一的内置变量 `threadIdx`。
2. **线程层次结构**：`threadIdx` 是三维向量（x, y, z），线程可以组织为一维、二维或三维的线程块。每个块的线程数上限为 1024。
3. **网格结构**：多个相同形状的线程块组织为一维、二维或三维的网格。每个块通过内置变量 `blockIdx` 唯一标识，`blockDim` 给出块的维度。
4. **线程映射**：全局线程索引的常用计算公式：`int i = blockIdx.x * blockDim.x + threadIdx.x`。
5. **块独立性要求**：线程块必须能够以任意顺序、并行或串行方式执行。这种独立执行特性是实现自动可扩展性的基础。

### 章节内容大纲
1. **第一个 CUDA 内核** — 向量加法示例：`__global__` 声明、`<<<1, N>>>` 启动语法、`threadIdx.x` 的使用。
2. **多维线程组织** — 矩阵加法示例：二维 `threadIdx`、`dim3` 类型的使用、二维 block 和 grid 的配置。
3. **多块内核** — 如何处理超过一个块能容纳的数据量，`blockIdx` 与 `blockDim` 的组合使用，边界检查的重要性。
4. **线程块内协作** — `__syncthreads()` 屏障函数的基本介绍，块内共享内存协作概念（为第3章铺垫）。

### 参考来源
- 旧版 Guide: 2.1 (Kernels), 2.2 (Thread Hierarchy)
- 新版 Guide: 5.1 (Kernels), 5.2 (Thread Hierarchy)

---

## 第3章：CUDA内存层次结构与非对称编程

### 本章目标
- 掌握 CUDA 的多级内存层次结构及各级内存的特性
- 理解主机（Host）与设备（Device）的非对称编程模型
- 了解统一内存（Unified Memory）的基本概念
- 掌握 Compute Capability（计算能力）的概念及其作用

### 核心知识点
1. **内存层次结构**：每个线程有私有的局部内存（local memory），每个线程块有共享内存（shared memory）对所有块内线程可见且与块同生命周期，所有线程都可以访问全局内存（global memory）。
2. **只读内存空间**：常量内存（constant memory）和纹理内存（texture memory）对所有线程可见且只读，各有不同的优化特性。全局、常量和纹理内存在同一应用的多次内核启动之间持久存在。
3. **异构编程模型**：CUDA 假设 CUDA 线程运行在物理上分离的设备（如 GPU）上，作为主机（CPU）的协处理器。主机和设备各自维护独立的内存空间（host memory 与 device memory），程序通过 CUDA 运行时 API 管理设备内存分配和主机-设备之间的数据传输。
4. **统一内存**：提供托管内存（managed memory）桥接主机和设备内存空间，形成单一、一致的内存镜像，消除显式数据镜像的需求。
5. **Compute Capability**：计算能力是 NVIDIA GPU 硬件版本的标识，由主版本号 X 和次版本号 y 组成（如 X.y），决定设备支持的硬件特性和资源上限。

### 章节内容大纲
1. **CUDA 内存全景图** — 各级内存的位置、作用域、生命周期和典型访问速度对比。
2. **全局内存、常量内存与纹理内存** — 各自的特性和适用场景概述。
3. **主机与设备的分离模型** — Host 代码与 Device 代码的执行分工，Serial 代码在 CPU 执行、Parallel 代码在 GPU 执行的基本模式。
4. **统一内存简介** — 托管内存的动机和基本概念（为后续深入章节铺垫）。
5. **了解 Compute Capability** — 如何查询设备的计算能力，计算能力对硬件功能支持的指导意义。

### 参考来源
- 旧版 Guide: 2.3 (Memory Hierarchy), 2.4 (Heterogeneous Programming)
- 新版 Guide: 5.3 (Memory Hierarchy), 5.4 (Heterogeneous Programming), 5.6 (Compute Capability)

---

## 第4章：NVCC编译模型

### 本章目标
- 理解 NVCC 编译器的工作流程和工作原理
- 掌握离线编译（offline compilation）与即时编译（JIT compilation）的区别
- 理解二进制兼容性（binary compatibility）和 PTX 兼容性
- 学会使用 `-arch`、`-code`、`-gencode` 编译选项生成多架构兼容的代码

### 核心知识点
1. **NVCC 编译流程**：NVCC 将源文件中的设备代码与主机代码分离，设备代码编译为 PTX 汇编或 cubin 二进制，主机代码中的 `<<<...>>>` 语法被替换为 CUDA 运行时函数调用。
2. **PTX（Parallel Thread Execution）**：CUDA 指令集架构的汇编形式，是 GPU 代码的中间表示。PTX 代码在运行时由设备驱动进一步 JIT 编译为目标硬件的二进制代码。
3. **离线编译与 JIT 编译**：离线编译直接生成特定架构的二进制代码（cubin）；JIT 编译在应用加载时将 PTX 代码编译为二进制，允许应用在应用编译时尚不存在的未来硬件上运行。
4. **Fat Binary（胖二进制）**：通过 `-gencode` 选项可在一个可执行文件中嵌入多组二进制代码和 PTX 代码，运行时系统自动选择最匹配的版本执行。
5. **兼容性规则**：二进制兼容性保证从较旧次版本到较新次版本的向前兼容（X.y 的二进制可在 X.z 上运行，其中 z >= y）；PTX 代码向前兼容性更强（为较低计算能力生成的 PTX 可以 JIT 编译到更高计算能力的硬件）。

### 章节内容大纲
1. **NVCC 编译器概述** — NVCC 的角色、命令行基本用法、与标准 C++ 编译器的关系。
2. **编译工作流** — 主机代码与设备代码分离，设备代码编译为 PTX/cubin，主机代码转换的完整流程。
3. **离线编译** — 直接生成目标硬件二进制代码的方式和适用场景。
4. **即时编译（JIT）** — PTX 代码运行时编译机制，计算缓存（compute cache）的作用。
5. **二进制兼容性与 PTX 兼容性** — 兼容性规则详解，`-arch` 和 `-code` 选项的含义。
6. **胖二进制与多架构支持** — `-gencode` 的实际使用示例，`__CUDA_ARCH__` 宏的条件编译。
7. **应用程序兼容性实践** — 如何确保应用在多种 GPU 架构上都能运行。

### 参考来源
- 旧版 Guide: 3.1 (Compilation with NVCC), 3.1.1 (Compilation Workflow), 3.1.1.1 (Offline Compilation), 3.1.1.2 (Just-in-Time Compilation), 3.1.2 (Binary Compatibility), 3.1.3 (PTX Compatibility), 3.1.4 (Application Compatibility)
- 新版 Guide: 6.1 (Compilation with NVCC), 6.1.1 (Compilation Workflow), 6.1.1.1 (Offline Compilation), 6.1.1.2 (Just-in-Time Compilation), 6.1.2 (Binary Compatibility), 6.1.3 (PTX Compatibility), 6.1.4 (Application Compatibility)

---

## 第5章：CUDA运行时——设备内存管理

### 本章目标
- 理解 CUDA 运行时（Runtime）的基本结构和 API 前缀惯例
- 掌握设备内存的分配（cudaMalloc）、释放（cudaFree）和数据传输（cudaMemcpy）
- 学会使用线性内存的二维/三维分配（cudaMallocPitch / cudaMalloc3D）
- 理解运行时初始化机制

### 核心知识点
1. **CUDA 运行时**：实现在 `cudart` 库中，所有入口函数以 `cuda` 为前缀。提供内存管理、数据传输、设备管理等核心功能。
2. **设备线性内存**：通过 `cudaMalloc()` 分配、`cudaFree()` 释放、`cudaMemcpy()` 传输。线性内存分配在统一的地址空间中，允许指针相互引用。
3. **对齐与 Pitch**：`cudaMallocPitch()` 和 `cudaMalloc3D()` 用于二维和三维数组分配，自动确保对齐和 padding 满足合并访问的要求。返回的 pitch（步长）必须用于访问数组元素。
4. **运行时初始化**：CUDA 运行时在第一次调用运行时 API 时自动初始化，为系统中的每个设备创建 CUDA 上下文。
5. **显存地址空间**：从 Compute Capability 6.0 (Pascal) 开始，支持高达 47-49 bit 的地址空间。

### 章节内容大纲
1. **CUDA Runtime 概述** — 运行时库（cudart）的结构、静态链接与动态链接方式、API 命名约定。
2. **设备内存分配与释放** — `cudaMalloc()` 和 `cudaFree()` 的完整使用示例，指针参数模式。
3. **主机与设备间的数据传输** — `cudaMemcpy()` 函数详解、`cudaMemcpyKind` 枚举类型（cudaMemcpyHostToDevice / cudaMemcpyDeviceToHost）。
4. **完整的向量加法程序** — 从分配内存、初始化数据、拷贝到设备、启动内核、拷贝回主机、释放内存的完整流程。
5. **二维/三维内存分配** — `cudaMallocPitch()` 和 `cudaMalloc3D()` 的使用场景和 API 详解，pitch 的概念。
6. **运行时初始化详解** — 默认初始化机制、显式初始化的使用场景。

### 参考来源
- 旧版 Guide: 3.2 (CUDA Runtime), 3.2.1 (Initialization), 3.2.2 (Device Memory)
- 新版 Guide: 6.2 (CUDA Runtime), 6.2.1 (Initialization), 6.2.2 (Device Memory)

---

## 第6章：共享内存与页锁定主机内存

### 本章目标
- 掌握共享内存（shared memory）的声明和使用方法
- 理解静态共享内存与动态共享内存的区别
- 学会使用共享内存优化矩阵乘法等经典算法
- 理解页锁定主机内存（page-locked memory）的优势和使用方法
- 了解零拷贝映射内存（mapped memory）的概念

### 核心知识点
1. **共享内存**：使用 `__shared__` 声明，位于 GPU 芯片上，延迟远低于全局内存。作为用户管理的缓存（scratchpad memory），适合线程块内的数据共享和重用。
2. **共享内存使用模式**：加载数据从全局内存到共享内存 → `__syncthreads()` 同步 → 处理共享内存中的数据 → 再次同步 → 写回全局内存。
3. **静态与动态共享内存**：静态共享内存在编译时声明大小；动态共享内存通过 `extern __shared__` 声明并在内核启动时指定大小。
4. **页锁定主机内存**：通过 `cudaMallocHost()` 分配，确保物理内存页不会被操作系统换出，数据不会被标记为交换空间（swap space），支持异步并发执行。可提升主机与设备间的数据传输带宽。
5. **零拷贝映射内存**：将页锁定主机内存映射到设备地址空间，内核可以直接访问主机内存而无需显式拷贝。在集成 GPU 上（主机和设备内存物理相同），应始终使用映射内存以避免冗余拷贝。

### 章节内容大纲
1. **共享内存基础** — `__shared__` 声明语法、共享内存的生命周期、与全局内存的性能对比。
2. **矩阵乘法——无共享内存版本** — 朴素矩阵乘法的实现，分析其全局内存访问模式的效率问题。
3. **矩阵乘法——使用共享内存优化** — 分块算法（tiling）通过共享内存减少全局内存访问的详细实现，图示分块数据重用模式。
4. **动态共享内存** — `extern __shared__` 声明方式、内核启动时指定共享内存大小、多个动态共享数组时的指针管理。
5. **页锁定主机内存** — `cudaMallocHost()` 和 `cudaFreeHost()` 的使用、性能优势、额外内存开销的考量。
6. **零拷贝映射内存** — `cudaHostAlloc()` 带 `cudaHostAllocMapped` 标志、`cudaHostGetDevicePointer()` 获取设备端指针、适用场景和同步要求。

### 参考来源
- 旧版 Guide: 3.2.4 (Shared Memory), 3.2.5 (Page-Locked Host Memory)
- 新版 Guide: 6.2.4 (Shared Memory), 6.2.6 (Page-Locked Host Memory), 6.2.6.3 (Mapped Memory)

---

## 第7章：流（Streams）、事件（Events）与异步并发执行

### 本章目标
- 理解 CUDA 中同步执行与异步执行的区别
- 掌握流（stream）的创建、销毁和基本使用
- 学会使用事件（event）进行精确的计时和同步
- 理解内核并发执行与数据传输重叠的实现原理
- 掌握默认流与显式流的行为差异

### 核心知识点
1. **异步执行模型**：CUDA 的内核启动是异步的——`<<<...>>>` 语法启动内核后，主机代码立即继续执行，不等待内核完成。同样，许多 GPU-GPU 和 GPU-CPU 的数据传输也可以是异步的。
2. **流（Stream）**：流是在设备上按顺序执行的一系列操作。不同流中的操作可以并发执行。默认流（stream 0）是隐式创建的同步流。
3. **数据传输与内核执行重叠**：使用页锁定内存和非默认流，可以实现主机到设备的数据传输与内核执行并发进行（即"复制与计算重叠"模式）。
4. **事件（Event）**：`cudaEvent_t` 用于流间同步和主机端计时。`cudaEventRecord()` 在流中记录事件，`cudaEventSynchronize()` 在主机端等待，`cudaStreamWaitEvent()` 实现流间依赖。
5. **隐式同步**：当任何非默认流操作有待处理时，默认流会等待这些操作完成。操作同一设备上的 CUDA 访存 API（如 cudaMemcpy 不带 Async 后缀）也会导致隐式同步。

### 章节内容大纲
1. **同步与异步操作** — 同步操作的阻塞特性，异步操作的非阻塞特性与执行时间线。
2. **流的基本概念** — 流的创建（`cudaStreamCreate`）、销毁（`cudaStreamDestroy`），在流中启动内核和拷贝数据。
3. **默认流与显式流** — 默认流的行为特点，显式流的创建和使用，流的并发执行条件。
4. **数据传输与内核重叠** —"复制与计算重叠"模式的实际实现示例，使用 Nsight 或可视化工具观察执行时间线。
5. **事件的使用** — 创建事件（`cudaEventCreate`）、记录事件（`cudaEventRecord`）、同步事件（`cudaEventSynchronize`）、计算时间间隔（`cudaEventElapsedTime`）。
6. **并发内核执行** — 多流中同时启动多个内核的条件和限制，最大并发内核数。
7. **主机端同步** — `cudaDeviceSynchronize()`、`cudaStreamSynchronize()`、`cudaEventSynchronize()` 的使用场景对比。

### 参考来源
- 旧版 Guide: 3.2.6 (Asynchronous Concurrent Execution), 3.2.6.5 (Streams), 3.2.6.6 (Events)
- 新版 Guide: 6.2.8 (Asynchronous Concurrent Execution), 6.2.8.5 (Streams), 相关的 Events 子节

---

## 第8章：多设备系统与统一虚拟地址空间

### 本章目标
- 掌握多 GPU 设备的枚举和选择方法
- 理解设备间点对点（Peer-to-Peer）通信
- 理解统一虚拟地址空间（UVA）的概念和作用
- 学会在多设备环境中分配和管理内存
- 掌握错误检查的基本方法

### 核心知识点
1. **设备枚举**：通过 `cudaGetDeviceCount()` 获取 GPU 数量，`cudaGetDeviceProperties()` 查询设备属性，`cudaSetDevice()` 选择当前操作设备。
2. **设备选择**：每个主机线程可以选择不同的当前设备。CUDA 运行时 API 调用作用于当前选中的设备。
3. **点对点（P2P）通信**：通过 `cudaDeviceEnablePeerAccess()` 启用设备间直接访问，允许一个 GPU 的内核直接读取另一个 GPU 的内存。P2P 内存拷贝（`cudaMemcpyPeer()`）支持设备间直接数据传输。
4. **统一虚拟地址空间**：从 Compute Capability 2.0 开始支持，所有主机内存和设备内存在单一虚拟地址空间中统一编址，通过指针值的地址范围即可判断数据所在位置。这使得 `cudaMemcpy` 可以不再需要指定方向。
5. **错误检查**：大多数 CUDA API 调用返回 `cudaError_t` 类型的错误码。内核启动需要通过 `cudaGetLastError()` 获取启动错误。由于内核异步执行，错误可能在稍后返回，需要通过 `cudaDeviceSynchronize()` 或流同步来捕获。

### 章节内容大纲
1. **设备枚举与查询** — 获取系统中所有 CUDA 设备的数量和属性（计算能力、显存大小、SM 数量等）。
2. **设备选择与上下文** — `cudaSetDevice()` 的使用、不同主机线程的设备选择、设备上下文的基本概念。
3. **点对点通信** — P2P 支持的检测、P2P 内存访问的启用、`cudaMemcpyPeer()` 和 `cudaMemcpyPeerAsync()` 的使用。
4. **统一虚拟地址空间** — UVA 的原理、`cudaPointerGetAttributes()` 查询指针对应的内存位置、UVA 对 `cudaMemcpyDefault` 的支持。
5. **错误检查机制** — `cudaGetLastError()` 获取内核启动错误、运行时 API 返回值检查、编写 `checkCudaErrors()` 宏的最佳实践。
6. **异步错误处理** — 为什么内核错误和异步拷贝错误不会立即返回，如何通过同步点捕获异步错误。

### 参考来源
- 旧版 Guide: 3.2.7 (Multi-Device System), 3.2.8 (Unified Virtual Address Space), 3.2.10 (Error Checking)
- 新版 Guide: 6.2.9 (Multi-Device System), 6.2.10 (Unified Virtual Address Space), 6.2.12 (Error Checking)

---

## 第9章：硬件实现——SIMT架构与Warp调度

### 本章目标
- 理解 NVIDIA GPU 的硬件架构：流多处理器（SM）的可扩展阵列
- 掌握 SIMT（Single-Instruction, Multiple-Thread）架构与 SIMD 的区别
- 理解 Warp 的概念、Warp 调度机制和 Warp Divergence
- 了解硬件多线程（Hardware Multithreading）如何隐藏延迟
- 理解 Volta 架构引入的独立线程调度（Independent Thread Scheduling）的影响

### 核心知识点
1. **SM 架构**：GPU 由可扩展的流多处理器（SM）阵列构成。每个 SM 被设计为并发执行数百个线程。线程块分配到 SM 上，一个 SM 可以同时驻留多个线程块。
2. **SIMT 架构**：SM 以 32 个线程为一组（称为 warp）来创建、管理和调度线程。一个 warp 中的所有线程从相同的程序地址开始执行，但各自拥有独立的指令地址计数器和寄存器状态，可以自由分支和执行。
3. **Warp 调度**：一个 warp 每次执行一条公共指令。当 warp 中的所有 32 个线程在相同执行路径上时达到最高效率。如果线程因数据相关的条件分支产生分歧（warp divergence），warp 会串行执行各条分支路径，禁用不在当前路径上的线程。
4. **SIMT 与 SIMD 的区别**：SIMD 向量组织将 SIMD 宽度暴露给软件（需要编程者显式管理向量操作），而 SIMT 指令指定单个线程的执行和分支行为，程序员可以从单线程角度编写代码。
5. **硬件多线程**：每个 warp 的执行上下文（程序计数器、寄存器等）在 warp 的整个生命周期中保持在芯片上。从一个执行上下文切换到另一个上下文的成本为零，warp 调度器可以在每个指令发射周期选择就绪的 warp 执行指令。
6. **独立线程调度（Volta 及更新架构）**：从 Volta 架构开始，GPU 为每个线程维护独立的程序计数器和调用栈，允许以单线程粒度进行调度，支持线程间更灵活的并发和子 warp 级别的发散/重汇聚。

### 章节内容大纲
1. **GPU 硬件拓扑** — SM 阵列、GPC（Graphics Processing Cluster）的概念、内存分区的物理布局。
2. **Warp 的概念** — Warp 的 32 线程固定大小、线程块如何划分为 warp（按线程 ID 连续递增分组）、warp 是调度的基本单位。
3. **SIMT 执行模型** — 单指令多线程的执行方式、warp divergence 对性能的影响（数据依赖分支导致串行化）、分支发散仅发生在 warp 内部（不同 warp 独立执行）。
4. **SIMT 与 SIMD 对比** — 关键差异分析：SIMD 向软件暴露并行宽度 vs. SIMT 隐藏并行宽度，各自对编程模型的影响。
5. **硬件多线程与延迟隐藏** — 零开销上下文切换的机制、warp 就绪条件（操作数可用性）、需要多少活跃 warp 才能隐藏各种延迟。
6. **独立线程调度** — Volta 架构及以后的变化：每个线程独立 PC 和调用栈、子 warp 发散与重汇聚、对 warp 同步代码（warp-synchronous programming）的兼容性影响。
7. **寄存器文件与共享内存对驻留 warp 数的限制** — 内核的寄存器和共享内存使用如何影响 SM 上能同时驻留的 block 和 warp 数量。

### 参考来源
- 旧版 Guide: 4 (Hardware Implementation), 4.1 (SIMT Architecture), 4.2 (Hardware Multithreading)
- 新版 Guide: 7 (Hardware Implementation), 7.1 (SIMT Architecture), 7.2 (Hardware Multithreading)

---

## 第10章：性能优化基础——占用率与内存访问优化

### 本章目标
- 理解 CUDA 性能优化的四大基本策略
- 掌握占用率（Occupancy）的概念、计算方法和优化手段
- 理解全局内存的合并访问（Memory Coalescing）机制
- 学会避免共享内存 Bank Conflict
- 了解如何使用 Occupancy Calculator API 辅助优化

### 核心知识点
1. **性能优化四大策略**：（1）最大化并行执行以提高利用率；（2）优化内存使用以获得最大内存吞吐量；（3）优化指令使用以获得最大指令吞吐量；（4）最小化内存抖动（memory thrashing）。
2. **占用率（Occupancy）**：占用率是每个 SM 上活跃 warp 数与最大可能 warp 数之比。更高占用率意味着 SM 有更多就绪 warp 可以隐藏延迟。占用率受每个线程块线程数、每个线程使用的寄存器数量、每个块使用的共享内存量的影响。
3. **全局内存合并访问**：全局内存通过 32、64 或 128 字节的内存事务访问。当 warp 中所有线程访问的地址落在尽可能少的内存事务内时，访问被"合并"（coalesced），达到最高带宽利用率。访问模式越分散，吞吐量损失越大。
4. **对齐要求**：全局内存访问的数据类型大小（1/2/4/8/16 字节）必须与其自然对齐边界对齐，否则会编译为多条指令并降低合并效率。`cudaMalloc` 返回的地址始终对齐到至少 256 字节。
5. **共享内存 Bank Conflict**：共享内存被分为 32 个 bank（内存库），同一 warp 的多个线程同时访问同一 bank 的不同地址时会发生 bank conflict，导致访问串行化。特殊情况：同一 bank 的同一地址可以广播，不会产生冲突。

### 章节内容大纲
1. **性能优化概述** — 四步优化策略全景、性能瓶颈分析方法论、使用 CUDA Profiler 识别性能限制器。
2. **占用率深入理解** — 占用率的定义和物理意义、寄存器压力与占用率的权衡、block 大小选择与占用率的关系。
3. **Occupancy Calculator 使用** — `cudaOccupancyMaxActiveBlocksPerMultiprocessor` API 的使用、`cudaOccupancyMaxPotentialBlockSize` 自动计算最优配置、电子表格版占用率计算器作为学习工具。
4. **全局内存访问模式** — 内存事务概念、合并访问与分散访问的吞吐量对比（一个 warp 访问需要 1 个事务 vs. 32 个事务）、数据结构对齐与 Padding 策略。
5. **二维数组访问优化** — `cudaMallocPitch()` 确保行宽为 warp size 倍数以实现合并访问的机制、padding 示例。
6. **共享内存 Bank 架构** — Bank 分布规则（连续的 4 字节字分配到连续的 bank）、Bank Conflict 的类型（2-way / 4-way / 8-way）、避免 Bank Conflict 的常用技巧（padding、交错访问模式）。

### 参考来源
- 旧版 Guide: 5.1 (Overall Performance Optimization Strategies), 5.2 (Maximize Utilization), 5.2.3 (Multiprocessor Level), 5.2.3.1 (Occupancy Calculator), 5.3 (Maximize Memory Throughput), 5.3.2 (Device Memory Accesses — Global Memory 部分)
- 新版 Guide: 8.1 (Overall Performance Optimization Strategies), 8.2 (Maximize Utilization), 8.2.3 (Multiprocessor Level), 8.2.3.1 (Occupancy Calculator), 8.3 (Maximize Memory Throughput), 8.3.2 (Device Memory Accesses)

---

## 第11章：性能优化进阶——Warp Divergence与指令吞吐量

### 本章目标
- 深入理解 Warp Divergence 对性能的影响及缓解策略
- 掌握主机与设备间数据传输优化技巧
- 理解指令吞吐量（instruction throughput）的考量
- 学会应用层面的并行优化策略
- 建立完整的性能调优思维框架

### 核心知识点
1. **Warp Divergence 优化**：数据依赖的 if-else 分支导致 warp 内部线程走不同路径时，warp 必须串行执行各分支（先执行 if 路径，禁用走 else 的线程；再执行 else 路径，禁用走 if 的线程）。尽量减少 warp 内部的分支分歧，或将分支条件设计为与 warp 边界对齐。
2. **主机-设备数据传输优化**：（1）尽量减少主机与设备间的数据传输——将更多代码移到设备端，即使这意味着运行不足以完全高效利用设备的 kernel；（2）将多次小传输合并为一次大传输以减少传输开销；（3）使用页锁定内存以获得更高的传输带宽。
3. **指令吞吐量优化**：不同类型的算术指令有不同的吞吐量（如单精度 vs 双精度、普通算术 vs 超越函数）。在性能敏感代码中优先使用高吞吐量指令，避免低吞吐量操作成为瓶颈。
4. **应用层并行策略**：不同线程块间如果需要共享数据，应通过全局内存和两次独立的 kernel 启动（一次写一次读），但这比块内共享内存协作效率低得多。应尽量将需要线程间通信的计算组织在同一个线程块内。
5. **设备级并行**：多个 kernel 可以通过流在同一设备上并发执行，充分利用 GPU 的并行能力。

### 章节内容大纲
1. **Warp Divergence 深度分析** — 不同分支模式的性能分析、避免 Divergence 的代码重构技巧、利用 warp 级别操作（warp shuffle）替代分支的方法。
2. **数据传输优化策略** — 减少 Host-Device 传输的技术（在设备端创建中间数据结构、合并小传输）、页锁定内存的性能对比数据。
3. **指令吞吐量考量** — 常见算术指令的吞吐量对比、使用内建函数的时机（如 `__sinf()` vs `sinf()`）、控制流指令（如 `__syncthreads()`）的开销。
4. **应用层与设备层并行** — 从应用宏观架构到 SM 微观执行的各层级并行最大化策略总结。
5. **调试与性能分析工具简介** — CUDA Profiler、Nsight Systems、Nsight Compute 的基本使用概要。
6. **综合优化实例** — 以一个完整小项目为例，展示从识别瓶颈到逐步优化的全过程。

### 参考来源
- 旧版 Guide: 5.1 (Overall Performance Optimization Strategies), 5.2.1 (Application Level), 5.2.2 (Device Level), 5.3.1 (Data Transfer between Host and Device), 5.4 (Maximize Instruction Throughput)
- 新版 Guide: 8.1 (Overall Performance Optimization Strategies), 8.2.1 (Application Level), 8.2.2 (Device Level), 8.3.1 (Data Transfer between Host and Device), 8.4 (Maximize Instruction Throughput)

---

## 进阶章节（基于 CUDA 13.0 新增内容）

以下章节将由后续部分详细编写，这里仅列出摘要供了解课程全貌：

1. **线程块集群（Thread Block Clusters，CC 9.0+）** — GPU Processing Cluster (GPC) 级别的调度新层次，集群内块的硬件同步与分布式共享内存。
2. **Blocks as Clusters（CC 12.0+）** — `__block_size__` 属性与按集群数启动 kernel 的新编程模型。
3. **异步 SIMT 编程模型** — `cuda::barrier` 和 `cuda::pipeline` 同步原语，`cuda::memcpy_async` 异步数据拷贝，异步操作的线程作用域（thread scope）。
4. **分布式共享内存（Distributed Shared Memory）** — 跨线程块集群的直接共享内存访问。
5. **Tensor Memory Accelerator (TMA)** — 硬件加速的异步数据拷贝与时域交换模式（swizzle）。
6. **内存同步域（Memory Synchronization Domains）** — 更精细的同步域管理。
7. **统一内存编程进阶** — Page Migration 引擎优化、`cudaMemAdvise` 和 `cudaPrefetchAsync` 的使用。
8. **CUDA Graphs** — 图编程模型及其与流的互操作。
9. **虚拟内存管理 API** — 细粒度的虚拟地址空间管理。
10. **流序内存分配器（Stream Ordered Memory Allocator）** — 基于流的异步内存分配/释放。

---

> **附录说明**：  
> 本核心课程大纲涵盖了 CUDA Programming Guide 11.5.1 版本与 13.0 版本中**共同存在且内容基本一致**的基础知识部分。  
> 两版 Guide 在核心概念的描述上保持一致（13.0 重编号了章节，新增了 Changelog 章节和新特性），本大纲在引用时同时标注了新旧版本的章节编号以便对照查阅。
