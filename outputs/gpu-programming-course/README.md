# GPU编程：从零开始的CUDA实战课程

<p align="center">
  <strong>GPU Programming: CUDA Hands-on Course from Scratch</strong>
</p>

## 项目简介

本课程是面向零基础学习者的GPU编程入门课程，以 **NVIDIA CUDA** 为核心，系统讲解GPU并行编程的理论与实践。课程内容源自 [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 的核心章节，并结合经典的矩阵乘法优化案例，帮助学习者从"写出第一个CUDA内核"到"能够进行性能调优"。

<strong>关键词</strong>：GPU编程、CUDA、并行计算、性能优化、NVIDIA

## 课程特色

- <strong>扎实的理论基础</strong>：以NVIDIA官方CUDA Programming Guide为蓝本，覆盖编程模型、内存层次、硬件架构、性能优化等核心主题
- <strong>渐进式实验设计</strong>：以矩阵乘法（SGEMM）的逐步优化为主线，从朴素实现到接近cuBLAS性能，每一步优化对应具体的CUDA概念
- <strong>动手实践导向</strong>：每个章节配备可运行的代码示例和实验，配套在线评测系统（需NVIDIA GPU）提供性能反馈
- <strong>开源社区规范</strong>：遵循开源社区课程规范，支持社区贡献和协作创作

## 学习路线

本课程分为 <strong>基础篇</strong> 和 <strong>进阶篇</strong> 两部分：

### 基础篇（第1-11章）

| 阶段 | 章节 | 内容 |
|------|------|------|
| 入门 | 第1章 | GPU计算与CUDA入门 |
| 编程模型 | 第2章 | 内核函数与线程层次结构 |
| | 第3章 | 内存层次结构与非对称编程 |
| 工具链 | 第4章 | NVCC编译模型 |
| | 第5章 | 设备内存管理 |
| 内存优化 | 第6章 | 共享内存与页锁定内存 |
| 并发 | 第7章 | 流、事件与异步并发执行 |
| | 第8章 | 多设备系统与统一虚拟地址空间 |
| 硬件与性能 | 第9章 | SIMT架构与Warp调度 |
| | 第10章 | 性能优化基础——占用率与内存访问优化 |
| | 第11章 | 性能优化进阶——指令吞吐量与Warp级优化 |

### 进阶篇（第12-17章）

涵盖Thread Block Clusters、异步SIMT编程模型、Tensor Memory Accelerator (TMA)、CUDA Graphs高级特性、Cooperative Groups扩展、现代GPU架构概览等。

### 实验

8个矩阵乘法优化实验贯穿整个课程，从全局内存合并访问到Warp Tiling，性能逐步提升至cuBLAS的90%以上。

## 前置要求

- C/C++ 编程基础（了解指针、函数、结构体）
- 基本的计算机体系结构知识（了解CPU、内存、缓存的概念）
- 一台配备NVIDIA GPU的计算机（推荐Compute Capability 6.0+，课程实验以RTX 3060 / sm_86为目标）
- 已安装CUDA Toolkit（推荐11.x或更新版本）

## 使用说明

### 在线阅读

本课程使用 Docsify 构建在线阅读站点。克隆仓库后在 `docs/` 目录下启动：

```bash
npm install -g docsify-cli
docsify serve docs/
```

### 运行代码

所有代码示例位于 `code/` 目录中，按章节组织。编译和运行示例：

```bash
cd code/chapter2
nvcc -o vector_add vector_add.cu
./vector_add
```

### 在线评测

课程配套的在线评测系统位于 `outputs/eval_system/`，支持代码提交、自动编译、NCU性能分析和报告下载。详见该目录下的 `design.md`。

## 项目结构

```
gpu-programming-course/
├── README.md                    # 本文件
├── docs/                        # 课程文档
│   ├── _sidebar.md              # 侧边栏导航
│   ├── 前言.md                  # 前言
│   ├── images/                  # 图片资源
│   ├── chapter1/ ~ chapter11/   # 基础篇章节
│   └── advanced-chapter1/ ~ 6/  # 进阶篇章节
├── code/                        # 随章代码
├── Extra-Chapter/               # 补充资料与参考答案
└── outputs/                     # 规划文档与评测系统
```

## 贡献指南

本课程是一个开源社区项目，欢迎贡献！

- 发现错误或改进建议：提交 Issue
- 贡献内容：Fork 本仓库，提交 Pull Request
- 参与讨论：欢迎在 GitHub Discussions 中交流

## 许可证

本课程采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可证。代码示例采用 MIT 许可证。

## 致谢

- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) 提供核心参考资料
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) 提供矩阵乘法优化案例
- Hello-Agents 开源课程提供结构参考

## 参考文献

[1] NVIDIA Corporation. *CUDA C++ Programming Guide*, Version 11.5.1, 2021.
[2] NVIDIA Corporation. *CUDA C++ Programming Guide*, Version 13.0, 2025.
[3] Simon Boehm. *How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog*, 2022.
