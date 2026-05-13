# SGEMM CUDA 优化实验参考代码

本目录包含矩阵乘法（SGEMM）CUDA 性能优化实验的起始代码和参考解答。

## 文件说明

| 文件 | 说明 |
|------|------|
| `sgemm_common.h` | 公共工具函数（计时、错误检查、矩阵初始化、验证等） |
| `labN_start.cu` | 实验 N 的起始代码（包含 TODO 提示） |
| `labN_solution.cu` | 实验 N 的参考解答（完整实现） |
| `CMakeLists.txt` | CMake 构建配置 |

## 性能总览

以下是在 NVIDIA A6000 (Ampere, CC 8.6) 上矩阵大小 4096x4096 的性能数据：

| 实验 | 优化技术 | GFLOPS/s | 相对 cuBLAS |
|------|---------|----------|------------|
| 1 | 朴素矩阵乘法 | 309.0 | 1.3% |
| 2 | 全局内存合并访问 | 1986.5 | 8.5% |
| 3 | 共享内存缓存分块 | 2980.3 | 12.8% |
| 4 | 一维 Block Tile | 8474.7 | 36.5% |
| 5 | 二维 Block Tile | 15971.7 | 68.7% |
| 6 | 向量化内存访问 | 18237.3 | 78.4% |
| 7 | 参数自动调优 | 19721.0 | 84.8% |
| 8 | Warp Tile 优化 | 21779.3 | 93.7% |
| cuBLAS | NVIDIA 官方库 | 23249.6 | 100.0% |

## 构建与运行

### 环境要求

- NVIDIA GPU（Compute Capability 6.0 以上，推荐 Ampere+）
- CUDA Toolkit 11.0+
- CMake 3.19+
- C++17 编译器
- cuBLAS 库（随 CUDA Toolkit 安装）

### 构建所有实验

```bash
cd code/labs
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

如果你的 GPU 不是 A6000（Compute Capability 8.6），在 cmake 时指定：
```bash
cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_COMPUTE_CAPABILITY=80 ..
# 80 for A100, 86 for A6000/RTX3090, 89 for RTX4090, 90 for H100
```

### 运行单个实验

```bash
# 运行实验1的参考解答
./lab1_solution

# 运行实验1的起始代码
./lab1_start
```

## 代码结构

每个实验的起始代码（`labN_start.cu`）包含：
- 完整的 main 函数和数据准备代码
- 用 TODO 标记的 kernel 实现区域
- 编译和运行所需的全部框架代码

每个实验的参考解答（`labN_solution.cu`）包含：
- 完整的 kernel 实现
- 基准测试和正确性验证
- 性能输出

### 实验递进关系

```
实验1 (朴素实现)
  └→ 实验2 (合并访问)         ← 修改线程映射
      └→ 实验3 (共享内存)      ← 引入 SMEM 缓存
          └→ 实验4 (1D Tile)   ← 每线程多结果
              └→ 实验5 (2D Tile) ← Register Blocking
                  └→ 实验6 (向量化) ← float4 访问
                      └→ 实验7 (调优) ← Warp Tile 雏形 + 参数搜索
                          └→ 实验8 (Warp Tile) ← 显式 Warp Tiling
```

## 常见问题

### Q: 我的 GPU 不是 A6000，性能数据对不上怎么办？

不同 GPU 的峰值性能不同。请参考以下峰值：
- A100 (80GB): ~19.5 TFLOPS FP32
- A6000: ~38.7 TFLOPS FP32 (但不含 Tensor Core)
- RTX 4090: ~82.6 TFLOPS FP32 (含 Tensor Core)
- H100: ~67 TFLOPS FP32 (不含 Tensor Core)

如果性能差距较大，请检查：
1. CUDA Compute Capability 设置是否正确
2. 是否以 Release 模式编译
3. 是否开启了 ECC 内存（会降低 GMEM 带宽约 10%）

### Q: 编译时出现 "static_assert failed" 错误？

这说明你选择的 Tile 参数不满足约束条件。检查：
- `NUM_THREADS * 4` 是否整除 BK 和 BN
- `BM * BK` 是否整除 `4 * NUM_THREADS`

### Q: 如何在 Nsight Compute 中分析性能？

```bash
ncu --set full -o profile_report ./lab8_solution
```

重点关注指标：
- Memory Workload Analysis
- Scheduler Statistics (Warp Stall Reasons)
- Speed of Light (compute vs memory bound)

## 参考资料

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) - Simon Boehm
- [SGEMM_CUDA 源代码](https://github.com/siboehm/SGEMM_CUDA)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
