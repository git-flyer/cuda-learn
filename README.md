# cuda_code

我的 CUDA 学习笔记仓库，记录 CUDA 基础语法、GPU 架构知识以及各类优化示例代码。

---

## 目录结构

```
cuda_code/
└── matmul/                  # 矩阵乘法系列示例
    ├── naive_matmul.cu              # 朴素矩阵乘法
    ├── shared_memory_matmul.cu     # 共享内存分块优化
    ├── thread_tile_matmul.cu       # Thread Tile（寄存器分块）优化
    └── Makefile
```

---

## matmul 系列

矩阵乘法 C = A × B（A: M×K，B: K×N，C: M×N）是 CUDA 优化的经典教学案例，
三个版本展示了从"能跑"到"跑快"的逐步优化过程。

### 1. 朴素矩阵乘法 (`naive_matmul.cu`)

**每个线程计算 C 中的一个元素。**

- 访存模式：每次乘加都直接读取**全局显存**（global memory）。
- 瓶颈：全局显存延迟约 200 cycle，内存带宽利用率低，算术强度 O(1) FLOP/Byte。
- 适合入门，理解线程/块/网格的基本映射方式。

关键参数：`BLOCK_SIZE = 32`

### 2. 共享内存分块矩阵乘法 (`shared_memory_matmul.cu`)

**利用共享内存（on-chip SRAM）缓存 tile，降低全局显存访问次数。**

优化思路：
1. 将矩阵按 `TILE_SIZE × TILE_SIZE` 分块。
2. 每个线程块协作地将 A_tile 和 B_tile 从全局显存载入**共享内存**。
3. 用 `__syncthreads()` 同步后，全程在共享内存中完成本 tile 的计算。
4. 沿 K 方向滑动 tile，累加所有部分和。

- 每个全局显存元素被重用 `TILE_SIZE` 次。
- 算术强度从 O(1) 提升至 O(TILE_SIZE) FLOP/Byte。

关键参数：`TILE_SIZE = 32`

### 3. Thread Tile（寄存器分块）矩阵乘法 (`thread_tile_matmul.cu`)

**让每个线程计算 TM × TN 个输出元素，充分利用寄存器，进一步提升算术强度。**

优化思路：
1. 每个线程块处理 `BM × BN` 大小的输出子块，K 方向步长为 `BK`。
2. 线程块内共 `(BM/TM) × (BN/TN)` 个线程，每个线程负责 `TM × TN` 个输出元素。
3. 内层循环中，每个线程将共享内存中的一列（`regA[TM]`）和一行（`regB[TN]`）
   读入**寄存器**，然后做外积，累加到 `accC[TM][TN]`（纯寄存器操作）。
4. 共享内存访问次数从 O(BK × BM × BN) 降至 O(BK × (BM + BN))，
   寄存器复用率大幅提升。

关键参数：`BM=64, BN=64, BK=8, TM=8, TN=8`（每个线程块 64 个线程，每线程 64 次输出）

---

## 编译与运行

```bash
cd matmul

# 根据你的 GPU 修改 Makefile 中的 -arch 参数
# Ampere: sm_80  Turing: sm_75  Volta: sm_70  Pascal: sm_60

make all          # 编译全部
./naive_matmul
./shared_memory_matmul
./thread_tile_matmul
```

---

## 学习路线

```
朴素版（理解基本 CUDA 线程模型）
  ↓
共享内存版（理解 tiling + __syncthreads + 访存优化）
  ↓
Thread Tile 版（理解寄存器分块 + 算术强度 + 外积累加）
  ↓
（后续）向量化加载（float4）、warp-level 优化、Tensor Core（wmma）...
```
