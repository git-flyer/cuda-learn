# CUDA Learn

CUDA 学习仓库，包含基础语法、GPU 架构知识和各类优化示例代码。

---

## 目录结构

```
cuda-learn/
├── kernels/
│   ├── matmul/              # 矩阵乘法系列
│   │   ├── naive_matmul.cu              # 朴素版
│   │   ├── shared_memory_matmul.cu      # 共享内存优化
│   │   ├── thread_tile_matmul.cu        # 寄存器分块优化
│   │   └── double_buffering_matmul.cu   # 双缓冲优化
│   ├── transpose/            # 矩阵转置
│   │   └── matrix_transpose.cu
│   └── histogram/           # 直方图统计
│       └── histogram_shared.cu
├── CMakeLists.txt
└── build.sh
```

---

## 示例详解

### matmul 系列

矩阵乘法 C = A × B（A: M×K，B: K×N，C: M×N）是 CUDA 优化的经典教学案例。

#### 1. 朴素矩阵乘法 (`naive_matmul.cu`)

每个线程计算 C 中的一个元素，直接访问全局显存。

- 访存模式：每次乘加都读取全局显存
- 瓶颈：全局显存延迟约 200 cycle，内存带宽利用率低
- `BLOCK_SIZE = 32`

#### 2. 共享内存分块 (`shared_memory_matmul.cu`)

利用共享内存（on-chip SRAM）缓存 tile，降低全局显存访问次数。

- 将矩阵按 `TILE_SIZE × TILE_SIZE` 分块
- 每个线程块协作地将 A_tile 和 B_tile 载入共享内存
- 全局显存元素重用 `TILE_SIZE` 次
- `TILE_SIZE = 32`

#### 3. Thread Tile 寄存器分块 (`thread_tile_matmul.cu`)

每个线程计算 TM × TN 个输出元素，充分利用寄存器。

- 线程块处理 `BM × BN` 输出子块，K 方向步长为 `BK`
- 每线程负责 `TM × TN` 个输出元素
- 共享内存访问次数从 O(BK × BM × BN) 降至 O(BK × (BM + BN))
- `BM=64, BN=64, BK=8, TM=8, TN=8`

#### 4. 双缓冲 (`double_buffering_matmul.cu`)

通过 double buffering 消除线程块等待数据加载的空泡期，进一步提升occupancy。

---

### transpose (`matrix_transpose.cu`)

矩阵转置，演示 bank conflict 优化和共享内存访问模式设计。

---

### histogram (`histogram_shared.cu`)

利用共享内存进行并行直方图统计，减少全局原子操作竞争。

---

## 编译与运行

```bash
mkdir build && cd build
cmake .. && make

# 运行各示例
./naive_matmul
./shared_memory_matmul
./thread_tile_matmul
./double_buffering_matmul
./matrix_transpose
./histogram_shared
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
双缓冲版（理解 double buffering + occupancy 优化）
  ↓
（扩展）向量化加载（float4）、warp-level 优化、Tensor Core（wmma）...
```
