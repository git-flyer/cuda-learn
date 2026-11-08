# CUDA Reduce Sum

计算一维连续 FP32 数组的和：`output[0] = sum(input[0:n])`，空数组的和为零。
这是整个数组归约到一个标量，不是按行/列归约；输入、累加器、输出都使用 `float`。

## 来源和修正

主要从 `cuda_code/course4/reduce_v4.cu` 的 grid-stride + warp shuffle 写法整理，
共享内存树形对照版借鉴 `course4/reduce_v2.cu`。原目录保留，便于对照学习。

| 原实现 | 检查到的问题 | 本目录的处理 |
| --- | --- | --- |
| `course4/reduce.cu` | 使用固定全局 `N`，不能表达每轮输入的实际长度 | 每轮显式传入当前长度 |
| `course4/reduce_v1/v2/v3.cu` | 第二轮有 512 个 partial，却启动 512 个线程、每线程读两个数，后半段读取越界；尾部也缺少检查 | 每次加载检查当前长度，固定线程数，逐轮归约 |
| `course4/reduce_v3.cu` | `int v` 截断浮点中间结果，全 1 测试掩盖错误 | 保持 FP32，加入带小数和负数的测试 |
| `course4/reduce_v4.cu` | 第二轮线程数等于第一轮 block 数；可能超过 1024，也可能出现不完整 warp | 每轮固定 256 个线程，数据不足的 lane 输入零 |
| `course2/reduce_smem.cu` | 输出只分配 1 个数却按 block 写入；共享数组只有 256 项却启动 512 个线程 | 未直接搬入该实现，采用独立的有效对照版 |

## 文件

- `reduce.cuh`：核心 kernel、workspace 大小计算和异步启动接口。
- `reduce.cu`：CPU double 参考值、GPU 正确性检查、两种实现的计时。

## 实现过程

### 1. 每个线程先在寄存器中累加

一个 block 有 256 个线程，一轮处理 1024 个数。线程 `t` 读取：

```text
input[block * 1024 + t]
input[block * 1024 + t + 256]
input[block * 1024 + t + 512]
input[block * 1024 + t + 768]
```

同一次加载中，相邻线程访问相邻元素，适合合并访存。
四个独立累加器在循环末尾合并，给编译器提供更多指令并行机会。
这里是展开后的标量加载，不是 `float4`，因此不额外要求输入地址满足 16 字节对齐。

最多启动 4096 个 block；更多输入通过 `gridDim.x * 1024` 步长继续读取。
这样既能处理大数组，也把第一轮 partial 和临时显存的规模限制住。
256 线程、4 个累加器、4096 个 block 都是可调起点，尚未针对具体 GPU 调优。

### 2. 每个 warp 用 shuffle 得到局部和

```cpp
for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
}
```

每一步让当前 lane 获取编号比自己大 `offset` 的 lane 的值，然后相加。
经过 `16 → 8 → 4 → 2 → 1` 五步，lane 0 得到该 warp 的 32 个线程的和。
**只有 lane 0 的结果是完整 warp 和**，其他 lane 不用于输出。

使用全掩码的前提是整个 warp 都执行这些调用。
所以越界线程只跳过加载、保留零，不能提前 `return`；不足 32 个输入也照样启动完整 warp。
这符合 [NVIDIA 的 warp 同步与 mask 约束](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#warp-sync-intrinsic-constraints)。

### 3. 共享内存连接 block 内的 8 个 warp

每个 warp 的 lane 0 将结果写入 `warp_sums[warp_id]`。
一次 `__syncthreads()` 后，warp 0 的前 8 个 lane 读这 8 个数，另外 24 个 lane 填零，
再执行一次 warp reduce；最终由线程 0 写出一个 partial。

| block 内归约方式 | 共享内存 | block 级同步次数 |
| --- | --- | --- |
| `Shared` 对照版 | 256 个 float，1024 字节 | 1 次初始化同步 + 8 层树形同步 = 9 次 |
| `Shuffle` 优化版 | 8 个 float，32 字节 | 1 次 |

两版使用相同的寄存器累加和加载布局，主要比较 block 内通信开销。
shuffle 优化降低了共享内存访问和同步次数，实际耗时仍受数据量、带宽和 GPU 架构影响。
参考：[NVIDIA 的多级 reduction 说明](https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/)。

### 4. 多次 kernel launch 完成全局归约

不同 block 不能依靠 `__syncthreads()` 互相同步。每轮将 partial 写回显存，
下一轮在同一 stream 中读取，直到只剩一个结果。例如：

```text
N = 1,000,003
1,000,003 个输入 → 977 个 partial → 1 个结果

N = 16,777,216
16,777,216 个输入 → 4096 个 partial → 4 个 partial → 1 个结果
```

中间结果在 workspace 两半之间交替写入，避免同一轮不同 block 读写互相覆盖。
最后一轮直接写用户提供的 `output`。这里不用全局 `atomicAdd`，不需要预先清零输出，
求和次序也不依赖 block 完成顺序。空输入单独用 `cudaMemsetAsync` 写零。

## 复用接口

```cpp
#include "reduce.cuh"

// d_input 已有 n 个 float，d_output 至少容纳 1 个 float，stream 已创建。
const size_t count = cuda_learn::reduce::workspace_elements(n);
float* workspace = nullptr;
if (count) cudaMalloc(reinterpret_cast<void**>(&workspace), count * sizeof(float));
cudaError_t status = cuda_learn::reduce::reduce_sum(
    d_input, d_output, n, workspace,
    cuda_learn::reduce::Method::Shuffle, stream);
// 实际使用时像 reduce.cu 一样检查每个 CUDA 返回值。
// 在同一 stream 上继续消费结果，或同步后拷回 CPU；完成后才能释放 workspace。
```

- workspace 大小接口返回 **float 个数**；最大 `2 * 4096 * 4 = 32768` 字节。
- `n <= 1024` 无需 workspace；`n == 0` 时 input 也可为空，但 output 必须有效。
- input、output、workspace 不得重叠，所有内存须在 stream 完成前有效。
- 连续调用可以复用 workspace；并发 stream 应分别使用自己的 workspace 和 output。
- kernel 启动函数不分配显存、不进行 CPU 同步；异步执行错误在同步时检查。
- 未做补偿求和。浮点加法不满足结合律，结果可能与 CPU 顺序相加或另一种归约树不同。
  严重抵消、大动态范围、溢出和非有限值不属于本示例的精度保证范围。

## 编译和验证

需要 Linux/Windows 上的 CUDA Toolkit 和 NVIDIA GPU。本次开发环境为 Apple Silicon，
没有 `nvcc` 和 NVIDIA GPU，因此尚未完成真实 CUDA 编译、GPU 正确性检查及性能测量。
本地已通过基于临时 CUDA 声明的 Clang 主机端/设备端语法检查，以及 2056 个长度的
独立 CPU 归约模拟和索引覆盖检查；这些检查不能替代真实 CUDA 编译和 GPU 同步检查。

在 `cuda-learn` 根目录执行（`86` 请换成目标 GPU 对应的计算能力）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target reduce -j
./build/reduce --test
ctest --test-dir build -R '^reduce_correctness$' --output-on-failure
./build/reduce 16777216 100
```

也可以单独编译：

```bash
nvcc -O3 -std=c++14 -arch=sm_86 kernels/reduce/reduce.cu -o /tmp/cuda-reduce
/tmp/cuda-reduce --test
```

测试覆盖空数组、单元素、warp/block/tile 边界、非 2 的幂、三轮归约、grid-stride 尾部，
包括全 1、零、负数、小数、随机正负数和仅最后一项非零的数据。
可精确表示的测试要求零误差，随机测试用 CPU double 参考值和绝对/相对误差阈值。
两种实现均检查输出复用、重复调用和非默认 stream；失败返回非零退出码。

建议在 GPU 机器上额外运行 CUDA 的检查工具：

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/reduce --test
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/reduce --test
compute-sanitizer --tool synccheck --error-exitcode 1 ./build/reduce --test
```

计时先预热 10 次，再用 CUDA Events 测多次完整归约的平均耗时，包含所有归约轮次，
不包含内存分配、CPU 求和及主机/设备拷贝。极小输入可能受 launch 和主机提交间隙影响。
输出的 `input_effective_bandwidth` 是原始输入字节数 / 总时间，不包含 partial 流量，
也不等同于实际 DRAM 带宽；重复读取可能命中缓存。
`shared/shuffle time ratio` 仅与本目录的对照版比较，不代表相对原始 v4 或 CUB 的加速比。
