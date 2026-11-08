#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>

// 从 cuda_code/course4/reduce_v4.cu 整理而来：FP32 一维数组 -> FP32 标量。
namespace cuda_learn {
namespace reduce {

constexpr int kBlockSize = 256;  // 一个 block 启动 256 个线程。
constexpr int kWarpSize = 32;    // 一个 warp 按 32 个线程组织。
constexpr int kItemsPerThread = 4;    // 每个线程在一次外层循环中处理 4 个元素。
constexpr std::size_t kTileSize = kBlockSize * kItemsPerThread;  // 一个 block 一次处理 256 × 4 = 1024 个元素，把这批数据称为一个 tile。
// 教学版的可调参数，不代表所有 GPU 上的最优配置。
constexpr unsigned int kMaxBlocks = 4096;   // 一轮 kernel 最多启动 4096 个 block。超出的数据由已有线程循环处理。
static_assert(kBlockSize >= kWarpSize && kBlockSize <= 1024,
              "Block size must be between one warp and 1024 threads");
static_assert(kBlockSize % kWarpSize == 0, "Full warps are required");
static_assert((kBlockSize & (kBlockSize - 1)) == 0,
              "The shared-memory tree requires a power-of-two block size");
static_assert(kItemsPerThread == 4, "Update the accumulator merge when changing this");
static_assert(kMaxBlocks > 0, "At least one block is required");

enum class Method { Shared, Shuffle };

// 所有 32 个 lane 都必须参与；只有 lane 0 的返回值是整个 warp 的和。
__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// 只有 thread 0 的返回值有效。每个 block 只需一次 __syncthreads()。
__device__ __forceinline__ float block_sum_shuffle(float value) {
  __shared__ float warp_sums[kBlockSize / kWarpSize];
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  value = warp_sum(value);
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();  // 发布各 warp 的局部和，再由 warp 0 读取。

  if (warp == 0) {
    // 8 个有效值 + 24 个零，整个 warp 仍然执行相同的 shuffle。
    value = lane < kBlockSize / kWarpSize ? warp_sums[lane] : 0.0f;
    value = warp_sum(value);
  }
  return value;
}

// 对照版：与 shuffle 版使用相同的加载方式，只改变 block 内的归约。
__device__ __forceinline__ float block_sum_shared(float value) {
  __shared__ float values[kBlockSize];
  const int tid = threadIdx.x;
  values[tid] = value;
  __syncthreads();
#pragma unroll
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) values[tid] += values[tid + stride];
    __syncthreads();
  }
  return values[0];
}



template <bool UseShuffle>
__global__ void reduce_pass(const float* __restrict__ input,
                            float* __restrict__ partials, std::size_t n) {
  // 四条独立累加链，降低单条浮点加法依赖链对指令并行度的限制。
  float sums[kItemsPerThread] = {};
  const std::size_t grid_stride = std::size_t(gridDim.x) * kTileSize;
  for (std::size_t base = std::size_t(blockIdx.x) * kTileSize + threadIdx.x;
       base < n; base += grid_stride) {
#pragma unroll
    for (int item = 0; item < kItemsPerThread; ++item) {
      const std::size_t index = base + item * kBlockSize;
      // 同一轮里相邻线程读相邻地址；尾部越界元素等价于零。
      if (index < n) sums[item] += input[index];
    }
  }
  float value = (sums[0] + sums[1]) + (sums[2] + sums[3]);
  // 不能让没有输入的线程提前 return：它们也要参与同步和 shuffle。
  value = UseShuffle ? block_sum_shuffle(value) : block_sum_shared(value);
  if (threadIdx.x == 0) partials[blockIdx.x] = value;
}

inline unsigned int block_count(std::size_t n) {
  if (n == 0) return 0;
  const std::size_t blocks = 1 + (n - 1) / kTileSize;
  return static_cast<unsigned int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

// 返回 float 元素数，不是字节数；两半 workspace 交替作为中间输出。
inline std::size_t workspace_elements(std::size_t n) {
  const unsigned int blocks = block_count(n);
  return blocks > 1 ? 2 * std::size_t(blocks) : 0;
}

// 异步接口：input[n]、output[1]、workspace[workspace_elements(n)] 均在 GPU。
// 三段内存不得重叠，并须保持有效直到 stream 完成；并发调用需独立 workspace。
// n == 0 时 input/workspace 可为 nullptr，结果定义为 0；小输入无需 workspace。
// 返回启动错误；异步执行错误由调用者在 stream 同步时检查。
inline cudaError_t reduce_sum(const float* input, float* output, std::size_t n,
                              float* workspace,
                              Method method = Method::Shuffle,
                              cudaStream_t stream = nullptr) {
  if (!output || n > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      (n > 0 && !input) || (block_count(n) > 1 && !workspace) ||
      (method != Method::Shared && method != Method::Shuffle)) {
    return cudaErrorInvalidValue;
  }
  if (n == 0) return cudaMemsetAsync(output, 0, sizeof(float), stream);

  const unsigned int capacity = block_count(n);
  unsigned int next_half = 0;
  while (true) {
    const unsigned int blocks = block_count(n);
    float* destination = blocks == 1 ? output : workspace + next_half * capacity;
    if (method == Method::Shuffle) {
      reduce_pass<true><<<blocks, kBlockSize, 0, stream>>>(input, destination, n);
    } else {
      reduce_pass<false><<<blocks, kBlockSize, 0, stream>>>(input, destination, n);
    }
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    if (blocks == 1) return cudaSuccess;
    input = destination;
    n = blocks;
    next_half ^= 1;
  }
}

}  // namespace reduce
}  // namespace cuda_learn
