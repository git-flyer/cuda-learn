/**
 * shared_memory_matmul.cu — 共享内存分块矩阵乘法
 *                           (Tiled Matrix Multiplication with Shared Memory)
 *
 * 优化动机：
 *   朴素版本中每次循环都访问全局显存，延迟极高。
 *   共享内存（shared memory）位于片上，延迟约为全局显存的 1/30，带宽数倍于全局显存。
 *
 * 算法思路（分块/Tiling）：
 *   将 A 和 B 按 TILE_SIZE × TILE_SIZE 大小分块。
 *   每个线程块在一轮迭代中：
 *     1. 协作地将一块 A_tile 和 B_tile 从全局显存载入共享内存。
 *     2. __syncthreads() 确保所有线程都完成加载。
 *     3. 用共享内存中的数据计算部分和（无全局显存访问）。
 *     4. __syncthreads() 确保本轮计算完成后再覆盖共享内存。
 *   沿 K 维度滑动，累加所有部分和，最终写回 C。
 *
 * 性能提升：
 *   每个全局显存元素被重用 TILE_SIZE 次，显著降低全局显存流量。
 *   理论算术强度从 O(1) FLOP/Byte 提升至 O(TILE_SIZE) FLOP/Byte。
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// ----------------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------------
__global__ void shared_matmul_kernel(const float *A, const float *B, float *C,
                                     int M, int K, int N) {
    // 每个线程块对应 C 的一个 TILE_SIZE × TILE_SIZE 子块
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // 声明共享内存 tile
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // 沿 K 维度分块迭代
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        // ---------- 协作加载 A_tile ----------
        int a_col = t * TILE_SIZE + threadIdx.x;  // A 列索引
        if (row < M && a_col < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // ---------- 协作加载 B_tile ----------
        int b_row = t * TILE_SIZE + threadIdx.y;  // B 行索引
        if (b_row < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // 等待块内所有线程完成数据加载
        __syncthreads();

        // ---------- 用共享内存中的 tile 计算部分和 ----------
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        // 等待本轮计算完成，再进入下一轮覆盖共享内存
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ----------------------------------------------------------------------------
// Host utility：CPU 参考实现，用于结果验证
// ----------------------------------------------------------------------------
static void cpu_matmul(const float *A, const float *B, float *C,
                       int M, int K, int N) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0.f;
            for (int k = 0; k < K; ++k)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main() {
    const int M = 512, K = 512, N = 512;

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    float *hA = (float *)malloc(sizeA);
    float *hB = (float *)malloc(sizeB);
    float *hC = (float *)malloc(sizeC);
    float *hC_ref = (float *)malloc(sizeC);

    for (int i = 0; i < M * K; ++i) hA[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) hB[i] = (float)rand() / RAND_MAX;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);

    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);
    shared_matmul_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);

    cpu_matmul(hA, hB, hC_ref, M, K, N);
    float max_err = 0.f;
    for (int i = 0; i < M * N; ++i)
        max_err = fmaxf(max_err, fabsf(hC[i] - hC_ref[i]));
    printf("[shared_memory_matmul] max error = %e  (%s)\n", max_err,
           max_err < 1e-3f ? "PASS" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hC_ref);
    return 0;
}
