/**
 * naive_matmul.cu — 朴素矩阵乘法（Naive Matrix Multiplication）
 *
 * 算法思路：
 *   C = A * B，其中 A(M×K)，B(K×N)，C(M×N)
 *
 * 线程映射：
 *   每个 CUDA 线程负责计算输出矩阵 C 中的**一个元素**。
 *   线程块大小：BLOCK_SIZE × BLOCK_SIZE
 *   网格大小：ceil(N/BLOCK_SIZE) × ceil(M/BLOCK_SIZE)
 *
 * 缺点：
 *   每次读取 A 和 B 的元素都直接访问全局显存（global memory），
 *   全局显存延迟高（~100-200 cycle），导致内存带宽成为瓶颈。
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

// ----------------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------------
__global__ void naive_matmul_kernel(const float *A, const float *B, float *C,
                                    int M, int K, int N) {
    // 计算当前线程对应的输出元素坐标
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 行索引
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 列索引

    if (row >= M || col >= N) return;  // 边界保护

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        // 每次循环都从全局显存读取，无任何数据复用
        sum += A[row * K + k] * B[k * N + col];
    }
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

    // 主机端内存
    float *hA = (float *)malloc(sizeA);
    float *hB = (float *)malloc(sizeB);
    float *hC = (float *)malloc(sizeC);
    float *hC_ref = (float *)malloc(sizeC);

    // 随机初始化
    for (int i = 0; i < M * K; ++i) hA[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) hB[i] = (float)rand() / RAND_MAX;

    // 设备端内存
    float *dA, *dB, *dC;
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);

    cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice);

    // 启动 kernel
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    naive_matmul_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);

    // 结果验证
    cpu_matmul(hA, hB, hC_ref, M, K, N);
    float max_err = 0.f;
    for (int i = 0; i < M * N; ++i)
        max_err = fmaxf(max_err, fabsf(hC[i] - hC_ref[i]));
    printf("[naive_matmul] max error = %e  (%s)\n", max_err,
           max_err < 1e-3f ? "PASS" : "FAIL");

    // 释放资源
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hC_ref);
    return 0;
}
