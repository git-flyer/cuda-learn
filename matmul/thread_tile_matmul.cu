/**
 * thread_tile_matmul.cu — Thread Tile（寄存器分块）矩阵乘法
 *
 * 优化动机：
 *   共享内存版本中，每个线程仅计算 C 的一个元素，线程间对共享内存的访问
 *   存在大量重复。通过让每个线程计算 TM × TN 个输出元素，可以：
 *     1. 更充分地利用寄存器（register），减少共享内存读取次数。
 *     2. 提高算术强度（FLOP/Byte），更好地隐藏访存延迟。
 *     3. 增大每个线程块处理的输出规模，减少 __syncthreads() 调用频率。
 *
 * 线程/块/网格映射（以 BM=BN=128，BK=8，TM=TN=8 为例）：
 *   - 线程块大小：(BM/TM) × (BN/TN) = 16 × 16 = 256 threads
 *   - 每个线程负责计算 C 中 TM × TN = 8×8 = 64 个输出元素
 *   - 网格大小：ceil(M/BM) × ceil(N/BN)
 *
 * 关键数据结构：
 *   - smA[BK][BM]：A 的共享内存 tile（列主序，便于线程在 M 方向上读取连续数据）
 *   - smB[BK][BN]：B 的共享内存 tile（行主序）
 *   - regA[TM]、regB[TN]：每个线程的寄存器缓存
 *   - accC[TM][TN]：每个线程的累加寄存器
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ---------- 超参数 ----------
#define BM 64   // 线程块负责的输出行数
#define BN 64   // 线程块负责的输出列数
#define BK 8    // K 方向分块大小
#define TM 8    // 每个线程负责的输出行数
#define TN 8    // 每个线程负责的输出列数

// 线程块中的线程数量
#define THREADS_PER_BLOCK ((BM / TM) * (BN / TN))  // = 64

// ----------------------------------------------------------------------------
// Kernel
// ----------------------------------------------------------------------------
__global__ void thread_tile_matmul_kernel(const float *A, const float *B,
                                          float *C, int M, int K, int N) {
    // 当前线程块负责的 C 子块的起始行列
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // 线程在块内的位置（行优先排布）
    int tid = threadIdx.x;                    // 0 .. THREADS_PER_BLOCK-1
    int thread_row = tid / (BN / TN);         // 块内线程行
    int thread_col = tid % (BN / TN);         // 块内线程列

    // 共享内存 tile
    // smA 以列主序存储，方便每个线程沿 M 方向连续访问
    __shared__ float smA[BK][BM];
    __shared__ float smB[BK][BN];

    // 每个线程的寄存器累加器，全部初始化为 0
    float accC[TM][TN] = {};

    // 用于协作加载共享内存的辅助索引
    // 线程块共 THREADS_PER_BLOCK 个线程，需要加载 BK*BM 个 A 元素和 BK*BN 个 B 元素
    // 每个线程每次加载若干个元素
    // BK*BM / THREADS_PER_BLOCK = 8*64/64 = 8 (每个线程加载 A 的 8 个元素)
    // BK*BN / THREADS_PER_BLOCK = 8*64/64 = 8 (每个线程加载 B 的 8 个元素)
    const int A_LOAD_PER_THREAD = (BK * BM) / THREADS_PER_BLOCK;  // = 8
    const int B_LOAD_PER_THREAD = (BK * BN) / THREADS_PER_BLOCK;  // = 8

    // 沿 K 方向分块迭代
    int num_tiles = (K + BK - 1) / BK;
    for (int t = 0; t < num_tiles; ++t) {
        int tile_k_start = t * BK;

        // ---------- 协作加载 A tile -> smA[BK][BM] ----------
        // 将 A[block_row : block_row+BM, tile_k_start : tile_k_start+BK]
        // 转置后存入 smA，即 smA[k_local][m_local] = A[block_row+m_local][tile_k_start+k_local]
        for (int i = 0; i < A_LOAD_PER_THREAD; ++i) {
            int idx  = tid + i * THREADS_PER_BLOCK;  // 展平索引
            int k_local = idx / BM;                  // smA 的行（K 方向）
            int m_local = idx % BM;                  // smA 的列（M 方向）
            int a_row = block_row + m_local;
            int a_col = tile_k_start + k_local;
            smA[k_local][m_local] = (a_row < M && a_col < K)
                                    ? A[a_row * K + a_col]
                                    : 0.0f;
        }

        // ---------- 协作加载 B tile -> smB[BK][BN] ----------
        // smB[k_local][n_local] = B[tile_k_start+k_local][block_col+n_local]
        for (int i = 0; i < B_LOAD_PER_THREAD; ++i) {
            int idx  = tid + i * THREADS_PER_BLOCK;
            int k_local = idx / BN;
            int n_local = idx % BN;
            int b_row = tile_k_start + k_local;
            int b_col = block_col + n_local;
            smB[k_local][n_local] = (b_row < K && b_col < N)
                                    ? B[b_row * N + b_col]
                                    : 0.0f;
        }

        __syncthreads();

        // ---------- 寄存器级计算 ----------
        // 每个线程从共享内存读取 regA[TM] 和 regB[TN]，
        // 做外积累加到 accC[TM][TN]
        float regA[TM], regB[TN];
        for (int k = 0; k < BK; ++k) {
            // 从共享内存读入寄存器（每次只读一行/列，后续全走寄存器）
            for (int m = 0; m < TM; ++m)
                regA[m] = smA[k][thread_row * TM + m];
            for (int n = 0; n < TN; ++n)
                regB[n] = smB[k][thread_col * TN + n];

            // 外积：accC[m][n] += regA[m] * regB[n]
            for (int m = 0; m < TM; ++m)
                for (int n = 0; n < TN; ++n)
                    accC[m][n] += regA[m] * regB[n];
        }

        __syncthreads();
    }

    // ---------- 写回结果 ----------
    for (int m = 0; m < TM; ++m) {
        int c_row = block_row + thread_row * TM + m;
        for (int n = 0; n < TN; ++n) {
            int c_col = block_col + thread_col * TN + n;
            if (c_row < M && c_col < N)
                C[c_row * N + c_col] = accC[m][n];
        }
    }
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

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    thread_tile_matmul_kernel<<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost);

    cpu_matmul(hA, hB, hC_ref, M, K, N);
    float max_err = 0.f;
    for (int i = 0; i < M * N; ++i)
        max_err = fmaxf(max_err, fabsf(hC[i] - hC_ref[i]));
    printf("[thread_tile_matmul] max error = %e  (%s)\n", max_err,
           max_err < 1e-3f ? "PASS" : "FAIL");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(hC_ref);
    return 0;
}
