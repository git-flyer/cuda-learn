#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

// ----------------------------------------------------------------------------
// Kernel: 结合了 Thread Tiling, 双缓冲 (Double Buffering) 和 向量化访存 (float4)
// ----------------------------------------------------------------------------
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void double_buffer_float4_matmul_kernel(const float *A, const float *B, float *C, int M, int K, int N) {

    // 线程块和线程索引
    const int b_x = blockIdx.x, b_y = blockIdx.y;
    const int t_x = threadIdx.x, t_y = threadIdx.y;

    // 线程级常量，编译时即确定 NUM_THREADS 的值
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN);
    const int tid = t_y * blockDim.x + t_x;

    // 计算每个线程需要搬运的 float4 数量
    // 强制要求可以整除，否则边界处理会很复杂
    // 编译期断言，如果不满足条件，直接编译报错
    // 一个线程负责计算TM * TN 个结果，每一行的TN
    // 需要被4整除，这样能够一次性打包4个float类型的
    // 数据写回 global memory
    static_assert((BM * BK) % (4 * NUM_THREADS) == 0, "A tile loading must be evenly divisible by threads * 4");
    static_assert((BK * BN) % (4 * NUM_THREADS) == 0, "B tile loading must be evenly divisible by threads * 4");
    static_assert(TN % 4 == 0, "TN must be a multiple of 4 for vectorized store");

    // 一个线程一次搬4个数据，A_LOADS_PER_THREAD 代表一个线程需要搬几次数据（才能填满一个As）
    // B_LOADS_PER_THREAD 同理，代表一个线程需要搬几次数据（才能填满一个Bs）
    constexpr int A_LOADS_PER_THREAD = (BM * BK) / (4 * NUM_THREADS);
    constexpr int B_LOADS_PER_THREAD = (BK * BN) / (4 * NUM_THREADS);

    // 1. 双缓冲共享内存分配 (第一维为 2)
    __shared__ float As[2][BM][BK];
    __shared__ float Bs[2][BK][BN];

    // 计算累加值的寄存器
    float accum[TM][TN] = {0.0f};

    // 用于计算的寄存器缓存
    float frag_a[TM], frag_b[TN];

    // 用于预取 (Prefetch) 全局内存数据的寄存器缓存 (float4向量化)
    // 一个线程需要预取的元素数组已经被开好了
    float4 prefetch_a[A_LOADS_PER_THREAD];
    float4 prefetch_b[B_LOADS_PER_THREAD];

    // ------------------------------------------------------------------------
    // Prologue: 初始化流水线，加载第 0 个 Tile 到共享内存 Buffer 0
    // ------------------------------------------------------------------------
    // 第一次必须先加载一个Tile到Buffer0，之后才能循环加载下一个并计算上一个
    #pragma unroll
    for (int i = 0; i < A_LOADS_PER_THREAD; ++i) {
        int load_idx = i * NUM_THREADS + tid;    // 第几波加载的第几个线程
        int r = load_idx / (BK / 4);             // 线程加载矩阵元素的行下标
        int c = (load_idx % (BK / 4)) * 4;       // 线程加载矩阵元素的列下标
        int a_row = b_y * BM + r;
        int a_col = c; // k_offset = 0
        // 向量化加载 A
        prefetch_a[i] = (a_row < M && a_col < K) ?
            reinterpret_cast<const float4*>(&A[a_row * K + a_col])[0] : make_float4(0.f, 0.f, 0.f, 0.f);
    }

    #pragma unroll
    for (int i = 0; i < B_LOADS_PER_THREAD; ++i) {
        int load_idx = i * NUM_THREADS + tid;
        int r = load_idx / (BN / 4);
        int c = (load_idx % (BN / 4)) * 4;
        int b_row = r; // k_offset = 0
        int b_col = b_x * BN + c;
        // 向量化加载 B
        prefetch_b[i] = (b_row < K && b_col < N) ?
            reinterpret_cast<const float4*>(&B[b_row * N + b_col])[0] : make_float4(0.f, 0.f, 0.f, 0.f);
    }

    // 将预取的数据写入 Buffer 0
    #pragma unroll
    for (int i = 0; i < A_LOADS_PER_THREAD; ++i) {
        int load_idx = i * NUM_THREADS + tid;
        int r = load_idx / (BK / 4);
        int c = (load_idx % (BK / 4)) * 4;
        As[0][r][c+0] = prefetch_a[i].x; As[0][r][c+1] = prefetch_a[i].y;
        As[0][r][c+2] = prefetch_a[i].z; As[0][r][c+3] = prefetch_a[i].w;
    }
    #pragma unroll
    for (int i = 0; i < B_LOADS_PER_THREAD; ++i) {
        int load_idx = i * NUM_THREADS + tid;
        int r = load_idx / (BN / 4);
        int c = (load_idx % (BN / 4)) * 4;
        Bs[0][r][c+0] = prefetch_b[i].x; Bs[0][r][c+1] = prefetch_b[i].y;
        Bs[0][r][c+2] = prefetch_b[i].z; Bs[0][r][c+3] = prefetch_b[i].w;
    }

    __syncthreads();

    // ------------------------------------------------------------------------
    // Main Loop: 双缓冲流水线
    // ------------------------------------------------------------------------
    int smem_idx = 0; // 当前计算读取的 buffer 索引

    for (int k_offset = 0; k_offset < K; k_offset += BK) {
        int next_k = k_offset + BK; // 下一个 Tile 的偏移量

        // 1. 预取下一个 Tile 到寄存器 (掩盖接下来的计算延迟)
        if (next_k < K) {
            #pragma unroll
            for (int i = 0; i < A_LOADS_PER_THREAD; ++i) {
                int load_idx = i * NUM_THREADS + tid;
                int r = load_idx / (BK / 4);
                int c = (load_idx % (BK / 4)) * 4;
                int a_row = b_y * BM + r;
                int a_col = next_k + c;
                prefetch_a[i] = (a_row < M && a_col < K) ?
                    reinterpret_cast<const float4*>(&A[a_row * K + a_col])[0] : make_float4(0.f, 0.f, 0.f, 0.f);
            }
            #pragma unroll
            for (int i = 0; i < B_LOADS_PER_THREAD; ++i) {
                int load_idx = i * NUM_THREADS + tid;
                int r = load_idx / (BN / 4);
                int c = (load_idx % (BN / 4)) * 4;
                int b_row = next_k + r;
                int b_col = b_x * BN + c;
                prefetch_b[i] = (b_row < K && b_col < N) ?
                    reinterpret_cast<const float4*>(&B[b_row * N + b_col])[0] : make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }

        // 2. 使用当前的 buffer (smem_idx) 进行矩阵乘累加计算
        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            #pragma unroll
            for (int j = 0; j < TM; ++j) frag_a[j] = As[smem_idx][t_y * TM + j][i];
            #pragma unroll
            for (int j = 0; j < TN; ++j) frag_b[j] = Bs[smem_idx][i][t_x * TN + j];

            #pragma unroll
            for (int j = 0; j < TM; ++j) {
                #pragma unroll
                for (int k = 0; k < TN; ++k) {
                    accum[j][k] += frag_a[j] * frag_b[k];
                }
            }
        }

        // 3. 将预取好的下一轮数据写入另一个 buffer (smem_idx ^ 1)
        if (next_k < K) {
            int next_smem_idx = smem_idx ^ 1;
            #pragma unroll
            for (int i = 0; i < A_LOADS_PER_THREAD; ++i) {
                int load_idx = i * NUM_THREADS + tid;
                int r = load_idx / (BK / 4);
                int c = (load_idx % (BK / 4)) * 4;
                As[next_smem_idx][r][c+0] = prefetch_a[i].x; As[next_smem_idx][r][c+1] = prefetch_a[i].y;
                As[next_smem_idx][r][c+2] = prefetch_a[i].z; As[next_smem_idx][r][c+3] = prefetch_a[i].w;
            }
            #pragma unroll
            for (int i = 0; i < B_LOADS_PER_THREAD; ++i) {
                int load_idx = i * NUM_THREADS + tid;
                int r = load_idx / (BN / 4);
                int c = (load_idx % (BN / 4)) * 4;
                Bs[next_smem_idx][r][c+0] = prefetch_b[i].x; Bs[next_smem_idx][r][c+1] = prefetch_b[i].y;
                Bs[next_smem_idx][r][c+2] = prefetch_b[i].z; Bs[next_smem_idx][r][c+3] = prefetch_b[i].w;
            }
        }

        // 翻转 buffer 索引并同步线程
        // 同步确保所有线程已经读完老 buffer，且完全写完新 buffer
        smem_idx ^= 1;
        __syncthreads();
    }

    // ------------------------------------------------------------------------
    // Epilogue: 向量化写回结果矩阵 C (利用 float4)
    // ------------------------------------------------------------------------
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int c_row = b_y * BM + t_y * TM + i;
        if (c_row < M) {
            #pragma unroll
            for (int j = 0; j < TN; j += 4) { // 按 4 的步长写入
                int c_col = b_x * BN + t_x * TN + j;
                if (c_col < N) {
                    // 打包为 float4 统一写回
                    float4 out = make_float4(accum[i][j], accum[i][j+1], accum[i][j+2], accum[i][j+3]);
                    reinterpret_cast<float4*>(&C[c_row * N + c_col])[0] = out;
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Host utility：CPU 参考实现，用于结果验证
// ----------------------------------------------------------------------------
static void cpu_matmul(const float *A, const float *B, float *C, int M, int K, int N){
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            float sum = 0;
            for(int k = 0; k < K; ++k){
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------
int main(){
    const int M = 512, K = 512, N = 512;
    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);
    float *hC_ref = (float *)malloc(sizeC);

    for (int i = 0; i < M * K; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    constexpr int BM = 32, BN = 32, BK = 32;
    constexpr int TM = 8, TN = 8;

    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    double_buffer_float4_matmul_kernel<BM,BN,BK,TM,TN><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    cpu_matmul(h_A, h_B, hC_ref, M, K, N);

    float max_err = 0.f;
    for(int i = 0; i < M * N; ++i){
        max_err = fmaxf(max_err, fabsf(h_C[i] - hC_ref[i]));
    }
    printf("[double_buffer_float4_matmul] max error = %e (%s)\n", max_err, max_err < 1e-3 ? "PASS":"FAIL");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(hC_ref);
    return 0;
}
