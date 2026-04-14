# include <cstdio>
# include <cstdlib>
# include <iostream>
# include <cuda_runtime.h>
# include <cmath>

# define BLOCK_SIZE 32
# define TILE_SIZE 8

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void thread_tile_matmul_kernel(const float *A, const float *B, float *C, int M, int K, int N){
    // thread_tile 矩阵乘优化


    // 当前线程块和线程的索引
    const int b_x = blockIdx.x, b_y = blockIdx.y;
    const int t_x = threadIdx.x, t_y = threadIdx.y;

    // tid是一个线程的一维坐标索引， num_threads 是一个线程块
    // 中线程的数量，一个线程块负责C矩阵中一个 BM * BN 大小的块的
    // 计算，每个线程负责其中TM * TN 大小的块的计算
    const int tid = t_y * blockDim.x + t_x;
    const int num_threads = blockDim.x * blockDim.y;

    // 申请共享内存，同时每个线程拥有一个TM * TN大小的寄存器
    // 组，里边存储其负责的C矩阵中的对应部分计算出来的累加值
    __shared__ float As[BM][BK], Bs[BK][BN];
    float accum[TM][TN] = {0.0f};

    // frag_a 和 frag_b 用于缓存从共享内存中读取的数据，最大化
    // 寄存器复用
    float frag_a[TM], frag_b[TN];
    // 在K维度上滑动，先搬运数据到 As 和 Bs （共享内存）
    // 之后每个线程搬运其感兴趣的As的列和Bs的行
    //
    for(int k_offset = 0; k_offset < K; k_offset += BK){
        // num_threads 是远小于 BM * BK 和 BK * BN 的，一次搬运num_threads
        // 个数据
        for(int i = tid; i < BM * BK; i += num_threads){
            int r = i / BK, c = i % BK;
            int a_row = b_y * BM + r,  a_col = k_offset + c;
            As[r][c] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }

        for(int i = tid; i < BK * BN; i += num_threads){
            int r = i / BN, c = i % BN;
            int b_row = k_offset + r, b_col = b_x * BN + c;
            Bs[r][c] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }

        //等待所有线程加载完毕
        __syncthreads();

        for(int i = 0; i < BK; ++i){
            for(int j = 0; j < TM; ++j){
                frag_a[j] = As[t_y * TM + j][i];
            }
            for(int j = 0; j < TN; ++j){
                frag_b[j] = Bs[i][t_x * TN + j];
            }

            for(int j = 0; j < TM; ++j){
                for(int k = 0; k < TN; ++k){
                    accum[j][k] += frag_a[j] * frag_b[k];
                }
            }
        }
        // 确保所有线程块都完成累加计算之后才能进行下一次
        // 的shared_memory的搬运工作
        __syncthreads();
    }
    // 全部完成之后要进行最后的往C结果矩阵里的写入工作
    for(int i = 0; i < TM; ++i){
        for(int j = 0; j < TN; ++j){
            int c_row = b_y * BM + t_y * TM + i;
            int c_col = b_x * BN + t_x * TN + j;
            if(c_row < M && c_col < N)
                C[c_row * N + c_col] = accum[i][j];
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

    // 主机端内存
    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);
    float *hC_ref = (float *)malloc(sizeC);

    // 随机初始化
    for (int i = 0; i < M * K; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = (float)rand() / RAND_MAX;

    // 设备端内存，在设备端为A,B,C矩阵开辟了内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    //搬运A,B矩阵到设备端
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // block 和 grid 两个dim3 变量的x,y维度的两个值被赋值


    constexpr int BM = 32, BN = 32, BK = 32;
    constexpr int TM = 8, TN = 8;

    // 这里要确保 BN / TN 和 BM / TM 都能整除
    dim3 block(BN / TN, BM / TM);

    // Grid 维度：覆盖整个输出矩阵 C
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    thread_tile_matmul_kernel<BM,BN,BK,TM,TN><<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();  // 等待gpu端执行完成
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);  //设备端拷贝到主机端

    // 结果验证,cpu的结果写到hC_ref里
    cpu_matmul(h_A, h_B, hC_ref, M, K, N);

    float max_err = 0.f;
    for(int i = 0; i < M * N; ++i){
        max_err = fmaxf(max_err, fabsf(h_C[i] - hC_ref[i]));
    }
    printf("[thread_tile_matmul] max error = %e (%s)\n", max_err, max_err < 1e-3 ? "PASS":"FAIL");

    // 释放资源
    cudaFree(d_A); cudaFree(d_B);
    cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    free(hC_ref);
    return 0;
}
