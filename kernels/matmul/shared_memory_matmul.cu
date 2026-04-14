# include <cstdio>
# include <cstdlib>
# include <iostream>
# include <cuda_runtime.h>
# include <cmath>

# define BLOCK_SIZE 32

__global__ void shared_memory_matmul_kernel(const float *A, const float *B, float *C, int M, int K, int N){
    // 整体的思路是先把global_memory中的 A 和 B 矩阵的数据以分块的
    // 形式搬到shared_memory中，其中一个线程块负责计算C矩阵中一个大小为
    // BM * BN的子矩阵，而每个线程负责其中的一个元素，这样相当于这个
    // 线程块要负责A矩阵中一个BM * K 大小的子矩阵的搬运，和B矩阵中一个
    // K * BN 大小的矩阵的搬运，同时在K维度上可以进一步进行分块
    // 比如A矩阵一次搬运一个BM * BK大小的矩阵进shared_memory
    // B矩阵一次搬运一个BK * BN 大小的矩阵进shared_memory

    const int BM = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;

    int b_x = blockIdx.x, b_y = blockIdx.y;
    int t_x = threadIdx.x, t_y = threadIdx.y;

    int row = b_y * BM + t_y;
    int col = b_x * BN + t_x;
    __shared__ float As[BM][BK], Bs[BK][BN];

    float sum = 0.;

    for(int k_0 = 0; k_0 < K; k_0 += BK){

        if(row < M && k_0 + t_x < K && t_x < BK){
            As[t_y][t_x] = A[row * K + k_0 + t_x];
        }
        else{
            As[t_y][t_x] = 0;
        }
        if(col < N && k_0 + t_y < K && t_y < BK){
            Bs[t_y][t_x] = B[col + (k_0 + t_y) * N];
        }
        else{
            Bs[t_y][t_x] = 0;
        }
        __syncthreads();
        // 确保线程块内的所有线程都完成了As和Bs矩阵的加载工作

        for(int k = 0; k < BK; ++k){
            sum += As[t_y][k] * Bs[k][t_x];
        }
        __syncthreads();
        // 确保线程块内的所有线程都已经完成了sum值的一次累加
        // 之后再进行 As 和 Bs 的搬运工作
    }

    if(row < M && col < N)
        C[row * N + col] = sum;

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
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((M + BLOCK_SIZE - 1)/BLOCK_SIZE, (N + BLOCK_SIZE - 1)/BLOCK_SIZE);

    shared_memory_matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();  // 等待gpu端执行完成
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);  //设备端拷贝到主机端

    // 结果验证,cpu的结果写到hC_ref里
    cpu_matmul(h_A, h_B, hC_ref, M, K, N);

    float max_err = 0.f;
    for(int i = 0; i < M * N; ++i){
        max_err = fmaxf(max_err, fabsf(h_C[i] - hC_ref[i]));
    }
    printf("[shared_memory_matmul] max error = %e (%s)\n", max_err, max_err < 1e-3 ? "PASS":"FAIL");

    // 释放资源
    cudaFree(d_A); cudaFree(d_B);
    cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    free(hC_ref);
    return 0;
}
