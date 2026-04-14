# include <cstdio>
# include <cstdlib>
# include <iostream>
# include <cuda_runtime.h>
# include <cmath>

# define BLOCK_SIZE 32

__global__ void naive_matmul_kernel(const float *A, const float *B, float *C, int M, int K, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    //至此已经计算出了一个线程负责C矩阵中元素的行和列的下标
    if(row >= M || col >= N)
        return;
    float sum = 0;
    for(int k = 0; k < K; ++k){
        sum += A[row * K + k] * B[k * N + col];
    }
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

    naive_matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();  // 等待gpu端执行完成
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);  //设备端拷贝到主机端

    // 结果验证,cpu的结果写到hC_ref里
    cpu_matmul(h_A, h_B, hC_ref, M, K, N);
    float max_err = 0.f;
    for(int i = 0; i < M * N; ++i){
        max_err = fmaxf(max_err, fabsf(h_C[i] - hC_ref[i]));
    }
    printf("[naive matmul] max error = %e (%s)\n", max_err, max_err < 1e-3 ? "PASS":"FAIL");

    // 释放资源
    cudaFree(d_A); cudaFree(d_B);
    cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    free(hC_ref);
    return 0;
}
