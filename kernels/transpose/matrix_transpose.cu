# include <cstdio>
# include <cstdlib>
# include <iostream>
# include <cuda_runtime.h>
# include <cmath>
using std::cout;
using std::endl;


// naive版本的矩阵转置，row版本，原始矩阵按行读取，转置后的矩阵按列写入
__global__ void naiveTransRow(float *d_output, float *d_input, int M, int N){
    // 输入矩阵是M* N的，M是行，N是列
    // 目的是把M * N 的矩阵转置成N * M;
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;
    if(x < N && y < M){
        d_output[x * M + y] = d_input[y * N + x];
    }
}

// 一个线程处理多个元素的情况
// 进阶版本的矩阵转置，col版本，原始矩阵按列读取，转置后的矩阵按行写入
// Bm是tile数据的行大小，Bn是列, 一般Bm要是blockDim.x的整数倍
// Bn要是blockDim.y的整数倍
// 实际上是实现了一个线程粗化的效果，一个线程处理多个元素，减少了线程调度的开销
// 线程级并行(Instruction-Level Parallelism)，因为一个线程现在实现多个元素的load
// + store，因此 GPU 能够同时发起多个 memory request，在等待时做别的计算
// 对于memory_bound 的算子，提升ILP能够有更多提升
template <int Bm, int Bn>
__global__ void naiveTransColNelements(float *d_output, float *d_input, int M, int N){
    // (r,c)代表tile数据左上角元素的坐标
    int r = blockIdx.x * Bm, c = blockIdx.y * Bn;
    int bx = blockDim.x, by = blockDim.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    for(int i = tx; i < Bm; i += bx){
        int r0 = r + i;
        if(r0 >= M)
            break;
        for(int j = ty; j < Bn; j += by){
            int c0 = c + j;
            if(c0 < N){
                d_output[c0 * M + r0] = d_input[r0 * N + c0];
            }
        }
    }
}

// 这里 BLOCK_SZ 的大小应该等于blockDim.x，也等于blockDim.y
// 线程块必须开的是方阵，否则sdata访问会越界
template <int BLOCK_SZ>
__global__ void SmemTrans_B_SZ(const float* idata, float* odata, int M, int N) {
    const int bx = blockIdx.x, by = blockIdx.y;
    const int tx = threadIdx.x, ty = threadIdx.y;

    __shared__ float sdata[BLOCK_SZ][BLOCK_SZ];

    int x = bx * BLOCK_SZ + tx;
    int y = by * BLOCK_SZ + ty;

    if (y < M && x < N) {
        sdata[ty][tx] = idata[y * N + x];
    }
    __syncthreads();  // 到这里搬运了所有原矩阵的数据到smem里

    x = by * BLOCK_SZ + tx;
    y = bx * BLOCK_SZ + ty;
    if (y < N && x < M) {    // 原始矩阵是按行搬运进smem的，这里按行写入odata，则应该按列读取smem
        odata[y * M + x] = sdata[tx][ty];
    }
}


// 一个线程处理一个矩阵元素的版本
// BLOCK_SZ_M = blockDim.y, BLOCK_SZ_N = blockDim.x
template <int BLOCK_SZ_M, int BLOCK_SZ_N>
__global__ void SmemTrans(float *d_output, const float *d_input, int M, int N) {
    // 1. 定义共享内存 (无 Padding)
    __shared__ float tile[BLOCK_SZ_M][BLOCK_SZ_N];

    // 2. 计算当前线程在输入矩阵 (M行 x N列) 中的全局坐标
    // 使用 blockDim 保证常规的线程索引逻辑
    int x = blockIdx.x * blockDim.x + threadIdx.x; // 列坐标
    int y = blockIdx.y * blockDim.y + threadIdx.y; // 行坐标
    int tx = threadIdx.x; // 线程在 Block 内的列索引
    int ty = threadIdx.y; // 线程在 Block 内的行索引

    // 3. 将数据从全局内存读入 Shared Memory (合并读取, Coalesced Read)
    if (x < N && y < M) {
        tile[ty][tx] = d_input[y * N + x];
    }

    // 确保整个 Tile 的数据都已写入 Shared Memory
    __syncthreads();

    // 4. 重新计算当前线程在 Block 内的一维索引
    // 使用 blockDim.x 保证计算线程块内的一维 ID 绝对正确和通用
    unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;

    // 5. 在非方阵下，重新分配线程在 Tile 里的新坐标
    // 这一步是为了让线程去按列读 tile，并按行写回 d_output，从而实现合并写入
    int new_tx = tid % blockDim.y; // 对应转置后的行（原 tile 的列）
    int new_ty = tid / blockDim.y; // 对应转置后的列（原 tile 的行）

    // 6. 计算输出矩阵 (N行 x M列) 的全局坐标
    // 让线程优先处理转置后的行（原 tile 的列），这样可以保证写回全局内存时的合并访问
    int out_x = blockIdx.y * blockDim.y + new_tx; // 转置后的列坐标
    int out_y = blockIdx.x * blockDim.x + new_ty; // 转置后的行坐标

    // 7. 将转置后的数据写回全局内存 (合并写入, Coalesced Write)
    // 转置后，全局宽度变成了 M，高度变成了 N
    if (out_x < M && out_y < N) {
        d_output[out_y * M + out_x] = tile[new_tx][new_ty];
    }
}

void call_naiveTrans(float *d_output, float *d_input, int M, int N){
    dim3 blockDim(8, 8);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    // naiveTransRow<<<gridDim, blockDim>>>(d_output, d_input, M, N);
    naiveTransColNelements<16, 16><<<gridDim, blockDim>>>(d_output, d_input, M, N);
}

void call_SmemTrans(float *d_output, float *d_input, int M, int N){
    dim3 blockDim(4, 2);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    // SmemTrans_B_SZ<16><<<gridDim, blockDim>>>(d_output, d_input, M, N);
    SmemTrans<4, 2><<<gridDim, blockDim>>>(d_output, d_input, M, N);
}

int main(){
    int M = 5, N = 5;
    size_t size = M * N * sizeof(float);

    // 主机端的输入输出数组的内存分配
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    // 随机初始化输入矩阵
    for(int i = 0; i < M * N; ++i){
        h_input[i] = (float)rand() / RAND_MAX;
    }

    // 设备内存分配
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // 将输入矩阵从主机端复制到设备端
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int warm_up_iter = 5;

    //
    for(int i = 0; i < warm_up_iter; ++i){
        call_naiveTrans(d_output, d_input, M, N);
    }

    int bench_iter = 5;

    //开始计时
    cudaEventRecord(start);
    for(int i = 0; i < bench_iter; ++i){
        call_SmemTrans(d_output, d_input, M, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    //



    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        cout << "CUDA error: " << cudaGetErrorString(err) << endl;
        return 0;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Naive transpose kernel execution time: "
            << milliseconds / float(bench_iter) << " ms" << std::endl;

    // 将输出矩阵从设备端复制到主机端
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);


    // 输出转置前的矩阵
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            cout << h_input[i * N + j] << " ";
        }
        cout << endl;
    }
    cout<<"-----------------------"<< endl;
    // 输出转置后的矩阵
    for(int i = 0; i < N; ++i){
        for(int j = 0; j < M; ++j){
            cout << h_output[i * M + j] << " ";
        }
        cout << endl;
    }

    // 释放主机端内存
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    cout << "Matrix transposition completed successfully!" << endl;

    return 0;
}
