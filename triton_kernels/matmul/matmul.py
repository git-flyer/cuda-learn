import triton 
import triton.language as tl
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

# CPU 矩阵乘法，用于在时间上做比较
def matrix_multiply(A, B):
    # A B 都是二维列表
    A_shape = A.shape
    B_shape = B.shape
    rows_A = A_shape[0] # A的行数
    cols_A = A_shape[1] # A的列数
    rows_B = B_shape[0] # B的行数
    cols_B = B_shape[1] # B的列数
    assert cols_A == rows_B # 要确保 A 的列数等于 B 的行数
    C = np.zeros((rows_A, cols_B)) # 生成一个全是0的矩阵，总计rows_A 行，cols_B 列
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C 


# 分块矩阵乘法
def matrix_multiply_blocked(A, B, BLOCK_SIZE):
    M,K = A.shape # 
    N = B.shape[1]
    C = np.zeros((M,N), dtype = np.float32)

    # 在 A 矩阵的 m 轴上遍历块
    for m in range(0, M, BLOCK_SIZE):
        # 在 B 矩阵的 n 轴上遍历块
        for n in range(0, N, BLOCK_SIZE):
            acc = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype = np.float32)
            # 在 A 矩阵的 k 轴上遍历块
            for k in range(0, K, BLOCK_SIZE):
                a = A[m: m + BLOCK_SIZE, k : k + BLOCK_SIZE]
                b = B[k: k + BLOCK_SIZE, n : n + BLOCK_SIZE]
                acc += matrix_multiply(a, b)
            C[m : m + BLOCK_SIZE, n : n + BLOCK_SIZE] = acc
    return C


@triton.jit
def triton_matrix_multiply(
    a_ptr, # A 矩阵首元素指针
    b_ptr, # B 矩阵首元素指针
    c_ptr, # A * B 的结果 C 矩阵的首元素指针
    M, N, K, # Matrix dimensions
    BLOCK_SIZE_M : tl.constexpr = 128, 
    BLOCK_SIZE_N : tl.constexpr = 128,
    BLOCK_SIZE_K : tl.constexpr = 64,
):
    # 每一个线程块都会执行该核函数
    # 一个 triton block的二维坐标
    pid_m = tl.program_id(0)  # program 的第 0 维用来处理行
    pid_n = tl.program_id(1)  # program 的第 1 维用来处理列
    # 一个triton block的处理范围（在M,N轴上),offs_m 最终是一个列向量
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[:, None] # 形状为 (BLOCK_SIZE_M, 1)。
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)[None, :] # 形状为 (1, BLOCK_SIZE_N)。

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32) # 用来累加结果的寄存器，大小是 BLOCK_SIZE_M × BLOCK_SIZE_N
    # 在k轴上进行一个分块规约
    for k in range(0, K, BLOCK_SIZE_K):
        a_k = tl.arange(0, BLOCK_SIZE_K)[None, :] + k  # a_k是一个 1 * BLOCK_SIZE_K 的行向量
        # 这里 offs_m 是一个列向量，a_k 是一个行向量，相加之后会广播成一个二维矩阵 
        a = tl.load(a_ptr + offs_m * K + a_k, mask=(offs_m < M) & (a_k < K), other=0.0) # 越界的元素位置填充0
        a = a.to(tl.float16)  # 转化成 float16 类型
        
        b_k = tl.arange(0, BLOCK_SIZE_K)[:, None] + k  # b_k 是一个 BLOCK_SIZE * 1 的列向量
        b = tl.load(b_ptr + b_k * N + offs_n, mask=(offs_n < N) & (b_k < K), other=0.0) 
        b = b.to(tl.float16)

        c = tl.dot(a, b, acc=c)

    # 一个 triton_block 计算出来的大小是 BLOCK_SIZE_M * BLOCK_SIZE_N
    c_offset = offs_m * N + offs_n
    c_mask = (offs_m < M) & (offs_n < N)

    tl.store(c_ptr + c_offset, c, mask=c_mask)


@torch.no_grad()
def call_triton_matmul(a, b):
    # out_shape_0 = a.shape[:-1]    # 记录除了最后一个维度以外的形状
    a = a.view((-1, a.shape[-1])) # 将输入张量重塑为二维矩阵，形状为 (M, K)，其中 M 是前面所有维度的乘积，K 是最后一个维度的大小
    M, K = a.shape     # M 是输入矩阵 a 的行数，K 是输入矩阵 a 的列数
    N = b.shape[1]   # 获取B权重矩阵的列数 N, 即输出矩阵的列数


    c = torch.empty((M, N), device=a.device, dtype=a.dtype) # 创建一个空的输出张量 z，形状为 (M, N)，与输入矩阵 a 的行数 M 和权重矩阵 b 的列数 N 相匹配

    BLOCK_SIZE_M = 64    # 在 M 轴上的块大小，表示每个 triton block 处理的行数
    BLOCK_SIZE_N = 64    # 在 N 轴上的块大小，表示每个 triton block 处理的列数
    BLOCK_SIZE_K = 32    # 在 K 轴上的块大小，表示每次处理的 K 维度的大小

    # 配置二维网格中，每个维度上的triton block数量
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N), 1) 

    triton_matrix_multiply[grid](
        a_ptr = a, # 输入数据矩阵首元素指针
        b_ptr = b, # 权重矩阵首元素指针
        c_ptr = c, # 输出结果地址
        M = M, N = N, K = K, # Matrix dimensions
        BLOCK_SIZE_M = BLOCK_SIZE_M, 
        BLOCK_SIZE_N = BLOCK_SIZE_N,
        BLOCK_SIZE_K = BLOCK_SIZE_K,
    )

    return c




    


if __name__ == '__main__':
    # 矩阵 A 是一个 32 * 64 的矩阵
    # 矩阵 B 是一个 64 * 128 的矩阵
    A = np.random.randn(32, 64).astype(np.float32)
    B = np.random.randn(64, 128).astype(np.float32)

    start = time.perf_counter()
    C_naive = matrix_multiply(A, B)
    t_naive = time.perf_counter() - start

    # 计时分块矩阵乘法
    start = time.perf_counter()
    C_blocked = matrix_multiply_blocked(A, B, BLOCK_SIZE = 16)
    t_blocked = time.perf_counter() - start

    if np.allclose(C_naive, C_blocked, atol=1e-5):
        print("PASS: 两种矩阵乘法结果一致")
        print(f"朴素矩阵乘法: {t_naive:.4f}s")
        print(f"分块矩阵乘法: {t_blocked:.4f}s")
        print(f"加速比: {t_naive / t_blocked:.2f}x")
    else:
        max_diff = np.max(np.abs(C_naive - C_blocked))
        print(f"FAIL: 结果不一致，最大差异 = {max_diff}")

    M, K, N = 128, 128, 64         
    A = torch.randn((M, K), device='cuda', dtype=torch.float16)
    B = torch.randn((K, N), device='cuda', dtype=torch.float16)
    

    for i in range(5):
        golden = A @ B
        output = call_triton_matmul(A, B)
        A = torch.randn((M, K), device='cuda', dtype=torch.float16)
        B = torch.randn((K, N), device='cuda', dtype=torch.float16)

    repeat_time = 5
    times_torch = []
    times_triton = []
    for i in range(repeat_time):
        # 重新生成输入
        A = torch.randn((M, K), device='cuda', dtype=torch.float16)
        B = torch.randn((K, N), device='cuda', dtype=torch.float16)
        torch.cuda.synchronize()   # 强制cpu暂停，直到gpu把手头的任务全部完成，才能开始计时，否则记录到的只是cpu发出
                                   # 计时指令的时间，而不是gpu实际执行的时间

        t1 = time.time()
        output = call_triton_matmul(A, B)
        torch.cuda.synchronize()
        t2 = time.time()
        print('triton time:{}'.format(t2 - t1))
        times_triton.append(t2 - t1)

        t1 = time.time()
        golden = A @ B
        torch.cuda.synchronize()
        t2 = time.time()
        times_torch.append(t2-t1)
        print('pytorch time:{}'.format(t2 - t1))

    # 将时间从秒转换为毫秒
    times_torch_ms = [t * 1000 for t in times_torch]
    times_triton_ms = [t * 1000 for t in times_triton]

    sizes = [i for i in range(repeat_time)]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_torch_ms, label='torch (matrix_multiply)', marker='o')
    plt.plot(sizes, times_triton_ms, label='triton (matrix_multiply)', marker='o')

    plt.xlabel('Run Index')
    plt.ylabel('Time (milliseconds)')
    plt.title('Matrix Multiplication Performance Comparison (Torch vs Triton)')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('cc.png')






