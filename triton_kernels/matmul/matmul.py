import triton 
import triton.language as tl
import numpy as np
import time

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

                





