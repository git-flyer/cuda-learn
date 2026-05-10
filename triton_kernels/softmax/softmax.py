import torch
import triton
import triton.language as tl
def softmax_cpu(x):
    # 每个元素减去它所在行的最大值，之后求指数，最后除以每行的指数和
    x_exp = torch.exp(x - torch.max(x, dim=-1, keepdim=True).values)
    return x_exp / torch.sum(x_exp, dim=-1, keepdim=True)

# BLOCK_SIZE被标记为编译期常量
# input_row_stride 是输入矩阵每行的元素个数
# output_row_stride 是输出矩阵每行的元素个数
# n_rows 是矩阵的总行数
# n_cols 是矩阵每行的列数
# BLOCK_SIZE 是编译期常量，代表一个 block 内的线程数量
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, 
output_row_stride, n_rows, n_cols, BLOCK_SIZE : tl.constexpr):
    # 每个 gpu program 处理 2 行数据
    row_len = 2    
    # 当前的 program_id * 2 得到该 block 负责的 input 数据的起始行号
    row_start = tl.program_id(0) * row_len    
    # 启动的线程块够多的时候就直接返回，什么也不做
    if row_start >= n_rows:
        return 
    # row_idx 是这个 block 处理的行的行号
    for row_idx in tl.range(row_start, row_start + row_len, 1):
        # 处理的某一行数据的行起始地址
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE) # arange生成的是一个张量，range 生成的是一个循环迭代器

        # 当前 block 处理某行中所有元素的指针
        data_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols

        # 加载一行中的所有元素, 越界的位置填充-inf,（这样 softmax 后 exp(-inf)=0，不影响结果）
        row = tl.load(data_ptrs, mask = mask, other=-float('inf'))

        # 每一行中的所有值减去该行中的最大值
        row_minus_max = row - tl.max(row, axis = 0)
        # 对每个减去最大值之后的值求一个指数
        numerator = tl.exp(row_minus_max)
        # 求完指数之后求和
        denominator = tl.sum(numerator, axis = 0)
        softmax_output = numerator / denominator

        # 输出数据的行起始地址
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_data_ptrs = output_row_start_ptr + col_offsets

        # softmax_output 就是最终输出，写回到最终的 output_data 指针里
        tl.store(output_data_ptrs, softmax_output, mask = mask)



if __name__ == '__main__':
    input_tensor = torch.randn(1000, 512, device = 'cuda') # 一个 1000 * 512 的随机输入张量
    output_tensor = torch.empty_like(input_tensor) # 用于存储输出的张量

    # 定义kernel的网格和块的大小
    n_rows, n_cols = input_tensor.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols) # 计算下一个大于等于n_cols的2的幂
    num_stages = 3

    # 调用kernel
    grid = lambda meta : (triton.cdiv(n_rows, 2), ) # 网格大小为n_rows/2向上取整
    softmax_kernel[grid](output_tensor, input_tensor, input_tensor.stride(0), output_tensor.stride(0),
    n_rows, n_cols,
    BLOCK_SIZE = BLOCK_SIZE,
    )
    output_custom = softmax_cpu(input_tensor)
    torch_output_custom = torch.softmax(input_tensor, dim=1)
    print("Triton Softmax 和 手写的 CPU Softmax 是否接近：", torch.allclose(output_tensor, output_custom, atol=1e-6))
    print("Triton Softmax 和 PyTorch Softmax 是否接近：", torch.allclose(output_tensor, torch_output_custom, atol=1e-6))
    

