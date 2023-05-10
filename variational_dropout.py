import math

import torch

import triton
import triton.language as tl


"""

"""
@triton.jit
def _variational_dropout(x_ptr, output_ptr, N, M, stride_n, stride_m, p, seed, 
        BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    """
    X is of shape (N, M)
    where M is the dimension along which the same dropout mask is always applied.
    """
    # compute memory offsets of elements handled by this instance
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))# % N
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))# % M
    offsets = (offs_n[:, None]*stride_n + offs_m[None, :]*stride_m)
   
    # load data from x
    mask = (offs_n[:, None] < N) * (offs_m[None, :] < M)
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # randomly prune it, applying the same mask to each column
    blk_seed = seed + pid_n                                                     # Use the same seed for every block in a row
    random = tl.rand(blk_seed, offs_n)
    x_keep = random > p
    x_keep = tl.broadcast_to(x_keep[:, None], (BLOCK_SIZE_N, BLOCK_SIZE_M))
    output = tl.where(x_keep, x / (1 - p), 0.0)
    
    # write-back
    tl.store(output_ptr + offsets, output, mask=mask)


def _near_pow_2(x):
    return 2**int(math.ceil(math.log2(x)))

def seeded_variational_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous() and len(x.shape) == 2
    NR_OF_CORES = 1408
    target = NR_OF_CORES 
    N, M = x.shape
    block_size_m = min(128, triton.next_power_of_2(M))
    tgt_nr_blocks_n = target / math.ceil((M/block_size_m))
    block_size_n = min(128, triton.next_power_of_2(math.ceil(N / tgt_nr_blocks_n)))
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(M, meta['BLOCK_SIZE_M']))
    _variational_dropout[grid](x, output, N, M, x.stride(0), x.stride(1), p, seed, BLOCK_SIZE_N=block_size_n, BLOCK_SIZE_M=block_size_m)
    return output


def torch_variational_dropout(x, p, seed):
    torch.manual_seed(seed)
    ones = x.data.new_ones(x.shape[0])
    dropout_mask = torch.nn.functional.dropout(ones, p=p)
    return dropout_mask.unsqueeze(1) * x


N = 64 * 768

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2 ** i for i in range(0, 14, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch', 'torch-compile'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch', 'Torch w/ compile'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='variational-dropout-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    size2 = (N, size)
    size = N * size
    print(size, size2)

    x = torch.rand(size2, device='cuda', dtype=torch.float32)
    p = 0.5
    seed = 1234
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_variational_dropout(x, p, seed), quantiles=quantiles)
    if provider == 'torch-compile':
        drop_fn = torch.compile(torch_variational_dropout)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: drop_fn(x, p, seed), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: seeded_variational_dropout(x, p, seed), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)



if __name__ == "__main__":

    benchmark.run(print_data=True, show_plots=True)

    # Do some kind of tests
    x = torch.ones((14,18), device='cuda')
    y = variational_dropout(x, 0.5, 123)

    torch.set_printoptions(linewidth=160)

    print(x)
    print(y)

