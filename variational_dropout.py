import math
import random 

import torch

import triton
import triton.language as tl


def _kernel_config_pruner(configs, nargs):
    """
    We don't want to use block sizes with dimensions larger than those of the tensor we are applying the dropout to.
    """
    n = max(triton.next_power_of_2(nargs['N']), 16)
    m = max(triton.next_power_of_2(nargs['M']), 1)

    used = set()
    for config in configs:
        block_size_m = min(m, int(config.kwargs['BLOCK_SIZE_M']))
        block_size_n = min(n, int(config.kwargs['BLOCK_SIZE_N']))

        if (block_size_m, block_size_n, config.num_stages, config.num_warps) in used:
            continue

        if block_size_n * block_size_m < 256:
            #print(block_size_n, block_size_m, block_size_n*block_size_m)
            continue

        used.add((block_size_m, block_size_n, config.num_stages, config.num_warps))
        yield triton.Config({'BLOCK_SIZE_N': block_size_n, 'BLOCK_SIZE_M': block_size_m}, 
                            num_stages=config.num_stages, num_warps=config.num_warps)


@triton.autotune(configs=[
#        triton.Config({'BLOCK_SIZE_N': 1,   'BLOCK_SIZE_M': 4096}, num_warps=4),     # 4096
        triton.Config({'BLOCK_SIZE_N': 1,   'BLOCK_SIZE_M': 8192}, num_warps=4),     # 8192
        triton.Config({'BLOCK_SIZE_N': 1,   'BLOCK_SIZE_M': 16384}, num_warps=4),     # 16384
#        triton.Config({'BLOCK_SIZE_N': 1,   'BLOCK_SIZE_M': 32768}, num_warps=4),     # 32768 
#        triton.Config({'BLOCK_SIZE_N': 2,   'BLOCK_SIZE_M': 2048}, num_warps=4),     # 4096
        triton.Config({'BLOCK_SIZE_N': 2,   'BLOCK_SIZE_M': 4096}, num_warps=4),     # 8129
        triton.Config({'BLOCK_SIZE_N': 2,   'BLOCK_SIZE_M': 8192}, num_warps=4),     # 16384
#        triton.Config({'BLOCK_SIZE_N': 2,   'BLOCK_SIZE_M': 16384}, num_warps=4),     # 32768 
#        triton.Config({'BLOCK_SIZE_N': 4,   'BLOCK_SIZE_M': 1024}, num_warps=4),     # 4096
        triton.Config({'BLOCK_SIZE_N': 4,   'BLOCK_SIZE_M': 2048}, num_warps=4),     # 8129
        triton.Config({'BLOCK_SIZE_N': 4,   'BLOCK_SIZE_M': 4096}, num_warps=4),     # 16384
#        triton.Config({'BLOCK_SIZE_N': 4,   'BLOCK_SIZE_M': 8192}, num_warps=4),     # 32768 
#        triton.Config({'BLOCK_SIZE_N': 8,   'BLOCK_SIZE_M': 512},  num_warps=4),     # 4096
        triton.Config({'BLOCK_SIZE_N': 8,   'BLOCK_SIZE_M': 1024}, num_warps=4),     # 8129
        triton.Config({'BLOCK_SIZE_N': 8,   'BLOCK_SIZE_M': 2048}, num_warps=4),     # 16384
#        triton.Config({'BLOCK_SIZE_N': 8,   'BLOCK_SIZE_M': 4096}, num_warps=4),     # 32768 
#        triton.Config({'BLOCK_SIZE_N': 16,  'BLOCK_SIZE_M': 256},  num_warps=4),     # 4096
        triton.Config({'BLOCK_SIZE_N': 16,  'BLOCK_SIZE_M': 512},  num_warps=4),     # 8129
        triton.Config({'BLOCK_SIZE_N': 16,  'BLOCK_SIZE_M': 1024}, num_warps=4),     # 16384
#        triton.Config({'BLOCK_SIZE_N': 16,  'BLOCK_SIZE_M': 2048}, num_warps=4),     # 32768 
#        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_M': 128},  num_warps=4),     # 4096
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_M': 256},  num_warps=4),     # 8129
        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_M': 512},  num_warps=4),     # 16384
#        triton.Config({'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_M': 1024}, num_warps=4),     # 32768 
#        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_M': 64},   num_warps=4),     # 4096
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_M': 128},  num_warps=4),     # 8192
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_M': 256},  num_warps=4),     # 16384
#        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_M': 512},  num_warps=4),    # 32768 
#        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 32},   num_warps=4),     # 4096
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 64},   num_warps=4),     # 8192
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 128},  num_warps=4),     # 16384
#        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 256},  num_warps=4),     # 32768 
#        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 16},   num_warps=4),     # 4096
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 32},   num_warps=4),     # 8192
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 64},   num_warps=4),     # 16384
#        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 128},  num_warps=4),     # 32768     
    ],
    key=['N', 'M'],
    prune_configs_by={
        'early_config_prune': _kernel_config_pruner,
        'perf_model': None,
        'top_k': None,
    },
)
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

def seeded_variational_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous() and len(x.shape) == 2
    N, M = x.shape
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(M, meta['BLOCK_SIZE_M']))
    _variational_dropout[grid](x, output, N, M, x.stride(0), x.stride(1), p, seed)
    return output

class SeededVariationalDropout(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, x, p, seed=None):
        if not seed:
            seed = random.randrange(int(1e3))
        ctx.p = p
        ctx.seed = seed
        return seeded_variational_dropout(x, p, seed)

    @classmethod
    def backward(cls, ctx, dy):
        p = ctx.p
        seed = ctx.seed
        return seeded_variational_dropout(dy, p, seed), None, None



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
            #int(2 ** (i + random.random())) for i in range(0, 12, 1)
            2 ** i for i in range(0, 12, 1)
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
    print(f"{provider:<15} {size2} ({size})")

    #x = torch.rand(size2, device='cuda', dtype=torch.float16)
    #x = torch.rand(size2, device='cuda', dtype=torch.float32)
    x = torch.rand(size2, device='cuda', dtype=torch.float64)
    p = 0.5
    seed = 1234
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_variational_dropout(x, p, seed), quantiles=quantiles)
    if provider == 'torch-compile':
        drop_fn = torch.compile(torch_variational_dropout)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: drop_fn(x, p, seed), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: SeededVariationalDropout.apply(x, p, seed), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)



if __name__ == "__main__":

    benchmark.run(print_data=True, show_plots=True)

    # Do some tests
    seed = random.randrange(int(1e6))

    #x = torch.ones((14,18), device='cuda', dtype=torch.double, requires_grad=True)
    x = torch.randn(14, 18, dtype=torch.double, requires_grad=True, device='cuda')
    y = seeded_variational_dropout(x, 0.5, seed)

    torch.set_printoptions(precision=3, threshold=1000, linewidth=240)

    print(x)
    print(y)

    # gradient check
    dropout_fn = lambda input_: SeededVariationalDropout.apply(input_, 0.5, seed)
    result = torch.autograd.gradcheck(dropout_fn, x, eps=1e-6, atol=1e-4)
    print(f"grad check: {result}")

