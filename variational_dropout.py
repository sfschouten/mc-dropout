import math
import random 

import torch

import triton
import triton.language as tl


# == 2D kernel implementation ==

MAX_NR_ELEM_BITS = 15

def _kernel_config_pruner(configs, nargs):
    """
    We don't want to use block sizes with dimensions larger than those of the tensor we are applying the dropout to.
    """
    n = max(triton.next_power_of_2(nargs['N']), 1)
    m = max(triton.next_power_of_2(nargs['M']), 1)

    used = set()
    for config in configs:
        block_size_n = min(n, int(config.kwargs['BLOCK_SIZE_N']))
        block_size_m = min(m, int(config.kwargs['BLOCK_SIZE_M']))

        if (block_size_m, block_size_n, config.num_stages, config.num_warps) in used:
            continue
        
        if (block_size_n * block_size_m) < 2**9 or (block_size_n * block_size_m) > 2**MAX_NR_ELEM_BITS: 
            #TODO test these boundaries on other GPUs
            continue

        used.add((block_size_m, block_size_n, config.num_stages, config.num_warps))
        yield triton.Config({'BLOCK_SIZE_N': block_size_n, 'BLOCK_SIZE_M': block_size_m}, num_stages=config.num_stages, num_warps=config.num_warps)


@triton.autotune(configs=[
        triton.Config({'BLOCK_SIZE_N': 2**n, 'BLOCK_SIZE_M': 2**m}, num_warps=4)
        for n in range(0, 10) 
        for m in range(0, MAX_NR_ELEM_BITS) 
    ],
    key=['BITS', 'N_O', 'M_O'],
    prune_configs_by={
        'early_config_prune': _kernel_config_pruner,
        'perf_model': None,
        'top_k': None,
    },
)
@triton.jit
def _variational_dropout_kernel_2D(x_ptr, output_ptr, N, M, stride_n, stride_m, p, seed, BITS, N_O, M_O,
                        BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr):
    """
    X is of shape (N, M)
    where M is the dimension along which the same dropout mask is always applied.
    """
    # compute memory offsets of elements handled by this instance
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offsets = (offs_n[:, None]*stride_n + offs_m[None, :]*stride_m)
   
    # load data from x
    mask = (offs_n[:, None] < N) * (offs_m[None, :] < M)
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # randomly prune it, applying the same mask to each column
    random = tl.rand(seed, offs_n)                                          # generate dropout mask of (N, 1), apply to each column
    x_keep = random > p
    x_keep = tl.broadcast_to(x_keep[:, None], (BLOCK_SIZE_N, BLOCK_SIZE_M))
    output = tl.where(x_keep, x / (1 - p), 0.0)
    
    # write-back
    tl.store(output_ptr + offsets, output, mask=mask)


def _variational_dropout_2D_launch(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous() and len(x.shape) == 2
    N, M = x.shape
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(M, meta['BLOCK_SIZE_M']))
    _variational_dropout_kernel_2D[grid](
            x, output, N, M, x.stride(0), x.stride(1), p, seed, torch.finfo(x.dtype).bits, int(math.log2(N)), int(math.log2(M)))
    return output

class SeededVariationalDropout2D(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, x, p, seed):
        ctx.p = p
        ctx.seed = seed
        return _variational_dropout_2D_launch(x, p, seed)

    @classmethod
    def backward(cls, ctx, dy):
        p = ctx.p
        seed = ctx.seed
        return _variational_dropout_2D_launch(dy, p, seed), None, None




# == 1D kernel implementation ==

@triton.autotune(configs=[
        triton.Config({'BLOCK_SIZE': 2**n}, num_warps=4)
        for n in range(4, 13) 
    ], key=['BITS', 'N_ELEM_O'] 
)
@triton.jit
def _variational_dropout_kernel_1D(x_ptr, output_ptr, n_elem, n_cols, p, seed, BITS, N_ELEM_O, BLOCK_SIZE: tl.constexpr):
    """

    """
    # compute memory offsets of elements handled by this instance
    pid_n = tl.program_id(axis=0)
    block_start = pid_n * BLOCK_SIZE 
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elem
    x = tl.load(x_ptr + offsets, mask=mask)

    if n_cols > 1:
        rand_offsets = offsets // n_cols
    else:
        rand_offsets = offsets
    # convert offsets into row indices, and use those resulting in the same seed being used for the same rows
    random = tl.rand(seed, rand_offsets)
    x_keep = random > p
    
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def _variational_dropout_1D_launch(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous() and len(x.shape) == 2
    N, M = x.shape
    n_elem = N * M
    grid = lambda meta: (triton.cdiv(n_elem, meta['BLOCK_SIZE']),)
    _variational_dropout_kernel_1D[grid](x, output, n_elem, M, p, seed, torch.finfo(x.dtype).bits, int(math.log2(n_elem)))
    return output



class SeededVariationalDropout1D(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, x, p, seed):
        ctx.p = p
        ctx.seed = seed
        return _variational_dropout_1D_launch(x, p, seed)

    @classmethod
    def backward(cls, ctx, dy):
        p = ctx.p
        seed = ctx.seed
        return _variational_dropout_1D_launch(dy, p, seed), None, None

def _torch_variational_dropout(x, p, seed=None):
    if not seed:
        seed = random.randrange(int(1e6))
        torch.manual_seed(seed)
    ones = x.data.new_ones(x.shape[0])
    dropout_mask = torch.nn.functional.dropout(ones, p=p)
    return dropout_mask.unsqueeze(1) * x

def variational_dropout(x, p, seed=None):
    if not seed:
        seed = random.randrange(int(1e6))

    if x.is_cuda:
        M = x.shape[-1]
        if M & (M-1) == 0: # check if power of 2
            return SeededVariationalDropout2D.apply(x, p, seed)
        else:
            return SeededVariationalDropout1D.apply(x, p, seed)
    else:
        return _torch_variational_dropout(x, p, seed)



# == testing code ==
N = 64 * 768


def _create_benchmark(x_vals, name, **kwargs):
    return triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=x_vals,
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton-1d', 'triton-2d', 'torch', 'torch-compile'],  # Possible values for `line_arg`.
        line_names=['Triton1D', 'Triton2D', 'Torch', 'Torch w/ compile'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('orange', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name=name,  # Name for the plot. Used also as a file name for saving the plot.
        args=kwargs # Values for function arguments not in `x_names` and `y_name`.
    )

def powers_of_two(start, end, noisy):
    return [int(2**(i+(random.random() if noisy else 0))) for i in range(start, end)]


@triton.testing.perf_report([
    _create_benchmark(powers_of_two(0,14,False), 'performance_fp16', dtype=torch.float16),
    _create_benchmark(powers_of_two(0,13,False), 'performance_fp32', dtype=torch.float32),
    _create_benchmark(powers_of_two(0,12,False), 'performance_fp64', dtype=torch.float64),
    _create_benchmark(powers_of_two(0,13,True),  'performance_fp16_noisy', dtype=torch.float16),
    _create_benchmark(powers_of_two(0,12,True),  'performance_fp32_noisy', dtype=torch.float32),
    _create_benchmark(powers_of_two(0,11,True),  'performance_fp64_noisy', dtype=torch.float64),
])
def _benchmark(size, provider, dtype):
    x = torch.empty((N,size), device='cuda', dtype=dtype)
    
    n_elements = x.shape[0] * x.shape[1]
    print(f"{provider:<15} {str(tuple(x.shape)):<14} ({n_elements})")

    p = 0.5
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: _torch_variational_dropout(x, p), quantiles=quantiles)
    if provider == 'torch-compile':
        drop_fn = torch.compile(_torch_variational_dropout)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: drop_fn(x, p), quantiles=quantiles)
    if provider == 'triton-1d':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: SeededVariationalDropout1D.apply(x, p, random.randrange(int(1e6))), quantiles=quantiles)
    if provider == 'triton-2d':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: SeededVariationalDropout2D.apply(x, p, random.randrange(int(1e6))), quantiles=quantiles)
    gbps = lambda ms: n_elements * (torch.finfo(dtype).bits / 8) / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":

    # Do some tests
    dropout_fns = [
        SeededVariationalDropout1D.apply,
        SeededVariationalDropout2D.apply,
    ]
    seed = random.randrange(int(1e6))
    for dropout_fn in dropout_fns:
        
        x = torch.ones((14,18), device='cuda', dtype=torch.double, requires_grad=True)
        #x = torch.randn(14, 18, dtype=torch.double, requires_grad=True, device='cuda')
        y = dropout_fn(x, 0.5, seed)

        torch.set_printoptions(precision=3, threshold=1000, linewidth=240)

        print(x)
        print(y)

        # gradient check
        _fn = lambda input_: dropout_fn(input_, 0.5, seed)
        result = torch.autograd.gradcheck(_fn, x, eps=1e-6, atol=1e-4)
        print(f"grad check: {result}")


    # test performance
    _benchmark.run(print_data=True, save_path='./performance_report/')
    
    import pprint
    for key, value in _variational_dropout_kernel_1D.cache.items(): 
        print(f"{key}: {value.kwargs}")
    for key, value in _variational_dropout_kernel_2D.cache.items(): 
        print(f"{key}: {value.kwargs}")



