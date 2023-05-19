import random

import torch

import triton

import kernel_1d
import kernel_2d_features
import kernel_2d_tokens


def _torch_variational_dropout(x, p, dim):
    mask_size = tuple(1 if i == dim else d for i,d in enumerate(x.size()))
    ones = x.data.new_ones(mask_size)
    dropout_mask = torch.nn.functional.dropout(ones, p=p)
    return dropout_mask * x


def variational_dropout(x, p, dim=0, seed=None):
    """
    Params:
        dim: The dimension along which the same dropout mask will be applied.
    """
    if not seed:
        seed = random.randrange(int(1e6))

    if x.is_cuda:
        #M = x.shape[dim]
        #if len(x.shape) == 2 and M & (M-1) == 0: # check if power of 2
        #    return kernel_2d_tokens.SeededVariationalDropout2D.apply(x, p, dim, seed)
        #else:
        return kernel_1d.SeededVariationalDropout1D.apply(x, p, dim, seed)
    else:
        torch.manual_seed(seed)
        return _torch_variational_dropout(x, p, dim)


# == testing code ==
N = 64
C = 768


def _create_benchmark(x_vals, name, **kwargs):
    return triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=x_vals,
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=[
            'triton-1d',
            #'triton-2d-tokens',
            'triton-2d-features',
            #'torch',
            'torch-compile'
        ],  # Possible values for `line_arg`.
        line_names=[
            'Triton1D',
            #'Triton2DTokens',
            'Triton2DFeatures',
            #'Torch',
            'Torch w/ compile'
        ],  # Label name for the lines.
        styles=[
            ('blue', '-'),
            #('green', '-'),
            ('red', '-'),
            #('orange', '-'),
            ('purple', '-'),
        ],  # Line styles.
        ylabel='GB/s',   # Label name for the y-axis.
        plot_name=name,  # Name for the plot. Used also as a file name for saving the plot.
        args=kwargs      # Values for function arguments not in `x_names` and `y_name`.
    )


def powers_of_two(start, end, noisy):
    return [int(2**(i+(random.random() if noisy else 0))) for i in range(start, end)]


@triton.testing.perf_report([
    _create_benchmark(powers_of_two(0, 12, False), 'performance_fp32', dtype=torch.float32),
    #_create_benchmark(powers_of_two(0, 12, True),  'performance_fp32_noisy', dtype=torch.float32),
    _create_benchmark(powers_of_two(0, 12, False), 'performance_fp16', dtype=torch.float16),
    #_create_benchmark(powers_of_two(0, 12, True),  'performance_fp16_noisy', dtype=torch.float16),
    #_create_benchmark(powers_of_two(0, 11, False), 'performance_fp64', dtype=torch.float64),
    #_create_benchmark(powers_of_two(0, 11, True),  'performance_fp64_noisy', dtype=torch.float64),
])
def _benchmark(size, provider, dtype):
    x = torch.empty((N, size, C), device='cuda', dtype=dtype)

    print(f"{provider:<15} {str(tuple(x.shape)):<14} ({x.numel()}) ({x.dtype})")

    seed = random.randrange(int(1e6))
    p = 0.5
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        torch.manual_seed(seed)
        fn = _torch_variational_dropout
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(x, p, 1), quantiles=quantiles)
    if provider == 'torch-compile':
        torch.manual_seed(seed)
        fn = torch.compile(_torch_variational_dropout)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(x, p, 1), quantiles=quantiles)
    if provider == 'triton-1d':
        fn = kernel_1d.SeededVariationalDropout1D.apply
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(x, p, 1, seed), quantiles=quantiles)
    if provider == 'triton-2d-tokens':
        fn = kernel_2d_tokens.SeededVariationalDropout2D.apply
        t = lambda _x: _x.transpose(1, 2).reshape(N*C, size)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(t(x), p, 1, seed), quantiles=quantiles)
    if provider == 'triton-2d-features':
        fn = kernel_2d_features.SeededVariationalDropout2D.apply
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: fn(x, p, 1, seed), quantiles=quantiles)

    gbps = lambda ms: x.numel() * (torch.finfo(dtype).bits / 8) / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":

    # Do some tests
    dropout_fns = [
        kernel_1d.SeededVariationalDropout1D.apply,
        #kernel_2d_tokens.SeededVariationalDropout2D.apply,
        #kernel_2d_features.SeededVariationalDropout2D.apply,
    ]
    seed = random.randrange(int(1e6))
    for dropout_fn in dropout_fns:
        
        #x = torch.ones((14,18), device='cuda', dtype=torch.double, requires_grad=True)
        x = torch.ones((3, 4, 5), device='cuda', dtype=torch.double, requires_grad=True)
        #x = torch.randn(14, 18, dtype=torch.double, requires_grad=True, device='cuda')
        y = dropout_fn(x, 0.5, 2, seed)

        torch.set_printoptions(precision=3, threshold=1000, linewidth=240)

        print(x)
        print(y)

        # gradient check
        _fn = lambda input_: dropout_fn(input_, 0.5, len(input_.size())-1, seed)
        result = torch.autograd.gradcheck(_fn, x, eps=1e-6, atol=1e-4)
        print(f"grad check: {result}")

    # test performance
    _benchmark.run(print_data=True, save_path='./performance_report/')
    
    for key, value in kernel_1d._variational_dropout_kernel_1d.cache.items():
        print(f"{key}: {value.kwargs}")
    for key, value in kernel_2d_features._variational_dropout_kernel_2d.cache.items():
        print(f"{key}: {value.kwargs}")



