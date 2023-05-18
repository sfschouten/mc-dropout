"""
Kernel which processes data in blocks of tokens.
For example, if we have a tensor with shape: sequence length, batch_size, embedding dimensionality (T, B, C),
then we process it as (T, B*C).
 (+) we can sample one dropout mask of size (1, B*C) and apply it to all tokens in the block, saving on sampling cost
 (-) token dimension needs to be last (or first, TODO), which often requires a transpose
"""
import math

import torch

import triton
import triton.language as tl


MAX_NR_ELEM_BITS = 15


def _kernel_config_pruner(configs, nargs):
    """
    We don't want to use block sizes with dimensions larger than those of the tensor we are applying the dropout to.
    """
    N, M = nargs['N'], nargs['M']
    n = max(triton.next_power_of_2(N), 1)
    m = max(triton.next_power_of_2(M), 1)

    used = set()
    for config in configs:
        block_size_n = min(n, int(config.kwargs['BLOCK_SIZE_N']))
        block_size_m = min(m, int(config.kwargs['BLOCK_SIZE_M']))

        if (block_size_m, block_size_n, config.num_stages, config.num_warps) in used:
            continue

        if (block_size_n * block_size_m) < min(N*M, 2 ** 9) or (block_size_n * block_size_m) > 2 ** MAX_NR_ELEM_BITS:
            # TODO test these boundaries on other GPUs
            continue

        used.add((block_size_m, block_size_n, config.num_stages, config.num_warps))
        yield triton.Config({'BLOCK_SIZE_N': block_size_n, 'BLOCK_SIZE_M': block_size_m}, num_stages=config.num_stages,
                            num_warps=config.num_warps)


@triton.autotune(configs=[
    triton.Config({'BLOCK_SIZE_N': 2 ** n, 'BLOCK_SIZE_M': 2 ** m}, num_warps=4)
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
def _variational_dropout_kernel_2d(x_ptr, output_ptr, N, M, stride_n, stride_m, p, seed, BITS, N_O, M_O,
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
    mask = (offs_n[:, None] < N) * (offs_m[None, :] < M)
    offs_n = offs_n * stride_n
    offs_m = offs_m * stride_m
    offsets = offs_n[:, None] + offs_m[None, :]

    # load data from x
    x = tl.load(x_ptr + offsets, mask=mask)

    # randomly prune it, applying the same mask to each column
    random = tl.rand(seed, offs_n)  # generate dropout mask of (N, 1), apply to each column
    x_keep = random > p
    x_keep = tl.broadcast_to(x_keep[:, None], (BLOCK_SIZE_N, BLOCK_SIZE_M))
    output = tl.where(x_keep, x / (1 - p), 0.0)

    # write-back
    tl.store(output_ptr + offsets, output, mask=mask)


def _variational_dropout_2d_launch(x, p, dim, seed):
    output = torch.empty_like(x)
    if dim not in (-1, len(x.size())-1):
        raise NotImplementedError('Right now kernel requires the tokens to be the last dimension.')
    x = x.view(-1, x.shape[dim])
    assert x.is_contiguous()
    N, M = x.shape
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(M, meta['BLOCK_SIZE_M']))
    _variational_dropout_kernel_2d[grid](
        x, output, N, M, x.stride(0), x.stride(1), p, seed, torch.finfo(x.dtype).bits, int(math.log2(N)),
        int(math.log2(M))
    )
    return output


class SeededVariationalDropout2D(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, x, p, dim, seed):
        ctx.p = p
        ctx.dim = dim
        ctx.seed = seed
        return _variational_dropout_2d_launch(x, p, dim, seed)

    @classmethod
    def backward(cls, ctx, dy):
        return _variational_dropout_2d_launch(dy, ctx.p, ctx.dim, ctx.seed), None, None, None

