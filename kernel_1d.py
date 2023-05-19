"""
Kernel which processes data in strings.
For example, if we have a tensor with shape: batch size, sequence length, embedding dimensionality (B, T, C),
then we process it as a vector of length B*T*C.
To ensure the same mask is applied to each token, we convert memory offsets as follows:
 - we ensure tensors are contiguous, so we know that an offset = b * T * C + t * C + c
 - we want to remove the influence of t, for which we need two values: T*C, C
   (the strides of the token dimension and the one preceding it)
 - then, we can do offset // T*C to get
"""
import math

import torch

import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({'BLOCK_SIZE': 2**n}, num_warps=4) for n in range(4, 13)],
    key=['BITS', 'N_ELEM_O']
)
@triton.jit
def _variational_dropout_kernel_1d(
        x_ptr, output_ptr, n_elem, stride_token_dim, stride_before, p, seed, BITS, N_ELEM_O, BLOCK_SIZE: tl.constexpr):
    # compute memory offsets of elements handled by this instance
    pid_n = tl.program_id(axis=0)
    block_start = pid_n * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elem
    x = tl.load(x_ptr + offsets, mask=mask)

    # create offsets that do not depend on the token index
    b = offsets // stride_before
    c = offsets % stride_token_dim
    rand_offsets = b * stride_before + c
    random = tl.rand(seed, rand_offsets)
    x_keep = random > p

    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def info(x):
    if torch.is_floating_point(x):
        return torch.finfo(x.dtype)
    else:
        return torch.iinfo(x.dtype)


def _variational_dropout_1d_launch(x, p, dim, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
    stride_token_dim = x.stride(dim)
    stride_before = x.stride(dim - 1) if dim > 0 else x.numel()
    _variational_dropout_kernel_1d[grid](
        x, output, x.numel(), stride_token_dim, stride_before, p, seed, info(x).bits, int(math.log2(x.numel())))
    return output


class SeededVariationalDropout1D(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, x, p, dim, seed):
        ctx.p = p
        ctx.dim = dim
        ctx.seed = seed
        return _variational_dropout_1d_launch(x, p, dim, seed)

    @classmethod
    def backward(cls, ctx, dy):
        return _variational_dropout_1d_launch(dy, ctx.p, ctx.dim, ctx.seed), None, None, None
