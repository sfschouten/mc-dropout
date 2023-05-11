# Seeded Variational Dropout

Implementation of seeded variational dropout (a.k.a. locked dropout) in Triton.

Most implementations of variational dropout first create a mask by applying the standard dropout procedure to a tensor of ones.
The downside is that it requires the mask to be kept in memory so it can be applied to the gradients during the backward pass.

A seeded implementation works by using the same seed in the same way, so that instead of keeping the mask in memory, we only need to remember the seed.
The potential memory savings are pretty modest, so it's important that the seeded implementation isn't any slower than the conventional implementation.

## Performance
Benchmark compares performance to conventional pytorch implementation , and to that same implementation but with `torch.compile`.

![Results of running benchmark for various sizes of tensors.](https://github.com/sfschouten/seeded-variational-dropout/blob/main/Figure_1.png)

The benchmark goes through tensors of shape [1 x 64 * 768] through [8192 x 64 * 768], simulating the application of dropout to representations of larger and larger contexts.

As can be seen, performance is initially a bit better than the baselines for smaller tensors, but then performs worse for larger tensors.
I'm not sure why that is, possibly I did something suboptimal in deciding the kernel block sizes, this is something I want to improve.

I've played around with the Triton's `autotune` feature a bit, but so far without success.

