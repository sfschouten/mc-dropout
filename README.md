# Seeded Variational Dropout

Implementation of seeded variational dropout in Triton.

Benchmark compares performance to pytorch implementation and that implementation with `torch.compile`, the result of which can be seen in the plot below.

![Results of running benchmark for various sizes of tensors.](https://github.com/sfschouten/seeded-variational-dropout/blob/main/Figure_1.png)

The benchmark goes through tensors of shape [1 x 64 * 768] through [8192 x 64 * 768], simulating the application of dropout to representations of larger and larger context windows.

As can be seen, performance is initially a bit better than the baselines for smaller tensors, but then performs worse for larger tensors.
I'm not sure why that is, possibly I did something suboptimal in deciding the kernel block sizes, this is something I want to improve.

