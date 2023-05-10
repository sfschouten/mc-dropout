# Seeded Variational Dropout

Implementation of seeded variational dropout in Triton.

Benchmark compares performance to pytorch implementation and that implementation with `torch.compile`, the result of which can be seen in the plot below.

![Results of running benchmark for various sizes of tensors.](https://github.com/sfschouten/seeded-variational-dropout/blob/main/Figure_1.png)
