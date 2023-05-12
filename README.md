# Seeded Variational Dropout

Implementation of seeded variational dropout (a.k.a. locked dropout) in Triton.

Most implementations (e.g. [1](https://github.com/allenai/allennlp/blob/main/allennlp/modules/input_variational_dropout.py), [2](https://github.com/s-nlp/certain-transformer/blob/main/src/ue4nlp/dropout_mc.py)) of variational dropout first create a mask by applying the standard dropout procedure to a tensor of ones.
The downside is that it requires the mask to be kept in memory so it can be applied to the gradients during the backward pass.

A seeded implementation works by using the same seed in the same way, so that instead of keeping the mask in memory, we only need to remember the seed.
The potential memory savings are pretty modest, so it's important that the seeded implementation isn't any slower than the conventional implementation.

This repository contains two implementations of seeded variational dropout, one based on a 1-dimensional kernel, and another based on a 2-dimensional kernel.


## Performance
Benchmark compares performance of seeded implementations to a conventional pytorch implementation.

training (block sizes are powers of two)                                                                        | inference
:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:
![](https://github.com/sfschouten/seeded-variational-dropout/blob/main/performance_report/performance_fp32.png) | ![](https://github.com/sfschouten/seeded-variational-dropout/blob/main/performance_report/performance_fp32_noisy.png)


The benchmark goes through tensors of shape [1 x 64 * 768] through [8192 x 64 * 768], simulating the application of dropout to representations of larger and larger contexts.
One variant uses only block sizes that are exact powers of two, and one simulates inference by randomly sampling block sizes.

As can be seen the 2-dimensional implementation suffers a lot from the block sizes not being exact powers of two, whereas the 1-dimensional implementation does not.
For both implementations, with the help of the autotune feature of Triton, performance is now at least as good as regular pytorch for all sizes.


The same graphs for dtype=torch.float16 and dtype=torch.float64.

torch.float16                                                                                                   | torch.float64
:--------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------:
![](https://github.com/sfschouten/seeded-variational-dropout/blob/main/performance_report/performance_fp16.png) | ![](https://github.com/sfschouten/seeded-variational-dropout/blob/main/performance_report/performance_fp64.png)

Not totally sure, but looks to me as though the torch implementation just ignores dtypes and performs the operation as though they are all torch.float32.
