# Seeded Variational Dropout

Implementation of seeded variational dropout (a.k.a. locked dropout) in Triton.

Most implementations ([1](https://github.com/allenai/allennlp/blob/main/allennlp/modules/input_variational_dropout.py), [2](https://github.com/s-nlp/certain-transformer/blob/main/src/ue4nlp/dropout_mc.py)) of variational dropout first create a mask by applying the standard dropout procedure to a tensor of ones.
The downside is that it requires the mask to be kept in memory so it can be applied to the gradients during the backward pass.

A seeded implementation works by using the same seed in the same way, so that instead of keeping the mask in memory, we only need to remember the seed.
The potential memory savings are pretty modest, so it's important that the seeded implementation isn't any slower than the conventional implementation.

## Performance
Benchmark compares performance to conventional pytorch implementation , and to that same implementation but with `torch.compile`.

![Results of running benchmark for various sizes of tensors.](https://github.com/sfschouten/seeded-variational-dropout/blob/main/performance_fp32.png)

The benchmark goes through tensors of shape [1 x 64 * 768] through [8192 x 64 * 768], simulating the application of dropout to representations of larger and larger contexts.

With the autotune feature of Triton performance is now at least as good as regular pytorch for all sizes.

Something interesting happens when I do the same thing for dtype=torch.float16 and dtype=torch.float64 though.

torch.float16                                                                                                 | torch.float64
:------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------:
![Results for FP16](https://github.com/sfschouten/seeded-variational-dropout/blob/main/performance_fp16.png)  | ![Results for FP64](https://github.com/sfschouten/seeded-variational-dropout/blob/main/performance_fp64.png)
