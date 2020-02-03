# DAPPLE: An Efficient Pipelined Data Parallel Approach for Large Models Training

DAPPLE is a distributed training framework which combines pipeline parallelism
and data parallelism to address aforementioned scheduling and planning challenges with synchronous training.
This framework features a profiler, a [planner](https://github.com/AlibabaPAI/DAPPLE/tree/master/planner)
and a runtime system.
The profiler takes a user’s DNN model as input, and profiles execution time, activation and parameter sizes for each layer.
Sample profiling results for some models are given in [profiling results](https://github.com/AlibabaPAI/DAPPLE/tree/master/profiling_results).
Taking profiling results as input, DAPPLE planner generates an optimized hybrid parallelization plan on a given global batch size,
which is further split into multiple micro-batches and scheduled for execution by DAPPLE runtime.

This repository contains the source code implementation of DAPPLE's planning results on
5 typical models:
[VGG19](https://github.com/AlibabaPAI/DAPPLE/tree/master/vgg19),
[AmoebaNet](https://github.com/AlibabaPAI/DAPPLE/tree/master/amoeba_net),
[BERT](https://github.com/AlibabaPAI/DAPPLE/tree/master/bert),
[GNMT](https://github.com/AlibabaPAI/DAPPLE/tree/master/gnmt),
and [XLNET](https://github.com/AlibabaPAI/DAPPLE/tree/master/xlnet).