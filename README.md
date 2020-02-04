# DAPPLE: An Efficient Pipelined Data Parallel Approach for Large Models Training

DAPPLE is a distributed training framework which combines pipeline parallelism
and data parallelism to address aforementioned scheduling and planning challenges with synchronous training.
This framework features a profiler, a [planner](https://github.com/AlibabaPAI/DAPPLE/tree/master/src)
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

## Running the DAPPLE experiments
Please see the launch script `run.sh` for each model for details.

## Using the Planner
### Install from Python PyPI, as a Python3 package
```bash
pip3 install HPGO
```

### Build from source
```bash
rustup default nightly
cargo build --release
maturin build --release
pip3 install xxx.whl
```

### Example Usage of Python API
```python
# Import HPGO Python API
import HPGO
# Construct the Conductor object
# conductor_from_torch_graph_and_seps(profile_filename, profile_batch_size, global_batch_size, devices)
conductor = HPGO.conductor_from_torch_graph_and_seps("./profiling_results/xlnet-36-pbs-1.txt", 1, 128, [8, 16])
result = conductor.py_orchestrate()
print(result)
```

## License
The DAPPLE Planner is open sourced under the terms of BSD-3-Clause, details of which can be found in the [`src/LICENSE.md`](src/LICENSE.md) file

The file [`src/input/torch_graph_py.rs`](`src/input/torch_graph_py.rs`) contains Python source code from [PipeDream](https://github.com/msr-fiddle/pipedream), which is licensed under the MIT License.
