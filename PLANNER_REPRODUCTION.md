# DAPPLE Planner Experiments Reproduction

## Note
* By default, the Planner will use all the available CPU cores in the system for faster planning. You can limit the parallel degree by setting the environment variable `RAYON_NUM_THREADS=4`
* The Python API of the planner does not release Python's Global Interpreter Lock (GIL), therefore cannot effectively use multiple cores at their full potential. The Rust API does not have this limitation.
* We ran all of our experiments on a 48c96t Xeon processor, YMMV
* The strategies and estimated speedups can then be verified with our DAPPLE Runtime, details of those experiments can be found in RUNTIME_EXPERIMENTS.md.
* All of our performance data was profiled on NVIDIA V100 GPUs, so for other cards you might need to redo the profiling.

## Installation (Python API)
### Prerequisites
The DAPPLE Planner requires the following package present in the system:
 
* Python 3 (>= 3.6)
* GraphViz
* Python-GraphViz

### Installing the Planner from PyPI
```bash
pip3 install HPGO==0.9.3
# version locked to 0.9.3 for reproducibility
```


## Installation (Rust API)
You will only need this if you want to call a Rust-API of the planner, normally due to performance concerns of Python's GIL. The Rust API allows full lock-free multi-threading for faster planning, while the Python API could only utilize a single core.

### Additional Requirements
To build the planner from source or install from Cargo, you will need the following packages in addition to those mentioned above:

* Rust (nightly > 2019-11-16)
* Cargo
* Maturin

### Install from Cargo

```bash
cargo install HPGO
```

### Build from source

```bash
# Switch to nightly channel
rustup default nightly
# Build All (library + examples + tests)
cargo build --release --all
# Build Python Wheel
maturin build --release
```

## Reproducing the Strategies

### Profiling
Currently our profiling is done offline, and the results are cached within the `profiles` folder

### Planning for a specific model at a single Global Batch Size

with the Python API

```python
# Import HPGO Python API
import HPGO
# Construct the Conductor object
# conductor_from_torch_graph_and_seps(profile_filename, profile_batch_size, global_batch_size, devices)
conductor = HPGO.conductor_from_torch_graph_and_seps("./profiling_results/xlnet-36-pbs-1.txt", 1, 128, [8, 16])
result = conductor.py_orchestrate()
print(result)
```

### Planning for a specific model at multiple Global Batch Sizes

multi-threaded planning with the Rust API. Here we use ResNet-50, GBS ranging from 32 - 4096.

```rust
#![allow(non_snake_case)]

use ordered_float::OrderedFloat;
use rayon::prelude::*;

use std::collections::BTreeSet;

use HPGO::environment::*;
use HPGO::input::*;
use HPGO::layerwise::model::*;
use HPGO::layerwise::orchestration::*;
use HPGO::layerwise::parallelism::*;

const VERBOSE: bool = true;

struct ModelConfig {
    gbs: Vec<u32>,    // GBS vector
    filename: String, // filename to TorchGraph txt
    optimizer_memory_scaling: u32,
    pbs: u32,
    mbs: u32,
    papb: f64,
}

fn test_speedup_at_all_bs(mc: ModelConfig, flat: bool) {
    let d: device::Devices;
    if flat {
        d = get_flat_devices();
    } else {
        d = get_hierarchical_devices();
    }

    // model at pbs
    if VERBOSE {
        println!("[main]\t Importing Model from TorchGraph...")
    }
    let tgi: torch_graph::TorchGraphImporter = LayerwiseModelImporter::new();
    let result = tgi.ImportFrom(&mc.filename);
    let (perf, states) = (result.0.unwrap(), result.1.unwrap());
    if VERBOSE {
        println!("[main]\t Constructing HPGO Model...")
    }
    let mut m0 = model::Model::new_from_model_perf(perf, states, mc.pbs, mc.pbs);
    m0.optimizer_memory_scaling = mc.optimizer_memory_scaling;
    m0.min_micro_batch_size = mc.mbs;
    if mc.papb > 0.0 {
        m0.peak_activation_per_batch = mc.papb;
    }

    if VERBOSE {
        println!("[main]\t Model Import Complete. Starting Parallel Planning...")
    }

    // Compute Multiple Batch Sizes in Parallel
    let res: Vec<_> = mc
        .gbs
        .par_iter()
        .map(|gbs| {
            if VERBOSE {
                println!("[main]\t Planning in parallel for bs = {} ...", *gbs);
            }
            let m1 = m0.normalized_copy(*gbs);

            // DP Speedups
            let dp_speedup = data_parallel::dp_speedup(&d, &m1);
            // let dp_p3_speedup = data_parallel::dp_p3_speedup(&d16, &model);
            let dp_ga_p3_speedup = gradient_accumulation::dp_cur_ga_p3_speedup(&d, &m1);
            let dp_ga_inner_overlap_speedup =
                gradient_accumulation::dp_cur_ga_inner_overlap_speedup(&d, &m1);

            // Hybrid Parallelism Speedups
            let mut c =
                orchestrate_async::AsyncOrchestrate::new_from_model_device(m1.clone(), d.clone());
            c.orchestrate();
            let mut pipeline_speedup = 0.0;
            let mut pipeline_stages: Vec<(u32, u32, u32, BTreeSet<u32>)> = vec![];

            let best_hp = c
                .res
                .into_par_iter()
                .max_by_key(|r| OrderedFloat(r.speedup))
                .unwrap();
            pipeline_speedup = best_hp.speedup;
            pipeline_stages = best_hp.stages;

            if VERBOSE {
                println!("[main]\t Got all results for bs = {} ...", *gbs);
            }

            // return gbs and all speedups
            (
                gbs,
                (
                    dp_speedup,
                    dp_ga_p3_speedup,
                    dp_ga_inner_overlap_speedup,
                    pipeline_speedup,
                    pipeline_stages,
                ),
            )
        })
        .collect();

    println!("Global Batch Size, DP No Overlap, DP+P3, DP+Normal Overlap, Best Hybrid Speedup | Best Hybrid Solution");
    for i in res {
        println!(
            "{}, {}, {}, {}, {} | {:?}",
            i.0,
            (i.1).0,
            (i.1).1,
            (i.1).2,
            (i.1).3,
            (i.1).4,
            // (i.1).7,
        );
    }
}
fn main() {
    test_speedup_at_all_bs(get_resnet50_model_config(), false);
}

/// Data Area

// Seps Array for Flat and Hierarchical
fn get_hierarchical_devices() -> device::Devices {
    device::Devices::new(16, vec![8, 16])
}

fn get_flat_devices() -> device::Devices {
    device::Devices::new(
        16,
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    )
}

fn get_resnet50_model_config() -> ModelConfig {
    let mut gbs = vec![32, 64];
    for i in 1..((4096 - 64) / 64) + 1 {
        gbs.push(64 + i * 64);
    }
    ModelConfig {
        gbs: gbs,
        filename: ["./profiles/", "resnet50", "/graph.txt"].join(""),
        optimizer_memory_scaling: 2,
        pbs: 32,
        mbs: 32,
        papb: 70000000.0,
    }
}

```

## Reproducing the Speedup Estimation

## Reproducing the Scalability Prediction

