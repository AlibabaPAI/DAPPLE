use environment::*;
use model::*;
use parallelism::*;
use rayon::prelude::*;
use std::collections::BTreeSet;

pub fn max_single_gpu_batch_size(m: &model::Model) -> u32 {
    // assume m is already under GBS
    let mut param_size: f64 = m.layers.par_iter().map(|s| s.parameter_size).sum();
    param_size *= m.optimizer_memory_scaling as f64;
    let activation_size: f64 = m.layers.par_iter().map(|s| s.activation_size).sum();
    println!("Parameter Size: {}", param_size);
    println!("Activation Size: {}", activation_size);
    println!("GBS = {}", m.global_batch_size);
    ((device::GPU_MEMORY - param_size) / m.peak_activation_per_batch) as u32
}
