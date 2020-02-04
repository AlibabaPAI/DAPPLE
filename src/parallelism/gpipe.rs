use environment::*;
use model::*;
use parallelism::*;
use std::collections::BTreeSet;

/// Calculate the speedup for the partition, assuming GPipe microbatch arrangement
pub fn gpipe_pipeline_speedup(
    d: &device::Devices,
    m: &model::Model,
    rp: u32,
    pipeline_time: f64,
    p: Vec<(u32, u32, u32, BTreeSet<u32>)>,
) -> f64 {
    unimplemented!()
}
