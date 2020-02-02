use analysis::*;
use environment::*;
use model::*;
use std::collections::BTreeSet;

pub fn all_reduce_time(d: &device::Devices, gids: &BTreeSet<u32>, size: f64) -> f64 {
    let b_cross = d.is_cross_machine_within(gids);
    let f_factor: f64 = (gids.len() - 1) as f64 / gids.len() as f64;
    match gids.len() {
        1 => 0.0,
        _ => match b_cross {
            true => size * 2.0 * f_factor / ethernet::BANDWIDTH_ETHERNET_NCCL,
            false => size * 2.0 * f_factor / nvlink::BANDWIDTH_NVLINK_P2P / (gids.len() / 2) as f64,
        },
    }
}

pub fn dp_speedup(d: &device::Devices, m: &model::Model) -> f64 {
    let comp_time = m.perf.compute_times[0][m.perf.compute_times[0].len() - 1];
    let param_size = m.perf.parameter_sizes[0][m.perf.parameter_sizes[0].len() - 1];
    let all_gids = d.all_gpus();
    let comm_time = all_reduce_time(d, &all_gids, param_size);

    comp_time / (comp_time / d.num_gpus as f64 + comm_time)
}

pub fn dp_p3_speedup(d: &device::Devices, m: &model::Model) -> f64 {
    let comp_time = m.perf.compute_times[0][m.perf.compute_times[0].len() - 1];

    let p3stats = cc_overlap::p3(d, m);
    comp_time / (comp_time / d.num_gpus as f64 + p3stats.offset)
}

pub fn dp_speedup_strong(d: &device::Devices, compute: f64, all_reduce: f64) -> f64 {
    compute / (compute / d.num_gpus as f64 + all_reduce)
}

pub fn dp_speedup_weak(d: &device::Devices, compute: f64, all_reduce: f64) -> f64 {
    d.num_gpus as f64 * (compute / (compute + all_reduce))
}
