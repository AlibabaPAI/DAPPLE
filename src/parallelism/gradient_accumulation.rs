use analysis::*;
use environment::*;
use model::*;
use parallelism::data_parallel;
use std::collections::BTreeSet;

pub fn dp_ga_speedup(d: &device::Devices, m: &model::Model) -> f64 {
    data_parallel::dp_speedup(d, m)
}

/// get the current GA iteration batch size per machine
pub fn current_ga_iter_size(d: &device::Devices, m: &model::Model) -> u32 {
    let max_bs = gpu_memory::max_single_gpu_batch_size(m);
    let gbs = m.global_batch_size;

    if max_bs >= gbs {
        gbs
    } else {
        let mut iter = gbs / max_bs;
        if iter * max_bs < gbs {
            iter += 1
        }
        while iter < gbs / 2 + 1 {
            if gbs % iter != 0 {
                iter += 1;
            } else {
                break;
            }
        }
        gbs / iter
    }
}

pub fn optimal_ga_iter_size(d: &device::Devices, m: &model::Model) -> u32 {
    let max_bs = gpu_memory::max_single_gpu_batch_size(m);
    let gbs = m.global_batch_size;
    if max_bs >= gbs {
        gbs
    } else if max_bs * 2 >= gbs {
        gbs / 2
    } else {
        max_bs
    }
}

pub fn dp_cur_ga_p3_speedup(d: &device::Devices, m: &model::Model) -> f64 {
    dp_ga_overlap_speedup(d, m, current_ga_iter_size(d, m), true)
}

pub fn dp_opt_ga_p3_speedup(d: &device::Devices, m: &model::Model) -> f64 {
    dp_ga_overlap_speedup(d, m, optimal_ga_iter_size(d, m), true)
}

pub fn dp_cur_ga_inner_overlap_speedup(d: &device::Devices, m: &model::Model) -> f64 {
    dp_ga_overlap_speedup(d, m, current_ga_iter_size(d, m), false)
}

pub fn dp_ga_overlap_speedup(
    d: &device::Devices,
    m: &model::Model,
    ga_size: u32,
    inter_batch_overlap: bool,
) -> f64 {
    let comp_time = m.perf.compute_times[0][m.perf.compute_times[0].len() - 1];
    let gbs = m.global_batch_size;
    let partial: f64 = ga_size as f64 / gbs as f64;

    let overlap_stats = cc_overlap::cc_overlap_partial(d, m, inter_batch_overlap, partial);
    comp_time / (comp_time / d.num_gpus as f64 + overlap_stats.offset)
}
