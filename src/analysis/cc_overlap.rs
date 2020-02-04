use environment::*;
use model::*;
use parallelism::*;
use rayon::prelude::*;
use std::collections::BTreeSet;

#[derive(Debug)]
struct DataBlock {
    available_time: f64,
    std_available_time: f64,
    size: f64,
    ETA: f64,
    comp_time: f64,
    need_time: f64,
    actual_start: f64,
    actual_end: f64,
    drift_back: f64,
}

#[derive(Debug)]
struct DataStats {
    blocks: Vec<DataBlock>,
    offset: f64,
    excess: f64,
}

#[derive(Debug)]
pub struct OverlapStats {
    pub offset: f64,
    pub speedup_percentage: f64,
}

pub fn p3(d: &device::Devices, m: &model::Model) -> OverlapStats {
    cc_overlap_partial(d, m, true, 1.0)
}

pub fn cc_overlap_partial(
    d: &device::Devices,
    m: &model::Model,
    inter_batch: bool,
    partial: f64,
) -> OverlapStats {
    assert_eq!((partial <= 1.0 && partial >= 0.0), true);
    let mut ds: DataStats = DataStats {
        blocks: vec![],
        offset: 0.0,
        excess: 0.0,
    };
    let all_gids = d.all_gpus();

    let mut cur_ts = 0.0;
    for i in (0..m.layers.len()).rev() {
        let l = &m.layers[i];
        let size = l.parameter_size;
        let ETA = data_parallel::all_reduce_time(d, &all_gids, size);
        // println!("ETA for {}: {}", i, ETA);
        cur_ts += l.compute_time * partial / 2.0; // should be back_time
        let available_time = cur_ts;
        ds.blocks.push(DataBlock {
            available_time: available_time,
            std_available_time: 0.0,
            size: size,
            ETA: ETA,
            comp_time: 0.0,
            need_time: 0.0,
            actual_start: 0.0,
            actual_end: 0.0,
            drift_back: 0.0,
        });
    }
    let n = ds.blocks.len();
    for i in 0..n {
        let b = &mut ds.blocks[i];
        // println!("std_available_time: {} - {}", b.available_time, cur_ts);
        b.std_available_time = b.available_time - cur_ts;
    }
    if inter_batch {
        cur_ts = 0.0;
        for i in 0..m.layers.len() {
            cur_ts += &m.layers[i].compute_time * partial / 2.0; // TODO: Fwd Time
            ds.blocks[i].need_time = cur_ts;
            ds.blocks[i].comp_time = &m.layers[i].compute_time * partial / 2.0;
        }
    }

    // println!("DS: {:?}", ds);

    ds.blocks[n - 1].drift_back += ds.blocks[n - 1].ETA;
    ds.offset += ds.blocks[n - 1].drift_back;

    for i in (0..n - 1).rev() {
        let prev_slot = ds.blocks[i + 1].available_time - ds.blocks[i].available_time;
        let after_slot = ds.blocks[i + 1].comp_time;
        if prev_slot + after_slot < ds.blocks[i].ETA {
            // need to further drift back
            if ds.blocks[i].ETA - (prev_slot + after_slot) < ds.excess {
                // use excess further
                // println!("Excess before use: {}", ds.excess);
                ds.excess -= ds.blocks[i].ETA - (prev_slot + after_slot);
            // println!("Excess after use: {}", ds.excess);
            } else {
                // println!("Excess before use: {}", ds.excess);
                ds.blocks[i].drift_back = ds.blocks[i].ETA - (prev_slot + after_slot) - ds.excess;
                ds.offset += ds.blocks[i].drift_back;
                ds.excess = 0.0;
                // ds.excess -= ds.blocks[i].ETA - (prev_slot + after_slot);
                // println!("Excess after use: {}", ds.excess);
            }
        } else {
            ds.excess += prev_slot + after_slot - ds.blocks[i].ETA;
        }
    }

    println!("Final Offset: {}", ds.offset);
    let comp_time: f64 = m.layers.par_iter().map(|s| s.compute_time).sum();
    let param_size: f64 = m.layers.par_iter().map(|s| s.parameter_size).sum();
    let comm_time: f64 = data_parallel::all_reduce_time(d, &all_gids, param_size);
    println!(
        "Pure Comp Time: {}\n Parameter Size: {}\n Pure Comm Time: {}",
        comp_time, param_size, comm_time
    );
    let max_speedup =
        ((comp_time + comm_time) - (comp_time + ds.offset)) / (comp_time + ds.offset) * 100.0;
    println!("Max Speedup Percentage: {}", max_speedup);
    OverlapStats {
        offset: ds.offset,
        speedup_percentage: max_speedup,
    }
}
