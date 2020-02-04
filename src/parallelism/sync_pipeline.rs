use environment::*;
use model::*;
use parallelism::*;
use std::collections::BTreeSet;

const VERBOSE: bool = false;

/// Calculate Synchronous Pipeline Length without full results
pub fn sync_pipeline_length_intermediate(
    max_pipeline_block: f64,
    m_batch: u32,
    half_stage_time_above: f64,
    half_compute_time: f64,
    cur_stage_allreduce: f64,
) -> f64 {
    half_compute_time
        + max_pipeline_block * ((m_batch - 1) as f64)
        + half_stage_time_above
        + cur_stage_allreduce
}

/// Synchronous Pipeline Speedup Estimation, Analytical Solution
pub fn sync_pipeline_speedup_analytical(
    d: &device::Devices,
    m: &model::Model,
    rp: u32,
    pipeline_time: f64,
    p: Vec<(u32, u32, u32, BTreeSet<u32>)>,
) -> f64 {
    if VERBOSE {
        println!("[sync_pipeline] analysing plan:\n{:?}", p);
    }

    let compute_times = &m.perf.compute_times;
    let activation_sizes = &m.perf.activation_sizes;
    let parameter_sizes = &m.perf.parameter_sizes;
    let output_activation_sizes = &m.perf.output_activation_sizes;

    let total_compute_time = compute_times[0][compute_times[0].len() - 1];
    let m_batch = m.global_batch_size / rp / m.min_micro_batch_size;
    if m_batch == 0 {
        return 0.0;
    }
    if VERBOSE {
        println!("[sync_pipeline]\t using m_batch = {}", m_batch);
    }

    if VERBOSE {
        for i in 0..p.len() {
            println!(
                "[sync_pipeline]\t Stage {} compute time: {}",
                i,
                compute_times[p[i].0 as usize][p[i].1 as usize] / p[i].2 as f64 / rp as f64
            );
        }
    }

    let block_time = pipeline_time / m_batch as f64;
    let pipeline_length_wout_dp =
        block_time * (m_batch - 1) as f64 + total_compute_time / rp as f64 / m_batch as f64;

    if VERBOSE {
        println!(
            "[sync_pipeline]\t block_time = {} | total/rp/m_batch = {}",
            block_time,
            total_compute_time / rp as f64 / m_batch as f64
        );
        println!(
            "[sync_pipeline]\t pipeline length without DP: {}",
            pipeline_length_wout_dp
        );
    }

    let mut pipeline_length_with_activations = pipeline_length_wout_dp;
    for i in 0..p.len() - 1 {
        let cut_activations =
            output_activation_sizes[(p[i].1 - 1) as usize] / rp as f64 / m_batch as f64;
        if VERBOSE {
            println!(
                "[sync_pipeline]\t cut_activations for stage {} ~ {} = {}, with original value = {}",
                i,
                i + 1,
                cut_activations,
                output_activation_sizes[(p[i].1 - 1) as usize]
            );
            println!(
                "[sync_pipeline] time needed for transmission = {}",
                split_concat::split_concat_all2all_time(d, &p[i].3, &p[i + 1].3, cut_activations)
            );
        }
        pipeline_length_with_activations +=
            split_concat::split_concat_all2all_time(d, &p[i].3, &p[i + 1].3, cut_activations);
    }

    if VERBOSE {
        println!(
            "[sync_pipeline] pipeline length after activations = {}",
            pipeline_length_with_activations
        );
    }

    let mut delta = 0.0;
    for i in 0..p.len() {
        let ARTime = data_parallel::all_reduce_time(
            d,
            &p[i].3,
            parameter_sizes[p[i].0 as usize][p[i].1 as usize],
        );
        if ARTime > i as f64 * block_time {
            delta = f64::max(ARTime - i as f64 * block_time, delta);
        }
    }

    if VERBOSE {
        println!(
            "[sync_pipeline] pipeline length after DP = {}",
            pipeline_length_with_activations + delta
        );
    }

    let res_speedup = total_compute_time / (pipeline_length_with_activations + delta);
    if VERBOSE {
        println!(
            "[sync_pipeline]\t speedup = {} / {}",
            total_compute_time,
            (pipeline_length_with_activations + delta)
        );
        println!(
            "[sync_pipeline]\t Estimated Speedup: {} ({:?})",
            res_speedup, p
        );
    }

    return res_speedup;
}

/// Synchronous Pipeline Speedup Calculation, Iterative Edition
pub fn sync_pipeline_speedup_recursive(
    d: &device::Devices,
    m: &model::Model,
    rp: u32,
    p: Vec<(u32, u32, u32, BTreeSet<u32>)>,
) -> f64 {
    // Shorthands
    let compute_times = &m.perf.compute_times;
    let activation_sizes = &m.perf.activation_sizes;
    let parameter_sizes = &m.perf.parameter_sizes;
    let output_activation_sizes = &m.perf.output_activation_sizes;
    let total_compute_time = compute_times[0][compute_times[0].len() - 1];

    let m_batch = m.global_batch_size / rp / m.min_micro_batch_size;

    // Construct F and B
    let mut F: Vec<f64> = vec![];
    let mut B: Vec<f64> = vec![];
    for i in 0..p.len() {
        let block_comp_time = compute_times[p[i].0 as usize][p[i].1 as usize]
            / p[i].2 as f64
            / rp as f64
            / m_batch as f64;
        F.push(block_comp_time / 2.0);
        B.push(block_comp_time / 2.0);
        if i != p.len() - 1 {
            let cut_activations = output_activation_sizes[(p[i].1 - 1) as usize] / m_batch as f64;
            let cut_comm_time =
                split_concat::split_concat_all2all_time(d, &p[i].3, &p[i + 1].3, cut_activations);
            F.push(cut_comm_time);
            B.push(cut_comm_time);
        }
    }
    if VERBOSE {
        println!("F: {:?}\nB: {:?}", F, B);
    }

    // Initialize f and b for recursive cache
    let mut f: Vec<Vec<Option<(f64, f64)>>> = vec![vec![None; F.len()]; m_batch as usize];
    let mut b: Vec<Vec<Option<(f64, f64)>>> = vec![vec![None; F.len()]; m_batch as usize];

    f[0][0] = Some((0.0, F[0]));
    for i in 1..F.len() {
        f[0][i] = Some(((f[0][i - 1].unwrap()).1, (f[0][i - 1].unwrap()).1 + F[i]));
    }

    let mut max_length = 0.0;
    for x in 0..F.len() {
        // Start Double Recursion
        let mut pipeline_end = sync_pipeline_speedup_resursive_helper_b(
            &mut f,
            &mut b,
            &F,
            &B,
            m_batch as i32 - 1,
            x as i32,
            m_batch as usize,
            F.len(),
            F.len(),
        )
        .1;
        if x % 2 == 0 {
            // Computation Block, x/2 to get to the actual comp stage number
            let ARTime = data_parallel::all_reduce_time(
                d,
                &p[x / 2].3,
                parameter_sizes[p[x / 2].0 as usize][p[x / 2].1 as usize],
            );
            pipeline_end += ARTime;
        }
        if pipeline_end > max_length {
            max_length = pipeline_end;
        }
    }

    let speedup = total_compute_time / max_length;
    if VERBOSE {
        println!(
            "[sync_pipeline]\t Recursive Pipeline Length = {}",
            max_length
        );
        println!("[sync_pipeline]\t Recursive Speedup = {}", speedup);
    }
    speedup
}

fn sync_pipeline_speedup_resursive_helper_f(
    f: &mut Vec<Vec<Option<(f64, f64)>>>,
    b: &mut Vec<Vec<Option<(f64, f64)>>>,
    F: &Vec<f64>,
    B: &Vec<f64>,
    i: i32,
    x: i32,
    M: usize,
    S: usize,
    phi: usize,
) -> (f64, f64) {
    if VERBOSE {
        println!("[sync_pipeline]\t Requesting f[{}][{}]", i, x);
    }

    if i < 0 || i >= M as i32 || x < 0 || x >= S as i32 {
        if VERBOSE {
            println!("[sync_pipeline]\t f[{}][{}] = {}", i, x, 0.0);
        }
        return (0.0, 0.0);
    }
    if f[i as usize][x as usize].is_some() {
        return f[i as usize][x as usize].unwrap();
    }
    let fm_1 =
        sync_pipeline_speedup_resursive_helper_b(f, b, F, B, i - phi as i32 + x, x, M, S, phi).1;
    let fm_2 = sync_pipeline_speedup_resursive_helper_f(f, b, F, B, i - 1, x, M, S, phi).1;
    let fm_3 = sync_pipeline_speedup_resursive_helper_f(f, b, F, B, i, x - 1, M, S, phi).1;

    let cur_f = f64::max(fm_1, f64::max(fm_2, fm_3));
    f[i as usize][x as usize] = Some((cur_f, cur_f + F[x as usize]));

    if VERBOSE {
        println!(
            "[sync_pipeline]\t f[{}][{}] = {:?}",
            i, x, f[i as usize][x as usize]
        );
    }

    return f[i as usize][x as usize].unwrap();
}

fn sync_pipeline_speedup_resursive_helper_b(
    f: &mut Vec<Vec<Option<(f64, f64)>>>,
    b: &mut Vec<Vec<Option<(f64, f64)>>>,
    F: &Vec<f64>,
    B: &Vec<f64>,
    i: i32,
    x: i32,
    M: usize,
    S: usize,
    phi: usize,
) -> (f64, f64) {
    if VERBOSE {
        println!("[sync_pipeline]\t Requesting b[{}][{}]", i, x);
    }

    if i < 0 || i >= M as i32 || x < 0 || x >= S as i32 {
        if VERBOSE {
            println!("[sync_pipeline]\t b[{}][{}] = {}", i, x, 0.0);
        }
        return (0.0, 0.0);
    }
    if b[i as usize][x as usize].is_some() {
        return b[i as usize][x as usize].unwrap();
    }
    let bm_1 = sync_pipeline_speedup_resursive_helper_b(f, b, F, B, i, x + 1, M, S, phi).1;
    // let bm_2 = sync_pipeline_speedup_resursive_helper_b(f, b, F, B, i - 1, x, M, S, phi).1;
    let bm_3 =
        sync_pipeline_speedup_resursive_helper_f(f, b, F, B, i + phi as i32 - x - 1, x, M, S, phi)
            .1;

    let cur_b = f64::max(bm_1, bm_3);
    b[i as usize][x as usize] = Some((cur_b, cur_b + B[x as usize]));

    if VERBOSE {
        println!(
            "[sync_pipeline]\t b[{}][{}] = {:?}",
            i, x, b[i as usize][x as usize]
        );
    }

    return b[i as usize][x as usize].unwrap();
}
