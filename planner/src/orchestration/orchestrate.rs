use environment::device;
use input::*;
use itertools::sorted;
use model::model;
use orchestration::{Orchestrate, OrchestrationResult};
use ordered_float::OrderedFloat;
use parallelism::*;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

const VERBOSE: bool = false;

pub type bitset = Vec<bool>;

#[derive(Debug)]
pub struct MatrixCell {
    pub current_value: Option<f64>,
    pub current_maxmin_block: Option<f64>,
    pub optimal_split: Option<(u32, u32)>,
    pub num_gpus_used: Option<u32>,
    pub availability_bitset: bitset,
    pub gpu_ids: BTreeSet<u32>,
}

pub type Matrix = Vec<Vec<RefCell<BTreeMap<bitset, MatrixCell>>>>;

/// Orchestration result returned by SyncConductor
#[pyclass]
#[derive(Debug, Clone)]
pub struct OrchestrateResult {
    pub speedup: f64,
    pub stages: Vec<(u32, u32, u32, BTreeSet<u32>)>,
    pub rp: u32,
}

/// Conductor for Synchronous Pipeline
#[pyclass]
#[derive(Debug, Clone)]
pub struct SyncOrchestrate {
    #[pyo3(get)]
    pub m: model::Model,
    #[pyo3(get)]
    pub d: device::Devices,
    #[pyo3(get)]
    pub res: Vec<OrchestrateResult>,
}

impl SyncOrchestrate {
    /// Construct the SyncConductor from TorchGraphImporter
    pub fn new_from_torch_graph(
        filename: &str,
        pbs: u32,
        gbs: u32,
        seps: Vec<u32>,
    ) -> SyncOrchestrate {
        let tgi: torch_graph::TorchGraphImporter = ModelImporter::new();
        let result = tgi.ImportFrom(filename);
        let (perf, states) = (result.0.unwrap(), result.1.unwrap());
        let model = model::Model::new_from_model_perf(perf, states, pbs, gbs);
        let n = seps[seps.len() - 1];
        let d = device::Devices::new(n, seps);
        SyncOrchestrate {
            m: model,
            d: d,
            res: vec![],
        }
    }

    /// Construct a SyncConductor from Model and Devices
    pub fn new_from_model_device(m: model::Model, d: device::Devices) -> SyncOrchestrate {
        SyncOrchestrate {
            m: m,
            d: d,
            res: vec![],
        }
    }

    /// Computations in the matrix to output raw planning metadata
    pub fn compute_plan_sync(&self, spa_size: u32, rp: u32, straight: bool) -> Matrix {
        // Shorthands
        let compute_times = &self.m.perf.compute_times;
        let activation_sizes = &self.m.perf.activation_sizes;
        let output_activation_sizes = &self.m.perf.output_activation_sizes;
        let parameter_sizes = &self.m.perf.parameter_sizes;
        let all_predecessor_ids = &self.m.perf.all_predecessor_ids;
        let mut A: Matrix = vec![];
        let d = &self.d;
        let num_machines = spa_size;

        // Initialize ctx matrix
        for _ in 0..compute_times[0].len() {
            let mut row_a: Vec<RefCell<BTreeMap<bitset, MatrixCell>>> = vec![];
            for _ in 0..num_machines {
                // let mut bt = ;
                row_a.push(RefCell::new(BTreeMap::new()));
            }
            A.push(row_a);
        }

        // Bitset placeholder
        let mut ph: bitset = vec![];
        let mut empty: bitset = vec![];
        for _ in 0..num_machines * rp + 1 {
            ph.push(true);
        }
        for _ in 0..d.num_gpus {
            empty.push(false);
        }

        // DP Initialization
        for j in 0..compute_times[0].len() {
            let cur_compute_time = compute_times[0][j];
            let cur_activation_size = activation_sizes[0][j];
            let cur_parameter_size = parameter_sizes[0][j];
            let max_m = if straight { 1 } else { num_machines };
            for m in 0..max_m {
                if VERBOSE {
                    println!("[orchestrate]\t Assigning DP to A, m = {}, j = {}", m, j);
                }
                let mut n = d.next_cards(empty.clone(), (m + 1) * rp)[0].clone();
                // Update n to prefer FF DP
                for rd in d.next_cards(empty.clone(), (m + 1) * rp) {
                    if rd.strategy == device::AllocationStrategy::FreshFirst {
                        n = rd.clone();
                        break;
                    }
                }
                // for n in d.next_cards(empty.clone(), (m + 1) * rp) {
                A[j][m as usize].get_mut().insert(
                    ph.clone(),
                    MatrixCell {
                        current_value: None,
                        current_maxmin_block: if cur_compute_time < -0.5 {
                            None
                        } else {
                            Some(
                                f64::max((cur_compute_time) / (m + 1) as f64 / rp as f64, 0.0)
                                    /*+ data_parallel::all_reduce_time(
                                        d,
                                        &n.gids,
                                        cur_parameter_size,
                                    ) * 3.0*/,
                            )
                        },
                        optimal_split: None,
                        num_gpus_used: if cur_compute_time < -0.5 {
                            None
                        } else {
                            Some(m + 1)
                        },
                        availability_bitset: if cur_compute_time < -0.5 {
                            ph.clone()
                        } else {
                            empty.clone()
                        },
                        gpu_ids: if cur_compute_time < -0.5 {
                            BTreeSet::new()
                        } else {
                            n.gids.clone()
                        },
                    },
                );
                A[j][m as usize].get_mut().insert(
                    n.occupied.clone(),
                    MatrixCell {
                        current_value: None,
                        current_maxmin_block: if cur_compute_time < -0.5 {
                            None
                        } else {
                            Some(
                                f64::max((cur_compute_time) / (m + 1) as f64 / rp as f64, 0.0)
                                    /*+ data_parallel::all_reduce_time(
                                        d,
                                        &n.gids.clone(),
                                        cur_parameter_size,
                                    ) * 2.0*/,
                            )
                        },
                        optimal_split: None,
                        num_gpus_used: if cur_compute_time < -0.5 {
                            None
                        } else {
                            Some(m + 1)
                        },
                        availability_bitset: if cur_compute_time < -0.5 {
                            ph.clone()
                        } else {
                            empty.clone()
                        },
                        gpu_ids: n.gids,
                    },
                );
            }
        }

        let min_m = 1;
        for m in min_m..num_machines {
            for j in 1..compute_times[0].len() {
                //                if VERBOSE {
                //                    println!("[orchestrate]\t pre-orchestration check, m = {}, j = {}", m, j);
                //                }
                //                if !A[j][m as usize].borrow().contains_key(&ph) && m > 0 {
                //                    continue;
                //                }
                if VERBOSE {
                    println!("[orchestrate]\t m = {}, j = {}", m, j);
                }

                let mut cur_A_bt = A[j][m as usize].borrow_mut();

                let empty_cell = MatrixCell {
                    current_value: None,
                    current_maxmin_block: None,
                    optimal_split: None,
                    num_gpus_used: None,
                    availability_bitset: vec![],
                    gpu_ids: BTreeSet::new(),
                };

                let cur_A: &MatrixCell;
                if cur_A_bt.contains_key(&ph) {
                    cur_A = cur_A_bt.get(&ph).unwrap();
                } else {
                    cur_A = &empty_cell;
                }

                // let cur_A = cur_A_bt.get(&ph).unwrap();
                let (
                    mut min_pipeline_time,
                    mut optimal_split,
                    mut optimal_num_machines,
                    mut last_from,
                    mut last_machines,
                ) = (
                    cur_A.current_maxmin_block,
                    cur_A.optimal_split,
                    cur_A.num_gpus_used,
                    cur_A.availability_bitset.clone(),
                    cur_A.gpu_ids.clone(),
                );

                for k in all_predecessor_ids[j].iter() {
                    if VERBOSE {
                        println!("[orchestrate]\t m = {}, j = {}, k = {}", m, j, k);
                    }
                    let max_mp = if straight { 2 } else { m + 1 };
                    for mp in 1..max_mp {
                        for (bs, cell) in A[*k as usize][(m - mp) as usize].borrow().iter() {
                            if bs.len() as u32 > num_machines * rp {
                                continue; // skip ph
                            }

                            let next_bs_all = d.next_cards_with_replica(bs.to_vec(), mp, rp);

                            for nbs in next_bs_all {
                                let from = &cell.gpu_ids;
                                let to = &nbs.gids;

                                let input_transfer_time = split_concat::split_concat_all2all_time(
                                    d,
                                    from,
                                    to,
                                    2.0 * output_activation_sizes[*k as usize],
                                );

                                let mut last_stage_time = compute_times[*k as usize + 1][j];
                                if last_stage_time < -0.5 {
                                    continue;
                                }
                                last_stage_time /= (mp * rp) as f64;

                                if !A[*k as usize][(m - mp) as usize].borrow().contains_key(&ph)
                                    || A[*k as usize][(m - mp) as usize]
                                        .borrow()
                                        .get(&ph)
                                        .unwrap()
                                        .current_maxmin_block
                                        .unwrap()
                                        < -0.5
                                {
                                    continue;
                                }

                                if VERBOSE {
                                    println!(
                                        "[orchestrate]\t last_stage_time for {},{},{},{} = {} | prev_time = {} | input_transfer_time = {}",
                                        m,
                                        j,
                                        k,
                                        mp,
                                        last_stage_time,
                                        A[*k as usize][(m - mp) as usize]
                                            .borrow()
                                            .get(bs)
                                            .unwrap()
                                            .current_maxmin_block
                                            .unwrap(),
                                        input_transfer_time
                                    );
                                }

                                let mut pipeline_time = f64::max(
                                    A[*k as usize][(m - mp) as usize]
                                        .borrow()
                                        .get(bs)
                                        .unwrap()
                                        .current_maxmin_block
                                        .unwrap(),
                                    last_stage_time,
                                );
                                pipeline_time = f64::max(pipeline_time, input_transfer_time);

                                if min_pipeline_time.is_none()
                                    || pipeline_time < min_pipeline_time.unwrap()
                                {
                                    optimal_split = Some((*k, m - mp));
                                    optimal_num_machines = Some(mp);
                                    min_pipeline_time = Some(pipeline_time);
                                    last_from = bs.clone();
                                    last_machines = nbs.gids.clone();
                                }

                                if !cur_A_bt.contains_key(&nbs.occupied)
                                    || pipeline_time
                                        < cur_A_bt
                                            .get(&nbs.occupied)
                                            .unwrap()
                                            .current_maxmin_block
                                            .unwrap()
                                {
                                    if VERBOSE {
                                        println!(
                                            "[orchestrate]\t Updating A[{}][{}][{:?}] \t| maxmin_block: {:.7}\t split: {:?}\t from_bs: {:?}\t gids: {:?} ",
                                            j,
                                            m,
                                            &nbs.occupied.iter().fold(String::new(), |acc, &b| acc
                                                                      + &(b as i32).to_string()),
                                            pipeline_time,
                                            (*k, m - mp),
                                            &bs.iter().fold(String::new(), |acc, &b| acc
                                                            + &(b as i32).to_string()),
                                            nbs.gids.clone(),
                                        );
                                    }
                                    cur_A_bt.insert(
                                        nbs.occupied.clone(),
                                        MatrixCell {
                                            current_value: None,
                                            current_maxmin_block: Some(pipeline_time),
                                            optimal_split: Some((*k, m - mp)),
                                            num_gpus_used: Some(mp),
                                            availability_bitset: bs.clone(),
                                            gpu_ids: nbs.gids.clone(),
                                        },
                                    );
                                }
                            }
                        }
                    }
                }

                cur_A_bt.insert(
                    ph.clone(),
                    MatrixCell {
                        current_value: None,
                        current_maxmin_block: min_pipeline_time,
                        optimal_split: optimal_split,
                        num_gpus_used: optimal_num_machines,
                        availability_bitset: last_from.clone(),
                        gpu_ids: last_machines.clone(),
                    },
                );
            }
        }
        A
    }

    /// Analyse the raw data from compute_plan_sync and output a human-readable plan
    pub fn analyse_plan_sync(
        &self,
        A: &Matrix,
        end: u32,
        num_machines: u32,
        rp: u32,
    ) -> Vec<(u32, u32, u32, BTreeSet<u32>)> {
        let mut res: Vec<(u32, u32, u32, BTreeSet<u32>)> = vec![];
        let mut ph: bitset = vec![];
        for _ in 0..num_machines * rp + 1 {
            ph.push(true);
        }
        let mut mt = A[end as usize - 1][num_machines as usize - 1].borrow();
        let mut metadata = mt.get(&ph).unwrap();

        let mut next_split = metadata.optimal_split;
        let mut last_machines = metadata.gpu_ids.clone();
        if last_machines.is_empty() {
            println!("Last Machines is EMPTY! \nFinal Context Matrix\n");
            SyncOrchestrate::print_matrix(A);
            panic!("last_machines.is_empty()");
        }
        let mut last_from = metadata.availability_bitset.clone();
        let mut prev_split = end - 1;

        while !next_split.is_none() {
            let num_machines_used = metadata.num_gpus_used.unwrap();
            res.push((
                next_split.unwrap().0 + 1,
                prev_split,
                num_machines_used,
                last_machines,
            ));
            prev_split = res[res.len() - 1].0;

            mt = A[next_split.unwrap().0 as usize][next_split.unwrap().1 as usize].borrow();
            metadata = mt.get(&last_from).unwrap();
            next_split = metadata.optimal_split;
            last_machines = metadata.gpu_ids.clone();
            last_from = metadata.availability_bitset.clone();
        }

        let num_machines_used = metadata.num_gpus_used.unwrap();
        res.push((0, prev_split, num_machines_used, last_machines));
        res.reverse();

        res
    }

    pub fn compute_bipartition(&self) -> OrchestrateResult {
        let compute_times = &self.m.perf.compute_times;
        let output_activation_sizes = &self.m.perf.output_activation_sizes;
        let all_predecessor_ids = &self.m.perf.all_predecessor_ids;
        let d = &self.d;
        let L = compute_times[0].len() as u32;

        let empty: bitset = vec![false; d.num_gpus as usize];

        // let bicards: Vec<(device::ReturnDevices, device::ReturnDevices)> = d
        //     .next_cards_with_replica_helper(empty, num_machines, 2)
        //     .par_iter()
        //     .map(move |v| (v[0].clone(), v[1].clone()))
        //     .collect();

        let vec_gpus: Vec<u32> = (1..d.num_gpus).collect();
        let vec_layers: Vec<u32> = (0..L - 1).collect();
        let res: Vec<_> = vec_layers
            .par_iter()
            .map(|l| {
                let res_inner: Vec<_> = vec_gpus
                    .par_iter()
                    .map(|g| {
                        let empty = vec![false; d.num_gpus as usize];
                        let first_round = d.next_cards(empty, *g);
                        let res_g1: Vec<_> = first_round
                            .par_iter()
                            .map(|g1| {
                                let cur_bs = g1.occupied.clone();
                                let sub_sr = d.next_cards(cur_bs, d.num_gpus - *g);
                                let res_g2: Vec<_> = sub_sr
                                    .par_iter()
                                    .map(|g2| {
                                        if VERBOSE {
                                            println!("[orchestrate]\t Planning for Layer {}", *l);
                                        }
                                        let comp_times = f64::max(
                                            compute_times[0][*l as usize] / g1.gids.len() as f64,
                                            compute_times[*l as usize + 1][L as usize - 1]
                                                / g2.gids.len() as f64,
                                        );
                                        let comm_times = split_concat::split_concat_all2all_time(
                                            &self.d,
                                            &g1.gids,
                                            &g2.gids,
                                            output_activation_sizes[*l as usize],
                                        );
                                        let ppl_time = f64::max(comp_times, comm_times);
                                        let p = vec![
                                            (0, l + 1, g1.gids.len() as u32, g1.gids.clone()),
                                            (l + 1, L - 1, g2.gids.len() as u32, g2.gids.clone()),
                                        ];
                                        let res_speedup =
                                            sync_pipeline::sync_pipeline_speedup_analytical(
                                                &self.d,
                                                &self.m,
                                                1,
                                                ppl_time,
                                                p.clone(),
                                            );
                                        if VERBOSE {
                                            println!(
                                                "[orchestrate]\n P = {:?}\nSpeedup = {}",
                                                p, res_speedup
                                            );
                                        }
                                        OrchestrateResult {
                                            speedup: res_speedup,
                                            stages: p,
                                            rp: 1,
                                        }
                                    })
                                    .collect();
                                res_g2
                            })
                            .collect();
                        res_g1
                    })
                    .collect();
                // let res_inner: Vec<_> = bicards
                //     .par_iter()
                //     .map(|(g1, g2)| {
                //     })
                //     .collect();
                res_inner
            })
            .collect();
        // we're working with a 4-dimensional array, flattening it to 1d.
        // NOTE: potential bottleneck for flatten
        let flattened_res: Vec<OrchestrateResult> =
            res.into_iter().flatten().flatten().flatten().collect();
        let best_result = flattened_res
            .into_par_iter()
            .max_by_key(|r| OrderedFloat(r.speedup))
            .unwrap();
        if VERBOSE {
            //println!("{:?}", flattened_res);
            println!("[orchestrate]\t best result = {:?}", best_result);
        }
        return best_result;
    }

    /// Shorthand for Planning each individual pipeline spin
    pub fn plan_for(&self, i: u32, straight: bool) -> OrchestrateResult {
        let num_gpus = self.d.num_gpus;
        // 2 <= i <= num_gpus
        let rp = num_gpus / i;
        if VERBOSE {
            println!("Planning for {} x {}, {}", i, rp, straight);
        }

        let A = self.compute_plan_sync(i, rp, straight);
        if VERBOSE {
            println!("Planning Done");
        }

        let mut ph: bitset = vec![];
        for _ in 0..i * rp + 1 {
            ph.push(true);
        }

        if VERBOSE {
            println!("========================");
            SyncOrchestrate::print_matrix(&A);
            println!("========================");
        }
        let pipeline_block_bt = A[self.m.perf.compute_times[0].len() - 1][i as usize - 1].borrow();
        let pipeline_block = pipeline_block_bt.get(&ph).unwrap();
        if VERBOSE {
            println!(
                "Accessing A[{}][{}][{:?}]",
                self.m.perf.compute_times[0].len() - 1,
                i as usize - 1,
                ph
            );
        }
        let pipeline_time = pipeline_block.current_maxmin_block.unwrap();
        if pipeline_time < 0.001 {
            panic!("ppl time error");
        }
        if VERBOSE {
            println!("Pipeline Time: {}", pipeline_time);
        }

        let res = self.analyse_plan_sync(&A, self.m.perf.compute_times[0].len() as u32, i, rp);
        let res_speedup = sync_pipeline::sync_pipeline_speedup_analytical(
            &self.d,
            &self.m,
            rp,
            pipeline_time,
            res.clone(),
        );
        OrchestrateResult {
            speedup: res_speedup,
            stages: res,
            rp: rp,
        }
    }

    pub fn print_matrix(A: &Matrix) {
        for j in 0..A.len() {
            for m in 0..A[j].len() {
                for (k, v) in A[j][m].borrow().iter() {
                    println!(
                        "A[{}][{}][{:?}] \t| maxmin_block: {:.7?}\t split: {:?}\t from_bs: {:?}\t gids: {:?} ",
                        j,
                        m,
                        k.iter().fold(String::new(), |acc, &b| acc
                            + &(b as i32).to_string()),
                        v.current_maxmin_block,
                        v.optimal_split,
                        v.availability_bitset.iter().fold(String::new(), |acc, &b| acc
                            + &(b as i32).to_string()),
                        v.gpu_ids,
                    );
                }
            }
        }
    }
}

/// SyncConductor Python API
#[pymethods]
impl SyncOrchestrate {
    fn py_orchestrate(&mut self) -> PyResult<(f64, Vec<(u32, u32, u32, Vec<u32>)>)> {
        self.orchestrate();
        let best_hp = &self
            .res
            .par_iter()
            .max_by_key(|r| OrderedFloat(r.speedup))
            .unwrap();
        let speedup = best_hp.speedup;
        let py_stages: Vec<(u32, u32, u32, Vec<u32>)>;
        py_stages = best_hp
            .stages
            .par_iter()
            .map(|s| (s.0, s.1, s.2, s.3.iter().cloned().collect()))
            .collect();
        Ok((speedup, py_stages))
    }
}

impl OrchestrationResult for OrchestrateResult {
    fn get_speedup(&self) -> Option<f64> {
        unimplemented!()
    }

    fn get_splits(&self) -> Option<Vec<u32>> {
        unimplemented!()
    }

    fn pretty_print(&self) -> Option<String> {
        unimplemented!()
    }
}

impl Orchestrate for SyncOrchestrate {
    fn orchestrate(&mut self) {
        let num_gpus = self.d.num_gpus;
        let vec_range: Vec<u32> = (2..num_gpus + 1).collect();
        let mut result: Vec<_> = vec_range
            .par_iter()
            .map(|i| self.plan_for(*i, false))
            .collect();

        let mut straight_vec: Vec<u32> = vec![];
        if num_gpus > 2 && num_gpus % 2 == 0 {
            straight_vec.push(2); // 1:1
            straight_vec.push(num_gpus); // all straight
        } else {
            straight_vec.push(num_gpus); // all straight
        }

        if VERBOSE {
            println!("Appending Orchestration for {:?}", straight_vec);
        }

        let result_straight: Vec<_> = straight_vec
            .par_iter()
            .map(|i| self.plan_for(*i, true))
            .collect();

        result.extend(result_straight);

        let result_bipartition = self.compute_bipartition();
        result.push(result_bipartition);

        self.res = result;
    }

    fn compute_plan(&mut self) {
        unimplemented!()
    }

    fn analyse_plan(&self) {
        unimplemented!()
    }

    fn return_plan(&self) -> Box<dyn OrchestrationResult> {
        unimplemented!()
    }
}
