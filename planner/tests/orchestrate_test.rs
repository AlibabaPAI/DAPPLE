extern crate HPGO;
extern crate ordered_float;
extern crate rayon;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use HPGO::environment::*;
use HPGO::input::*;
use HPGO::model::*;
use HPGO::orchestration::*;
use HPGO::parallelism::*;

#[test]
fn test_orchestrate_compute_plan() {
    let mut c = orchestrate::SyncOrchestrate::new_from_torch_graph(
        "./profiles/amoebanet/flattened.txt",
        8,
        1024,
        [1, 2, 3, 4, 5, 6, 7, 8].to_vec(),
    );
    let A = c.compute_plan_sync(8, 1, false);
    println!("\nFinal Context Matrix\n");
    for j in 0..A.len() {
        for m in 0..A[j].len() {
            for (k, v) in A[j][m].borrow().iter() {
                // println!(
                //     "{} {} {:?} | {:?}",
                //     j,
                //     m,
                //     k.iter()
                //         .fold(String::new(), |acc, &b| acc + &(b as i32).to_string()),
                //     v
                // );
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

#[test]
fn test_orchestrate_analyse_plan() {
    // Construct Model
    let tgi: torch_graph::TorchGraphImporter = ModelImporter::new();
    let result = tgi.ImportFrom(&["./profiles/", "xlnet", "/graph.txt"].join(""));
    let (perf, states) = (result.0.unwrap(), result.1.unwrap());
    let mut model = model::Model::new_from_model_perf(perf, states, 1, 256);
    model.optimizer_memory_scaling = 3;
    model.peak_activation_per_batch = 3942774528.0;
    model.min_micro_batch_size = 1;
    // Construct Devices
    let d16 = device::Devices::new(16, vec![8, 16]);

    let mut c = orchestrate::SyncOrchestrate::new_from_model_device(model, d16);
    let A = c.compute_plan_sync(16, 1, false);
    let res = c.analyse_plan_sync(&A, c.m.perf.compute_times[0].len() as u32, 16, 1);
    println!("{:?}", res);
}

#[test]
fn test_orchestrate_plan_straight() {
    // Construct Model
    let tgi: torch_graph::TorchGraphImporter = ModelImporter::new();
    let result = tgi.ImportFrom(&["./profiles/", "xlnet", "/graph.txt"].join(""));
    let (perf, states) = (result.0.unwrap(), result.1.unwrap());
    let mut model = model::Model::new_from_model_perf(perf, states, 1, 16);
    model.optimizer_memory_scaling = 3;
    model.peak_activation_per_batch = 3942774528.0;
    model.min_micro_batch_size = 1;
    // Construct Devices
    let d16 = device::Devices::new(16, vec![8, 16]);

    let mut c = orchestrate::SyncOrchestrate::new_from_model_device(model, d16);
    println!("{:?}", c.plan_for(2, true));
}

#[test]
fn test_orchestrate_plan_consistency() {
    // Construct Model
    let tgi: torch_graph::TorchGraphImporter = ModelImporter::new();
    let result = tgi.ImportFrom(&["./profiles/", "xlnet", "/graph.txt"].join(""));
    let (perf, states) = (result.0.unwrap(), result.1.unwrap());
    let mut model = model::Model::new_from_model_perf(perf, states, 1, 128);
    model.optimizer_memory_scaling = 3;
    model.peak_activation_per_batch = 3942774528.0;
    model.min_micro_batch_size = 1;
    // Construct Devices
    let d16 = device::Devices::new(16, vec![8, 16]);

    let mut c = orchestrate::SyncOrchestrate::new_from_model_device(model, d16);
    let mut planning_results: Vec<orchestrate::OrchestrateResult> = vec![];
    for _ in 0..5 {
        planning_results.push(c.plan_for(2, true));
    }
    for i in 0..planning_results.len() {
        println!("Try #{} -> {:?}", i, planning_results[i]);
    }
}

#[test]
fn test_orchestrate_orchestrate() {
    // Construct Model
    let tgi: torch_graph::TorchGraphImporter = ModelImporter::new();
    let result = tgi.ImportFrom(&["./profiles/", "xlnet", "/xlnet-36.txt"].join(""));
    let (perf, states) = (result.0.unwrap(), result.1.unwrap());
    let mut model = model::Model::new_from_model_perf(perf, states, 1, 16);
    model.optimizer_memory_scaling = 3;
    model.peak_activation_per_batch = 3942774528.0 * 1.5;
    model.min_micro_batch_size = 1;
    // Construct Devices
    let d16 = device::Devices::new(4, vec![2, 4]);

    let mut c = orchestrate::SyncOrchestrate::new_from_model_device(model, d16);
    c.orchestrate();
    let best_hp = c
        .res
        .into_par_iter()
        .max_by_key(|r| {
            if r.stages.len() == 1 {
                // throw away pseudo DPs
                OrderedFloat(0.0)
            } else {
                OrderedFloat(r.speedup)
            }
        })
        .unwrap();
    println!(
        "Best HP Speedup: {}, Stages: {:?}",
        best_hp.speedup, best_hp.stages
    );
}

#[test]
fn test_orchestrate_bipartition() {
    // Construct Model
    let tgi: torch_graph::TorchGraphImporter = ModelImporter::new();
    let result = tgi.ImportFrom(&["./profiles/", "xlnet", "/graph.txt"].join(""));
    let (perf, states) = (result.0.unwrap(), result.1.unwrap());
    let mut model = model::Model::new_from_model_perf(perf, states, 1, 256);
    model.optimizer_memory_scaling = 3;
    model.peak_activation_per_batch = 3942774528.0;
    model.min_micro_batch_size = 1;
    // Construct Devices
    let d16 = device::Devices::new(16, vec![8, 16]);

    let mut c = orchestrate::SyncOrchestrate::new_from_model_device(model, d16);
    c.compute_bipartition();
}

#[test]
fn test_orchestrate_sync_speedup() {
    // Construct Model
    let tgi: torch_graph::TorchGraphImporter = ModelImporter::new();
    let result = tgi.ImportFrom(&["./profiles/", "xlnet", "/xlnet-36.txt"].join(""));
    let (perf, states) = (result.0.unwrap(), result.1.unwrap());
    let mut model = model::Model::new_from_model_perf(perf, states, 1, 4);
    model.optimizer_memory_scaling = 3;
    model.peak_activation_per_batch = 3942774528.0 * 1.5;
    model.min_micro_batch_size = 1;
    // Construct Devices
    let d16 = device::Devices::new(2, vec![1, 2]);

    let mut c = orchestrate::SyncOrchestrate::new_from_model_device(model, d16);
    c.orchestrate();
    let best_hp = c
        .res
        .into_par_iter()
        .max_by_key(|r| {
            if r.stages.len() == 1 {
                // throw away pseudo DPs
                OrderedFloat(0.0)
            } else {
                OrderedFloat(r.speedup)
            }
        })
        .unwrap();
    println!(
        "Best HP Speedup: {}, Stages: {:?}",
        best_hp.speedup, best_hp.stages
    );

    let pipeline_recursive_speedup = sync_pipeline::sync_pipeline_speedup_recursive(
        &c.d,
        &c.m,
        best_hp.rp,
        best_hp.stages.clone(),
    );

    println!("Best HP Speedup Recursive: {}", pipeline_recursive_speedup);
}
