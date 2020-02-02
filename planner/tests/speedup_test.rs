extern crate HPGO;
extern crate rayon;
use rayon::prelude::*;
use HPGO::analysis::*;
use HPGO::environment::*;
use HPGO::input::*;
use HPGO::model::*;
use HPGO::orchestration::*;
use HPGO::parallelism::*;

#[test]
fn test_dp_speedup() {
    // Define Models: (name, PBS, GBS, optimizer scale, PAPB)
    let models = vec![
        ("vgg16", 32, 512, 1, -1.0),
        ("vgg19", 32, 512, 1, -1.0),
        ("xlnet", 1, 128, 3, 3942774528.0),
        ("amoebanet", 8, 512, 4, 250845152.0),
        ("bert_large", 2, 128, 3, 1733171968.0),
        ("gnmt_large", 32, 512, 3, -1.0),
        // ("resnet50", 32),
    ];

    // Devices
    let d16 = device::Devices::new(16, vec![8, 16]);

    // Compute Max Batch Size in Parallel
    let res: Vec<_> = models
        .par_iter()
        .map(|(s, pbs, gbs, opt_scale, papb)| {
            let tgi: torch_graph::TorchGraphImporter = ModelImporter::new();
            let result = tgi.ImportFrom(&["./profiles/", s, "/graph.txt"].join(""));
            let (perf, states) = (result.0.unwrap(), result.1.unwrap());
            let mut model = model::Model::new_from_model_perf(perf, states, *pbs, *gbs);
            model.optimizer_memory_scaling = *opt_scale;
            if *papb > 0.0 {
                model.peak_activation_per_batch = *papb;
            }
            let dp_speedup = data_parallel::dp_speedup(&d16, &model);
            let dp_p3_speedup = data_parallel::dp_p3_speedup(&d16, &model);
            let dp_cur_ga_p3_speedup = gradient_accumulation::dp_cur_ga_p3_speedup(&d16, &model);
            let dp_opt_ga_p3_speedup = gradient_accumulation::dp_opt_ga_p3_speedup(&d16, &model);
            (
                s,
                (
                    dp_speedup,
                    dp_p3_speedup,
                    dp_cur_ga_p3_speedup,
                    dp_opt_ga_p3_speedup,
                ),
            )
        })
        .collect();

    println!("Speedups for \tDP+GA\tDP+P3(INF MEM)\tDP+UniformGA+P3\tDP+NonUniformGA+P3");
    for i in res {
        println!(
            "{:12}\t{}\t{}\t{}\t{}",
            i.0,
            (i.1).0,
            (i.1).1,
            (i.1).2,
            (i.1).3
        );
    }
}

#[test]
fn test_xlnet_speedup_at_all_bs() {
    // GBS
    let mut gbs = vec![1, 2, 4, 8, 16, 32, 64];
    for i in 1..((2048 - 64) / 64) {
        gbs.push(64 + i * 64);
    }

    // Compute Max Batch Size in Parallel
    let res: Vec<_> = gbs
        .par_iter()
        .map(|(gbs)| {
            // Construct Model
            let tgi: torch_graph::TorchGraphImporter = ModelImporter::new();
            let result = tgi.ImportFrom(&["./profiles/", "xlnet", "/graph.txt"].join(""));
            let (perf, states) = (result.0.unwrap(), result.1.unwrap());
            let mut model = model::Model::new_from_model_perf(perf, states, 1, *gbs);
            model.optimizer_memory_scaling = 3;
            model.peak_activation_per_batch = 3942774528.0;
            model.min_micro_batch_size = 1;
            // Construct Devices
            let d16 = device::Devices::new(16, vec![8, 16]);

            // DP Speedups
            let dp_speedup = data_parallel::dp_speedup(&d16, &model);
            // let dp_p3_speedup = data_parallel::dp_p3_speedup(&d16, &model);
            let dp_cur_ga_p3_speedup = gradient_accumulation::dp_cur_ga_p3_speedup(&d16, &model);
            let dp_opt_ga_p3_speedup = gradient_accumulation::dp_opt_ga_p3_speedup(&d16, &model);

            // Hybrid Parallel Speedups
            let mut c = orchestrate::SyncOrchestrate::new_from_model_device(model, d16);
            c.orchestrate();
            let pipeline_speedup = c.res.iter().map(|s| s.speedup).fold(0. / 0., f64::max);

            // return gbs and all speedups
            (
                gbs,
                (
                    dp_speedup,
                    dp_cur_ga_p3_speedup,
                    dp_opt_ga_p3_speedup,
                    pipeline_speedup,
                ),
            )
        })
        .collect();

    for i in res {
        println!("{},{},{},{},{}", i.0, (i.1).0, (i.1).1, (i.1).2, (i.1).3);
    }
}
