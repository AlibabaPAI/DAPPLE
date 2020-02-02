extern crate HPGO;
extern crate rayon;
use rayon::prelude::*;
use HPGO::analysis::*;
use HPGO::environment::*;
use HPGO::input::*;
use HPGO::model::*;
use HPGO::parallelism::*;

#[test]
fn test_cur_ga_iter_size() {
    // Define Models: (name, PBS, optimizer scale, PAPB)
    let models = vec![
        ("vgg16", 32, 1, -1.0),
        ("vgg19", 32, 1, -1.0),
        ("xlnet", 1, 3, 3942774528.0),
        ("amoebanet", 8, 4, 250845152.0),
        ("bert_large", 2, 3, 1733171968.0),
        ("gnmt_large", 32, 3, -1.0),
        // ("resnet50", 32),
    ];

    // Devices
    let d16 = device::Devices::new(16, vec![8, 16]);

    // Compute Max Batch Size in Parallel
    let res: Vec<_> = models
        .iter()
        .map(|(s, bs, opt_scale, papb)| {
            let tgi: torch_graph::TorchGraphImporter = ModelImporter::new();
            let result = tgi.ImportFrom(&["./profiles/", s, "/graph.txt"].join(""));
            let (perf, states) = (result.0.unwrap(), result.1.unwrap());
            let mut model = model::Model::new_from_model_perf(perf, states, *bs, 1024);
            model.optimizer_memory_scaling = *opt_scale;
            if *papb > 0.0 {
                model.peak_activation_per_batch = *papb;
            }
            let cur_ga_size = gradient_accumulation::current_ga_iter_size(&d16, &model);
            let opt_ga_size = gradient_accumulation::optimal_ga_iter_size(&d16, &model);
            (s, (cur_ga_size, opt_ga_size))
        })
        .collect();

    println!("\nCurrent / Optimal GA Per iteration Max Batch Size:");
    for i in res {
        println!("{:12}: {} vs {}", i.0, (i.1).0, (i.1).1);
    }
}
