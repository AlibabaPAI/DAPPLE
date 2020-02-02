extern crate HPGO;
extern crate rayon;
use rayon::prelude::*;
use HPGO::analysis::*;
use HPGO::input::*;
use HPGO::model::*;

#[test]
fn test_model_max_batch_size() {
    // Define Models: (name, PBS, optimizer scale, PAPB)
    let models = vec![
        ("vgg16", 32, 1, -1.0),
        ("vgg19", 32, 1, -1.0),
        ("xlnet", 1, 3, 3942774528.0),
        ("amoebanet", 8, 4, 250845152.0),
        ("bert_large", 2, 3, 1733171968.0),
        ("gnmt_large", 32, 3, -1.0),
        ("amoebanet_18", 8, 4, 250845152.0 * 1.5),
        ("amoebanet_36", 1, 4, 250845152.0 * 3.0), // ("resnet50", 32),
    ];

    // Compute Max Batch Size in Parallel
    let res: Vec<_> = models
        .par_iter()
        .map(|(s, bs, opt_scale, papb)| {
            let tgi: torch_graph::TorchGraphImporter = ModelImporter::new();
            let result = tgi.ImportFrom(&["./profiles/", s, "/graph.txt"].join(""));
            let (perf, states) = (result.0.unwrap(), result.1.unwrap());
            let mut model = model::Model::new_from_model_perf(perf, states, *bs, 32);
            model.optimizer_memory_scaling = *opt_scale;
            if *papb > 0.0 {
                model.peak_activation_per_batch = *papb;
            }
            (s, gpu_memory::max_single_gpu_batch_size(&model))
        })
        .collect();

    println!("\nMax Single GPU Batch Size:");
    for i in res {
        println!("{:12}: {}", i.0, i.1);
    }
}
