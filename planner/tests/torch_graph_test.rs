extern crate HPGO;
use HPGO::input::*;

#[test]
fn test_python_env() {
    // should not crash
    let tgi: torch_graph::TorchGraphImporter = ModelImporter::new();
    tgi.ImportFrom("./profiles/xlnet/graph.txt");
}

#[test]
fn test_python_import_basic() {
    let tgi: torch_graph::TorchGraphImporter = ModelImporter::new();
    let result = tgi.ImportFrom("./profiles/vgg16/graph.txt");
    match result {
        (Some(x), Some(y)) => {
            println!("Got result successfully, printing all fields...");
            // NOTE: could've just print x, as it derives Debug
            println!("compute_times: {:?}", x.compute_times);
            println!("activation_sizes: {:?}", x.activation_sizes);
            println!("parameter_sizes: {:?}", x.parameter_sizes);
            println!("output_activation_sizes: {:?}", x.output_activation_sizes);
            println!("all_predecessor_ids: {:?}", x.all_predecessor_ids);

            // println!("model_states: {:?}, {:?}", y.len(), x.compute_times[0].len())
        }
        _ => {
            panic!();
        }
    }
}
