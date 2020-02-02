// import from PyTorch Profiler graph
pub mod torch_graph;
mod torch_graph_py;

// import from TensorFlow Timeline
pub mod tensorflow_timeline;

use model::model_perf;

pub trait ModelImporter {
    fn new() -> Self;
    fn ImportFrom(
        &self,
        filename: &str,
    ) -> (
        Option<model_perf::ModelPerf>,
        Option<model_perf::ModelStates>,
    );
}
