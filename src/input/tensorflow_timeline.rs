use super::ModelImporter;
use model::model_perf;

struct TensorflowTimelineImporter {}

impl ModelImporter for TensorflowTimelineImporter {
    fn new() -> TensorflowTimelineImporter {
        TensorflowTimelineImporter {}
    }
    fn ImportFrom(
        &self,
        filename: &str,
    ) -> (
        Option<model_perf::ModelPerf>,
        Option<model_perf::ModelStates>,
    ) {
        unimplemented!();
    }
}
