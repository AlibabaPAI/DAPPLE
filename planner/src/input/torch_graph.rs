use super::ModelImporter;
use input::torch_graph_py;
use model::model_perf;
use pyo3::prelude::*;
use pyo3::types::PyModule;

const VERBOSE: bool = true;

pub struct TorchGraphImporter {}

impl ModelImporter for TorchGraphImporter {
    fn new() -> TorchGraphImporter {
        TorchGraphImporter {}
    }

    fn ImportFrom(
        &self,
        filename: &str,
    ) -> (
        Option<model_perf::ModelPerf>,
        Option<model_perf::ModelStates>,
    ) {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let graph = PyModule::from_code(
            py,
            torch_graph_py::TORCH_GRAPH_PY,
            "torch_graph.py",
            "torch_graph",
        )
        .map_err(|e| {
            e.print_and_set_sys_last_vars(py);
        })
        .unwrap();

        if VERBOSE {
            println!("[python]\t Entering Prepare()...");
        }

        let result: (
            PyObject,
            PyObject,
            Vec<Vec<f64>>,
            Vec<Vec<f64>>,
            Vec<Vec<f64>>,
            Vec<f64>,
            Vec<Vec<u32>>,
        ) = graph
            .call1("prepare", (filename,))
            .map_err(|e| {
                e.print_and_set_sys_last_vars(py);
            })
            .unwrap()
            .extract()
            .map_err(|e| {
                e.print_and_set_sys_last_vars(py);
            })
            .unwrap();
        // TODO: no error handling at all

        // NOTE: process states object into Rust
        let py_states: Vec<PyObject> = result.1.extract(py).unwrap();
        let mut states: model_perf::ModelStates = vec![];

        if VERBOSE {
            println!("[python]\t Prepare() done, States.len(): {}", py_states.len());
        }
        for s in py_states {
            let id: Option<u32> = s
                .getattr(py, "node_id")
                .ok()
                .and_then(|x| x.extract(py).ok());
            let name: Option<String> = s
                .getattr(py, "node_name")
                .ok()
                .and_then(|x| x.extract(py).ok());
            let desc: Option<String> = s
                .getattr(py, "node_desc")
                .ok()
                .and_then(|x| x.extract(py).ok());

            if VERBOSE {
                println!("[python]\t @state: {:?}", desc);
            }
            
            // below are required
            let compute_time: f64 = s.getattr(py, "compute_time").unwrap().extract(py).unwrap();
            let activation_size: f64 = s
                .getattr(py, "activation_size")
                .unwrap()
                .extract(py)
                .unwrap();
            let output_activation_size: f64 = s
                .getattr(py, "output_activation_size")
                .unwrap()
                .extract(py)
                .unwrap();
            let parameter_size: f64 = s
                .getattr(py, "parameter_size")
                .unwrap()
                .extract(py)
                .unwrap();
            states.push(model_perf::ModelState {
                id: id,
                name: name,
                desc: desc,
                compute_time: compute_time,
                activation_size: activation_size,
                output_activation_size: output_activation_size,
                parameter_size: parameter_size,
                stage_id: None,
            });
        }
        // println!("{:?}", states);
        let perf = model_perf::ModelPerf {
            compute_times: result.2,
            activation_sizes: result.3,
            parameter_sizes: result.4,
            output_activation_sizes: result.5,
            all_predecessor_ids: result.6,
        };
        (Some(perf), Some(states))
    }
}
