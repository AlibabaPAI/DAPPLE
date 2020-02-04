use model::model_perf;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
pub struct Layer {
    #[pyo3(get)]
    pub id: Option<u32>,
    #[pyo3(get)]
    pub name: Option<String>,
    #[pyo3(get)]
    pub desc: Option<String>,
    #[pyo3(get)]
    pub compute_time: f64,
    #[pyo3(get)]
    pub activation_size: f64,
    #[pyo3(get)]
    pub output_activation_size: f64,
    #[pyo3(get)]
    pub parameter_size: f64,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Model {
    #[pyo3(get)]
    pub layers: Vec<Layer>,
    pub perf: model_perf::ModelPerf,
    pub states: model_perf::ModelStates,
    pub global_batch_size: u32,
    pub profile_batch_size: u32,
    #[pyo3(get, set)]
    pub min_micro_batch_size: u32,
    pub use_async: bool,
    #[pyo3(get, set)]
    pub optimizer_memory_scaling: u32,
    #[pyo3(get, set)]
    pub peak_activation_per_batch: f64,
}

const VERBOSE: bool = false;

impl Model {
    pub fn new() -> Model {
        println!("For now, don't call new(), construct from perf results instead");
        panic!()
    }
    pub fn new_from_model_perf(
        perf: model_perf::ModelPerf,
        states: model_perf::ModelStates,
        pbs: u32,
        gbs: u32,
    ) -> Model {
        // WIP
        let layers: Vec<Layer> = vec![];

        if VERBOSE {
            // Format predecessor IDs
            println!("All Predecessor IDs");
            for i in 0..perf.compute_times[0].len() {
                println!("pred[{}]: {:?}", i, perf.all_predecessor_ids[i]);
            }
            println!("Constructing Model m");
        }

        let mut m = Model {
            layers: layers,
            perf: perf,
            states: states,
            global_batch_size: gbs,
            profile_batch_size: pbs,
            min_micro_batch_size: 1,
            use_async: false,
            optimizer_memory_scaling: 1,
            peak_activation_per_batch: 0.0,
        };

        m.peak_activation_per_batch =
            m.perf.activation_sizes[0][m.perf.activation_sizes[0].len() - 1] / pbs as f64;

        if VERBOSE {
            println!(
                "Setting model peak activation per unit batch: {}",
                m.peak_activation_per_batch
            );
            println!("Profiling Results before normalization:");
            for i in 0..m.perf.compute_times[0].len() {
                println!(
                    "C = {:.5} \t A = {:.5} \t OA = {:.5} \t P = {:.5}",
                    m.perf.compute_times[i][i],
                    m.perf.activation_sizes[i][i],
                    m.perf.output_activation_sizes[i],
                    m.perf.parameter_sizes[i][i]
                );
            }
            println!("Normalizing to GBS");
        }

        m.normalize(pbs, gbs);

        // Model Performance Stats
        for i in 0..m.perf.compute_times[0].len() {
            m.layers.push(Layer {
                id: Some(i as u32),
                name: None,
                desc: m.states[i].desc.clone(),
                compute_time: m.perf.compute_times[i][i],
                activation_size: m.perf.activation_sizes[i][i],
                output_activation_size: m.perf.output_activation_sizes[i],
                parameter_size: m.perf.parameter_sizes[i][i],
            });
        }
        // Format Compute Times
        if VERBOSE && m.perf.compute_times.len() < 30 {
            println!("Compute Times matrix: ");
            for ct in &m.perf.compute_times {
                for i in ct {
                    print!("{:2.5}\t", if i < &-0.5 { &0.0 } else { i });
                }
                println!();
            }
        }

        m
    }
    fn normalize(&mut self, from: u32, to: u32) {
        let factor: f64 = to as f64 / from as f64;
        self.states.par_iter_mut().for_each(|s| {
            s.compute_time *= factor;
            s.activation_size *= factor;
            s.output_activation_size *= factor;
        });
        self.perf.compute_times.par_iter_mut().for_each(|row_c| {
            row_c.par_iter_mut().for_each(|c| {
                *c *= factor;
            });
        });
        self.perf.activation_sizes.par_iter_mut().for_each(|row_c| {
            row_c.par_iter_mut().for_each(|c| {
                *c *= factor;
            });
        });
        self.perf
            .output_activation_sizes
            .par_iter_mut()
            .for_each(|c| {
                *c *= factor;
            });
    }

    pub fn normalized_copy(&self, bs: u32) -> Model {
        let mut m = self.clone();
        // normalize from gbs to bs
        m.normalize(self.global_batch_size, bs);
        m.global_batch_size = bs;
        // reconstruct & override the original layers
        m.layers = vec![];
        for i in 0..m.perf.compute_times[0].len() {
            m.layers.push(Layer {
                id: Some(i as u32),
                name: None,
                desc: m.states[i].desc.clone(),
                compute_time: m.perf.compute_times[i][i],
                activation_size: m.perf.activation_sizes[i][i],
                output_activation_size: m.perf.output_activation_sizes[i],
                parameter_size: m.perf.parameter_sizes[i][i],
            });
        }

        m
    }
}
