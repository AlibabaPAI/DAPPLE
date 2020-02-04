use pyo3::prelude::*;
#[pyclass]
#[derive(Debug, Clone)]
pub struct ModelPerf {
    pub compute_times: Vec<Vec<f64>>,
    pub activation_sizes: Vec<Vec<f64>>,
    pub parameter_sizes: Vec<Vec<f64>>,
    pub output_activation_sizes: Vec<f64>,
    pub all_predecessor_ids: Vec<Vec<u32>>,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct ModelState {
    pub id: Option<u32>,
    pub name: Option<String>,
    pub desc: Option<String>,
    pub compute_time: f64,
    pub activation_size: f64,
    pub output_activation_size: f64,
    pub parameter_size: f64,
    pub stage_id: Option<u32>,
}

pub type ModelStates = Vec<ModelState>;
