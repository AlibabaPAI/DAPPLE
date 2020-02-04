//! HPGO!

#![allow(non_snake_case)]

extern crate itertools;
extern crate ordered_float;
extern crate pyo3;
extern crate rayon;

/// HPGO Model Analysis
pub mod analysis;

/// HPGO Hardware Environment: Ethernet, GPU, NVLink, etc.
pub mod environment;

/// HPGO Model Importer, currently TorchGraphImporter only
pub mod input;

/// HPGO Model Abstract Definition
pub mod model;

/// HPGO Orchestration Variations
pub mod orchestration;

/// HPGO Parallelism Definitions and Helpers
pub mod parallelism;

/// HPGO Public API: C & Python
pub mod api;
