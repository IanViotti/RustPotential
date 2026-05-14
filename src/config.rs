#![allow(non_snake_case)]

use serde::Deserialize;

/// Main configuration structure that holds all parameters required for the 
/// grid generation process. This struct is populated by reading the `job_config.toml` file.
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    // Basic identifiers
    pub meshname: String,

    pub jobname: String,

    pub u_inf: f64,
    
}