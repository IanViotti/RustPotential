#![allow(non_snake_case)]

use serde::Deserialize;

/// Main configuration structure that holds all parameters required for the 
/// grid generation process. This struct is populated by reading the `job_config.toml` file.
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    // Basic identifiers
    pub meshname: String,

    pub jobname: String,

    pub export_metrics: bool,

    pub Mach_inf: f64,

    pub max_iterations: usize,

    pub convergence_residual: f64,

    pub gamma: f64,

    pub omega: f64,

    pub beta_mult: f64,

    pub alpha_sequence_length: usize,

    pub C: f64,

    #[serde(skip)]
    pub u_inf: f64,

    #[serde(skip)]    
    pub rho_inf: f64,
    
}

impl Config {
    /// Método para popular as variáveis que dependem da física após a leitura do arquivo
    pub fn compute_derived_variables(&mut self) {
        // Inifinite velocity
        self.u_inf = ((self.gamma + 1.0) / (self.gamma - 1.0 + 2.0 / self.Mach_inf.powi(2))).sqrt();
        
        // Infinite density (usando a relação isentrópica para fluxo compressível)
        self.rho_inf = (1.0 - (self.gamma - 1.0) / (self.gamma + 1.0) * self.u_inf.powi(2)).powf(1.0 / (self.gamma - 1.0));
    }
}