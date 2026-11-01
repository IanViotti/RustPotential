#![allow(non_snake_case)]

use serde::Deserialize;

fn default_rg() -> f64 {
    0.5
}

/// Main configuration structure that holds all parameters required for the 
/// solver process. This struct is populated by reading the `job_config.toml` file.
#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    // Basic identifiers
    pub meshname: String,

    pub jobname: String,

    pub export_metrics: bool,

    pub Mach_inf: f64,

    #[serde(default, alias = "alpha_deg", alias = "alpha", alias = "aoa")]
    pub AoA: f64,

    #[serde(default = "default_rg", alias = "r_g", alias = "rg")]
    pub Rg: f64,

    pub max_iterations: usize,

    pub convergence_residual: f64,

    pub gamma: f64,

    pub omega: f64,

    pub beta_mult: f64,

    pub alpha_sequence_length: usize,

    pub C: f64,

    #[serde(skip)]
    pub output_dir: String,

    #[serde(skip)]
    pub alpha_rad: f64,

    #[serde(skip)]
    pub q_inf: f64,

    #[serde(skip)]
    pub u_inf: f64,

    #[serde(skip)]
    pub v_inf: f64,

    #[serde(skip)]    
    pub rho_inf: f64,
}

impl Config {
    /// Populates derived physical variables after reading the configuration file.
    pub fn compute_derived_variables(&mut self) {
        self.alpha_rad = self.AoA.to_radians();

        // Non-dimensional freestream speed magnitude (normalized by critical sound speed a*)
        self.q_inf = ((self.gamma + 1.0) * self.Mach_inf.powi(2)
            / (2.0 + (self.gamma - 1.0) * self.Mach_inf.powi(2)))
            .sqrt();

        // Cartesian freestream velocity components
        self.u_inf = self.q_inf * self.alpha_rad.cos();
        self.v_inf = self.q_inf * self.alpha_rad.sin();
        
        // Non-dimensional freestream density (normalized by stagnation density rho_t)
        self.rho_inf = (1.0 - (self.gamma - 1.0) / (self.gamma + 1.0) * self.q_inf.powi(2))
            .powf(1.0 / (self.gamma - 1.0));
    }
}
