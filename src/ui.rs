use std::io::{self, Write};
use std::time::Duration;
use ndarray::Array2;

use crate::config::Config;
use crate::job_utils::Grid;
use crate::solver_core::SolverStats;
use crate::solver_utils::PointData;

/// Displays the project visual headline and ASCII banner.
pub fn print_banner() {
    println!(r#"
================================================================================
    ______      _ _   _____       _             _   _       _ 
   |  ____|    | | | |  __ \     | |           | | (_)     | |
   | |__ _   _ | | | | |__) |___ | |_ ___ _ __ | |_ _  __ _| |
   |  __| | | || | | |  ___/ _ \| __/ _ \ '_ \| __| |/ _` | |
   | |  | |_| || | | | |  | (_) | ||  __/ | | | |_| | (_| | |
   |_|   \__,_||_|_| |_|   \___/ \__\___|_| |_|\__|_|\__,_|_|
                                                              
        Full Potential CFD Solver in Rust | AF2 Algorithm
================================================================================"#);
}

/// Formats and displays an organized configuration dashboard.
pub fn print_config_dashboard(config: &Config, grid: &Grid) {
    let total_nodes = grid.imax * grid.jmax;
    let regime = if config.Mach_inf < 0.8 {
        "Subsonic"
    } else if config.Mach_inf <= 1.2 {
        "Transonic"
    } else {
        "Supersonic"
    };

    println!(
r#"--------------------------------------------------------------------------------
  JOB & MESH SPECIFICATIONS
--------------------------------------------------------------------------------
  Job Name             : {}
  Mesh File            : {}
  Grid Dimensions      : {} x {} ({} total nodes)
  Flow Regime          : M_inf = {:.4} ({})
  Specific Heat Ratio  : gamma = {:.2}
  Freestream Speed     : u_inf = {:.6} (non-dimensional)
  Freestream Density   : rho_inf = {:.6} (non-dimensional)
--------------------------------------------------------------------------------
  AF2 ITERATION PARAMETERS
--------------------------------------------------------------------------------
  Max Iterations       : {}
  Tolerance (Max Res)  : {:.2e}
  Relaxation Factor    : omega = {:.2}
  Alpha Cycle Length   : M = {}
  Temporal Damping     : beta_sub = 0.30, beta_sup = {:.2}
  Holst Upwind Coeff   : C = {:.2}
--------------------------------------------------------------------------------"#,
        config.jobname,
        config.meshname,
        grid.imax,
        grid.jmax,
        total_nodes,
        config.Mach_inf,
        regime,
        config.gamma,
        config.u_inf,
        config.rho_inf,
        config.max_iterations,
        config.convergence_residual,
        config.omega,
        config.alpha_sequence_length,
        config.beta_mult,
        config.C
    );
}

/// Prints the table header for the iterative AF2 solver loop.
pub fn print_iteration_header() {
    println!(
r#"+-------+--------------+--------------+------------+--------------------+
| Iter  |   Max Res    |   RMS Res    |  Sup Beta  | Status             |
+-------+--------------+--------------+------------+--------------------+"#
    );
}

/// Prints a single row in the iteration table.
pub fn print_iteration_row(iteration: usize, max_res: f64, rms_res: f64, beta_sup: f64, status: &str) {
    println!(
        "| {:5} |   {:.4e}   |   {:.4e}   |    {:5.2}   | {:<18} |",
        iteration, max_res, rms_res, beta_sup, status
    );
}

/// Prints the footer closing line for the iteration table.
pub fn print_iteration_footer() {
    println!("+-------+--------------+--------------+------------+--------------------+");
}

/// Displays a comprehensive post-computation summary with aerodynamic figures of merit.
pub fn print_summary(
    config: &Config,
    grid: &Grid,
    domain: &Array2<PointData>,
    stats: &SolverStats,
    total_duration: Duration,
) {
    let j_wall = grid.jmax - 1;
    let gamma = config.gamma;
    let m_inf = config.Mach_inf;
    let rho_inf = config.rho_inf;
    let cp_multiplier = 2.0 / (gamma * m_inf.powi(2));

    // 1. Overall Mach range
    let mut max_mach = 0.0f64;
    let mut min_mach = f64::MAX;
    for pt in domain.iter() {
        let m = pt.flow_ij.M;
        if m > max_mach { max_mach = m; }
        if m < min_mach { min_mach = m; }
    }

    // 2. Wall Cp extrema
    let mut cp_min = f64::MAX;
    let mut x_cp_min = 0.0f64;
    let mut cp_max = -f64::MAX;
    let mut x_cp_max = 0.0f64;

    for i in 0..grid.imax - 1 {
        let x = grid.points[[i, j_wall]].x;
        let rho = domain[[i, j_wall]].flow_ij.rho;
        let cp = cp_multiplier * ((rho / rho_inf).powf(gamma) - 1.0);

        if cp < cp_min {
            cp_min = cp;
            x_cp_min = x;
        }
        if cp > cp_max {
            cp_max = cp;
            x_cp_max = x;
        }
    }

    // Exact theoretical isentropic stagnation Cp
    let p_ratio_stag = (1.0 + 0.5 * (gamma - 1.0) * m_inf.powi(2)).powf(gamma / (gamma - 1.0));
    let cp_stag_theory = (2.0 / (gamma * m_inf.powi(2))) * (p_ratio_stag - 1.0);

    let status_str = if stats.converged {
        format!("CONVERGED in {} iterations", stats.iterations)
    } else {
        format!("MAX ITERATIONS REACHED ({})", stats.iterations)
    };

    let total_ms = total_duration.as_secs_f64() * 1000.0;
    let ms_per_iter = if stats.iterations > 0 {
        total_ms / (stats.iterations as f64)
    } else {
        0.0
    };

    println!(
r#"
================================================================================
                           COMPUTATION SUMMARY
================================================================================
  Status               : {}
  Target Tolerance     : Max Res < {:.2e}
  Final Max Residual   : {:.4e}
  Final RMS Residual   : {:.4e}
  Total Solver Time    : {:.2} ms ({:.2} ms/iteration)
--------------------------------------------------------------------------------
                         AERODYNAMIC FIELD RESULTS
--------------------------------------------------------------------------------
  Freestream Mach      : {:.4}
  Flowfield Mach Range : [{:.4}, {:.4}]
  Stagnation Cp (Wall) : {:+.4} (Theoretical: {:+.4}) at x = {:.3}
  Suction Peak Cp_min  : {:+.4} at x = {:.3}
--------------------------------------------------------------------------------
                           OUTPUT FILES GENERATED
--------------------------------------------------------------------------------
  [VTK 3D Solution]    : job_files/{}/solution.vtk
  [Wall Cp Profile]    : job_files/{}/Cp_distribution.csv
================================================================================
Job completed successfully!"#,
        status_str,
        config.convergence_residual,
        stats.final_max_res,
        stats.final_rms_res,
        total_ms,
        ms_per_iter,
        m_inf,
        min_mach,
        max_mach,
        cp_max,
        cp_stag_theory,
        x_cp_max,
        cp_min,
        x_cp_min,
        config.jobname,
        config.jobname
    );
}

/// Pauses execution until the user presses Enter, preventing the terminal from auto-closing.
pub fn wait_for_user_exit() {
    print!("\nPress [Enter] to exit...");
    let _ = io::stdout().flush();
    let mut buffer = String::new();
    let _ = io::stdin().read_line(&mut buffer);
}
