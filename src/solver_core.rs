use ndarray::Array2;

use crate::config;
use crate::job_utils::Grid;
use crate::solver_utils;

/// Main entry point for the computational fluid dynamics solver.
/// Initializes the domain, calculates metrics, and triggers the iterative process.
pub fn solve(grid: &Grid, config: &config::Config) -> Array2<solver_utils::PointData> {
    println!("Starting the solver with mesh: {}", config.meshname);
    
    // 1. Initialize the single domain array containing all nodes, half-points, and metrics.
    // This creates a single source of truth for the entire grid.
    let mut domain = solver_utils::initialize_domain(config, grid);

    // 2. Calculate geometric metrics in-place.
    // The mutable reference '&mut domain' allows the function to write directly 
    // into the domain structures without allocating new arrays in memory.
    solver_utils::calculate_metrics(grid, &mut domain);

    // 3. Enter the main iterative solver loop (AF2 scheme).
    main_solver_loop(config, grid, &mut domain);
    
    println!("Solver completed for mesh: {}", config.meshname);
    
    // Return the fully computed field for post-processing and VTK export
    domain
}

/// Iterative loop to drive the density and potential fields to convergence
fn main_solver_loop(
    config: &config::Config, 
    grid: &Grid, 
    domain: &mut Array2<solver_utils::PointData>
) {
    println!("Entering main solver loop...");
    
    for iteration in 0..config.max_iterations {
        println!("Iteration: {}", iteration);
        
        // =========================================================================
        // TODO: IMPLEMENT HOLST AND BALLHAUS AF2 SCHEME HERE
        // =========================================================================
        // 
        // Expected Sequence per Iteration:
        //
        // 1. Compute Contravariant Velocities (U, V) at the half-points.
        // 2. Compute isentropic density (rho) at the half-points.
        // 3. Apply Holst's Artificial Density (rho_art) using Upwinding 
        //    (checks local Mach number to add artificial viscosity in supersonic zones).
        // 4. Calculate the Residual at each node (i,j) using conservative finite differences.
        // 5. Check for convergence (e.g., maximum absolute residual < tolerance).
        //    If converged, break the loop early.
        // 6. AF2 Sweep 1: Solve the first bidiagonal/tridiagonal system (Eta direction).
        // 7. AF2 Sweep 2: Solve the second tridiagonal system (Xi direction).
        // 8. Update the nodal velocity potential (phi) using the computed correction.
        // 
        // =========================================================================

        solve_field(config, grid, domain);
    }   
}

fn solve_field(config: &config::Config, grid: &Grid, domain: &mut Array2<solver_utils::PointData>) {
    // Placeholder function to represent the entire AF2 solution process for one iteration.
    // The actual implementation will involve multiple steps as outlined in the main_solver_loop comments.

    // 1. Compute contravariant velocities and densities at half-points
    // 2. Compute residuals at nodes
    // 3. Check for convergence
    // 4. Perform AF2 sweeps to solve for corrections
    // 5. Update nodal phi values

    // For demonstration, we will just call a placeholder function to update phi at one point.

    for i in 1..grid.imax-1 {
        for j in 1..grid.jmax-1 {
            let L_phi_ij = calc_L_phi_ij(config, grid, domain, i, j);
        }
    }
}

fn calc_L_phi_ij(config: &config::Config, grid: &Grid, domain: &mut Array2<solver_utils::PointData>, i: usize, j: usize) -> f64{

    let gamma = config.gamma;

    // === Parameters for ip12j ===
    // Phi derivatives
    let phi_xi_ip12j = domain[[i+1, j]].flow_ij.phi - domain[[i, j]].flow_ij.phi; 
    let phi_eta_ip12_j = 0.5 * (
        0.5 * (domain[[i, j+1]].flow_ij.phi + domain[[i, j-1]].flow_ij.phi) +
        0.5 * (domain[[i+1, j+1]].flow_ij.phi + domain[[i+1, j-1]].flow_ij.phi)
    );

    // Contravariant 
    domain[[i, j]].flow_ip12j.U = 
        domain[[i, j]].metrics_ip12j.A1 * phi_xi_ip12j + 
        domain[[i, j]].metrics_ip12j.A2 * phi_eta_ip12_j;

    domain[[i, j]].flow_ip12j.V = 
        domain[[i, j]].metrics_ip12j.A2 * phi_xi_ip12j + 
        domain[[i, j]].metrics_ip12j.A3 * phi_eta_ip12_j;

    // rho
    domain[[i, j]].flow_ip12j.rho = (1.0 - (gamma - 1.0 ) / (gamma + 1.0) * (domain[[i, j]].flow_ip12j.U * phi_xi_ip12j + domain[[i, j]].flow_ip12j.V * phi_eta_ip12_j)).powf(1.0 / (gamma - 1.0));

    // === Parameters for ijp12 ===
    // Phi derivatives
    let phi_xi_ijp12 = domain[[i, j+1]].flow_ij.phi - domain[[i, j]].flow_ij.phi; 
    let phi_eta_ijp12 = 0.5 * (
        0.5 * (domain[[i, j+1]].flow_ij.phi + domain[[i, j-1]].flow_ij.phi) +
        0.5 * (domain[[i+1, j+1]].flow_ij.phi + domain[[i+1, j-1]].flow_ij.phi)
    );

    // Contravariant 
    domain[[i, j]].flow_ijp12.U = 
        domain[[i, j]].metrics_ijp12.A1 * phi_xi_ijp12 + 
        domain[[i, j]].metrics_ijp12.A2 * phi_eta_ijp12;

    domain[[i, j]].flow_ijp12.V = 
        domain[[i, j]].metrics_ijp12.A2 * phi_xi_ijp12 + 
        domain[[i, j]].metrics_ijp12.A3 * phi_eta_ijp12;

    // rho
    domain[[i, j]].flow_ijp12.rho = (1.0 - (gamma - 1.0 ) / (gamma + 1.0) * (domain[[i, j]].flow_ijp12.U * phi_xi_ijp12 + domain[[i, j]].flow_ijp12.V * phi_eta_ijp12)).powf(1.0 / (gamma - 1.0));

    // == Integer Point Rho ==
    domain[[i,j]].flow_ij.rho = 0.25 * (domain[[i, j]].flow_ip12j.rho + domain[[i-1, j]].flow_ip12j.rho + domain[[i, j]].flow_ijp12.rho + domain[[i, j-1]].flow_ijp12.rho);

    // Transition function
    let v_ip12j: f64;
    let v_ijp12: f64;
    let transition_funcion_ip12j: f64;
    let transition_funcion_ijp12: f64;
    let C1 = 0.633939; // For gamma = 1.4
    let C2 = 4.9325;
    let C = config.C;

    if domain[[i, j]].flow_ip12j.U >= 0.0 {
        transition_funcion_ip12j = (C1 - domain[[i,j]].flow_ij.rho) * C2 * C;
        v_ip12j = 0.0_f64.max(transition_funcion_ip12j);
    } else {
        transition_funcion_ip12j = (C1 - domain[[i+1,j]].flow_ij.rho) * C2 * C;
        v_ip12j = 0.0_f64.max(transition_funcion_ip12j);
    }

    if domain[[i, j]].flow_ijp12.V >= 0.0 {
        transition_funcion_ijp12 = (C1 - domain[[i,j]].flow_ij.rho) * C2 * C;
        v_ijp12 = 0.0_f64.max(transition_funcion_ijp12);
    } else {
        transition_funcion_ijp12 = (C1 - domain[[i,j+1]].flow_ij.rho) * C2 * C;
        v_ijp12 = 0.0_f64.max(transition_funcion_ijp12);
    }

    // Upwind direction for artificial density
    let r: i32;
    let s: i32;

    if domain[[i, j]].flow_ip12j.U >= 0.0 {
        r = -1;
    } else {
        r = 1;
    }

    if domain[[i, j]].flow_ijp12.V >= 0.0 {
        s = -1;
    } else {
        s = 1;
    }

    // Artificial Density Coefficients
    domain[[i,j]].flow_ip12j.rho_art = (1.0 - v_ip12j) * domain[[i, j]].flow_ip12j.rho + v_ip12j * domain[[i + r as usize, j]].flow_ip12j.rho;
    domain[[i,j]].flow_ijp12.rho_art = (1.0 - v_ijp12) * domain[[i, j]].flow_ijp12.rho + v_ijp12 * domain[[i, j + s as usize]].flow_ijp12.rho;

    // Residual operator
    let L_phi_ij = ((domain[[i,j]].flow_ip12j.rho_art * domain[[i,j]].flow_ip12j.U / domain[[i,j]].metrics_ip12j.J) - 
                    (domain[[i-1,j]].flow_ip12j.rho_art * domain[[i-1,j]].flow_ip12j.U / domain[[i-1,j]].metrics_ip12j.J)) + 
                    ((domain[[i,j]].flow_ijp12.rho_art * domain[[i,j]].flow_ijp12.V / domain[[i,j]].metrics_ijp12.J) - 
                    (domain[[i,j-1]].flow_ijp12.rho_art * domain[[i,j-1]].flow_ijp12.V / domain[[i,j-1]].metrics_ijp12.J));

    return L_phi_ij;
}