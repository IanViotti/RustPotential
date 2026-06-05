use ndarray::{Array2, iter};

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

    solver_utils::calculate_field(config, grid, &mut domain);
    
    println!("Solver completed for mesh: {}", config.meshname);
    
    // Return the fully computed field for post-processing and VTK export
    domain
}

fn main_solver_loop(
    config: &config::Config, 
    grid: &Grid, 
    domain: &mut Array2<solver_utils::PointData>
) {
    // Sequência ótima do parâmetro alpha de Holst para acelerar convergência
    let alphas = solver_utils::alpha_seq(config, grid); 
    let omega = config.omega; // Fator de relaxação

    for iteration in 1..config.max_iterations {
        let alpha = alphas[iteration % alphas.len()]; // Cicla os alphas

        enforce_boundary_conditions(config, grid, domain);
        
        // 2. Calcular o campo de Resíduos L(phi)
        let mut max_res = 0.0;
        let mut residual = Array2::<f64>::zeros((grid.imax, grid.jmax));
        
        // Calculate residuals in all points, except in outter boundary and i=0 (which is periodic with i=imax-1) 
        for i in 1..grid.imax {
            for j in 0..grid.jmax-1 {
                let res = calc_L_phi_ij(config, grid, domain, i, j);
                residual[[i, j]] = res;
                if res.abs() > max_res { max_res = res.abs(); }
            }
        }

        // Matrizes intermediárias para o AF2
        let mut f_inter = Array2::<f64>::zeros((grid.imax, grid.jmax));
        let mut delta_phi = Array2::<f64>::zeros((grid.imax, grid.jmax));

        // =========================================================
        // Passo 1 (AF2): Varredura em Eta (j) - Radial
        // =========================================================
        for i in 1..grid.imax {
            // Extrair coeficientes A, B, C e rhs (alpha * omega * residual) da linha 'i'
            // Aplicar Algoritmo de Thomas (TDMA) simples com condições de contorno
            // Salvar resultado em f_inter[[i, j]]
            solver_utils::solve_tdma_eta(i, alpha, omega, grid, domain, &residual, &mut f_inter);
        }

        // =========================================================
        // Passo 2 (AF2): Varredura em Xi (i) - Azimutal (Malha O)
        // =========================================================
        for j in 0..grid.jmax-1 {
            // Extrair coeficientes A, B, C e rhs (f_inter) do anel 'j'
            // ATENÇÃO: Aqui DEVE ser usado o Solver Tridiagonal Periódico (Cíclico)
            // Salvar o resultado final em delta_phi[[i, j]]
            solver_utils::solve_periodic_tdma_xi(j, alpha, grid, domain, &f_inter, &mut delta_phi);
        }

        // =========================================================
        // Atualização do Potencial Phi
        // =========================================================
        for i in 1..grid.imax {
            for j in 0..grid.jmax-1 {
                domain[[i, j]].flow_ij.phi += delta_phi[[i, j]];
            }
        }

        if iteration % 100 == 0 || iteration == config.max_iterations - 1  || iteration == 1{
            println!("Iteration: {}/{} | Max Residual: {:.2e}", iteration, config.max_iterations, max_res);
        }
        
        if max_res < config.convergence_residual {
            println!("Converged at iteration {} with max residual {:.2e}", iteration, max_res);
            break;
        }
        
    }   
}

fn calc_L_phi_ij(config: &config::Config, grid: &Grid, domain: &mut Array2<solver_utils::PointData>, i: usize, j: usize) -> f64{

    let gamma = config.gamma;

    // Mapeamento topológico da Malha O (Nós)
    // Evita panic (out of bounds) e garante periodicidade
    let i_plus = if i == grid.imax - 1 { 1 } else { i + 1 };
    let i_minus = if i == 1 { grid.imax - 1 } else { i - 1 };

    // =========================================================
    // === Parameters for ip12j (Face na direção Xi) ===
    // =========================================================
    let phi_xi_ip12j = domain[[i_plus, j]].flow_ij.phi - domain[[i, j]].flow_ij.phi; 
    let phi_eta_ip12_j = if j == 0 {
        let a2 = domain[[i, 0]].metrics_ip12j.A2;
        let a3 = domain[[i, 0]].metrics_ip12j.A3;
        -(a2 / a3) * phi_xi_ip12j
    } else {
        0.5 * (
            0.5 * (domain[[i, j+1]].flow_ij.phi - domain[[i, j-1]].flow_ij.phi) +
            0.5 * (domain[[i_plus, j+1]].flow_ij.phi - domain[[i_plus, j-1]].flow_ij.phi)
        )
    };

    domain[[i, j]].flow_ip12j.U = 
        domain[[i, j]].metrics_ip12j.A1 * phi_xi_ip12j + 
        domain[[i, j]].metrics_ip12j.A2 * phi_eta_ip12_j;

    domain[[i, j]].flow_ip12j.V = 
        domain[[i, j]].metrics_ip12j.A2 * phi_xi_ip12j + 
        domain[[i, j]].metrics_ip12j.A3 * phi_eta_ip12_j;

    let q2_ip12j = domain[[i, j]].flow_ip12j.U * phi_xi_ip12j + domain[[i, j]].flow_ip12j.V * phi_eta_ip12_j;
    let base_ip12j = 1.0 - ((gamma - 1.0) / (gamma + 1.0)) * q2_ip12j;
    domain[[i, j]].flow_ip12j.rho = base_ip12j.max(1e-5).powf(1.0 / (gamma - 1.0));

    // =========================================================
    // === Parameters for ijp12 (Face na direção Eta) ===
    // =========================================================
    let phi_eta_ijp12 = domain[[i, j+1]].flow_ij.phi - domain[[i, j]].flow_ij.phi;
    let phi_xi_ijp12 = 0.5 * (
        0.5 * (domain[[i_plus, j]].flow_ij.phi - domain[[i_minus, j]].flow_ij.phi) +
        0.5 * (domain[[i_plus, j+1]].flow_ij.phi - domain[[i_minus, j+1]].flow_ij.phi)
    );

    domain[[i, j]].flow_ijp12.U = 
        domain[[i, j]].metrics_ijp12.A1 * phi_xi_ijp12 + 
        domain[[i, j]].metrics_ijp12.A2 * phi_eta_ijp12;

    domain[[i, j]].flow_ijp12.V = 
        domain[[i, j]].metrics_ijp12.A2 * phi_xi_ijp12 + 
        domain[[i, j]].metrics_ijp12.A3 * phi_eta_ijp12;

    let q2_ijp12 = domain[[i, j]].flow_ijp12.U * phi_xi_ijp12 + domain[[i, j]].flow_ijp12.V * phi_eta_ijp12;
    let base_ijp12 = 1.0 - ((gamma - 1.0) / (gamma + 1.0)) * q2_ijp12;
    domain[[i, j]].flow_ijp12.rho = base_ijp12.max(1e-5).powf(1.0 / (gamma - 1.0));

    // =========================================================
    // === Integer Point Rho (Média das 4 faces) ===
    // =========================================================
    let rho_bottom = if j == 0 {
        domain[[i, 0]].flow_ijp12.rho 
    } else {
        domain[[i, j-1]].flow_ijp12.rho
    };
    
    // Mapeamento topológico da face esquerda (i-1/2)
    let rho_left = if i == 1 {
        domain[[grid.imax - 1, j]].flow_ip12j.rho
    } else {
        domain[[i-1, j]].flow_ip12j.rho
    };

    domain[[i, j]].flow_ij.rho = 0.25 * (
        domain[[i, j]].flow_ip12j.rho + 
        rho_left + 
        domain[[i, j]].flow_ijp12.rho + 
        rho_bottom
    );

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
        transition_funcion_ip12j = (C1 - domain[[i_plus,j]].flow_ij.rho) * C2 * C;
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
    let r: i32 = if domain[[i, j]].flow_ip12j.U >= 0.0 { -1 } else { 1 };
    let s: i32 = if domain[[i, j]].flow_ijp12.V >= 0.0 { -1 } else { 1 };

    let j_upwind = if j == 0 && s == -1 {
        0 
    } else {
        (j as i32 + s) as usize
    };

    // Mapeamento topológico da face Upwind (i+r)
    let mut i_upwind = i as i32 + r;
    if i_upwind == grid.imax as i32 {
        i_upwind = 1; // Face imax+1/2 é fisicamente idêntica à Face 1+1/2
    } else if i_upwind == 0 {
        i_upwind = (grid.imax - 1) as i32; // Face 0+1/2 é fisicamente idêntica à Face (imax-1)+1/2
    }
    let i_upwind_mapped = i_upwind as usize;

    // Artificial Density Coefficients
    domain[[i,j]].flow_ip12j.rho_art = (1.0 - v_ip12j) * domain[[i, j]].flow_ip12j.rho + v_ip12j * domain[[i_upwind_mapped, j]].flow_ip12j.rho;
    domain[[i,j]].flow_ijp12.rho_art = (1.0 - v_ijp12) * domain[[i, j]].flow_ijp12.rho + v_ijp12 * domain[[i, j_upwind]].flow_ijp12.rho;

    // Mapeamento topológico dos Fluxos para garantir a Conservação da Massa
    let flux_eta_minus = if j == 0 {
        0.0 
    } else {
        domain[[i,j-1]].flow_ijp12.rho_art * domain[[i,j-1]].flow_ijp12.V / domain[[i,j-1]].metrics_ijp12.J
    };

    let flux_xi_minus = if i == 1 {
        domain[[grid.imax - 1, j]].flow_ip12j.rho_art * domain[[grid.imax - 1, j]].flow_ip12j.U / domain[[grid.imax - 1, j]].metrics_ip12j.J
    } else {
        domain[[i-1, j]].flow_ip12j.rho_art * domain[[i-1, j]].flow_ip12j.U / domain[[i-1, j]].metrics_ip12j.J
    };

    let flux_xi_plus = domain[[i,j]].flow_ip12j.rho_art * domain[[i,j]].flow_ip12j.U / domain[[i,j]].metrics_ip12j.J;
    let flux_eta_plus = domain[[i,j]].flow_ijp12.rho_art * domain[[i,j]].flow_ijp12.V / domain[[i,j]].metrics_ijp12.J;

    // Residual operator
    let L_phi_ij = (flux_xi_plus - flux_xi_minus) + (flux_eta_plus - flux_eta_minus);
    
    return L_phi_ij;
}

pub fn enforce_boundary_conditions(config: &config::Config, grid: &Grid, domain: &mut Array2<solver_utils::PointData>) {
    let imax = grid.imax;
    let jmax = grid.jmax;

    // ====================================================================
    // 2. Condição de Periodicidade no "Wake Cut" (i = 0 e i = imax-1)
    // ====================================================================
    for j in 0..jmax {
        // Copia os dados do nó inteiro
        domain[[0, j]].flow_ij = domain[[imax - 1, j]].flow_ij.clone();
        
        // Copia os dados das faces (meio-pontos) para manter a integridade total
        domain[[0, j]].flow_ip12j = domain[[imax - 1, j]].flow_ip12j.clone();
        domain[[0, j]].flow_ijp12 = domain[[imax - 1, j]].flow_ijp12.clone();
    }

    // ====================================================================
    // 3. Condição de Farfield (j = jmax - 1)
    // ====================================================================
    for i in 0..imax {
        let x = grid.points[[i, jmax - 1]].x;
        // let y = grid.points[[i, jmax - 1]].y; // Descomente caso o ângulo de ataque seja != 0
        
        // Assumindo alfa = 0 (Escoamento livre alinhado com o eixo X)
        domain[[i, jmax - 1]].flow_ij.phi = config.u_inf * x; 
        domain[[i, jmax - 1]].flow_ij.rho = config.rho_inf;

    }

}