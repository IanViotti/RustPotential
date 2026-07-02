use ndarray::{Array2, iter};

use crate::config;
use crate::job_utils::Grid;
use crate::solver_utils;
use crate::af2;

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
    // Optimal sequence of Holst's alpha parameter to accelerate convergence
    let alphas = solver_utils::alpha_seq(config, grid); 
    let omega = config.omega; // Relaxation factor
    let m_cycle = alphas.len(); // Length of the alpha cycle (M)

    // Initialization of Beta control variables (As per Point 1)
    let beta_sub: f64 = 0.3;
    let mut beta_sup: f64 = config.beta_mult;
    let b_high = beta_sup + 1.0;
    let b_low = beta_sup - 1.0;

    // Residual history to compare iteration n with (n - M)
    let mut prev_rms_res: f64 = 1.0; // Prevents division by zero on the first pass
    let mut prev_max_res: f64 = 1.0;

    // It's standard to start at 0 so modulo arithmetic for cycles works perfectly
    for iteration in 1..config.max_iterations {
        let alpha = alphas[iteration % m_cycle]; // Cycle through the alphas

        // =========================================================
        // PASSO A: Atualizar o Estado Físico em Todo o Domínio
        // =========================================================
        update_thermodynamics(config, grid, domain);

        // =========================================================
        // PASSO B: Sincronizar Condições de Contorno e Wake Cut
        // (Isso copiará os rho_art e Fluxos recém-calculados de imax-1 para 0)
        // =========================================================
        enforce_boundary_conditions(config, grid, domain);
        
        // =========================================================
        // 1. Calculate the Residual field L(phi)
        // =========================================================
        let mut max_res = 0.0;
        let mut sum_sq_res = 0.0; // Sum of squares for the L2 norm
        let mut count = 0.0;
        let mut residual = Array2::<f64>::zeros((grid.imax, grid.jmax));

        for i in 1..grid.imax {
            for j in 0..grid.jmax - 1 {
                let res = calc_L_phi_ij(grid, domain, i, j);
                residual[[i, j]] = res;
                
                let abs_res = res.abs();
                if abs_res > max_res { 
                    max_res = abs_res; 
                }
                
                // Accumulate the squared residual for the L2 norm
                sum_sq_res += res * res;
                count += 1.0;
            }
        }

        // Apply Periodicity Condition (Mathematical stitching of the Wake Cut)
        for j in 0..grid.jmax - 1 {
            residual[[0, j]] = residual[[grid.imax - 1, j]];
        }

        // Calculate the RMS (Root Mean Square / Average L2 norm)
        let rms_res = (sum_sq_res / count).sqrt();

        // =========================================================
        // 2. Dynamic Update Logic (Convergence Feedback)
        // =========================================================
        // Occurs only at the end of a complete alpha cycle (every M iterations)
        if iteration > 0 && iteration % m_cycle == 0 {
            let ratio = (rms_res / prev_rms_res) + (max_res / prev_max_res);

            if ratio < 2.0 {
                beta_sup *= 0.98; // Good convergence: accelerate the solver
            } else if ratio > 2.1 {
                beta_sup *= 1.1;  // Risk of instability: increase damping immediately
            }

            // Impose safety limits
            beta_sup = beta_sup.clamp(b_low, b_high);

            // Update the history for the next cycle comparison
            prev_rms_res = rms_res;
            prev_max_res = max_res;
        }

        // =========================================================
        // 3. Execute the AF2 Scheme
        // =========================================================
        // Pass both beta_sub and beta_sup so the solver handles them internally
        af2::execute_af2(grid, domain, &residual, alpha, beta_sub, beta_sup, omega);

        // =========================================================
        // 4. Convergence Check & Logging
        // =========================================================
        if iteration == 1 || iteration % 10 == 0 || iteration == config.max_iterations - 1 {
            println!(
                "Iteration: {}/{} | Max Res: {:.2e} | RMS Res: {:.2e} | Sup Beta: {:.2}", 
                iteration, config.max_iterations, max_res, rms_res, beta_sup
            );
        }
        
        if max_res < config.convergence_residual {
            println!("Converged at iteration {} with max residual {:.2e}", iteration, max_res);
            break;
        }
    }   
}

pub fn update_thermodynamics(config: &config::Config, grid: &Grid, domain: &mut Array2<solver_utils::PointData>) {
    let gamma = config.gamma;
    let c_rho = (gamma - 1.0) / (gamma + 1.0);
    let mach_factor = 2.0 / (gamma + 1.0);

    // =========================================================
    // CICLO 1: Propriedades Isentrópicas, Mach e Velocidades
    // =========================================================
    for i in 1..grid.imax {
        for j in 0..grid.jmax - 1 {
            let i_plus = if i == grid.imax - 1 { 1 } else { i + 1 };
            let i_minus = if i == 1 { grid.imax - 1 } else { i - 1 };

            // --- Faces Xi (i+1/2, j) ---
            let phi_xi_ip12j = domain[[i_plus, j]].flow_ij.phi - domain[[i, j]].flow_ij.phi;
            let phi_eta_ip12_j = if j == 0 {
                -(domain[[i, 0]].metrics_ip12j.A2 / domain[[i, 0]].metrics_ip12j.A3) * phi_xi_ip12j
            } else {
                0.25 * (
                    (domain[[i, j+1]].flow_ij.phi - domain[[i, j-1]].flow_ij.phi) +
                    (domain[[i_plus, j+1]].flow_ij.phi - domain[[i_plus, j-1]].flow_ij.phi)
                )
            };

            domain[[i, j]].flow_ip12j.U = domain[[i, j]].metrics_ip12j.A1 * phi_xi_ip12j + domain[[i, j]].metrics_ip12j.A2 * phi_eta_ip12_j;
            domain[[i, j]].flow_ip12j.V = domain[[i, j]].metrics_ip12j.A2 * phi_xi_ip12j + domain[[i, j]].metrics_ip12j.A3 * phi_eta_ip12_j;
            
            let q2_ip12j = domain[[i, j]].flow_ip12j.U * phi_xi_ip12j + domain[[i, j]].flow_ip12j.V * phi_eta_ip12_j;
            let base_ip12j = 1.0 - c_rho * q2_ip12j;
            domain[[i, j]].flow_ip12j.rho = base_ip12j.max(1e-5).powf(1.0 / (gamma - 1.0));

            // --- Faces Eta (i, j+1/2) ---
            let phi_eta_ijp12 = domain[[i, j+1]].flow_ij.phi - domain[[i, j]].flow_ij.phi;
            let phi_xi_ijp12 = 0.25 * (
                (domain[[i_plus, j]].flow_ij.phi - domain[[i_minus, j]].flow_ij.phi) +
                (domain[[i_plus, j+1]].flow_ij.phi - domain[[i_minus, j+1]].flow_ij.phi)
            );

            domain[[i, j]].flow_ijp12.U = domain[[i, j]].metrics_ijp12.A1 * phi_xi_ijp12 + domain[[i, j]].metrics_ijp12.A2 * phi_eta_ijp12;
            domain[[i, j]].flow_ijp12.V = domain[[i, j]].metrics_ijp12.A2 * phi_xi_ijp12 + domain[[i, j]].metrics_ijp12.A3 * phi_eta_ijp12;
            
            let q2_ijp12 = domain[[i, j]].flow_ijp12.U * phi_xi_ijp12 + domain[[i, j]].flow_ijp12.V * phi_eta_ijp12;
            let base_ijp12 = 1.0 - c_rho * q2_ijp12;
            domain[[i, j]].flow_ijp12.rho = base_ijp12.max(1e-5).powf(1.0 / (gamma - 1.0));

            // --- Ponto Inteiro (i, j) para Sensor Mach e Velocidade Contravariante ---
            let phi_xi_ij = 0.5 * (domain[[i_plus, j]].flow_ij.phi - domain[[i_minus, j]].flow_ij.phi);
            let phi_eta_ij = if j == 0 {
                -(domain[[i, j]].metrics_ij.A2 / domain[[i, j]].metrics_ij.A3.max(1e-12)) * phi_xi_ij
            } else {
                0.5 * (domain[[i, j+1]].flow_ij.phi - domain[[i, j-1]].flow_ij.phi)
            };

            let u_contra = domain[[i, j]].metrics_ij.A1 * phi_xi_ij + domain[[i, j]].metrics_ij.A2 * phi_eta_ij;
            let v_contra = domain[[i, j]].metrics_ij.A2 * phi_xi_ij + domain[[i, j]].metrics_ij.A3 * phi_eta_ij;
            let q2_ij = u_contra * phi_xi_ij + v_contra * phi_eta_ij;

            let base_ij = 1.0 - c_rho * q2_ij;
            domain[[i, j]].flow_ij.rho = base_ij.max(1e-5).powf(1.0 / (gamma - 1.0));
            domain[[i, j]].flow_ij.M = (mach_factor * (q2_ij / base_ij.max(1e-5))).sqrt();
            
            // Salvamos a contravariante para o estabilizador AF2 ler corretamente
            domain[[i, j]].flow_ij.u = u_contra; 
        }
    }

    // Sincronizamos preventivamente os nós isentrópicos antes de calcular o upwind da densidade artificial
    enforce_boundary_conditions(config, grid, domain);

    // =========================================================
    // CICLO 2: Densidade Artificial (Upwind de Holst)
    // =========================================================
    let C1 = 0.633939; // Para gamma = 1.4
    let C2 = 4.9325;
    let C = config.C;

    for i in 1..grid.imax {
        for j in 0..grid.jmax - 1 {
            let i_plus = if i == grid.imax - 1 { 1 } else { i + 1 };

            // Transição em Xi
            let v_ip12j = if domain[[i, j]].flow_ip12j.U >= 0.0 {
                0.0_f64.max((C1 - domain[[i,j]].flow_ij.rho) * C2 * C).min(1.0)
            } else {
                0.0_f64.max((C1 - domain[[i_plus,j]].flow_ij.rho) * C2 * C).min(1.0)
            };

            // Transição em Eta
            let v_ijp12 = if domain[[i, j]].flow_ijp12.V >= 0.0 {
                0.0_f64.max((C1 - domain[[i,j]].flow_ij.rho) * C2 * C).min(1.0)
            } else {
                0.0_f64.max((C1 - domain[[i,j+1]].flow_ij.rho) * C2 * C).min(1.0)
            };

            // Direção Upwind
            let r: i32 = if domain[[i, j]].flow_ip12j.U >= 0.0 { -1 } else { 1 };
            let s: i32 = if domain[[i, j]].flow_ijp12.V >= 0.0 { -1 } else { 1 };

            let j_upwind = if j == 0 && s == -1 { 0 } else { (j as i32 + s) as usize };
            let mut i_upwind = i as i32 + r;
            if i_upwind == grid.imax as i32 { i_upwind = 1; } 
            else if i_upwind == 0 { i_upwind = (grid.imax - 1) as i32; }
            
            let i_upwind_mapped = i_upwind as usize;

            // Densidades Artificiais Finais
            domain[[i,j]].flow_ip12j.rho_art = (1.0 - v_ip12j) * domain[[i, j]].flow_ip12j.rho + v_ip12j * domain[[i_upwind_mapped, j]].flow_ip12j.rho;
            domain[[i,j]].flow_ijp12.rho_art = (1.0 - v_ijp12) * domain[[i, j]].flow_ijp12.rho + v_ijp12 * domain[[i, j_upwind]].flow_ijp12.rho;
        }
    }
}

fn calc_L_phi_ij(grid: &Grid, domain: &Array2<solver_utils::PointData>, i: usize, j: usize) -> f64 {
    // Fluxo na face inferior (j - 1/2)
    let flux_eta_minus = if j == 0 {
        0.0 // Parede impermeável
    } else {
        domain[[i, j-1]].flow_ijp12.rho_art * domain[[i, j-1]].flow_ijp12.V / domain[[i, j-1]].metrics_ijp12.J
    };

    // Fluxo na face esquerda (i - 1/2)
    // O nó 'i-1' garante automaticamente o fechamento periódico graças às condições de contorno pré-aplicadas!
    let flux_xi_minus = domain[[i-1, j]].flow_ip12j.rho_art * domain[[i-1, j]].flow_ip12j.U / domain[[i-1, j]].metrics_ip12j.J;

    // Fluxo na face direita (i + 1/2)
    let flux_xi_plus = domain[[i, j]].flow_ip12j.rho_art * domain[[i, j]].flow_ip12j.U / domain[[i, j]].metrics_ip12j.J;
    
    // Fluxo na face superior (j + 1/2)
    let flux_eta_plus = domain[[i, j]].flow_ijp12.rho_art * domain[[i, j]].flow_ijp12.V / domain[[i, j]].metrics_ijp12.J;

    // Operador Resíduo
    (flux_xi_plus - flux_xi_minus) + (flux_eta_plus - flux_eta_minus)
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