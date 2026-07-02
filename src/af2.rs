use ndarray::Array2;
use crate::job_utils::Grid;
use crate::solver_utils::PointData;

/// Main orchestrator for the AF2 numerical scheme
pub fn execute_af2(
    grid: &Grid,
    domain: &mut Array2<PointData>,
    residual: &Array2<f64>,
    alpha: f64,
    beta_sub: f64,
    beta_sup: f64,
    omega: f64
) {
    let mut f_inter = Array2::<f64>::zeros((grid.imax, grid.jmax));
    let mut delta_phi = Array2::<f64>::zeros((grid.imax, grid.jmax));

    // =========================================================
    // Memory pre-allocation for Step 2 (Periodic Tridiagonal)
    // N is the number of unique nodes in the O-grid ring
    // =========================================================
    let n = grid.imax - 1;
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];
    
    // Auxiliary buffers for Sherman-Morrison
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    // =========================================================
    // STEP 1: Radial Sweep (Eta direction)
    // Solves the Bidiagonal system from Farfield to Wall
    // =========================================================
    for i in 1..grid.imax {
        step1_bidiagonal_eta(i, alpha, omega, grid, domain, residual, &mut f_inter);
    }

    // =========================================================
    // STEP 2: Azimuthal Sweep (Xi direction)
    // Solves the Periodic Tridiagonal system injecting Temporal Damping
    // =========================================================
    for j in 0..grid.jmax - 1 {
        step2_tridiagonal_xi(
            j, alpha, beta_sub, beta_sup, grid, domain, &f_inter, &mut delta_phi,
            &mut a, &mut b, &mut c, &mut d, 
            &mut y, &mut z, &mut c_prime, &mut d_prime
        );
    }

    // =========================================================
    // Final Update of the Velocity Potential
    // =========================================================
    for i in 1..grid.imax {
        for j in 0..grid.jmax - 1 {
            domain[[i, j]].flow_ij.phi += delta_phi[[i, j]];
        }
    }
}


/// Solves the first step of AF2. 
/// Employs a forward difference operator, so the sweep must be backwards (j decreasing).
fn step1_bidiagonal_eta(
    i: usize,
    alpha: f64,
    omega: f64,
    grid: &Grid,
    domain: &Array2<PointData>,
    residual: &Array2<f64>,
    f_inter: &mut Array2<f64>,
) {
    // Dirichlet boundary condition at the Farfield
    f_inter[[i, grid.jmax - 1]] = 0.0;

    // Reverse Sweep (From farfield towards the airfoil surface)
    for j in (0..grid.jmax - 1).rev() {
        
        // Extract A3 metric mapped at the j+1/2 face
        let a_plus = domain[[i, j]].flow_ijp12.rho_art 
                   * domain[[i, j]].metrics_ijp12.A3 
                   / domain[[i, j]].metrics_ijp12.J;

        let rhs = alpha * omega * residual[[i, j]];

        // Direct bidiagonal substitution (f_{i, j+1} is already computed)
        f_inter[[i, j]] = (rhs + a_plus * f_inter[[i, j + 1]]) / (alpha + a_plus);
    }
}


/// Solves the second step of AF2 incorporating the temporal damping (Beta).
/// Uses Sherman-Morrison to enforce exact periodicity at the wake cut.
fn step2_tridiagonal_xi(
    j: usize,
    alpha: f64,
    beta_sub: f64,
    beta_sup: f64,
    grid: &Grid,
    domain: &Array2<PointData>,
    f_inter: &Array2<f64>,
    delta_phi: &mut Array2<f64>,
    a: &mut [f64], b: &mut [f64], c: &mut [f64], d: &mut [f64],
    y: &mut [f64], z: &mut [f64], c_prime: &mut [f64], d_prime: &mut [f64]
) {
    let n = grid.imax - 1; 

    // 1. Build the matrix coefficients for the ring 'j'
    for idx in 0..n {
        let i = idx + 1; 
        
        // Topological mapping for O-Grid periodicity
        let i_plus = if i == grid.imax - 1 { 1 } else { i + 1 };
        let i_minus = if i == 1 { grid.imax - 1 } else { i - 1 };

        // Spatial central difference operators based on A1 metric
        let b_plus = domain[[i, j]].flow_ip12j.rho_art * domain[[i, j]].metrics_ip12j.A1 / domain[[i, j]].metrics_ip12j.J;
        let b_minus = domain[[i_minus, j]].flow_ip12j.rho_art * domain[[i_minus, j]].metrics_ip12j.A1 / domain[[i_minus, j]].metrics_ip12j.J;

        a[idx] = -b_minus;
        b[idx] = alpha + b_plus + b_minus;
        c[idx] = -b_plus;
        d[idx] = f_inter[[i, j]];

        // =======================================================
        // TEMPORAL DAMPING INJECTION (Beta Control)
        // Introduces the phi_{xi, t} cross-derivative for stability
        // =======================================================
        let mach_local = domain[[i, j]].flow_ij.M;
        let u_contra = domain[[i, j]].flow_ij.u; // Contravariant velocity
        
        // Differentiate damping intensity based on flow regime
        let beta_local = if mach_local > 1.0 { beta_sup } else { beta_sub };
        
        // Scale damping by contravariant velocity magnitude
        let stabilizer = alpha * beta_local * u_contra.abs();

        // Strict Upwind Discretization
        if u_contra >= 0.0 {
            // Forward flow: apply backward difference (retarded)
            b[idx] += stabilizer; // Reinforce main diagonal
            a[idx] -= stabilizer; // Subtract from upstream neighbor
        } else {
            // Reverse flow: apply forward difference (retarded)
            b[idx] += stabilizer; // Reinforce main diagonal
            c[idx] -= stabilizer; // Subtract from downstream neighbor
        }
    }

    // 2. Sherman-Morrison Modification for Cyclic Tridiagonal System
    let gamma = -b[0]; 
    
    b[0] -= gamma;
    b[n - 1] -= c[n - 1] * a[0] / gamma;

    let mut d_z = vec![0.0; n];
    d_z[0] = gamma;
    d_z[n - 1] = c[n - 1];

    // 3. Solve auxiliary systems in-place (Zero heap allocation)
    solve_standard_tdma_inplace(a, b, c, d, y, c_prime, d_prime);
    solve_standard_tdma_inplace(a, b, c, &d_z, z, c_prime, d_prime);

    // 4. Recombine using Sherman-Morrison formula
    let fact = (y[0] + a[0] * y[n - 1] / gamma) / (1.0 + z[0] + a[0] * z[n - 1] / gamma);

    for idx in 0..n {
        let i = idx + 1;
        delta_phi[[i, j]] = y[idx] - fact * z[idx];
    }

    // 5. Enforce periodicity at the exact Wake Cut overlap
    delta_phi[[0, j]] = delta_phi[[grid.imax - 1, j]];
}


/// Helper: Highly optimized standard Thomas Algorithm.
/// Uses pre-allocated slices to avoid any dynamic memory allocation during sweeps.
fn solve_standard_tdma_inplace(
    a: &[f64], b: &[f64], c: &[f64], d: &[f64], 
    x: &mut [f64], c_prime: &mut [f64], d_prime: &mut [f64]
) {
    let n = b.len();

    // Forward Sweep
    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let denom = b[i] - a[i] * c_prime[i - 1];
        if denom.abs() < 1e-14 {
            panic!("Ill-conditioned matrix encountered at index {}. Check alpha sequence.", i);
        }
        c_prime[i] = c[i] / denom;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom;
    }

    // Backward Substitution
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
}