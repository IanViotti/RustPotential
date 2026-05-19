use crate::config;
use crate::job_utils::Grid;
use ndarray::Array2;

// ==========================================
// 1. STRUCTURE DEFINITIONS
// ==========================================

/// Structure for Geometric Metrics
/// Holds the Jacobian and the transformation coefficients (A1, A2, A3)
#[derive(Clone, Copy, Debug, Default)]
pub struct Metrics {
    pub J: f64,
    pub A1: f64,
    pub A2: f64,
    pub A3: f64,
}

/// Physical variables evaluated exactly at the grid node (i, j)
#[derive(Clone, Copy, Debug, Default)]
pub struct NodalFlow {
    pub phi: f64, // Velocity potential (the primary unknown solved by AF2)
    pub M: f64,   // Local Mach number (computed at the end for VTK export)
    pub rho: f64, // Local density (computed at the end for VTK export)
    pub u: f64,   // Cartesian velocity u (computed at the end for VTK export)
    pub v: f64,   // Cartesian velocity v (computed at the end for VTK export)
}

/// Physical variables evaluated at the intermediate half-points (i+1/2, j) or (i, j+1/2)
/// This is essential to form the compact finite difference stencils required by the Holst scheme.
#[derive(Clone, Copy, Debug, Default)]
pub struct HalfPointFlow {
    pub U: f64, // Represents 'U' at i+1/2, 
    pub V: f64, // Represents 'V' at i+1/2
    pub rho: f64,               // Isentropic density at the half-point
    pub rho_art: f64,           // Artificial density at the half-point (Holst's Upwinding)
}

/// Master structure grouping data for the node (i,j) and its adjacent forward half-points
#[derive(Clone, Copy, Debug, Default)]
pub struct PointData {
    // ----------------------------------------------------
    // Integer Point Data: (i, j)
    // ----------------------------------------------------
    pub flow_ij: NodalFlow,
    pub metrics_ij: Metrics,

    // ----------------------------------------------------
    // Half-Point Data in Xi direction: (i+1/2, j)
    // ----------------------------------------------------
    pub flow_ip12j: HalfPointFlow,
    pub metrics_ip12j: Metrics,

    // ----------------------------------------------------
    // Half-Point Data in Eta direction: (i, j+1/2)
    // ----------------------------------------------------
    pub flow_ijp12: HalfPointFlow,
    pub metrics_ijp12: Metrics,
}


// ==========================================
// 2. INITIALIZATION AND CALCULATION FUNCTIONS
// ==========================================

/// Creates the data grid and initializes the nodal fields (e.g., freestream potential)
pub fn initialize_domain(config: &config::Config, grid: &Grid) -> Array2<PointData> {
    println!("Initializing computational domain...");
    
    // Using the Default trait to allocate the array with zeroed/empty structures
    let mut domain: ndarray::prelude::ArrayBase<ndarray::OwnedRepr<PointData>, ndarray::prelude::Dim<[usize; 2]>, PointData> = Array2::default((grid.imax, grid.jmax));

    for i in 0..grid.imax {
        for j in 0..grid.jmax {
            domain[[i, j]].flow_ij = NodalFlow {
                // Initialize velocity potential assuming uniform flow (0 deg angle of attack)
                phi: config.u_inf * grid.points[[i, j]].x, 
                M: 0.0,
                rho: 1.0,
                u: config.u_inf,
                v: 0.0,
            };
        }
    }
        
    domain
}

/// Computes all geometric metrics using finite differences and stores them "in-place"
pub fn calculate_metrics(grid: &Grid, domain: &mut Array2<PointData>) {
    println!("Calculating mesh metrics...");

    // Loop over the internal points of the structured grid
    for i in 1..grid.imax - 1 {
        for j in 1..grid.jmax - 1 {

            // ----------------------------------------------------
            // 1. Metrics at the Integer Point: (i, j)
            // Evaluated using 2nd order central differences
            // ----------------------------------------------------
            let x_xi = (grid.points[[i+1, j]].x - grid.points[[i-1, j]].x) / 2.0; 
            let y_xi = (grid.points[[i+1, j]].y - grid.points[[i-1, j]].y) / 2.0; 
            let x_eta = (grid.points[[i, j+1]].x - grid.points[[i, j-1]].x) / 2.0; 
            let y_eta = (grid.points[[i, j+1]].y - grid.points[[i, j-1]].y) / 2.0; 

            // Inverse transformation Jacobian at (i, j)
            let J_ij = 1.0 / (x_xi * y_eta - x_eta * y_xi);

            let xi_x = J_ij * y_eta;
            let xi_y = -J_ij * x_eta;
            let eta_x = -J_ij * y_xi;
            let eta_y = J_ij * x_xi;

            domain[[i, j]].metrics_ij = Metrics {
                J: J_ij,
                A1: xi_x.powi(2) + xi_y.powi(2),
                A2: xi_x * eta_x + xi_y * eta_y,
                A3: eta_x.powi(2) + eta_y.powi(2),
            };

            // ----------------------------------------------------
            // 2. Metrics at the Xi Half-Point: (i + 1/2, j)
            // Xi derivatives are exact (compact); Eta derivatives are averaged central
            // ----------------------------------------------------
            let x_xi_ip12 = grid.points[[i+1, j]].x - grid.points[[i, j]].x; 
            let y_xi_ip12 = grid.points[[i+1, j]].y - grid.points[[i, j]].y; 
            
            let x_eta_ip12 = 0.5 * ((grid.points[[i+1, j+1]].x - grid.points[[i+1, j-1]].x) / 2.0 + 
                                    (grid.points[[i, j+1]].x - grid.points[[i, j-1]].x) / 2.0); 
            let y_eta_ip12 = 0.5 * ((grid.points[[i+1, j+1]].y - grid.points[[i+1, j-1]].y) / 2.0 + 
                                    (grid.points[[i, j+1]].y - grid.points[[i, j-1]].y) / 2.0); 

            let J_ip12 = 1.0 / (x_xi_ip12 * y_eta_ip12 - x_eta_ip12 * y_xi_ip12);

            let xi_x_ip12 = J_ip12 * y_eta_ip12;
            let xi_y_ip12 = -J_ip12 * x_eta_ip12;
            let eta_x_ip12 = -J_ip12 * y_xi_ip12;
            let eta_y_ip12 = J_ip12 * x_xi_ip12;

            domain[[i, j]].metrics_ip12j = Metrics {
                J: J_ip12,
                A1: xi_x_ip12.powi(2) + xi_y_ip12.powi(2),
                A2: xi_x_ip12 * eta_x_ip12 + xi_y_ip12 * eta_y_ip12,
                A3: eta_x_ip12.powi(2) + eta_y_ip12.powi(2),
            };

            // ----------------------------------------------------
            // 3. Metrics at the Eta Half-Point: (i, j + 1/2)
            // Eta derivatives are exact (compact); Xi derivatives are averaged central
            // ----------------------------------------------------
            let x_xi_jp12 = 0.5 * ((grid.points[[i+1, j+1]].x - grid.points[[i-1, j+1]].x) / 2.0 + 
                                   (grid.points[[i+1, j]].x - grid.points[[i-1, j]].x) / 2.0); 
            let y_xi_jp12 = 0.5 * ((grid.points[[i+1, j+1]].y - grid.points[[i-1, j+1]].y) / 2.0 + 
                                   (grid.points[[i+1, j]].y - grid.points[[i-1, j]].y) / 2.0); 
            
            let x_eta_jp12 = grid.points[[i, j+1]].x - grid.points[[i, j]].x; 
            let y_eta_jp12 = grid.points[[i, j+1]].y - grid.points[[i, j]].y; 

            let J_jp12 = 1.0 / (x_xi_jp12 * y_eta_jp12 - x_eta_jp12 * y_xi_jp12);

            let xi_x_jp12 = J_jp12 * y_eta_jp12;
            let xi_y_jp12 = -J_jp12 * x_eta_jp12;
            let eta_x_jp12 = -J_jp12 * y_xi_jp12;
            let eta_y_jp12 = J_jp12 * x_xi_jp12;

            domain[[i, j]].metrics_ijp12 = Metrics {
                J: J_jp12,
                A1: xi_x_jp12.powi(2) + xi_y_jp12.powi(2),
                A2: xi_x_jp12 * eta_x_jp12 + xi_y_jp12 * eta_y_jp12,
                A3: eta_x_jp12.powi(2) + eta_y_jp12.powi(2),
            };
        }
    }
}