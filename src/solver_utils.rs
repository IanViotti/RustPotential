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
                M: config.Mach_inf,
                rho: config.rho_inf,
                u: config.u_inf,
                v: 0.0,
            };

            domain[[i, j]].flow_ip12j = HalfPointFlow {
                // Initialize velocity potential assuming uniform flow (0 deg angle of attack)
                U: 1.0,
                V: 0.0,
                rho: config.rho_inf,
                rho_art: config.rho_inf
            };

            domain[[i, j]].flow_ijp12 = HalfPointFlow {
                // Initialize velocity potential assuming uniform flow (0 deg angle of attack)
                U: 1.0,
                V: 0.0,
                rho: config.rho_inf,
                rho_art: config.rho_inf
            };

        }
    }
        
    domain
}
pub fn calculate_metrics(grid: &Grid, domain: &mut Array2<PointData>) {
    println!("Calculating mesh metrics...");

    // Usamos a periodicidade total na direção Xi (i)
    for j in 0..grid.jmax {
        for i in 0..grid.imax {
            
            // Topologia O-grid: O salto periódico resolve a singularidade do bordo afiado
            let i_plus = if i == grid.imax - 1 { 1 } else { i + 1 };
            let i_minus = if i == 0 { grid.imax - 2 } else { i - 1 };

            // 1. Direção Xi (Sempre central para cruzar a esteira ortogonalmente)
            let x_xi = (grid.points[[i_plus, j]].x - grid.points[[i_minus, j]].x) / 2.0;
            let y_xi = (grid.points[[i_plus, j]].y - grid.points[[i_minus, j]].y) / 2.0;

            // 2. Direção Eta (Diferença lateral exigida apenas na parede e no farfield)
            let (x_eta, y_eta) = if j == 0 {
                (grid.points[[i, 1]].x - grid.points[[i, 0]].x, 
                 grid.points[[i, 1]].y - grid.points[[i, 0]].y)
            } else if j == grid.jmax - 1 {
                (grid.points[[i, j]].x - grid.points[[i, j - 1]].x, 
                 grid.points[[i, j]].y - grid.points[[i, j - 1]].y)
            } else {
                ((grid.points[[i, j + 1]].x - grid.points[[i, j - 1]].x) / 2.0, 
                 (grid.points[[i, j + 1]].y - grid.points[[i, j - 1]].y) / 2.0)
            };

            let J_ij = 1.0 / (x_xi * y_eta - x_eta * y_xi);
            domain[[i, j]].metrics_ij = Metrics {
                J: J_ij,
                A1: (J_ij * y_eta).powi(2) + (-J_ij * x_eta).powi(2),
                A2: (J_ij * y_eta) * (-J_ij * y_xi) + (-J_ij * x_eta) * (J_ij * x_xi),
                A3: (-J_ij * y_xi).powi(2) + (J_ij * x_xi).powi(2),
            };


            // ----------------------------------------------------
            // Faces (Meio-Pontos)
            // ----------------------------------------------------
            let x_xi_ip12 = grid.points[[i_plus, j]].x - grid.points[[i, j]].x;
            let y_xi_ip12 = grid.points[[i_plus, j]].y - grid.points[[i, j]].y;

            let (x_eta_ip12, y_eta_ip12) = if j == 0 {
                (0.5 * ((grid.points[[i_plus, 1]].x - grid.points[[i_plus, 0]].x) + (grid.points[[i, 1]].x - grid.points[[i, 0]].x)),
                 0.5 * ((grid.points[[i_plus, 1]].y - grid.points[[i_plus, 0]].y) + (grid.points[[i, 1]].y - grid.points[[i, 0]].y)))
            } else if j == grid.jmax - 1 {
                (0.5 * ((grid.points[[i_plus, j]].x - grid.points[[i_plus, j-1]].x) + (grid.points[[i, j]].x - grid.points[[i, j-1]].x)),
                 0.5 * ((grid.points[[i_plus, j]].y - grid.points[[i_plus, j-1]].y) + (grid.points[[i, j]].y - grid.points[[i, j-1]].y)))
            } else {
                (0.5 * ((grid.points[[i_plus, j+1]].x - grid.points[[i_plus, j-1]].x)/2.0 + (grid.points[[i, j+1]].x - grid.points[[i, j-1]].x)/2.0),
                 0.5 * ((grid.points[[i_plus, j+1]].y - grid.points[[i_plus, j-1]].y)/2.0 + (grid.points[[i, j+1]].y - grid.points[[i, j-1]].y)/2.0))
            };

            let J_ip12 = 1.0 / (x_xi_ip12 * y_eta_ip12 - x_eta_ip12 * y_xi_ip12);
            domain[[i, j]].metrics_ip12j = Metrics {
                J: J_ip12,
                A1: (J_ip12 * y_eta_ip12).powi(2) + (-J_ip12 * x_eta_ip12).powi(2),
                A2: (J_ip12 * y_eta_ip12) * (-J_ip12 * y_xi_ip12) + (-J_ip12 * x_eta_ip12) * (J_ip12 * x_xi_ip12),
                A3: (-J_ip12 * y_xi_ip12).powi(2) + (J_ip12 * x_xi_ip12).powi(2),
            };

            if j < grid.jmax - 1 {
                let x_eta_jp12 = grid.points[[i, j+1]].x - grid.points[[i, j]].x;
                let y_eta_jp12 = grid.points[[i, j+1]].y - grid.points[[i, j]].y;

                let x_xi_jp12 = 0.5 * ((grid.points[[i_plus, j+1]].x - grid.points[[i_minus, j+1]].x)/2.0 + (grid.points[[i_plus, j]].x - grid.points[[i_minus, j]].x)/2.0);
                let y_xi_jp12 = 0.5 * ((grid.points[[i_plus, j+1]].y - grid.points[[i_minus, j+1]].y)/2.0 + (grid.points[[i_plus, j]].y - grid.points[[i_minus, j]].y)/2.0);

                let J_jp12 = 1.0 / (x_xi_jp12 * y_eta_jp12 - x_eta_jp12 * y_xi_jp12);
                domain[[i, j]].metrics_ijp12 = Metrics {
                    J: J_jp12,
                    A1: (J_jp12 * y_eta_jp12).powi(2) + (-J_jp12 * x_eta_jp12).powi(2),
                    A2: (J_jp12 * y_eta_jp12) * (-J_jp12 * y_xi_jp12) + (-J_jp12 * x_eta_jp12) * (J_jp12 * x_xi_jp12),
                    A3: (-J_jp12 * y_xi_jp12).powi(2) + (J_jp12 * x_xi_jp12).powi(2),
                };
            }
        }
    }

    // Fechamento exato para o plot no VTK (O nó 0 dita o estado físico do imax-1)
    for j in 0..grid.jmax {
        domain[[grid.imax - 1, j]].metrics_ij = domain[[0, j]].metrics_ij.clone();
        domain[[grid.imax - 1, j]].metrics_ip12j = domain[[0, j]].metrics_ip12j.clone();
        if j < grid.jmax - 1 {
            domain[[grid.imax - 1, j]].metrics_ijp12 = domain[[0, j]].metrics_ijp12.clone();
        }
    }
}

pub fn calculate_field(config: &config::Config, grid: &Grid, domain: &mut Array2<PointData>) {
    println!("Calculating final flow field variables...");
    
    let gamma = config.gamma;
    let m_inf = config.Mach_inf;
    let c_rho = (gamma - 1.0) / (gamma + 1.0); // Fator de normalização de Holst

    // Calculamos apenas para o interior e a parede (i de 1 até imax-1, j de 0 até jmax-1)
    for j in 0..grid.jmax - 1 {
        for i in 1..grid.imax {
            
            // =========================================================
            // 1. Mapeamento Topológico (Malha O)
            // =========================================================
            let i_plus = if i == grid.imax - 1 { 1 } else { i + 1 };
            let i_minus = if i == 1 { grid.imax - 1 } else { i - 1 };

            // =========================================================
            // 2. Derivadas da Malha nos Nós Inteiros (i, j)
            // =========================================================
            let x_xi = (grid.points[[i_plus, j]].x - grid.points[[i_minus, j]].x) / 2.0;
            let y_xi = (grid.points[[i_plus, j]].y - grid.points[[i_minus, j]].y) / 2.0;

            let (x_eta, y_eta) = if j == 0 {
                // Parede (j=0): Diferença avançada para não sair do domínio
                (
                    grid.points[[i, 1]].x - grid.points[[i, 0]].x,
                    grid.points[[i, 1]].y - grid.points[[i, 0]].y
                )
            } else {
                // Interior: Diferença central
                (
                    (grid.points[[i, j+1]].x - grid.points[[i, j-1]].x) / 2.0,
                    (grid.points[[i, j+1]].y - grid.points[[i, j-1]].y) / 2.0
                )
            };

            let j_inv = x_xi * y_eta - x_eta * y_xi;
            let jacobian = 1.0 / j_inv;

            let xi_x = jacobian * y_eta;
            let xi_y = -jacobian * x_eta;
            let eta_x = -jacobian * y_xi;
            let eta_y = jacobian * x_xi;

            // =========================================================
            // 3. Derivadas do Potencial Sincronizadas
            // =========================================================
            let i_plus = if i == grid.imax - 1 { 1 } else { i + 1 };
            let i_minus = if i == 0 { grid.imax - 2 } else { i - 1 };

            // Derivada EXATAMENTE equivalente ao cálculo do x_xi nas métricas
            let phi_xi = (domain[[i_plus, j]].flow_ij.phi - domain[[i_minus, j]].flow_ij.phi) / 2.0;

            let phi_eta = if j == 0 {
                // Tangência na parede. Adicionamos uma proteção para divisões delicadas
                let a2_local = domain[[i, j]].metrics_ij.A2;
                let a3_local = domain[[i, j]].metrics_ij.A3;
                
                if a3_local.abs() > 1e-12 {
                    -(a2_local / a3_local) * phi_xi
                } else {
                    domain[[i, 1]].flow_ij.phi - domain[[i, 0]].flow_ij.phi
                }
            } else if j == grid.jmax - 1 {
                domain[[i, j]].flow_ij.phi - domain[[i, j - 1]].flow_ij.phi
            } else {
                (domain[[i, j + 1]].flow_ij.phi - domain[[i, j - 1]].flow_ij.phi) / 2.0
            };

            // =========================================================
            // 4. Velocidades Cartesianas
            // =========================================================
            let u = xi_x * phi_xi + eta_x * phi_eta;
            let v = xi_y * phi_xi + eta_y * phi_eta;
            let q2 = u * u + v * v;

            // =========================================================
            // 5. Termodinâmica (Sincronizada com o calc_L_phi_ij)
            // =========================================================
            // Na normalização de Holst baseada na velocidade crítica a*:
            // a^2 = 1.0 - [(gamma-1)/(gamma+1)] * q^2
            let a_squared = 1.0 - c_rho * q2;
            
            let rho = a_squared.powf(1.0 / (gamma - 1.0));
            
            // Mach number: M^2 = q^2 / a^2
            let mach_factor = 2.0 / (gamma + 1.0);
            let mach = (mach_factor * (q2 / a_squared)).sqrt();

            domain[[i, j]].flow_ij.u = u;
            domain[[i, j]].flow_ij.v = v;
            domain[[i, j]].flow_ij.rho = rho;
            domain[[i, j]].flow_ij.M = mach;
        }
    }

    // =========================================================
    // 6. Fechamento das Fronteiras Restantes para o Plot VTK
    // =========================================================
    for j in 0..grid.jmax {
        // A. Espelhamento do Wake Cut (Nó fantasma i=0 recebe cópia de i=imax-1)
        domain[[0, j]].flow_ij.u = domain[[grid.imax - 1, j]].flow_ij.u;
        domain[[0, j]].flow_ij.v = domain[[grid.imax - 1, j]].flow_ij.v;
        domain[[0, j]].flow_ij.rho = domain[[grid.imax - 1, j]].flow_ij.rho;
        domain[[0, j]].flow_ij.M = domain[[grid.imax - 1, j]].flow_ij.M;

        // B. Condição de Farfield (j = jmax - 1)
        if j == grid.jmax - 1 {
            for i in 0..grid.imax {
                domain[[i, j]].flow_ij.u = config.u_inf;
                domain[[i, j]].flow_ij.v = 0.0;
                domain[[i, j]].flow_ij.rho = config.rho_inf;
                domain[[i, j]].flow_ij.M = m_inf;
            }
        }
    }
}

pub fn alpha_seq(config: &config::Config, grid: &Grid) -> Vec<f64> {

    let min_spacing = compute_min_spacing(grid);

    // Valores extremos de Alpha baseados na recomendação de Holst
    let alpha_h = 1.0 / min_spacing;
    let alpha_l = 1.0; // Pode ser reduzido para 0.1 dependendo da estabilidade
    
    // O tamanho da sequência M (Holst costumava usar entre 4 e 8).
    // Se você não tiver isso no seu arquivo toml, 8 é um excelente padrão.
    let m = config.alpha_sequence_length; // Idealmente: config.alpha_sequence_length
    
    let mut alpha_k = Vec::with_capacity(m);
    
    if m <= 1 {
        alpha_k.push(alpha_h);
        return alpha_k;
    }

    // Fator da progressão geométrica: ratio = (alpha_L / alpha_H)^(1 / (M - 1))
    let ratio = (alpha_l / alpha_h).powf(1.0 / (m as f64 - 1.0));
    
    // Constrói a sequência de alfa_H (alto) até alfa_L (baixo)
    for k in 0..m {
        let alpha = alpha_h * ratio.powi(k as i32);
        alpha_k.push(alpha);
    }
    
    alpha_k
}


pub fn compute_min_spacing(grid: &Grid) -> f64 {
    let mut min_spacing = f64::MAX;

    // A parede do perfil está em j = 0
    let j = 0;
    
    for i in 0..grid.imax {
        // 1. Espaçamento Azimutal (Ao longo do perfil - Direção Xi)
        // Usa a periodicidade para fechar o bordo de fuga (i = imax - 1 conecta com i = 1)
        let i_plus = if i == grid.imax - 1 { 1 } else { i + 1 };
        
        let dx_xi = grid.points[[i_plus, j]].x - grid.points[[i, j]].x;
        let dy_xi = grid.points[[i_plus, j]].y - grid.points[[i, j]].y;
        let ds_xi = (dx_xi.powi(2) + dy_xi.powi(2)).sqrt();
        
        if ds_xi < min_spacing {
            min_spacing = ds_xi;
        }

        // 2. Espaçamento Normal (Afastando da parede - Direção Eta)
        // Checa a distância do nó na parede (j=0) para o seu vizinho imediato (j=1)
        let dx_eta = grid.points[[i, 1]].x - grid.points[[i, 0]].x;
        let dy_eta = grid.points[[i, 1]].y - grid.points[[i, 0]].y;
        let ds_eta = (dx_eta.powi(2) + dy_eta.powi(2)).sqrt();
        
        if ds_eta < min_spacing {
            min_spacing = ds_eta;
        }
    }
    
    // Prevenção de segurança para malhas degeneradas
    if min_spacing < 1e-16 {
        panic!("O espaçamento da malha é muito próximo de zero. Verifique o gerador de malha.");
    }

    min_spacing
}