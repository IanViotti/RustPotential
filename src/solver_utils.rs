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
                phi: 1.0 * grid.points[[i, j]].x, 
                M: config.Mach_inf,
                rho: 1.0,
                u: 1.0,
                v: 0.0,
            };

            domain[[i, j]].flow_ip12j = HalfPointFlow {
                // Initialize velocity potential assuming uniform flow (0 deg angle of attack)
                U: 1.0,
                V: 0.0,
                rho: 1.0,
                rho_art: 1.0
            };

            domain[[i, j]].flow_ijp12 = HalfPointFlow {
                // Initialize velocity potential assuming uniform flow (0 deg angle of attack)
                U: 1.0,
                V: 0.0,
                rho: 1.0,
                rho_art: 1.0
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

    // ====================================================================
    // Tratamento de Fronteira: Métricas na Parede do Perfil (j = 0)
    // ====================================================================
    for i in 1..grid.imax - 1 {
        // 1. Métricas na face j+1/2 (ijp12) avaliadas em j=0
        // A derivada em eta é exata (compacta) e usa apenas j=1 e j=0.
        let x_eta_jp12 = grid.points[[i, 1]].x - grid.points[[i, 0]].x; 
        let y_eta_jp12 = grid.points[[i, 1]].y - grid.points[[i, 0]].y; 
        
        // A derivada em xi na face j+1/2 é a média das diferenças em j=1 e j=0
        let x_xi_jp12 = 0.5 * ((grid.points[[i+1, 1]].x - grid.points[[i-1, 1]].x) / 2.0 + 
                               (grid.points[[i+1, 0]].x - grid.points[[i-1, 0]].x) / 2.0); 
        let y_xi_jp12 = 0.5 * ((grid.points[[i+1, 1]].y - grid.points[[i-1, 1]].y) / 2.0 + 
                               (grid.points[[i+1, 0]].y - grid.points[[i-1, 0]].y) / 2.0); 

        // Jacobiano e métricas contravariantes para a face da parede
        let J_jp12 = 1.0 / (x_xi_jp12 * y_eta_jp12 - x_eta_jp12 * y_xi_jp12);

        let xi_x_jp12 = J_jp12 * y_eta_jp12;
        let xi_y_jp12 = -J_jp12 * x_eta_jp12;
        let eta_x_jp12 = -J_jp12 * y_xi_jp12;
        let eta_y_jp12 = J_jp12 * x_xi_jp12;

        domain[[i, 0]].metrics_ijp12 = Metrics {
            J: J_jp12,
            A1: xi_x_jp12.powi(2) + xi_y_jp12.powi(2),
            A2: xi_x_jp12 * eta_x_jp12 + xi_y_jp12 * eta_y_jp12,
            A3: eta_x_jp12.powi(2) + eta_y_jp12.powi(2),
        };
    }

    // ====================================================================
    // Tratamento de Fronteira: Periodicidade no Wake Cut (i = 0 e i = imax-1)
    // ====================================================================
    for j in 0..grid.jmax {
        // Copia as métricas da face logo acima do bordo de fuga para o nó fantasma 0
        domain[[0, j]].metrics_ip12j = domain[[grid.imax - 2, j]].metrics_ip12j.clone();
        domain[[0, j]].metrics_ijp12 = domain[[grid.imax - 2, j]].metrics_ijp12.clone();
        domain[[0, j]].metrics_ij    = domain[[grid.imax - 2, j]].metrics_ij.clone();

        // Copia as métricas da face logo abaixo do bordo de fuga para o nó fantasma imax-1
        domain[[grid.imax - 1, j]].metrics_ip12j = domain[[1, j]].metrics_ip12j.clone();
        domain[[grid.imax - 1, j]].metrics_ijp12 = domain[[1, j]].metrics_ijp12.clone();
        domain[[grid.imax - 1, j]].metrics_ij    = domain[[1, j]].metrics_ij.clone();
    }

}

/// Resolve o 1º Passo do esquema AF2 na direção Eta (Radial) utilizando o Algoritmo de Thomas.
/// Este sistema é resolvido para uma linha fixa `i`.
pub fn solve_tdma_eta(
    i: usize,
    alpha: f64,
    omega: f64,
    grid: &Grid,
    domain: &Array2<PointData>,
    residual: &Array2<f64>,
    f_inter: &mut Array2<f64>,
) {
    let j_start = 1;
    let j_end = grid.jmax - 1;
    let n = grid.jmax;

    // Vetores para os coeficientes da matriz tridiagonal: a (sub), b (diagonal principal), c (super) e d (RHS)
    // Nota: Em cenários de alta performance, estes vetores podem ser pré-alocados fora do loop para evitar alocações repetidas.
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];

    // =========================================================================
    // 1. Construção dos Coeficientes da Matriz
    // =========================================================================
    for j in j_start..j_end {
        // Coeficiente A_tilde em j+1/2 (Utilizando a métrica A3 referente à direção eta)
        let a_tilde_plus = domain[[i, j]].flow_ijp12.rho_art 
                         * domain[[i, j]].metrics_ijp12.A3 
                         / domain[[i, j]].metrics_ijp12.J;

        // Coeficiente A_tilde em j-1/2 
        // Equivale a pegar o valor j+1/2 do nó anterior (j-1)
        let a_tilde_minus = domain[[i, j - 1]].flow_ijp12.rho_art 
                          * domain[[i, j - 1]].metrics_ijp12.A3 
                          / domain[[i, j - 1]].metrics_ijp12.J;

        // Montagem da linha da matriz tridiagonal
        a[j] = -a_tilde_minus;                            // Sub-diagonal
        b[j] = alpha + a_tilde_plus + a_tilde_minus;      // Diagonal principal
        c[j] = -a_tilde_plus;                             // Super-diagonal
        
        // Termo independente (RHS)
        d[j] = alpha * omega * residual[[i, j]];
    }

    // =========================================================================
    // 2. Aplicação das Condições de Fronteira no Vetor Intermediário 'f'
    // =========================================================================
    // Na parede do perfil (j = 1), a condição de impermeabilidade (Neumann, dPhi/dEta = 0)
    // reflete-se frequentemente no vetor f como f_0 = f_1.
    // Isso significa que somamos o coeficiente 'a' ao coeficiente 'b' e anulamos 'a'.
    b[1] += a[1];
    a[1] = 0.0;

    // No Farfield (j = jmax - 2), normalmente a perturbação é nula (Dirichlet, f_jmax-1 = 0).
    // Anulamos o coeficiente que multiplicaria o ponto fora do domínio de cálculo.
    c[j_end - 1] = 0.0;

    // =========================================================================
    // 3. Algoritmo de Thomas (TDMA) - Fase Forward Sweep
    // =========================================================================
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    c_prime[j_start] = c[j_start] / b[j_start];
    d_prime[j_start] = d[j_start] / b[j_start];

    for j in (j_start + 1)..j_end {
        let denominator = b[j] - a[j] * c_prime[j - 1];
        
        // Proteção contra divisão por zero em malhas mal condicionadas
        if denominator.abs() < 1e-14 {
            panic!("TDMA falhou: zero na diagonal no índice i={}, j={}", i, j);
        }

        c_prime[j] = c[j] / denominator;
        d_prime[j] = (d[j] - a[j] * d_prime[j - 1]) / denominator;
    }

    // =========================================================================
    // 4. Algoritmo de Thomas (TDMA) - Fase Backward Substitution
    // =========================================================================
    // O último ponto do domínio calcula-se diretamente
    f_inter[[i, j_end - 1]] = d_prime[j_end - 1];

    // Os restantes calculam-se de trás para a frente
    for j in (j_start..(j_end - 1)).rev() {
        f_inter[[i, j]] = d_prime[j] - c_prime[j] * f_inter[[i, j + 1]];
    }
}


/// Resolve o 2º Passo do esquema AF2 na direção Xi (Azimutal) utilizando 
/// o Algoritmo Tridiagonal Cíclico (baseado em Sherman-Morrison).
/// Este sistema é resolvido para um anel fixo `j`.
pub fn solve_periodic_tdma_xi(
    j: usize,
    alpha: f64,
    grid: &Grid,
    domain: &Array2<PointData>,
    f_inter: &Array2<f64>,
    delta_phi: &mut Array2<f64>,
) {
    // Numa malha tipo O, os nós 1 e imax coincidem (wake cut).
    // O número de pontos únicos a resolver é N = imax - 1.
    let n = grid.imax - 1; 

    // Alocação dos vetores do sistema linear (tamanho N, com índice de 0 a N-1)
    // Em produção, pré-aloque estes vetores numa struct de SolverState para máxima performance.
    let mut a = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];

    // =========================================================================
    // 1. Construção dos Coeficientes do Sistema Cíclico
    // =========================================================================
    // Assumimos que A1 é a métrica correspondente à direção xi (nabla xi * nabla xi)
    // e beta é o coeficiente escalar do operador em xi.
    let beta = 1.0; 

    for idx in 0..n {
        let i = idx + 1; // Índice real na grelha do domínio (1 a imax-1)
        
        // Determinar o vizinho "anterior" respeitando a periodicidade
        let i_minus = if i == 1 { n } else { i - 1 };

        // Coeficiente B_tilde em i+1/2 
        let b_tilde_plus = domain[[i, j]].flow_ip12j.rho_art 
                         * domain[[i, j]].metrics_ip12j.A1 
                         / domain[[i, j]].metrics_ip12j.J;

        // Coeficiente B_tilde em i-1/2 (igual ao B_tilde_plus do nó anterior)
        let b_tilde_minus = domain[[i_minus, j]].flow_ip12j.rho_art 
                          * domain[[i_minus, j]].metrics_ip12j.A1 
                          / domain[[i_minus, j]].metrics_ip12j.J;

        // Preenchimento dos vetores da matriz
        a[idx] = -beta * b_tilde_minus;                      // Sub-diagonal
        b[idx] = alpha + beta * (b_tilde_plus + b_tilde_minus); // Diagonal principal
        c[idx] = -beta * b_tilde_plus;                       // Super-diagonal
        
        // O RHS é o vetor f_inter calculado no Passo 1 (direção eta)
        d[idx] = f_inter[[i, j]];
    }

    // =========================================================================
    // 2. Resolução do Sistema Tridiagonal Cíclico (Sherman-Morrison)
    // =========================================================================
    // Modificamos a matriz para extrair os cantos (que quebram o formato tridiagonal)
    let gamma = -b[0]; 
    
    // Guardamos as diagonais originais que serão alteradas
    let mut b_prime = b.clone();
    b_prime[0] = b[0] - gamma;
    b_prime[n - 1] = b[n - 1] - c[n - 1] * a[0] / gamma;

    // Vetores independentes para os dois sistemas TDMA normais
    let mut d_y = d.clone();
    let mut d_z = vec![0.0; n];
    d_z[0] = gamma;
    d_z[n - 1] = c[n - 1];

    // Resolvemos os dois sistemas (A' * y = d) e (A' * z = u)
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    
    // Chamadas a um solver TDMA padrão para sistemas não-periódicos
    solve_standard_tdma(&a, &b_prime, &c, &d_y, &mut y);
    // Para resolver o vetor z, temos de garantir que o canto inferior esquerdo
    // a[0] e o canto superior direito c[n-1] do nosso "A'" virtual sejam zero.
    // O helper solve_standard_tdma assume sempre a=0 no primeiro elemento e c=0 no último.
    solve_standard_tdma(&a, &b_prime, &c, &d_z, &mut z);

    // =========================================================================
    // 3. Recombinação de Sherman-Morrison para obter a solução cíclica real
    // =========================================================================
    let fact = (y[0] + a[0] * y[n - 1] / gamma) / (1.0 + z[0] + a[0] * z[n - 1] / gamma);

    for idx in 0..n {
        let i = idx + 1;
        // Solução final do incremento de potencial para este ponto
        let delta = y[idx] - fact * z[idx];
        delta_phi[[i, j]] = delta;
    }

    // =========================================================================
    // 4. Fecho da Malha (Wake Cut)
    // =========================================================================
    // O último nó (imax) recebe exatamente o mesmo incremento que o nó 1,
    // garantindo a periodicidade física e numérica no corte da malha.
    delta_phi[[0, j]] = delta_phi[[grid.imax - 1, j]];
}

/// Helper: Algoritmo de Thomas Clássico para os passos intermediários de Sherman-Morrison.
/// Recebe arrays de tamanho N e devolve o resultado em `x`.
fn solve_standard_tdma(a: &[f64], b: &[f64], c: &[f64], d: &[f64], x: &mut [f64]) {
    let n = b.len();
    let mut c_prime = vec![0.0; n];
    let mut d_prime = vec![0.0; n];

    c_prime[0] = c[0] / b[0];
    d_prime[0] = d[0] / b[0];

    for i in 1..n {
        let denom = b[i] - a[i] * c_prime[i - 1];
        if denom.abs() < 1e-14 {
            // Se falhar aqui, o pseudo-tempo alpha pode estar mal parametrizado
            panic!("TDMA Padrão falhou: zero na diagonal no índice {}.", i);
        }
        c_prime[i] = c[i] / denom;
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom;
    }

    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }
}