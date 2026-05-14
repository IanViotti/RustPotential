use crate::config;
use crate::job_utils::Grid;
use ndarray::Array2;


#[derive(Clone, Copy, Debug)]
pub struct Field {
    pub phi: f64,
    pub M: f64,
    pub rho: f64,
    pub u: f64,
    pub v: f64,
}

pub fn initialize_field(config: &config::Config, grid: &Grid) -> Array2<Field> {
    // Placeholder for field initialization logic.
    // This function should create and return a 2D vector representing the initial field values based on the grid dimensions.
    
    println!("Initializing field");

    let initial_field_value = Field {
        phi: 0.0,
        M: 0.0,
        rho: 1.0,
        u: 1.0,
        v: 0.0,
     };
    
    // For demonstration, we will just create a dummy field matrix filled with zeros.
    let mut field = Array2::from_elem((grid.imax, grid.jmax), initial_field_value);

    for i in 0..grid.imax {
        for j in 0..grid.jmax {
            field[[i, j]] = Field {
                phi: config.u_inf * grid.points[[i, j]].x, // Exemplo de potencial linear
                M: 0.0,
                rho: 1.0,
                u: 1.0,
                v: 0.0,
            };
        }
    }
        
    field
}