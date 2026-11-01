#![allow(non_snake_case)]

// External modules
use std::time::Instant;

// Import internal modules
mod config; 
mod job_utils;
mod solver_core;
mod solver_utils;
mod af2;
mod ui;

fn main() {
    // 1. Visual project headline banner
    ui::print_banner();

    // Start timer for overall execution time
    let start = Instant::now();

    // 2. Read configuration and initialize directories
    let config = job_utils::read_config("job_config.toml");
    job_utils::init_job_directory(&config);

    // 3. Load mesh from VTK
    let grid = job_utils::read_vtk_structured_grid(&config.meshname);

    // 4. Formatted configuration and flight parameters dashboard
    ui::print_config_dashboard(&config, &grid);

    // 5. Run the AF2 Full Potential solver
    let (field, stats) = solver_core::solve(&grid, &config);

    // 6. Export solution fields
    job_utils::export_vtk_fields(&config, &grid, &field);
    job_utils::export_cp_csv(&config, &grid, &field);

    // 7. Comprehensive summary with aerodynamic metrics
    let duration = start.elapsed();
    ui::print_summary(&config, &grid, &field, &stats, duration);

    // 8. Prevent terminal from auto-closing; wait for user input
    ui::wait_for_user_exit();
}
