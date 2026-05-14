#![allow(non_snake_case)]

use std::time::Instant;
use toml;

// Import internal modules
mod config; 
mod job_utils;
mod solver_core;
mod solver_utils;

fn main() {

    // Start a timer to track the total execution time of the mesh generation
    let start = Instant::now();

    let config = job_utils::read_config("job_config.toml");

    // Initialize the job directory structure based on the job name specified in the configuration.
    job_utils::init_job_directory(&config);

    // Read mesh
    let grid = job_utils::read_vtk_structured_grid(&format!("{}", config.meshname));

    // Call main function to solve and get the field solution
    let field = solver_core::solve(&grid, &config);

    // Export results to VTK file
    job_utils::export_vtk_fields(&format!("job_files/{}/solution.vtk", config.jobname), &grid, &field);

    // Print final completion message and elapsed execution time.
    println!("Job completed!");
    
    let duration = start.elapsed();
    println!("Total running time: {:?} \n", duration);
}
