use crate::config;
use crate::solver_utils;

use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::io::{Write, BufWriter};
use ndarray::Array2; 

// ==========================================
// 1. MESH DATA STRUCTURES
// ==========================================

/// Struct representing a 2D physical coordinate in the mesh
#[derive(Clone, Copy, Debug, Default)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

/// Grid struct holding grid dimensions and the 2D array of physical points
pub struct Grid {
    pub imax: usize,
    pub jmax: usize,
    pub points: Array2<Point>,
}

// ==========================================
// 2. CONFIGURATION AND INITIALIZATION
// ==========================================

pub fn read_config(filename: &str) -> config::Config {
    // 1. Read the contents of the TOML file into a string.
    let config_content = fs::read_to_string(filename)
        .expect(&format!("Could not read the {} file. Please make sure it exists and is in the correct path.\n", filename));

    // 2. Deserialize using the 'toml' crate
    let mut config: config::Config = toml::from_str(&config_content)
        .expect("Error parsing the configuration file. Please check the format and values.\n");

    config.compute_derived_variables();

    // Display job startup information
    println!("\n
            +-----------------------------+\n
            Starting job: {}\n
            +-----------------------------+\n", config.jobname);

    println!("Configuration loaded: \n{:#?}\n", config);


    config
}

pub fn init_job_directory(config: &config::Config) {
    let jobname = &config.jobname;
    std::fs::create_dir_all(&format!("job_files/{}", jobname)).unwrap();
}

// ==========================================
// 3. VTK I/O OPERATIONS
// ==========================================

/// Reads a structured grid .vtk file and reconstructs the coordinate matrices.
/// Triggers a panic if the file cannot be read or parsed properly.
pub fn read_vtk_structured_grid(filepath: &str) -> Grid {
    let file = File::open(filepath).expect(&format!("Failed to open VTK file: {}", filepath));
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut imax = 0;
    let mut jmax = 0;
    
    let mut points_array = Array2::<Point>::default((0, 0));

    while let Some(Ok(line)) = lines.next() {
        let line = line.trim();

        if line.starts_with("DIMENSIONS") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            imax = parts[1].parse::<usize>().expect("Failed to read imax dimension from VTK file");
            jmax = parts[2].parse::<usize>().expect("Failed to read jmax dimension from VTK file");
        }

        if line.starts_with("POINTS") {
            points_array = Array2::from_elem((imax, jmax), Point { x: 0.0, y: 0.0 });

            for j in 0..jmax {
                for i in 0..imax {
                    if let Some(Ok(point_line)) = lines.next() {
                        let coords: Vec<f64> = point_line
                            .split_whitespace()
                            .map(|s| s.parse().unwrap_or(0.0))
                            .collect();
                        
                        points_array[[i, j]].x = coords[0];
                        points_array[[i, j]].y = coords[1];
                    }
                }
            }
            break; 
        }
    }

    println!("Mesh {} loaded successfully: imax = {}, jmax = {}", filepath, imax, jmax);

    Grid {
        imax,
        jmax,
        points: points_array,
    }
}


pub fn export_vtk_fields(config: &config::Config, grid: &Grid, domain: &Array2<solver_utils::PointData>) {
    let file_path = &format!("job_files/{}/solution.vtk", config.jobname);
    let file = File::create(file_path).expect(&format!("Failed to create VTK output file: {}", config.jobname));
    let mut writer = BufWriter::new(file);

    // Internal closure to use the '?' operator without dirtying the main function signature
    let mut write_data = || -> std::io::Result<()> {
        let imax = grid.imax;
        let jmax = grid.jmax;
        let total_points = imax * jmax;

        // 1. VTK Header
        writeln!(writer, "# vtk DataFile Version 3.0")?;
        writeln!(writer, "Full Potential CFD Results")?;
        writeln!(writer, "ASCII")?;
        writeln!(writer, "DATASET STRUCTURED_GRID")?;
        writeln!(writer, "DIMENSIONS {} {} 1", imax, jmax)?;

        // 2. Points (Geometry extracted from Grid)
        writeln!(writer, "POINTS {} float", total_points)?;
        for j in 0..jmax {
            for i in 0..imax {
                let pt = &grid.points[[i, j]];
                writeln!(writer, "{:.6} {:.6} 0.000000", pt.x, pt.y)?;
            }
        }

        // 3. Point Data (Results mapped directly from NodalFlow)
        writeln!(writer, "POINT_DATA {}", total_points)?;

        // --- Scalar Field: Potential (phi) ---
        writeln!(writer, "SCALARS Potential float 1")?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for j in 0..jmax {
            for i in 0..imax {
                writeln!(writer, "{:.6}", domain[[i, j]].flow_ij.phi)?;
            }
        }

        // --- Scalar Field: Mach Number (M) ---
        writeln!(writer, "SCALARS Mach_Number float 1")?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for j in 0..jmax {
            for i in 0..imax {
                writeln!(writer, "{:.6}", domain[[i, j]].flow_ij.M)?;
            }
        }

        // --- Scalar Field: Density (rho) ---
        writeln!(writer, "SCALARS Density float 1")?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for j in 0..jmax {
            for i in 0..imax {
                writeln!(writer, "{:.6}", domain[[i, j]].flow_ij.rho)?;
            }
        }

        // --- Vector Field: Velocity (u, v) ---
        writeln!(writer, "VECTORS Velocity float")?;
        for j in 0..jmax {
            for i in 0..imax {
                writeln!(writer, "{:.6} {:.6} 0.000000", domain[[i, j]].flow_ij.u, domain[[i, j]].flow_ij.v)?;
            }
        }

        // =========================================================
        // 4. Métricas Geométricas Opcionais
        // =========================================================
        if config.export_metrics {
            // --- Scalar Field: Jacobian (J) ---
            writeln!(writer, "SCALARS Jacobian float 1")?;
            writeln!(writer, "LOOKUP_TABLE default")?;
            for j in 0..jmax {
                for i in 0..imax {
                    writeln!(writer, "{:.8}", domain[[i, j]].metrics_ij.J)?;
                }
            }

            // --- Scalar Field: Metric A1 ---
            writeln!(writer, "SCALARS Metric_A1 float 1")?;
            writeln!(writer, "LOOKUP_TABLE default")?;
            for j in 0..jmax {
                for i in 0..imax {
                    writeln!(writer, "{:.8}", domain[[i, j]].metrics_ij.A1)?;
                }
            }

            // --- Scalar Field: Metric A2 ---
            writeln!(writer, "SCALARS Metric_A2 float 1")?;
            writeln!(writer, "LOOKUP_TABLE default")?;
            for j in 0..jmax {
                for i in 0..imax {
                    writeln!(writer, "{:.8}", domain[[i, j]].metrics_ij.A2)?;
                }
            }

            // --- Scalar Field: Metric A3 ---
            writeln!(writer, "SCALARS Metric_A3 float 1")?;
            writeln!(writer, "LOOKUP_TABLE default")?;
            for j in 0..jmax {
                for i in 0..imax {
                    writeln!(writer, "{:.8}", domain[[i, j]].metrics_ij.A3)?;
                }
            }
        }

        Ok(())
    };

    // Execute the write operation and handle potential errors cleanly
    write_data().expect("Failed during data writing to VTK file");

    println!("Results {} exported successfully", file_path);
}
