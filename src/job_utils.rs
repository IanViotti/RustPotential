use crate::config;
use crate::solver_utils;
use crate::solver_utils::PointData;

use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::io::{Write, BufWriter};
use std::time::SystemTime;
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
        .unwrap_or_else(|_| panic!("Could not read the {} file. Please make sure it exists and is in the correct path.\n", filename));

    // 2. Deserialize using the 'toml' crate
    let mut config: config::Config = toml::from_str(&config_content)
        .expect("Error parsing the configuration file. Please check the format and values.\n");

    config.compute_derived_variables();

    config
}

/// Generates a timestamp string YYYY-MM-DD_HH-MM-SS from system time.
fn current_timestamp_string() -> String {
    let now = SystemTime::now();
    let duration = now
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let total_secs = duration.as_secs();

    // Deconstruct seconds into date and time
    let sec = total_secs % 60;
    let total_min = total_secs / 60;
    let min = total_min % 60;
    let total_hours = total_min / 60;
    let hour = total_hours % 24;

    // Convert days since UNIX epoch (1970-01-01) to Year-Month-Day
    let mut days = (total_hours / 24) as i64;
    // 1970-01-01 was a Thursday
    let mut year = 1970i32;

    loop {
        let leap = is_leap_year(year);
        let days_in_year = if leap { 366 } else { 365 };
        if days >= days_in_year {
            days -= days_in_year;
            year += 1;
        } else {
            break;
        }
    }

    let leap = is_leap_year(year);
    let days_in_months = [
        31, if leap { 29 } else { 28 }, 31, 30, 31, 30,
        31, 31, 30, 31, 30, 31,
    ];

    let mut month = 1usize;
    for &dim in &days_in_months {
        if days >= dim as i64 {
            days -= dim as i64;
            month += 1;
        } else {
            break;
        }
    }
    let day = (days + 1) as usize;

    format!("{:04}-{:02}-{:02}_{:02}-{:02}-{:02}", year, month, day, hour, min, sec)
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Creates a unique output directory incorporating jobname and date/timestamp.
pub fn init_job_directory(config: &mut config::Config) {
    let timestamp = current_timestamp_string();
    let folder_name = format!("{}_{}", config.jobname, timestamp);
    let output_dir = format!("job_files/{}", folder_name);
    
    std::fs::create_dir_all(&output_dir)
        .unwrap_or_else(|_| panic!("Failed to create job output directory: {}", output_dir));
    
    config.output_dir = output_dir;
}

/// Exports solver configuration and mesh information into a human-readable config summary file.
pub fn save_solver_configuration(config: &config::Config, grid: &Grid) {
    let summary_path = format!("{}/solver_config.txt", config.output_dir);
    let file = File::create(&summary_path)
        .unwrap_or_else(|_| panic!("Failed to create solver configuration file: {}", summary_path));
    let mut writer = BufWriter::new(file);

    let total_nodes = grid.imax * grid.jmax;
    let regime = if config.Mach_inf < 0.8 {
        "Subsonic"
    } else if config.Mach_inf <= 1.2 {
        "Transonic"
    } else {
        "Supersonic"
    };

    writeln!(writer, "================================================================================").unwrap();
    writeln!(writer, "                    RUSTPOTENTIAL SOLVER CONFIGURATION                          ").unwrap();
    writeln!(writer, "================================================================================").unwrap();
    writeln!(writer, "Job Name                : {}", config.jobname).unwrap();
    writeln!(writer, "Output Directory        : {}", config.output_dir).unwrap();
    writeln!(writer, "").unwrap();
    writeln!(writer, "--------------------------------------------------------------------------------").unwrap();
    writeln!(writer, "  MESH SPECIFICATIONS                                                           ").unwrap();
    writeln!(writer, "--------------------------------------------------------------------------------").unwrap();
    writeln!(writer, "Mesh File               : {}", config.meshname).unwrap();
    writeln!(writer, "Grid Dimensions (imax)  : {} (Azimuthal / Xi direction)", grid.imax).unwrap();
    writeln!(writer, "Grid Dimensions (jmax)  : {} (Radial / Eta direction)", grid.jmax).unwrap();
    writeln!(writer, "Total Grid Nodes        : {}", total_nodes).unwrap();
    writeln!(writer, "Wake Cut Node Pair      : i = 0 <-> i = {}", grid.imax - 1).unwrap();
    writeln!(writer, "Farfield Boundary       : j = 0").unwrap();
    writeln!(writer, "Airfoil Wall Boundary   : j = {}", grid.jmax - 1).unwrap();
    writeln!(writer, "").unwrap();
    writeln!(writer, "--------------------------------------------------------------------------------").unwrap();
    writeln!(writer, "  FLOW & FLIGHT PARAMETERS                                                      ").unwrap();
    writeln!(writer, "--------------------------------------------------------------------------------").unwrap();
    writeln!(writer, "Freestream Mach (M_inf) : {:.6} ({})", config.Mach_inf, regime).unwrap();
    writeln!(writer, "Angle of Attack (AoA)   : {:.4} deg ({:.6} rad)", config.AoA, config.alpha_rad).unwrap();
    writeln!(writer, "Specific Heat Ratio (g) : {:.4}", config.gamma).unwrap();
    writeln!(writer, "Freestream Speed (q_inf): {:.6} (non-dimensional)", config.q_inf).unwrap();
    writeln!(writer, "Freestream Velocity u   : {:.6}", config.u_inf).unwrap();
    writeln!(writer, "Freestream Velocity v   : {:.6}", config.v_inf).unwrap();
    writeln!(writer, "Freestream Density (rho): {:.6}", config.rho_inf).unwrap();
    writeln!(writer, "").unwrap();
    writeln!(writer, "--------------------------------------------------------------------------------").unwrap();
    writeln!(writer, "  AF2 NUMERICAL SCHEME CONTROLS                                                 ").unwrap();
    writeln!(writer, "--------------------------------------------------------------------------------").unwrap();
    writeln!(writer, "Max Iterations          : {}", config.max_iterations).unwrap();
    writeln!(writer, "Convergence Tolerance   : {:.2e}", config.convergence_residual).unwrap();
    writeln!(writer, "Relaxation Factor (w)   : {:.4}", config.omega).unwrap();
    writeln!(writer, "Circulation Relax (Rg)  : {:.4}", config.Rg).unwrap();
    writeln!(writer, "Alpha Sequence Length M : {}", config.alpha_sequence_length).unwrap();
    writeln!(writer, "Supersonic Damping Beta : {:.4}", config.beta_mult).unwrap();
    writeln!(writer, "Holst Upwind Coeff (C)  : {:.4}", config.C).unwrap();
    writeln!(writer, "Export Geometric Metrics: {}", config.export_metrics).unwrap();
    writeln!(writer, "================================================================================").unwrap();

    // Also write a copy of job_config.toml into the folder for exact reproduction
    if let Ok(config_copy) = fs::read_to_string("job_config.toml") {
        let toml_dest = format!("{}/job_config.toml", config.output_dir);
        let _ = fs::write(toml_dest, config_copy);
    }

    println!("Solver configuration saved to: {}", summary_path);
}

// ==========================================
// 3. VTK I/O OPERATIONS
// ==========================================

/// Reads a structured grid .vtk file and reconstructs the coordinate matrices.
/// Reorients into the course lecture note convention for O-grids:
/// - j = 0 at farfield, j = jmax - 1 at airfoil surface (eta points inward)
/// - i = 0 at TE, i increases counter-clockwise (upper surface -> LE -> lower surface -> TE)
pub fn read_vtk_structured_grid(filepath: &str) -> Grid {
    let file = File::open(filepath).unwrap_or_else(|_| panic!("Failed to open VTK file: {}", filepath));
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

            for j_raw in 0..jmax {
                let j = jmax - 1 - j_raw;
                for i_raw in 0..imax {
                    let i = imax - 1 - i_raw;
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
    let file_path = format!("{}/solution.vtk", config.output_dir);
    let file = File::create(&file_path).unwrap_or_else(|_| panic!("Failed to create VTK output file: {}", file_path));
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
        // 4. Geometric Metrics (Optional)
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


pub fn export_cp_csv(config: &config::Config, grid: &Grid, domain: &Array2<PointData>) {
    println!("Exporting Cp distribution to CSV...");

    let filename = format!("{}/Cp_distribution.csv", config.output_dir);
    let mut file = File::create(&filename).unwrap_or_else(|_| panic!("Failed to create CSV file: {}", filename));

    writeln!(file, "x,y,Cp").unwrap();

    let gamma = config.gamma;
    let m_inf = config.Mach_inf;
    let rho_inf = config.rho_inf;
    let cp_multiplier = 2.0 / (gamma * m_inf.powi(2));

    // Airfoil surface is at j = jmax - 1
    let j_wall = grid.jmax - 1;

    // Loop from 0 to imax-2 to complete profile without duplicating trailing edge
    for i in 0..grid.imax - 1 {
        let x = grid.points[[i, j_wall]].x;
        let y = grid.points[[i, j_wall]].y;
        let rho_local = domain[[i, j_wall]].flow_ij.rho;

        // Isentropic relation for pressure coefficient Cp
        let cp = cp_multiplier * ((rho_local / rho_inf).powf(gamma) - 1.0);

        writeln!(file, "{:.6},{:.6},{:.6}", x, y, cp).unwrap();
    }

    println!("CSV file exported successfully: {}", filename);
}
