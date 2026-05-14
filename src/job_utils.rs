use crate::config;
use crate::solver_utils::Field;

use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::io::{Write, BufWriter};
use ndarray::Array2; // Assumindo o uso da crate ndarray


// 1. Novo struct para representar uma coordenada na malha
#[derive(Clone, Copy, Debug, Default)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

// 2. Struct Grid atualizado para conter o Array2
pub struct Grid {
    pub imax: usize,
    pub jmax: usize,
    pub points: Array2<Point>,
}

pub fn read_config(filename: &str) -> config::Config {
    // 1. Read the contents of the `job_config.toml` file into a string.
    let config_content = fs::read_to_string("job_config.toml")
        .expect("It was not possible to read the job_config.toml file. Please make sure it exists and is in the correct path.\n");

    // 2. Deserialize using the 'toml' crate
    // Maps the raw string into a strongly-typed `config::Config` struct.
    let config: config::Config = toml::from_str(&config_content)
        .expect("Error parsing the job_config.toml file. Please check the format and values.\n");

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


/// Lê um arquivo .vtk de malha estruturada e reconstrói as matrizes de coordenadas.
/// Em caso de erro na leitura, a execução do programa será interrompida (panic).
pub fn read_vtk_structured_grid(filepath: &str) -> Grid {
    // Usamos .expect() no lugar do operador '?'. 
    // Se o arquivo não existir, o programa para aqui e mostra essa mensagem.
    let file = File::open(filepath).expect(&format!("Falha ao abrir o arquivo VTK: {}", filepath));
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut imax = 0;
    let mut jmax = 0;
    
    let mut points_array = Array2::<Point>::default((0, 0));

    while let Some(Ok(line)) = lines.next() {
        let line = line.trim();

        if line.starts_with("DIMENSIONS") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            // Também substituímos o '?' por .expect() nas conversões numéricas
            imax = parts[1].parse::<usize>().expect("Falha ao ler dimensão imax do arquivo");
            jmax = parts[2].parse::<usize>().expect("Falha ao ler dimensão jmax do arquivo");
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

    // Retornamos a struct diretamente, sem envolver no Ok()
    Grid {
        imax,
        jmax,
        points: points_array,
    }
}



/// Salva a malha e os campos calculados em um arquivo VTK.
/// Interrompe a execução (panic) em caso de erro de IO.
pub fn export_vtk_fields(filepath: &str, grid: &Grid, fields: &Array2<Field>) {
    let file = File::create(filepath).expect(&format!("Falha ao criar o arquivo VTK: {}", filepath));
    let mut writer = BufWriter::new(file);

    // Closure interna para usar o operador '?' sem sujar a assinatura da função principal
    let mut write_data = || -> std::io::Result<()> {
        let imax = grid.imax;
        let jmax = grid.jmax;
        let total_points = imax * jmax;

        // 1. Cabeçalho VTK
        writeln!(writer, "# vtk DataFile Version 3.0")?;
        writeln!(writer, "Resultados CFD")?;
        writeln!(writer, "ASCII")?;
        writeln!(writer, "DATASET STRUCTURED_GRID")?;
        writeln!(writer, "DIMENSIONS {} {} 1", imax, jmax)?;

        // 2. Pontos (Geometria extraída do Grid)
        writeln!(writer, "POINTS {} float", total_points)?;
        for j in 0..jmax {
            for i in 0..imax {
                let pt = &grid.points[[i, j]];
                writeln!(writer, "{:.6} {:.6} 0.000000", pt.x, pt.y)?;
            }
        }

        // 3. Resultados (Dados dos Pontos)
        writeln!(writer, "POINT_DATA {}", total_points)?;

        // --- Campo Escalar: Potencial (phi) ---
        writeln!(writer, "SCALARS Potencial float 1")?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for j in 0..jmax {
            for i in 0..imax {
                writeln!(writer, "{:.6}", fields[[i, j]].phi)?;
            }
        }

        // --- Campo Escalar: Mach (M) ---
        writeln!(writer, "SCALARS Mach float 1")?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for j in 0..jmax {
            for i in 0..imax {
                writeln!(writer, "{:.6}", fields[[i, j]].M)?;
            }
        }

        // --- Campo Escalar: Densidade (rho) ---
        writeln!(writer, "SCALARS Densidade float 1")?;
        writeln!(writer, "LOOKUP_TABLE default")?;
        for j in 0..jmax {
            for i in 0..imax {
                writeln!(writer, "{:.6}", fields[[i, j]].rho)?;
            }
        }

        // --- Campo Vetorial: Velocidade (u, v) ---
        writeln!(writer, "VECTORS Velocidade float")?;
        for j in 0..jmax {
            for i in 0..imax {
                writeln!(writer, "{:.6} {:.6} 0.000000", fields[[i, j]].u, fields[[i, j]].v)?;
            }
        }

        Ok(())
    };

    // Executa a escrita e, se algo falhar, dá o panic informando o problema
    write_data().expect("Falha durante a escrita dos dados no arquivo VTK");

    println!("Results {} exported successfully", filepath);
}