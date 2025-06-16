// src/bin/fss_analysis.rs - Analyze results from multiple system sizes

use std::fs::File;
use std::path::PathBuf;
use clap::Parser;
use csv::{Reader, ReaderBuilder, WriterBuilder};
use scan::finite_size::{FiniteSizeAnalysis, FSAData};

#[derive(Parser)]
struct Cli {
    /// Input CSV files from different system sizes
    #[arg(long, value_delimiter = ',')]
    inputs: Vec<PathBuf>,
    
    /// Output file for FSS results
    #[arg(long, default_value = "fss_results.csv")]
    output: PathBuf,
    
    /// Expected critical beta (initial guess)
    #[arg(long, default_value = "3.0")]
    beta_c: f64,
    
    /// Expected critical alpha (initial guess)
    #[arg(long, default_value = "1.5")]
    alpha_c: f64,
}

#[derive(Debug)]
struct DataPoint {
    beta: f64,
    alpha: f64,
    mean_w: f64,
    mean_cos: f64,
    susceptibility: f64,
    mean_action: f64,
    // Add specific heat when available
}

fn read_data_file(path: &PathBuf) -> Result<(usize, Vec<DataPoint>), Box<dyn std::error::Error>> {
    // Extract system size from filename (e.g., "results_n48.csv" -> 48)
    let filename = path.file_stem().unwrap().to_str().unwrap();
    let size: usize = filename
        .split('n')
        .last()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);
    
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;
    
    let mut data = Vec::new();
    for result in rdr.records() {
        let record = result?;
        data.push(DataPoint {
            beta: record[0].parse()?,
            alpha: record[1].parse()?,
            mean_w: record[2].parse()?,
            mean_cos: record[4].parse()?,
            susceptibility: record[6].parse()?,
            mean_action: record[7].parse().unwrap_or(0.0),
        });
    }
    
    Ok((size, data))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();
    
    let mut fsa = FiniteSizeAnalysis::new();
    
    // Read all data files
    for input_path in &args.inputs {
        println!("Reading {}", input_path.display());
        let (size, data) = read_data_file(input_path)?;
        
        // Convert to FSAData format
        for dp in data {
            // Calculate Binder cumulant if we have the raw data
            // For now, use a placeholder
            let binder = 0.5; // This would need 4th moment data
            
            // Estimate specific heat from action fluctuations
            // This is simplified - proper calculation needs time series
            let specific_heat = 0.0; // Placeholder
            
            fsa.add_data(FSAData {
                lattice_size: size,
                beta: dp.beta,
                alpha: dp.alpha,
                susceptibility: dp.susceptibility,
                specific_heat,
                binder_cumulant: binder,
                correlation_length: 0.0, // Would need eigenvalue calculation
            });
        }
    }
    
    // Find critical point from Binder cumulant crossings
    let (beta_c, alpha_c) = fsa.find_critical_point();
    println!("\nEstimated critical point: β_c = {:.3}, α_c = {:.3}", beta_c, alpha_c);
    
    // Extract critical exponents
    let gamma_over_nu = fsa.extract_gamma_over_nu(beta_c, alpha_c);
    let alpha_over_nu = fsa.extract_alpha_over_nu(beta_c, alpha_c);
    
    println!("\nCritical exponents:");
    println!("  γ/ν = {:.3}", gamma_over_nu);
    println!("  α/ν = {:.3}", alpha_over_nu);
    
    // Write results
    let mut wtr = WriterBuilder::new()
        .from_path(&args.output)?;
    
    wtr.write_record(&["quantity", "value"])?;
    wtr.write_record(&["beta_c", &beta_c.to_string()])?;
    wtr.write_record(&["alpha_c", &alpha_c.to_string()])?;
    wtr.write_record(&["gamma_over_nu", &gamma_over_nu.to_string()])?;
    wtr.write_record(&["alpha_over_nu", &alpha_over_nu.to_string()])?;
    
    // Compare to known universality classes
    println!("\nUniversality class comparison:");
    println!("  3D Ising: γ/ν ≈ 1.96, α/ν ≈ 0.11");
    println!("  4D Ising: γ/ν = 2.00, α/ν = 0.00");
    println!("  3D XY:    γ/ν ≈ 1.97, α/ν ≈ -0.01");
    
    Ok(())
}