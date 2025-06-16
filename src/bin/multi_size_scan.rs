// src/bin/multi_size_scan.rs - Run scans at multiple system sizes

use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use clap::Parser;

#[derive(Parser)]
struct Cli {
    /// System sizes to scan
    #[arg(long, default_value = "24,48,96", value_delimiter = ',')]
    sizes: Vec<usize>,
    
    /// Beta range (min,max,step)
    #[arg(long, default_value = "2.8,3.2,0.02", value_delimiter = ',')]
    beta_range: Vec<f64>,
    
    /// Alpha range (min,max,step)
    #[arg(long, default_value = "1.3,1.7,0.02", value_delimiter = ',')]
    alpha_range: Vec<f64>,
    
    /// Number of replicas per point
    #[arg(long, default_value = "10")]
    replicas: usize,
    
    /// Output directory
    #[arg(long, default_value = "fss_data")]
    output_dir: PathBuf,
}

fn generate_pairs(beta_range: &[f64], alpha_range: &[f64]) -> Vec<(f64, f64)> {
    let mut pairs = Vec::new();
    
    let beta_min = beta_range[0];
    let beta_max = beta_range[1];
    let beta_step = beta_range[2];
    
    let alpha_min = alpha_range[0];
    let alpha_max = alpha_range[1];
    let alpha_step = alpha_range[2];
    
    let mut beta = beta_min;
    while beta <= beta_max + 1e-10 {
        let mut alpha = alpha_min;
        while alpha <= alpha_max + 1e-10 {
            pairs.push((beta, alpha));
            alpha += alpha_step;
        }
        beta += beta_step;
    }
    
    pairs
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();
    
    // Create output directory
    fs::create_dir_all(&args.output_dir)?;
    
    // Generate parameter pairs
    let pairs = generate_pairs(&args.beta_range, &args.alpha_range);
    println!("Generated {} (β,α) pairs", pairs.len());
    
    // Write pairs file
    let pairs_path = args.output_dir.join("pairs.csv");
    let mut pairs_file = File::create(&pairs_path)?;
    for (beta, alpha) in &pairs {
        writeln!(pairs_file, "{},{}", beta, alpha)?;
    }
    
    // Create config files for each system size
    for &size in &args.sizes {
        let config_path = args.output_dir.join(format!("config_n{}.toml", size));
        let mut config_file = File::create(&config_path)?;
        
        writeln!(config_file, "[mc]")?;
        writeln!(config_file, "n_nodes = {}", size)?;
        writeln!(config_file, "n_steps = {}", 200_000 * (48.0 / size as f64).sqrt() as usize)?;
        writeln!(config_file, "equil_steps = {}", 80_000 * (48.0 / size as f64).sqrt() as usize)?;
        writeln!(config_file, "sample_every = 10")?;
        writeln!(config_file, "n_rep = {}", args.replicas)?;
        writeln!(config_file, "tune_win = 200")?;
        writeln!(config_file, "tune_tgt = 0.3")?;
        writeln!(config_file, "tune_band = 0.05")?;
    }
    
    // Create run script
    let script_path = args.output_dir.join("run_all.sh");
    let mut script = File::create(&script_path)?;
    writeln!(script, "#!/bin/bash")?;
    writeln!(script, "set -e")?;
    writeln!(script)?;
    
    for &size in &args.sizes {
        let output_file = format!("results_n{}.csv", size);
        writeln!(script, "echo 'Running n={} scan...'", size)?;
        writeln!(script, "# Note: You'll need to modify improved_narrow_scan to accept --n-nodes parameter")?;
        writeln!(script, "# Or create a version that reads the config file")?;
        writeln!(script, "cargo run --release --bin improved_narrow_scan -- \\")?;
        writeln!(script, "  --pairs {} \\", pairs_path.display())?;
        writeln!(script, "  --output {}", args.output_dir.join(&output_file).display())?;
        writeln!(script, "# TODO: Add --n-nodes {} parameter", size)?;
        writeln!(script)?;
    }
    
    writeln!(script, "echo 'Running FSS analysis...'")?;
    writeln!(script, "cargo run --release --bin fss_analysis -- \\")?;
    write!(script, "  --inputs ")?;
    for (i, &size) in args.sizes.iter().enumerate() {
        if i > 0 { write!(script, ",")?; }
        write!(script, "{}", args.output_dir.join(format!("results_n{}.csv", size)).display())?;
    }
    writeln!(script, " \\")?;
    writeln!(script, "  --output {}", args.output_dir.join("fss_results.csv").display())?;
    
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&script_path, fs::Permissions::from_mode(0o755))?;
    }
    
    println!("\nSetup complete! To run the full FSS analysis:");
    println!("  cd {}", args.output_dir.display());
    println!("  ./run_all.sh");
    
    Ok(())
}