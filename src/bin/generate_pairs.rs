// src/bin/generate_pairs.rs - Generate parameter pairs for scanning

use std::fs::File;
use std::io::Write;
use clap::Parser;

#[derive(Parser)]
struct Cli {
    /// Beta center
    #[arg(long, default_value = "3.0")]
    beta_center: f64,
    
    /// Alpha center
    #[arg(long, default_value = "1.5")]
    alpha_center: f64,
    
    /// Beta width (±)
    #[arg(long, default_value = "0.3")]
    beta_width: f64,
    
    /// Alpha width (±)
    #[arg(long, default_value = "0.3")]
    alpha_width: f64,
    
    /// Step size
    #[arg(long, default_value = "0.02")]
    step: f64,
    
    /// Output file
    #[arg(long, default_value = "pairs.csv")]
    output: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Cli::parse();
    
    let beta_min = args.beta_center - args.beta_width;
    let beta_max = args.beta_center + args.beta_width;
    let alpha_min = args.alpha_center - args.alpha_width;
    let alpha_max = args.alpha_center + args.alpha_width;
    
    let mut file = File::create(&args.output)?;
    let mut count = 0;
    
    let mut beta = beta_min;
    while beta <= beta_max + 1e-10 {
        let mut alpha = alpha_min;
        while alpha <= alpha_max + 1e-10 {
            writeln!(file, "{:.3},{:.3}", beta, alpha)?;
            count += 1;
            alpha += args.step;
        }
        beta += args.step;
    }
    
    println!("Generated {} pairs in {}", count, args.output);
    println!("Beta range: [{:.2}, {:.2}]", beta_min, beta_max);
    println!("Alpha range: [{:.2}, {:.2}]", alpha_min, alpha_max);
    
    // Also generate a coarse version for quick tests
    if args.output == "pairs.csv" {
        let mut coarse = File::create("pairs_coarse.csv")?;
        let coarse_step = args.step * 5.0;
        let mut coarse_count = 0;
        
        let mut beta = beta_min;
        while beta <= beta_max + 1e-10 {
            let mut alpha = alpha_min;
            while alpha <= alpha_max + 1e-10 {
                writeln!(coarse, "{:.2},{:.2}", beta, alpha)?;
                coarse_count += 1;
                alpha += coarse_step;
            }
            beta += coarse_step;
        }
        
        println!("\nAlso generated {} coarse pairs in pairs_coarse.csv", coarse_count);
    }
    
    Ok(())
}