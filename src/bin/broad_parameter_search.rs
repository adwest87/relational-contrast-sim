// Broad parameter search for phase transition
use scan::graph_fast::{FastGraph, BatchedObservables};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("🔍 BROAD PARAMETER SEARCH");
    println!("========================");
    
    let n = 48;
    let equilibration_steps = 30_000;
    let production_steps = 30_000;
    let measure_interval = 100;
    
    // Broader parameter ranges
    let beta_values: Vec<f64> = vec![0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0];
    let alpha_values: Vec<f64> = vec![0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0];
    
    println!("System: N={}", n);
    println!("β range: {:?}", beta_values);
    println!("α range: {:?}", alpha_values);
    println!("Quick scan: {}k eq + {}k prod per point\n", 
        equilibration_steps/1000, production_steps/1000);
    
    let mut csv_file = File::create("broad_parameter_search.csv").unwrap();
    writeln!(csv_file, "beta,alpha,chi,mean_cos,mean_abs_cos,binder,acceptance").unwrap();
    
    let mut max_chi = 0.0;
    let mut max_chi_params = (0.0, 0.0);
    
    for &beta in &beta_values {
        println!("β = {:.1}:", beta);
        
        for &alpha in &alpha_values {
            print!("  α={:.1}: ", alpha);
            std::io::stdout().flush().unwrap();
            
            let mut rng = Pcg64::seed_from_u64(42);
            let mut graph = FastGraph::new(n, 12345);
            
            // Equilibration
            let mut accepted = 0;
            for _ in 0..equilibration_steps {
                let info = graph.metropolis_step(alpha, beta, 0.5, 0.5, &mut rng);
                if info.accept {
                    accepted += 1;
                }
            }
            
            // Production
            let mut chi_values = Vec::new();
            let mut cos_values = Vec::new();
            let mut observable_calc = BatchedObservables::new();
            
            accepted = 0;
            for step in 0..production_steps {
                let info = graph.metropolis_step(alpha, beta, 0.5, 0.5, &mut rng);
                if info.accept {
                    accepted += 1;
                }
                
                if step % measure_interval == 0 {
                    let obs = observable_calc.measure(&graph, alpha, beta);
                    chi_values.push(obs.susceptibility);
                    cos_values.push(obs.mean_cos);
                }
            }
            
            let acceptance = accepted as f64 / production_steps as f64 * 100.0;
            let mean_chi = chi_values.iter().sum::<f64>() / chi_values.len() as f64;
            let mean_cos = cos_values.iter().sum::<f64>() / cos_values.len() as f64;
            let mean_abs_cos = cos_values.iter().map(|x| x.abs()).sum::<f64>() 
                / cos_values.len() as f64;
            
            // Binder cumulant
            let m2_values: Vec<f64> = cos_values.iter().map(|&m| m * m).collect();
            let m4_values: Vec<f64> = cos_values.iter().map(|&m| m.powi(4)).collect();
            let mean_m2 = m2_values.iter().sum::<f64>() / m2_values.len() as f64;
            let mean_m4 = m4_values.iter().sum::<f64>() / m4_values.len() as f64;
            let binder = 1.0 - mean_m4 / (3.0 * mean_m2.powi(2));
            
            println!("χ={:.1}, |M|={:.3}, U₄={:.3}, acc={:.0}%", 
                mean_chi, mean_abs_cos, binder, acceptance);
            
            writeln!(csv_file, "{:.1},{:.1},{:.3},{:.6},{:.6},{:.3},{:.1}",
                beta, alpha, mean_chi, mean_cos, mean_abs_cos, binder, acceptance).unwrap();
            
            if mean_chi > max_chi {
                max_chi = mean_chi;
                max_chi_params = (beta, alpha);
            }
        }
    }
    
    println!("\n📊 RESULTS:");
    println!("===========");
    println!("Maximum susceptibility: χ = {:.1}", max_chi);
    println!("Found at: (β, α) = ({:.1}, {:.1})", max_chi_params.0, max_chi_params.1);
    
    if max_chi < 10.0 {
        println!("\n⚠️ WARNING: Maximum susceptibility is very low!");
        println!("This suggests:");
        println!("  1. No phase transition in this parameter range");
        println!("  2. System size N={} is too small", n);
        println!("  3. The model may not have a conventional phase transition");
    }
    
    println!("\nData saved to broad_parameter_search.csv");
}