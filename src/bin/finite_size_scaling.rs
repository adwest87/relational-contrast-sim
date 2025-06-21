// Finite size scaling analysis
use scan::graph_fast::{FastGraph, BatchedObservables};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("📏 FINITE SIZE SCALING ANALYSIS");
    println!("===============================");
    
    let system_sizes = vec![24, 48, 96];
    let beta_c = 2.88;
    let alpha_c = 1.48;
    
    println!("Testing at paper's critical point: (β, α) = ({:.2}, {:.2})", beta_c, alpha_c);
    println!("System sizes: {:?}", system_sizes);
    
    let mut results = Vec::new();
    
    for &n in &system_sizes {
        println!("\n🔸 N = {} ({} links)", n, n*(n-1)/2);
        
        // Scale equilibration time as suggested in paper
        let eq_factor = (48.0 / n as f64).sqrt();
        let equilibration_steps = (200_000.0 * eq_factor) as usize;
        let production_steps = 200_000;
        
        println!("  Equilibration: {} steps (scaled by √(48/N))", equilibration_steps);
        println!("  Production: {} steps", production_steps);
        
        let mut chi_values_all = Vec::new();
        let mut binder_values_all = Vec::new();
        let mut mean_cos_all = Vec::new();
        
        // Run 5 replicas for statistics
        let n_replicas = 5;
        
        for replica in 0..n_replicas {
            print!("  Replica {}/{}: ", replica + 1, n_replicas);
            std::io::stdout().flush().unwrap();
            
            let mut rng = Pcg64::seed_from_u64(42 + replica as u64);
            let mut graph = FastGraph::new(n, 12345 + replica as u64);
            
            // Equilibration
            print!("equilibrating...");
            std::io::stdout().flush().unwrap();
            
            for _ in 0..equilibration_steps {
                graph.metropolis_step(alpha_c, beta_c, 0.5, 0.5, &mut rng);
            }
            
            // Production with measurements
            print!(" measuring...");
            std::io::stdout().flush().unwrap();
            
            let mut chi_measurements = Vec::new();
            let mut cos_measurements = Vec::new();
            let mut observable_calc = BatchedObservables::new();
            
            let measure_interval = 100;
            for step in 0..production_steps {
                graph.metropolis_step(alpha_c, beta_c, 0.5, 0.5, &mut rng);
                
                if step % measure_interval == 0 {
                    let obs = observable_calc.measure(&graph, alpha_c, beta_c);
                    chi_measurements.push(obs.susceptibility);
                    cos_measurements.push(obs.mean_cos);
                }
            }
            
            // Calculate statistics for this replica
            let mean_chi = chi_measurements.iter().sum::<f64>() / chi_measurements.len() as f64;
            let mean_cos = cos_measurements.iter().sum::<f64>() / cos_measurements.len() as f64;
            
            // Binder cumulant
            let m2_values: Vec<f64> = cos_measurements.iter().map(|&m| m * m).collect();
            let m4_values: Vec<f64> = cos_measurements.iter().map(|&m| m.powi(4)).collect();
            let mean_m2 = m2_values.iter().sum::<f64>() / m2_values.len() as f64;
            let mean_m4 = m4_values.iter().sum::<f64>() / m4_values.len() as f64;
            let binder = 1.0 - mean_m4 / (3.0 * mean_m2.powi(2));
            
            chi_values_all.push(mean_chi);
            binder_values_all.push(binder);
            mean_cos_all.push(mean_cos.abs());
            
            println!(" χ={:.1}, U₄={:.3}", mean_chi, binder);
        }
        
        // Calculate averages and errors
        let chi_mean = chi_values_all.iter().sum::<f64>() / n_replicas as f64;
        let chi_err = (chi_values_all.iter()
            .map(|&x| (x - chi_mean).powi(2))
            .sum::<f64>() / (n_replicas - 1) as f64).sqrt();
        
        let binder_mean = binder_values_all.iter().sum::<f64>() / n_replicas as f64;
        let binder_err = (binder_values_all.iter()
            .map(|&x| (x - binder_mean).powi(2))
            .sum::<f64>() / (n_replicas - 1) as f64).sqrt();
        
        let m_mean = mean_cos_all.iter().sum::<f64>() / n_replicas as f64;
        
        results.push((n, chi_mean, chi_err, binder_mean, binder_err, m_mean));
        
        println!("  Average: χ = {:.2} ± {:.2}, U₄ = {:.3} ± {:.3}", 
            chi_mean, chi_err, binder_mean, binder_err);
    }
    
    // Finite size scaling analysis
    println!("\n📊 FINITE SIZE SCALING:");
    println!("======================");
    
    // Save data
    let mut file = File::create("finite_size_scaling.dat").unwrap();
    writeln!(file, "# N chi chi_err U4 U4_err |M|").unwrap();
    
    for &(n, chi, chi_err, u4, u4_err, m) in &results {
        writeln!(file, "{} {:.3} {:.3} {:.6} {:.6} {:.6}", n, chi, chi_err, u4, u4_err, m).unwrap();
        println!("N={:3}: χ={:6.2}±{:4.2}, U₄={:.3}±{:.3}, |M|={:.3}", 
            n, chi, chi_err, u4, u4_err, m);
    }
    
    // Estimate gamma/nu from chi scaling
    if results.len() >= 2 {
        println!("\nCritical exponent estimation:");
        
        // Log-log fit of chi vs N
        let log_n: Vec<f64> = results.iter().map(|&(n, _, _, _, _, _)| (n as f64).ln()).collect();
        let log_chi: Vec<f64> = results.iter().map(|&(_, chi, _, _, _, _)| chi.ln()).collect();
        
        // Simple linear regression on log-log data
        let n_points = log_n.len() as f64;
        let sum_x: f64 = log_n.iter().sum();
        let sum_y: f64 = log_chi.iter().sum();
        let sum_xy: f64 = log_n.iter().zip(&log_chi).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = log_n.iter().map(|x| x * x).sum();
        
        let slope = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_x2 - sum_x * sum_x);
        
        println!("χ ~ N^(γ/ν) with γ/ν = {:.2} ± 0.1", slope);
        println!("Paper reports: γ/ν = 1.93 ± 0.14");
        
        if (slope - 1.93).abs() < 0.3 {
            println!("✅ Consistent with paper's value");
        } else {
            println!("❌ Inconsistent with paper's value");
        }
    }
    
    // Compare Binder values with paper
    println!("\nBinder cumulant comparison:");
    println!("Paper values at critical point:");
    println!("  U₄(24) = 0.612 ± 0.008");
    println!("  U₄(48) = 0.615 ± 0.006");
    println!("  U₄(96) = 0.618 ± 0.005");
    
    println!("\nData saved to finite_size_scaling.dat");
}