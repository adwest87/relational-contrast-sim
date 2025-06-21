// Validate critical point against paper's results
use scan::graph_fast::{FastGraph, BatchedObservables};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("🎯 CRITICAL POINT VALIDATION");
    println!("============================");
    println!("Testing (β_c, α_c) = (2.88, 1.48) from paper");
    
    let n = 48;
    let beta_c = 2.88;
    let alpha_c = 1.48;
    let equilibration_steps = 200_000;
    let production_steps = 200_000;
    let n_replicas = 10;
    
    println!("System: N={}", n);
    println!("Critical point: β={:.3}, α={:.3}", beta_c, alpha_c);
    println!("Steps: {}k equilibration + {}k production", 
        equilibration_steps/1000, production_steps/1000);
    println!("Replicas: {}", n_replicas);
    
    let mut all_binder_values = Vec::new();
    let mut all_chi_values = Vec::new();
    let mut all_mean_cos_values = Vec::new();
    
    for replica in 0..n_replicas {
        print!("\nReplica {}/{}: ", replica + 1, n_replicas);
        std::io::stdout().flush().unwrap();
        
        // Different seed for each replica
        let mut rng = Pcg64::seed_from_u64(42 + replica as u64);
        let mut graph = FastGraph::new(n, 12345 + replica as u64);
        
        // Adaptive move sizes
        let mut delta_z = 0.5;
        let mut delta_theta = 0.5;
        let mut accepted = 0;
        let mut total = 0;
        
        // Equilibration with adaptive moves
        print!("equilibrating");
        for step in 0..equilibration_steps {
            let info = graph.metropolis_step(alpha_c, beta_c, delta_z, delta_theta, &mut rng);
            if info.accept {
                accepted += 1;
            }
            total += 1;
            
            // Adapt move sizes every 10k steps
            if step > 0 && step % 10000 == 0 {
                let acceptance_rate = accepted as f64 / total as f64;
                
                // Adjust move sizes to target 50% acceptance
                if acceptance_rate < 0.4 {
                    delta_z *= 0.9;
                    delta_theta *= 0.9;
                } else if acceptance_rate > 0.6 {
                    delta_z *= 1.1;
                    delta_theta *= 1.1;
                }
                
                // Reset counters
                accepted = 0;
                total = 0;
                
                if step % 50000 == 0 {
                    print!(".");
                    std::io::stdout().flush().unwrap();
                }
            }
        }
        
        let final_acceptance = accepted as f64 / total as f64 * 100.0;
        print!(" [acc={:.1}%]", final_acceptance);
        
        // Production run with measurements
        print!(" measuring");
        let mut cos_theta_values = Vec::new();
        let mut chi_measurements = Vec::new();
        let mut observable_calc = BatchedObservables::new();
        
        // Measure every 100 steps
        let measure_interval = 100;
        
        for step in 0..production_steps {
            let _info = graph.metropolis_step(alpha_c, beta_c, delta_z, delta_theta, &mut rng);
            
            if step % measure_interval == 0 {
                let obs = observable_calc.measure(&graph, alpha_c, beta_c);
                cos_theta_values.push(obs.mean_cos);
                chi_measurements.push(obs.susceptibility);
                
                if step % 50000 == 0 && step > 0 {
                    print!(".");
                    std::io::stdout().flush().unwrap();
                }
            }
        }
        
        // Calculate Binder cumulant
        let mean_cos = cos_theta_values.iter().sum::<f64>() / cos_theta_values.len() as f64;
        let m2_values: Vec<f64> = cos_theta_values.iter().map(|&m| m * m).collect();
        let m4_values: Vec<f64> = cos_theta_values.iter().map(|&m| m.powi(4)).collect();
        
        let mean_m2 = m2_values.iter().sum::<f64>() / m2_values.len() as f64;
        let mean_m4 = m4_values.iter().sum::<f64>() / m4_values.len() as f64;
        
        let binder = 1.0 - mean_m4 / (3.0 * mean_m2.powi(2));
        let mean_chi = chi_measurements.iter().sum::<f64>() / chi_measurements.len() as f64;
        
        all_binder_values.push(binder);
        all_chi_values.push(mean_chi);
        all_mean_cos_values.push(mean_cos);
        
        println!(" U_4={:.3}, χ={:.1}, |M|={:.3}", binder, mean_chi, mean_cos.abs());
        println!("  Final move sizes: δz={:.3}, δθ={:.3}", delta_z, delta_theta);
    }
    
    // Calculate statistics
    let mean_binder = all_binder_values.iter().sum::<f64>() / n_replicas as f64;
    let std_binder = (all_binder_values.iter()
        .map(|&x| (x - mean_binder).powi(2))
        .sum::<f64>() / (n_replicas - 1) as f64).sqrt();
    
    let mean_chi = all_chi_values.iter().sum::<f64>() / n_replicas as f64;
    let std_chi = (all_chi_values.iter()
        .map(|&x| (x - mean_chi).powi(2))
        .sum::<f64>() / (n_replicas - 1) as f64).sqrt();
    
    println!("\n📊 RESULTS:");
    println!("===========");
    println!("Binder cumulant U_4 = {:.3} ± {:.3}", mean_binder, std_binder);
    println!("Susceptibility χ = {:.1} ± {:.1}", mean_chi, std_chi);
    
    println!("\n📋 COMPARISON WITH PAPER:");
    println!("========================");
    println!("Paper: U_4(48) = 0.615 ± 0.006");
    println!("Us:    U_4(48) = {:.3} ± {:.3}", mean_binder, std_binder);
    
    let deviation = ((mean_binder - 0.615) / 0.006).abs();
    if deviation < 3.0 {
        println!("✅ PASS: Within {} sigma of paper's value", deviation as i32);
    } else {
        println!("❌ FAIL: {:.1} sigma deviation from paper", deviation);
    }
    
    // Save raw data
    let mut file = File::create("critical_point_validation.dat").unwrap();
    writeln!(file, "# Replica U_4 chi |M|").unwrap();
    for i in 0..n_replicas {
        writeln!(file, "{} {:.6} {:.3} {:.6}", 
            i+1, all_binder_values[i], all_chi_values[i], all_mean_cos_values[i].abs()).unwrap();
    }
    
    println!("\nData saved to critical_point_validation.dat");
}