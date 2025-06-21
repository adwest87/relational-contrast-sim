// Test thermalization at different temperatures
use scan::graph_fast::{FastGraph, BatchedObservables};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("🌡️ THERMALIZATION TEST");
    println!("=====================");
    println!("Testing high T → low T behavior");
    
    let n = 48;
    let alpha = 1.5; // Fixed coupling ratio
    
    // Test temperatures from high to low
    let test_points = vec![
        (0.1, "Very high T"),
        (0.5, "High T"),
        (1.0, "Medium-high T"),
        (2.0, "Medium T"),
        (5.0, "Medium-low T"),
        (10.0, "Low T"),
        (50.0, "Very low T"),
    ];
    
    let equilibration_steps = 100_000;
    let production_steps = 100_000;
    let measure_interval = 100;
    
    println!("System: N={}, α={}", n, alpha);
    println!("Steps: {}k equilibration + {}k production\n", 
        equilibration_steps/1000, production_steps/1000);
    
    let mut csv_file = File::create("thermalization_test.csv").unwrap();
    writeln!(csv_file, "beta,T,mean_cos,std_cos,mean_abs_cos,chi,binder,label").unwrap();
    
    for (beta, label) in test_points {
        let temperature = 1.0 / beta;
        print!("{} (β={:.1}, T={:.2}): ", label, beta, temperature);
        std::io::stdout().flush().unwrap();
        
        let mut rng = Pcg64::seed_from_u64(42);
        let mut graph = FastGraph::new(n, 12345);
        
        // Equilibration
        print!("equilibrating...");
        std::io::stdout().flush().unwrap();
        
        for _ in 0..equilibration_steps {
            graph.metropolis_step(alpha, beta, 0.5, 0.5, &mut rng);
        }
        
        // Production
        print!(" measuring...");
        std::io::stdout().flush().unwrap();
        
        let mut cos_theta_values = Vec::new();
        let mut chi_values = Vec::new();
        let mut observable_calc = BatchedObservables::new();
        
        for step in 0..production_steps {
            graph.metropolis_step(alpha, beta, 0.5, 0.5, &mut rng);
            
            if step % measure_interval == 0 {
                let obs = observable_calc.measure(&graph, alpha, beta);
                cos_theta_values.push(obs.mean_cos);
                chi_values.push(obs.susceptibility);
            }
        }
        
        // Calculate statistics
        let mean_cos = cos_theta_values.iter().sum::<f64>() / cos_theta_values.len() as f64;
        let mean_abs_cos = cos_theta_values.iter().map(|x| x.abs()).sum::<f64>() 
            / cos_theta_values.len() as f64;
        let std_cos = (cos_theta_values.iter()
            .map(|&x| (x - mean_cos).powi(2))
            .sum::<f64>() / cos_theta_values.len() as f64).sqrt();
        
        let mean_chi = chi_values.iter().sum::<f64>() / chi_values.len() as f64;
        
        // Binder cumulant
        let m2_values: Vec<f64> = cos_theta_values.iter().map(|&m| m * m).collect();
        let m4_values: Vec<f64> = cos_theta_values.iter().map(|&m| m.powi(4)).collect();
        let mean_m2 = m2_values.iter().sum::<f64>() / m2_values.len() as f64;
        let mean_m4 = m4_values.iter().sum::<f64>() / m4_values.len() as f64;
        let binder = 1.0 - mean_m4 / (3.0 * mean_m2.powi(2));
        
        println!("\n  <cos θ> = {:.4} ± {:.4}", mean_cos, std_cos);
        println!("  <|cos θ|> = {:.4}", mean_abs_cos);
        println!("  χ = {:.2}, U₄ = {:.3}", mean_chi, binder);
        
        writeln!(csv_file, "{:.1},{:.3},{:.6},{:.6},{:.6},{:.3},{:.3},{}",
            beta, temperature, mean_cos, std_cos, mean_abs_cos, mean_chi, binder, label).unwrap();
        
        // Check physical behavior
        if beta < 1.0 && mean_abs_cos > 0.1 {
            println!("  ⚠️ Warning: Order parameter too large for high T!");
        }
        if beta > 10.0 && mean_abs_cos < 0.5 {
            println!("  ⚠️ Warning: Order parameter too small for low T!");
        }
    }
    
    println!("\n📊 PHYSICS CHECKS:");
    println!("==================");
    
    // Read back the data for analysis
    println!("\n1. Temperature dependence:");
    println!("   - High T (β→0): <|M|> should → 0 (disorder)");
    println!("   - Low T (β→∞): <|M|> should → 1 (order)");
    
    println!("\n2. Binder cumulant:");
    println!("   - High T: U₄ → 0 (Gaussian fluctuations)");
    println!("   - Low T: U₄ → 2/3 (ordered state)");
    println!("   - Critical point: U₄ ≈ 0.61 (3D Ising)");
    
    println!("\nData saved to thermalization_test.csv");
    
    // Create plotting script
    let mut plot_script = File::create("plot_thermalization.py").unwrap();
    writeln!(plot_script, r#"#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('thermalization_test.csv')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Order parameter vs temperature
ax1.semilogx(df['T'], df['mean_abs_cos'], 'o-', markersize=8)
ax1.set_xlabel('Temperature T')
ax1.set_ylabel('<|cos θ|>')
ax1.set_title('Order Parameter')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# Susceptibility vs temperature
ax2.loglog(df['T'], df['chi'], 'o-', markersize=8)
ax2.set_xlabel('Temperature T')
ax2.set_ylabel('χ')
ax2.set_title('Susceptibility')
ax2.grid(True, alpha=0.3)

# Binder cumulant vs temperature
ax3.semilogx(df['T'], df['binder'], 'o-', markersize=8)
ax3.axhline(y=2/3, color='red', linestyle='--', label='2/3 (ordered)')
ax3.axhline(y=0.61, color='green', linestyle='--', label='0.61 (3D Ising)')
ax3.set_xlabel('Temperature T')
ax3.set_ylabel('U₄')
ax3.set_title('Binder Cumulant')
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_ylim(-0.1, 0.8)

plt.tight_layout()
plt.savefig('thermalization_test.png', dpi=150)
plt.show()

# Print summary
print("Temperature limits:")
print(f"High T (T=10): <|M|> = {{df[df['T']==10]['mean_abs_cos'].values[0]:.3f}}")
print(f"Low T (T=0.02): <|M|> = {{df[df['T']==0.02]['mean_abs_cos'].values[0]:.3f}}")
"#).unwrap();
    
    println!("Plot with: python plot_thermalization.py");
}