// Phase diagram scan to find critical point
use scan::graph_fast::{FastGraph, BatchedObservables};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("🗺️ PHASE DIAGRAM SCAN");
    println!("====================");
    
    let n = 48;
    let beta_min = 2.7;
    let beta_max = 3.0;
    let beta_step = 0.02;
    
    let alpha_min = 1.3;
    let alpha_max = 1.6;
    let alpha_step = 0.02;
    
    let equilibration_steps = 50_000;
    let production_steps = 50_000;
    let measure_interval = 100;
    
    println!("System: N={}", n);
    println!("β range: [{:.2}, {:.2}] step {:.2}", beta_min, beta_max, beta_step);
    println!("α range: [{:.2}, {:.2}] step {:.2}", alpha_min, alpha_max, alpha_step);
    println!("Steps: {}k eq + {}k prod per point", 
        equilibration_steps/1000, production_steps/1000);
    
    let beta_values: Vec<f64> = (0..=((beta_max - beta_min) / beta_step) as usize)
        .map(|i| beta_min + i as f64 * beta_step)
        .collect();
    
    let alpha_values: Vec<f64> = (0..=((alpha_max - alpha_min) / alpha_step) as usize)
        .map(|i| alpha_min + i as f64 * alpha_step)
        .collect();
    
    let total_points = beta_values.len() * alpha_values.len();
    println!("Total grid points: {}", total_points);
    
    let mut csv_file = File::create("phase_diagram.csv").unwrap();
    writeln!(csv_file, "beta,alpha,chi,chi_err,binder,mean_cos,acceptance").unwrap();
    
    let mut max_chi = 0.0;
    let mut max_chi_beta = 0.0;
    let mut max_chi_alpha = 0.0;
    
    let mut point_number = 0;
    
    for &beta in &beta_values {
        for &alpha in &alpha_values {
            point_number += 1;
            print!("[{}/{}] β={:.2}, α={:.2}: ", point_number, total_points, beta, alpha);
            std::io::stdout().flush().unwrap();
            
            let mut rng = Pcg64::seed_from_u64(42);
            let mut graph = FastGraph::new(n, 12345);
            
            // Fixed move sizes for consistency
            let delta_z = 0.5;
            let delta_theta = 0.5;
            
            // Equilibration
            let mut accepted = 0;
            for _ in 0..equilibration_steps {
                let info = graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
                if info.accept {
                    accepted += 1;
                }
            }
            let eq_acceptance = accepted as f64 / equilibration_steps as f64 * 100.0;
            
            // Production with measurements
            let mut chi_measurements = Vec::new();
            let mut cos_theta_values = Vec::new();
            let mut observable_calc = BatchedObservables::new();
            
            accepted = 0;
            for step in 0..production_steps {
                let info = graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
                if info.accept {
                    accepted += 1;
                }
                
                if step % measure_interval == 0 {
                    let obs = observable_calc.measure(&graph, alpha, beta);
                    chi_measurements.push(obs.susceptibility);
                    cos_theta_values.push(obs.mean_cos);
                }
            }
            let prod_acceptance = accepted as f64 / production_steps as f64 * 100.0;
            
            // Calculate statistics
            let mean_chi = chi_measurements.iter().sum::<f64>() / chi_measurements.len() as f64;
            let chi_err = (chi_measurements.iter()
                .map(|&x| (x - mean_chi).powi(2))
                .sum::<f64>() / (chi_measurements.len() - 1) as f64).sqrt()
                / (chi_measurements.len() as f64).sqrt();
            
            let mean_cos = cos_theta_values.iter().sum::<f64>() / cos_theta_values.len() as f64;
            
            // Binder cumulant
            let m2_values: Vec<f64> = cos_theta_values.iter().map(|&m| m * m).collect();
            let m4_values: Vec<f64> = cos_theta_values.iter().map(|&m| m.powi(4)).collect();
            let mean_m2 = m2_values.iter().sum::<f64>() / m2_values.len() as f64;
            let mean_m4 = m4_values.iter().sum::<f64>() / m4_values.len() as f64;
            let binder = 1.0 - mean_m4 / (3.0 * mean_m2.powi(2));
            
            println!("χ={:.1}±{:.1}, U₄={:.3}, acc={:.0}%", 
                mean_chi, chi_err, binder, prod_acceptance);
            
            writeln!(csv_file, "{:.3},{:.3},{:.3},{:.3},{:.6},{:.6},{:.1}",
                beta, alpha, mean_chi, chi_err, binder, mean_cos, prod_acceptance).unwrap();
            
            if mean_chi > max_chi {
                max_chi = mean_chi;
                max_chi_beta = beta;
                max_chi_alpha = alpha;
            }
        }
    }
    
    csv_file.flush().unwrap();
    
    println!("\n📊 RESULTS:");
    println!("===========");
    println!("Maximum susceptibility: χ = {:.1}", max_chi);
    println!("Located at: (β, α) = ({:.2}, {:.2})", max_chi_beta, max_chi_alpha);
    println!("\nData saved to phase_diagram.csv");
    
    // Create Python plotting script
    let mut plot_script = File::create("plot_phase_diagram.py").unwrap();
    writeln!(plot_script, r#"#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read data
df = pd.read_csv('phase_diagram.csv')

# Create pivot tables for heatmaps
chi_pivot = df.pivot(index='alpha', columns='beta', values='chi')
binder_pivot = df.pivot(index='alpha', columns='beta', values='binder')

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Susceptibility heatmap
im1 = ax1.imshow(chi_pivot, origin='lower', aspect='auto', 
                 extent=[df['beta'].min(), df['beta'].max(), 
                        df['alpha'].min(), df['alpha'].max()],
                 cmap='hot')
ax1.set_xlabel('β')
ax1.set_ylabel('α')
ax1.set_title('Susceptibility χ')
plt.colorbar(im1, ax=ax1)

# Mark paper's critical point
ax1.plot(2.88, 1.48, 'b*', markersize=15, label='Paper (2.88, 1.48)')

# Mark our maximum
max_idx = df['chi'].idxmax()
ax1.plot(df.loc[max_idx, 'beta'], df.loc[max_idx, 'alpha'], 'w+', 
         markersize=15, markeredgewidth=2, label=f'Our max ({{df.loc[max_idx, "beta"]:.2f}}, {{df.loc[max_idx, "alpha"]:.2f}})')
ax1.legend()

# Binder cumulant heatmap
im2 = ax2.imshow(binder_pivot, origin='lower', aspect='auto',
                 extent=[df['beta'].min(), df['beta'].max(), 
                        df['alpha'].min(), df['alpha'].max()],
                 cmap='viridis', vmin=0, vmax=0.7)
ax2.set_xlabel('β')
ax2.set_ylabel('α')
ax2.set_title('Binder Cumulant U₄')
plt.colorbar(im2, ax=ax2)

# Add contour line at U4=0.615
CS = ax2.contour(chi_pivot.columns, chi_pivot.index, binder_pivot, 
                 levels=[0.615], colors='red', linewidths=2)
ax2.clabel(CS, inline=1, fontsize=10)

plt.tight_layout()
plt.savefig('phase_diagram.png', dpi=150)
plt.show()

# Print critical point estimates
print(f"Paper critical point: (β, α) = (2.88, 1.48)")
print(f"Our χ maximum at: (β, α) = ({{df.loc[max_idx, 'beta']:.3f}}, {{df.loc[max_idx, 'alpha']:.3f}})")
print(f"χ at paper's point: {{df[(df['beta']==2.88) & (df['alpha']==1.48)]['chi'].values}}")
"#).unwrap();
    
    println!("\nPlot script saved to plot_phase_diagram.py");
    println!("Run: python plot_phase_diagram.py");
}