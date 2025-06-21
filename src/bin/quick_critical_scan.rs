// Quick critical scan to rapidly explore the critical ridge
// Focused on finding the critical points with reasonable statistics

use scan::graph_ultra_optimized::UltraOptimizedGraph;
use scan::error_analysis::ErrorAnalysis;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

#[derive(Debug, Clone)]
struct QuickResult {
    alpha: f64,
    beta: f64,
    n: usize,
    susceptibility: f64,
    susceptibility_err: f64,
    specific_heat: f64,
    binder_cumulant: f64,
    acceptance_rate: f64,
}

fn quick_measure(
    graph: &mut UltraOptimizedGraph, 
    alpha: f64, 
    beta: f64, 
    rng: &mut Pcg64
) -> QuickResult {
    let n = graph.n();
    
    // Short thermalization
    let n_therm = 500 * n;
    let n_measure = 1000 * n;
    
    // Adaptive step sizes
    let mut delta_z = 0.1;
    let mut delta_theta = 0.1;
    
    // Thermalization
    let mut accepts = 0;
    for i in 0..n_therm {
        let accept = graph.metropolis_step(alpha, beta, 0.0, delta_z, delta_theta, rng);
        if accept { accepts += 1; }
        
        if i > 0 && i % 100 == 0 {
            let acc_rate = accepts as f64 / 100.0;
            if acc_rate > 0.6 {
                delta_z *= 1.05;
                delta_theta *= 1.05;
            } else if acc_rate < 0.4 {
                delta_z *= 0.95;
                delta_theta *= 0.95;
            }
            delta_z = delta_z.clamp(0.01, 0.3);
            delta_theta = delta_theta.clamp(0.01, 0.3);
            accepts = 0;
        }
    }
    
    // Measurement
    let mut action_samples = Vec::with_capacity(n_measure);
    let mut magnetization_samples = Vec::with_capacity(n_measure);
    let mut total_accepts = 0;
    
    for _ in 0..n_measure {
        let accept = graph.metropolis_step(alpha, beta, 0.0, delta_z, delta_theta, rng);
        if accept { total_accepts += 1; }
        
        let action = graph.action(alpha, beta, 0.0);
        let magnetization = graph.cos_theta.iter().sum::<f64>() / n as f64;
        
        action_samples.push(action);
        magnetization_samples.push(magnetization);
    }
    
    let acceptance_rate = total_accepts as f64 / n_measure as f64;
    
    // Quick susceptibility calculation
    let mean_m = magnetization_samples.iter().sum::<f64>() / n_measure as f64;
    let mean_m2 = magnetization_samples.iter().map(|&m| m * m).sum::<f64>() / n_measure as f64;
    let susceptibility = n as f64 * (mean_m2 - mean_m * mean_m);
    let susceptibility_err = susceptibility * (2.0 / n_measure as f64).sqrt(); // Rough estimate
    
    // Quick specific heat
    let action_analysis = ErrorAnalysis::new(action_samples);
    let specific_heat = action_analysis.variance() / n as f64;
    
    // Quick Binder cumulant
    let mean_m4 = magnetization_samples.iter().map(|&m| m.powi(4)).sum::<f64>() / n_measure as f64;
    let binder_cumulant = if mean_m2 > 1e-10 {
        1.0 - mean_m4 / (3.0 * mean_m2 * mean_m2)
    } else {
        0.0
    };
    
    QuickResult {
        alpha,
        beta,
        n,
        susceptibility,
        susceptibility_err,
        specific_heat,
        binder_cumulant,
        acceptance_rate,
    }
}

fn main() {
    println!("=== Quick Critical Ridge Scan ===\n");
    
    let start_time = Instant::now();
    
    // Multiple system sizes for finite-size scaling
    let system_sizes = vec![8, 12, 16, 20, 24];
    
    // Focus on the critical ridge region
    let beta_values: Vec<f64> = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    
    let mut all_results = Vec::new();
    
    for &n in &system_sizes {
        println!("System size N = {}:", n);
        
        for &beta in &beta_values {
            // Expected ridge: α ≈ 0.06β + 1.31
            let alpha_center = 0.06 * beta + 1.31;
            
            // Scan around the ridge
            let alpha_offsets = vec![-0.15, -0.075, 0.0, 0.075, 0.15];
            
            for &offset in &alpha_offsets {
                let alpha = alpha_center + offset;
                if alpha <= 0.0 { continue; }
                
                let mut graph = UltraOptimizedGraph::new(n, 42 + (alpha * 1000.0) as u64);
                let mut rng = Pcg64::seed_from_u64(12345 + (n * 1000) as u64);
                
                let result = quick_measure(&mut graph, alpha, beta, &mut rng);
                
                println!("  β={:.1}, α={:.3}: χ={:.3}±{:.3}, C={:.3}, U4={:.3}, acc={:.1}%",
                         beta, alpha, result.susceptibility, result.susceptibility_err,
                         result.specific_heat, result.binder_cumulant, 100.0 * result.acceptance_rate);
                
                all_results.push(result);
            }
        }
        
        println!("  Completed N={} in {:.1}s\n", n, start_time.elapsed().as_secs_f64());
    }
    
    // Save results
    save_quick_results(&all_results, "quick_critical_data.csv").expect("Failed to save data");
    
    // Find critical points
    find_critical_points(&all_results);
    
    // Finite-size scaling analysis
    finite_size_analysis(&all_results);
    
    println!("Total time: {:.1} minutes", start_time.elapsed().as_secs_f64() / 60.0);
    println!("Results saved to: quick_critical_data.csv");
}

fn save_quick_results(results: &[QuickResult], filename: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    
    writeln!(file, "alpha,beta,n,susceptibility,susceptibility_err,specific_heat,binder_cumulant,acceptance_rate")?;
    
    for r in results {
        writeln!(file, "{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
                 r.alpha, r.beta, r.n, r.susceptibility, r.susceptibility_err,
                 r.specific_heat, r.binder_cumulant, r.acceptance_rate)?;
    }
    
    Ok(())
}

fn find_critical_points(results: &[QuickResult]) {
    println!("\n=== Critical Point Analysis ===");
    
    // Find susceptibility maxima for each (n, beta) combination
    let mut critical_points = Vec::new();
    
    for &n in &[8, 12, 16, 20, 24] {
        for &beta in &[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] {
            let subset: Vec<_> = results.iter()
                .filter(|r| r.n == n && (r.beta - beta).abs() < 0.01)
                .collect();
            
            if let Some(max_chi_point) = subset.iter()
                .max_by(|a, b| a.susceptibility.partial_cmp(&b.susceptibility).unwrap()) {
                critical_points.push((*max_chi_point).clone());
            }
        }
    }
    
    // Print critical points
    println!("Susceptibility Maxima (Critical Points):");
    println!("N    β     α_c      χ_max    C_max    U4       Error%");
    println!("---- ----- -------- -------- -------- -------- ------");
    
    for point in &critical_points {
        let rel_error = 100.0 * point.susceptibility_err / point.susceptibility;
        println!("{:4} {:5.1} {:8.3} {:8.3} {:8.3} {:8.3} {:6.1}",
                 point.n, point.beta, point.alpha, point.susceptibility,
                 point.specific_heat, point.binder_cumulant, rel_error);
    }
    
    // Fit critical line for each system size
    println!("\nCritical Ridge Fits:");
    for &n in &[8, 12, 16, 20, 24] {
        let size_points: Vec<_> = critical_points.iter().filter(|p| p.n == n).collect();
        
        if size_points.len() >= 3 {
            // Linear regression: α_c = a*β + b
            let n_pts = size_points.len() as f64;
            let sum_beta: f64 = size_points.iter().map(|p| p.beta).sum();
            let sum_alpha: f64 = size_points.iter().map(|p| p.alpha).sum();
            let sum_beta2: f64 = size_points.iter().map(|p| p.beta * p.beta).sum();
            let sum_alpha_beta: f64 = size_points.iter().map(|p| p.alpha * p.beta).sum();
            
            let slope = (n_pts * sum_alpha_beta - sum_alpha * sum_beta) / 
                       (n_pts * sum_beta2 - sum_beta * sum_beta);
            let intercept = (sum_alpha - slope * sum_beta) / n_pts;
            
            println!("N={:2}: α_c = {:.4}β + {:.4}", n, slope, intercept);
        }
    }
    
    // Check for quantum spin liquid signatures
    println!("\nQuantum Spin Liquid Analysis:");
    let large_system_u4: Vec<_> = critical_points.iter()
        .filter(|p| p.n >= 16)
        .map(|p| p.binder_cumulant)
        .collect();
    
    if !large_system_u4.is_empty() {
        let avg_u4 = large_system_u4.iter().sum::<f64>() / large_system_u4.len() as f64;
        let min_u4 = large_system_u4.iter().fold(1.0f64, |a, &b| a.min(b));
        let max_u4 = large_system_u4.iter().fold(-1.0f64, |a, &b| a.max(b));
        
        println!("Large systems (N≥16): <U4> = {:.3}, range = [{:.3}, {:.3}]", 
                 avg_u4, min_u4, max_u4);
        
        if avg_u4 < 0.5 {
            println!("🔬 UNUSUAL CRITICAL BEHAVIOR: U4 < 0.5 suggests quantum spin liquid!");
        } else if min_u4 < 0.0 {
            println!("⚠ Some negative U4 values detected - possible exotic phase");
        } else {
            println!("✓ Conventional critical behavior observed");
        }
    }
}

fn finite_size_analysis(results: &[QuickResult]) {
    println!("\n=== Finite-Size Scaling Analysis ===");
    
    // Analyze scaling of χ_max with system size for each β
    for &beta in &[1.5, 2.0, 2.5, 3.0] {  // Focus on mid-range
        println!("\nβ = {:.1}:", beta);
        println!("N     χ_max    χ/N^(γ/ν)  log(χ)");
        
        for &n in &[8, 12, 16, 20, 24] {
            let size_results: Vec<_> = results.iter()
                .filter(|r| r.n == n && (r.beta - beta).abs() < 0.01)
                .collect();
            
            if let Some(max_chi_point) = size_results.iter()
                .max_by(|a, b| a.susceptibility.partial_cmp(&b.susceptibility).unwrap()) {
                
                // Assume γ/ν ~ 1.75 for 2D Ising-like (rough estimate)
                let scaling_exponent = 1.75;
                let scaled_chi = max_chi_point.susceptibility / (n as f64).powf(scaling_exponent);
                let log_chi = max_chi_point.susceptibility.ln();
                
                println!("{:4}  {:8.3}  {:9.3}  {:6.3}",
                         n, max_chi_point.susceptibility, scaled_chi, log_chi);
            }
        }
    }
    
    // Check correlation length scaling
    println!("\nCorrelation Length Estimates:");
    println!("N    β     ξ/L      Comment");
    println!("---- ----- -------- -------");
    
    for &n in &[8, 12, 16, 20, 24] {
        for &beta in &[2.0, 2.5, 3.0] {
            let size_results: Vec<_> = results.iter()
                .filter(|r| r.n == n && (r.beta - beta).abs() < 0.01)
                .collect();
            
            if let Some(max_chi_point) = size_results.iter()
                .max_by(|a, b| a.susceptibility.partial_cmp(&b.susceptibility).unwrap()) {
                
                // Rough correlation length: ξ ~ sqrt(χ/N)
                let xi = (max_chi_point.susceptibility / n as f64).sqrt();
                let xi_over_l = xi / (n as f64).sqrt();  // Rough for complete graph
                
                let comment = if xi_over_l > 0.3 {
                    "Large ξ"
                } else if xi_over_l > 0.1 {
                    "Moderate ξ"
                } else {
                    "Small ξ"
                };
                
                println!("{:4} {:5.1} {:8.3} {}", n, beta, xi_over_l, comment);
            }
        }
    }
}