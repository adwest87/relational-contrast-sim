// Focused critical point scan with efficient sampling and full error analysis
// Targets the critical ridge region based on previous knowledge

use scan::graph_ultra_optimized::UltraOptimizedGraph;
use scan::error_analysis::ErrorAnalysis;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

#[derive(Debug, Clone)]
struct CriticalPoint {
    alpha: f64,
    beta: f64,
    n: usize,
    susceptibility: f64,
    susceptibility_err: f64,
    specific_heat: f64,
    specific_heat_err: f64,
    binder_cumulant: f64,
    correlation_length: f64,
    tau_int: f64,
    n_eff: f64,
    acceptance_rate: f64,
}

fn measure_observables(
    graph: &mut UltraOptimizedGraph, 
    alpha: f64, 
    beta: f64, 
    n_therm: usize, 
    n_measure: usize,
    rng: &mut Pcg64
) -> CriticalPoint {
    let n = graph.n();
    
    // Adaptive Monte Carlo for optimal sampling
    let mut delta_z = 0.1;
    let mut delta_theta = 0.1;
    
    // Thermalization with tuning
    let mut accepts = 0;
    for i in 0..n_therm {
        let accept = graph.metropolis_step(alpha, beta, 0.0, delta_z, delta_theta, rng);
        if accept { accepts += 1; }
        
        // Tune every 200 steps
        if i > 0 && i % 200 == 0 {
            let acc_rate = accepts as f64 / 200.0;
            if acc_rate > 0.6 {
                delta_z *= 1.05;
                delta_theta *= 1.05;
            } else if acc_rate < 0.4 {
                delta_z *= 0.95;
                delta_theta *= 0.95;
            }
            delta_z = delta_z.clamp(0.01, 0.5);
            delta_theta = delta_theta.clamp(0.01, 0.5);
            accepts = 0;
        }
    }
    
    // Measurement phase
    let mut action_samples = Vec::with_capacity(n_measure);
    let mut magnetization_samples = Vec::with_capacity(n_measure);
    let mut total_accepts = 0;
    
    for _ in 0..n_measure {
        let accept = graph.metropolis_step(alpha, beta, 0.0, delta_z, delta_theta, rng);
        if accept { total_accepts += 1; }
        
        // Measure observables
        let action = graph.action(alpha, beta, 0.0);
        let magnetization = graph.cos_theta.iter().sum::<f64>() / n as f64;
        
        action_samples.push(action);
        magnetization_samples.push(magnetization);
    }
    
    let acceptance_rate = total_accepts as f64 / n_measure as f64;
    
    // Error analysis
    let action_analysis = ErrorAnalysis::new(action_samples);
    
    let action_errors = action_analysis.errors();
    
    // Calculate susceptibility with jackknife
    let (susceptibility, susceptibility_err) = calculate_susceptibility_jackknife(&magnetization_samples, n);
    
    // Calculate specific heat
    let specific_heat = action_analysis.variance() / n as f64;
    let specific_heat_err = specific_heat * (2.0 / action_errors.n_eff).sqrt();
    
    // Calculate Binder cumulant
    let binder_cumulant = calculate_binder_cumulant(&magnetization_samples);
    
    // Estimate correlation length
    let correlation_length = (susceptibility / n as f64).sqrt().min((n as f64).sqrt() / 2.0);
    
    CriticalPoint {
        alpha,
        beta,
        n,
        susceptibility,
        susceptibility_err,
        specific_heat,
        specific_heat_err,
        binder_cumulant,
        correlation_length,
        tau_int: action_errors.tau_int,
        n_eff: action_errors.n_eff,
        acceptance_rate,
    }
}

fn calculate_susceptibility_jackknife(magnetization_samples: &[f64], n: usize) -> (f64, f64) {
    if magnetization_samples.len() < 10 {
        return (0.0, 0.0);
    }
    
    let n_samples = magnetization_samples.len();
    
    // Full estimate
    let mean_m = magnetization_samples.iter().sum::<f64>() / n_samples as f64;
    let mean_m2 = magnetization_samples.iter().map(|&m| m * m).sum::<f64>() / n_samples as f64;
    let full_chi = n as f64 * (mean_m2 - mean_m * mean_m);
    
    // Jackknife estimates
    let mut jack_estimates = Vec::new();
    for i in 0..n_samples {
        let sum_m: f64 = magnetization_samples.iter().enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &m)| m)
            .sum();
        let sum_m2: f64 = magnetization_samples.iter().enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &m)| m * m)
            .sum();
        
        let jack_mean_m = sum_m / (n_samples - 1) as f64;
        let jack_mean_m2 = sum_m2 / (n_samples - 1) as f64;
        let jack_chi = n as f64 * (jack_mean_m2 - jack_mean_m * jack_mean_m);
        jack_estimates.push(jack_chi);
    }
    
    // Jackknife error
    let jack_mean = jack_estimates.iter().sum::<f64>() / n_samples as f64;
    let jack_var = jack_estimates.iter()
        .map(|&x| (x - jack_mean).powi(2))
        .sum::<f64>() * (n_samples - 1) as f64 / n_samples as f64;
    
    (full_chi, jack_var.sqrt())
}

fn calculate_binder_cumulant(magnetization_samples: &[f64]) -> f64 {
    if magnetization_samples.len() < 2 {
        return 0.0;
    }
    
    let mean_m2 = magnetization_samples.iter().map(|&m| m * m).sum::<f64>() / magnetization_samples.len() as f64;
    let mean_m4 = magnetization_samples.iter().map(|&m| m.powi(4)).sum::<f64>() / magnetization_samples.len() as f64;
    
    if mean_m2 > 1e-10 {
        1.0 - mean_m4 / (3.0 * mean_m2 * mean_m2)
    } else {
        0.0
    }
}

fn main() {
    println!("=== Focused Critical Ridge Scan ===\n");
    
    let start_time = Instant::now();
    
    // Focus on multiple system sizes with targeted sampling
    let system_sizes = vec![8, 12, 16, 20, 24, 32];
    
    // Target the critical ridge region more precisely
    // Ridge approximately: α ≈ 0.06β + 1.31
    let beta_values: Vec<f64> = vec![1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0];
    
    let mut all_results = Vec::new();
    
    for &n in &system_sizes {
        println!("Scanning system size N = {}...", n);
        
        for &beta in &beta_values {
            let alpha_center = 0.06 * beta + 1.31;
            
            // Scan around the predicted ridge
            let alpha_offsets = vec![-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2];
            
            for &offset in &alpha_offsets {
                let alpha = alpha_center + offset;
                if alpha <= 0.0 { continue; }
                
                let mut graph = UltraOptimizedGraph::new(n, 42 + (alpha * 1000.0) as u64);
                let mut rng = Pcg64::seed_from_u64(12345 + (n * 1000) as u64);
                
                // Scale sampling with system size
                let n_therm = 1000 * n;
                let n_measure = 2000 * n.min(16);
                
                print!("  α={:.3}, β={:.3}: ", alpha, beta);
                
                let result = measure_observables(&mut graph, alpha, beta, n_therm, n_measure, &mut rng);
                
                println!("χ={:.3}±{:.3}, C={:.3}±{:.3}, U4={:.3}, ξ={:.2}, τ={:.1}, acc={:.1}%",
                         result.susceptibility, result.susceptibility_err,
                         result.specific_heat, result.specific_heat_err,
                         result.binder_cumulant, result.correlation_length,
                         result.tau_int, 100.0 * result.acceptance_rate);
                
                all_results.push(result);
            }
        }
        
        println!("  Completed N={} in {:.1}s\n", n, start_time.elapsed().as_secs_f64());
    }
    
    // Save results
    save_results(&all_results, "focused_critical_data.csv").expect("Failed to save data");
    
    // Find and analyze critical points
    analyze_critical_ridge(&all_results);
    
    // Error analysis summary
    error_analysis_summary(&all_results);
    
    println!("\nTotal scan time: {:.1} minutes", start_time.elapsed().as_secs_f64() / 60.0);
    println!("Results saved to: focused_critical_data.csv");
}

fn save_results(results: &[CriticalPoint], filename: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    
    writeln!(file, "alpha,beta,n,susceptibility,susceptibility_err,specific_heat,specific_heat_err,binder_cumulant,correlation_length,tau_int,n_eff,acceptance_rate")?;
    
    for r in results {
        writeln!(file, "{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                 r.alpha, r.beta, r.n, r.susceptibility, r.susceptibility_err,
                 r.specific_heat, r.specific_heat_err, r.binder_cumulant,
                 r.correlation_length, r.tau_int, r.n_eff, r.acceptance_rate)?;
    }
    
    Ok(())
}

fn analyze_critical_ridge(results: &[CriticalPoint]) {
    println!("\n=== Critical Ridge Analysis ===");
    
    // Find susceptibility maxima for each size and beta
    let mut peak_data = Vec::new();
    
    for &n in &[8, 12, 16, 20, 24, 32] {
        for &beta in &[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] {
            let size_beta_results: Vec<_> = results.iter()
                .filter(|r| r.n == n && (r.beta - beta).abs() < 0.01)
                .collect();
            
            if let Some(max_point) = size_beta_results.iter()
                .max_by(|a, b| a.susceptibility.partial_cmp(&b.susceptibility).unwrap()) {
                peak_data.push(max_point.clone());
            }
        }
    }
    
    println!("Susceptibility Peaks:");
    println!("N    β     α_peak   χ_max      C_max      U4        ξ/L");
    println!("---- ----- -------- ---------- ---------- --------- -----");
    
    for point in &peak_data {
        let xi_over_l = point.correlation_length / point.n as f64;
        println!("{:4} {:5.1} {:8.3} {:10.3} {:10.3} {:9.3} {:5.3}",
                 point.n, point.beta, point.alpha,
                 point.susceptibility, point.specific_heat,
                 point.binder_cumulant, xi_over_l);
    }
    
    // Analyze critical ridge scaling
    if peak_data.len() > 5 {
        println!("\nCritical Ridge Fit Analysis:");
        
        // Group by system size and fit α vs β
        for &n in &[8, 12, 16, 20, 24, 32] {
            let size_peaks: Vec<_> = peak_data.iter().filter(|p| p.n == n).collect();
            
            if size_peaks.len() >= 3 {
                // Linear fit: α = a*β + b
                let n_points = size_peaks.len() as f64;
                let sum_beta: f64 = size_peaks.iter().map(|p| p.beta).sum();
                let sum_alpha: f64 = size_peaks.iter().map(|p| p.alpha).sum();
                let sum_beta2: f64 = size_peaks.iter().map(|p| p.beta * p.beta).sum();
                let sum_alpha_beta: f64 = size_peaks.iter().map(|p| p.alpha * p.beta).sum();
                
                let slope = (n_points * sum_alpha_beta - sum_alpha * sum_beta) / 
                           (n_points * sum_beta2 - sum_beta * sum_beta);
                let intercept = (sum_alpha - slope * sum_beta) / n_points;
                
                // Calculate R²
                let mean_alpha = sum_alpha / n_points;
                let ss_tot: f64 = size_peaks.iter()
                    .map(|p| (p.alpha - mean_alpha).powi(2))
                    .sum();
                let ss_res: f64 = size_peaks.iter()
                    .map(|p| (p.alpha - (slope * p.beta + intercept)).powi(2))
                    .sum();
                let r_squared = 1.0 - ss_res / ss_tot;
                
                println!("N={:2}: α = {:.4}β + {:.4} (R² = {:.3})",
                         n, slope, intercept, r_squared);
            }
        }
        
        // Finite-size scaling analysis
        println!("\nFinite-Size Scaling:");
        println!("Checking for quantum spin liquid signatures...");
        
        let avg_binder_by_size: Vec<_> = [8, 12, 16, 20, 24, 32].iter()
            .map(|&n| {
                let size_peaks: Vec<_> = peak_data.iter().filter(|p| p.n == n).collect();
                let avg_u4 = if !size_peaks.is_empty() {
                    size_peaks.iter().map(|p| p.binder_cumulant).sum::<f64>() / size_peaks.len() as f64
                } else {
                    0.0
                };
                (n, avg_u4)
            })
            .collect();
        
        for (n, u4) in avg_binder_by_size {
            println!("N={:2}: <U4> = {:.3}", n, u4);
        }
        
        // Check for unusual critical behavior
        let large_system_u4: Vec<_> = peak_data.iter()
            .filter(|p| p.n >= 20)
            .map(|p| p.binder_cumulant)
            .collect();
        
        if !large_system_u4.is_empty() {
            let avg_u4_large = large_system_u4.iter().sum::<f64>() / large_system_u4.len() as f64;
            
            if avg_u4_large < 0.5 {
                println!("\n⚠ UNUSUAL CRITICAL BEHAVIOR DETECTED:");
                println!("  Average U4 = {:.3} < 0.5 for large systems", avg_u4_large);
                println!("  This suggests quantum spin liquid-like behavior!");
            } else {
                println!("\n✓ Conventional critical behavior: U4 ≈ {:.3}", avg_u4_large);
            }
        }
    }
}

fn error_analysis_summary(results: &[CriticalPoint]) {
    println!("\n=== Error Analysis Summary ===");
    
    let n_points = results.len();
    
    // Statistical quality metrics
    let avg_tau = results.iter().map(|r| r.tau_int).sum::<f64>() / n_points as f64;
    let avg_n_eff = results.iter().map(|r| r.n_eff).sum::<f64>() / n_points as f64;
    let avg_acceptance = results.iter().map(|r| r.acceptance_rate).sum::<f64>() / n_points as f64;
    
    println!("Overall Statistics:");
    println!("  Total points: {}", n_points);
    println!("  Average τ_int: {:.2}", avg_tau);
    println!("  Average N_eff: {:.1}", avg_n_eff);
    println!("  Average acceptance: {:.1}%", 100.0 * avg_acceptance);
    
    // Error analysis by system size
    println!("\nError Quality by System Size:");
    println!("N    <τ_int>  <N_eff>  <χ_err/χ>  Quality");
    println!("---- -------- -------- ---------- -------");
    
    for &n in &[8, 12, 16, 20, 24, 32] {
        let size_results: Vec<_> = results.iter().filter(|r| r.n == n).collect();
        
        if !size_results.is_empty() {
            let avg_tau = size_results.iter().map(|r| r.tau_int).sum::<f64>() / size_results.len() as f64;
            let avg_n_eff = size_results.iter().map(|r| r.n_eff).sum::<f64>() / size_results.len() as f64;
            
            let rel_errors: Vec<_> = size_results.iter()
                .filter(|r| r.susceptibility > 0.0)
                .map(|r| r.susceptibility_err / r.susceptibility)
                .collect();
            
            let avg_rel_err = if !rel_errors.is_empty() {
                rel_errors.iter().sum::<f64>() / rel_errors.len() as f64
            } else {
                0.0
            };
            
            let quality = if avg_rel_err < 0.05 {
                "Excellent"
            } else if avg_rel_err < 0.1 {
                "Good"
            } else if avg_rel_err < 0.2 {
                "Fair"
            } else {
                "Poor"
            };
            
            println!("{:4} {:8.2} {:8.1} {:10.3} {}", 
                     n, avg_tau, avg_n_eff, avg_rel_err, quality);
        }
    }
    
    // Recommend improvements
    let high_tau_points = results.iter().filter(|r| r.tau_int > 10.0).count();
    let low_neff_points = results.iter().filter(|r| r.n_eff < 100.0).count();
    
    if high_tau_points > n_points / 10 {
        println!("\n⚠ High autocorrelation detected in {}% of points", 
                 100 * high_tau_points / n_points);
        println!("  Consider increasing thermalization time");
    }
    
    if low_neff_points > n_points / 10 {
        println!("\n⚠ Low effective sample size in {}% of points", 
                 100 * low_neff_points / n_points);
        println!("  Consider increasing measurement time");
    }
    
    if high_tau_points <= n_points / 20 && low_neff_points <= n_points / 20 {
        println!("\n✓ Excellent statistical quality across all measurements");
    }
}