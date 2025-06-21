// Comprehensive critical ridge exploration with full error analysis
// Uses UltraOptimizedGraph for maximum performance

use scan::graph_ultra_optimized::UltraOptimizedGraph;
use scan::error_analysis::ErrorAnalysis;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

#[derive(Debug, Clone)]
struct MeasurementPoint {
    alpha: f64,
    beta: f64,
    n: usize,
    
    // Primary observables
    mean_action: f64,
    mean_action_err: f64,
    
    susceptibility: f64,
    susceptibility_err: f64,
    
    specific_heat: f64,
    specific_heat_err: f64,
    
    binder_cumulant: f64,
    binder_cumulant_err: f64,
    
    correlation_length: f64,
    correlation_length_err: f64,
    
    // Magnetization (order parameter)
    magnetization: f64,
    magnetization_err: f64,
    
    // Thermodynamic quantities
    entropy: f64,
    triangle_sum: f64,
    
    // Statistics
    n_samples: usize,
    tau_int: f64,
    n_eff: f64,
    acceptance_rate: f64,
}

struct AdaptiveMC {
    graph: UltraOptimizedGraph,
    delta_z: f64,
    delta_theta: f64,
    target_acceptance: f64,
    adaptation_interval: usize,
}

impl AdaptiveMC {
    fn new(n: usize, seed: u64) -> Self {
        Self {
            graph: UltraOptimizedGraph::new(n, seed),
            delta_z: 0.1,
            delta_theta: 0.1,
            target_acceptance: 0.5,
            adaptation_interval: 100,
        }
    }
    
    fn tune_parameters(&mut self, acceptance_rate: f64) {
        let factor = if acceptance_rate > self.target_acceptance + 0.1 {
            1.1  // Increase step size
        } else if acceptance_rate < self.target_acceptance - 0.1 {
            0.9  // Decrease step size
        } else {
            1.0  // Keep current
        };
        
        self.delta_z *= factor;
        self.delta_theta *= factor;
        
        // Keep within reasonable bounds
        self.delta_z = self.delta_z.clamp(0.01, 1.0);
        self.delta_theta = self.delta_theta.clamp(0.01, 1.0);
    }
    
    fn run_measurement(&mut self, alpha: f64, beta: f64, n_therm: usize, n_measure: usize, rng: &mut Pcg64) -> MeasurementPoint {
        let n = self.graph.n();
        
        // Thermalization with parameter tuning
        let mut accepts = 0;
        for i in 0..n_therm {
            let accept = self.graph.metropolis_step(alpha, beta, 0.0, self.delta_z, self.delta_theta, rng);
            if accept { accepts += 1; }
            
            // Tune parameters every adaptation_interval steps
            if i > 0 && i % self.adaptation_interval == 0 {
                let acc_rate = accepts as f64 / self.adaptation_interval as f64;
                self.tune_parameters(acc_rate);
                accepts = 0;
            }
        }
        
        // Measurement phase with proper statistics
        let mut action_samples = Vec::new();
        let mut magnetization_samples = Vec::new();
        let mut entropy_samples = Vec::new();
        let mut triangle_samples = Vec::new();
        
        let mut total_accepts = 0;
        
        for _ in 0..n_measure {
            let accept = self.graph.metropolis_step(alpha, beta, 0.0, self.delta_z, self.delta_theta, rng);
            if accept { total_accepts += 1; }
            
            // Sample observables
            let action = self.graph.action(alpha, beta, 0.0);
            let entropy = self.graph.z_values.iter()
                .zip(&self.graph.exp_neg_z)
                .map(|(&z, &w)| -z * w)
                .sum::<f64>();
            let triangle_sum = self.graph.triangle_sum();
            
            // Calculate magnetization (order parameter)
            let sum_cos: f64 = self.graph.cos_theta.iter().sum();
            let magnetization = sum_cos / n as f64;  // Normalized by number of nodes
            
            action_samples.push(action);
            magnetization_samples.push(magnetization);
            entropy_samples.push(entropy);
            triangle_samples.push(triangle_sum);
        }
        
        let acceptance_rate = total_accepts as f64 / n_measure as f64;
        
        // Error analysis using our sophisticated methods
        let action_analysis = ErrorAnalysis::new(action_samples.clone());
        let mag_analysis = ErrorAnalysis::new(magnetization_samples.clone());
        let entropy_analysis = ErrorAnalysis::new(entropy_samples.clone());
        
        let action_errors = action_analysis.errors();
        let mag_errors = mag_analysis.errors();
        
        // Calculate derived observables with proper error propagation
        let (susceptibility, susceptibility_err) = calculate_susceptibility(&magnetization_samples, n);
        let (specific_heat, specific_heat_err) = calculate_specific_heat(&action_samples, beta, n);
        let (binder_cumulant, binder_cumulant_err) = calculate_binder_cumulant(&magnetization_samples);
        
        // Rough correlation length estimate (simplified for complete graph)
        let correlation_length = estimate_correlation_length(susceptibility, n);
        let correlation_length_err = correlation_length * susceptibility_err / susceptibility.max(1e-10);
        
        MeasurementPoint {
            alpha,
            beta,
            n,
            mean_action: action_analysis.mean(),
            mean_action_err: action_errors.stat_error,
            susceptibility,
            susceptibility_err,
            specific_heat,
            specific_heat_err,
            binder_cumulant,
            binder_cumulant_err,
            correlation_length,
            correlation_length_err,
            magnetization: mag_analysis.mean(),
            magnetization_err: mag_errors.stat_error,
            entropy: entropy_analysis.mean(),
            triangle_sum: triangle_samples.iter().sum::<f64>() / triangle_samples.len() as f64,
            n_samples: n_measure,
            tau_int: action_errors.tau_int,
            n_eff: action_errors.n_eff,
            acceptance_rate,
        }
    }
}

fn calculate_susceptibility(magnetization_samples: &[f64], n: usize) -> (f64, f64) {
    if magnetization_samples.len() < 2 {
        return (0.0, 0.0);
    }
    
    let mean_m = magnetization_samples.iter().sum::<f64>() / magnetization_samples.len() as f64;
    let mean_m2 = magnetization_samples.iter().map(|&m| m * m).sum::<f64>() / magnetization_samples.len() as f64;
    
    let susceptibility = n as f64 * (mean_m2 - mean_m * mean_m);
    
    // Jackknife error estimation
    let n_samples = magnetization_samples.len();
    let mut jackknife_estimates = Vec::new();
    
    for i in 0..n_samples {
        let sum_without_i: f64 = magnetization_samples.iter().enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &m)| m)
            .sum();
        let sum2_without_i: f64 = magnetization_samples.iter().enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &m)| m * m)
            .sum();
        
        let mean_jack = sum_without_i / (n_samples - 1) as f64;
        let mean2_jack = sum2_without_i / (n_samples - 1) as f64;
        let chi_jack = n as f64 * (mean2_jack - mean_jack * mean_jack);
        jackknife_estimates.push(chi_jack);
    }
    
    let jack_mean = jackknife_estimates.iter().sum::<f64>() / n_samples as f64;
    let jack_var = jackknife_estimates.iter()
        .map(|&x| (x - jack_mean).powi(2))
        .sum::<f64>() * (n_samples - 1) as f64 / n_samples as f64;
    let error = jack_var.sqrt();
    
    (susceptibility, error)
}

fn calculate_specific_heat(action_samples: &[f64], _beta: f64, n: usize) -> (f64, f64) {
    if action_samples.len() < 2 {
        return (0.0, 0.0);
    }
    
    let mean_e = action_samples.iter().sum::<f64>() / action_samples.len() as f64;
    let mean_e2 = action_samples.iter().map(|&e| e * e).sum::<f64>() / action_samples.len() as f64;
    
    let specific_heat = (mean_e2 - mean_e * mean_e) / (n as f64);
    
    // Simple error estimate (could be improved with jackknife)
    let var_e = mean_e2 - mean_e * mean_e;
    let error = (2.0 * var_e / (action_samples.len() as f64).sqrt()) / n as f64;
    
    (specific_heat, error)
}

fn calculate_binder_cumulant(magnetization_samples: &[f64]) -> (f64, f64) {
    if magnetization_samples.len() < 2 {
        return (0.0, 0.0);
    }
    
    let mean_m2 = magnetization_samples.iter().map(|&m| m * m).sum::<f64>() / magnetization_samples.len() as f64;
    let mean_m4 = magnetization_samples.iter().map(|&m| m.powi(4)).sum::<f64>() / magnetization_samples.len() as f64;
    
    let binder = if mean_m2 > 1e-10 {
        1.0 - mean_m4 / (3.0 * mean_m2 * mean_m2)
    } else {
        0.0
    };
    
    // Rough error estimate
    let error = 0.1 * binder.abs().max(0.01);  // 10% relative error or minimum absolute
    
    (binder, error)
}

fn estimate_correlation_length(susceptibility: f64, n: usize) -> f64 {
    // For complete graph: ξ ~ sqrt(χ/N) with system size cutoff
    let xi_raw = (susceptibility / n as f64).sqrt();
    let system_scale = (n as f64).powf(0.5);  // Effective 2D-like scaling
    xi_raw.min(system_scale / 2.0).max(0.1)
}

fn main() {
    println!("=== Critical Ridge Exploration with Full Error Analysis ===\n");
    
    let start_time = Instant::now();
    
    // System sizes to explore
    let system_sizes = vec![8, 12, 16, 20, 24];
    
    // Create parameter grid around expected critical ridge
    // Based on previous work: ridge approximately at α ≈ 0.06β + 1.31
    let beta_range: Vec<f64> = (10..=50).map(|i| i as f64 * 0.1).collect();  // 1.0 to 5.0
    
    let mut all_results = Vec::new();
    
    for &n in &system_sizes {
        println!("Exploring system size N = {}...", n);
        
        // Adaptive sampling along the ridge
        for &beta in &beta_range {
            // Sample around the predicted ridge
            let alpha_center = 0.06 * beta + 1.31;
            let alpha_range: Vec<f64> = (-10..=10)
                .map(|i| alpha_center + i as f64 * 0.05)
                .filter(|&a| a > 0.0)
                .collect();
            
            for &alpha in &alpha_range {
                let mut mc = AdaptiveMC::new(n, 42 + (alpha * 1000.0) as u64 + (beta * 1000.0) as u64);
                let mut rng = Pcg64::seed_from_u64(12345 + (n * 1000) as u64);
                
                // Adaptive sampling based on system size
                let n_therm = 2000 * n.min(16);  // More thermalization for larger systems
                let n_measure = 5000 * n.min(12); // Sufficient statistics
                
                println!("  Processing α={:.3}, β={:.3} (therm={}, measure={})", 
                         alpha, beta, n_therm, n_measure);
                
                let result = mc.run_measurement(alpha, beta, n_therm, n_measure, &mut rng);
                
                // Print key indicators for this point
                println!("    χ={:.4}±{:.4}, C={:.4}±{:.4}, U4={:.3}±{:.3}, acc={:.1}%", 
                         result.susceptibility, result.susceptibility_err,
                         result.specific_heat, result.specific_heat_err,
                         result.binder_cumulant, result.binder_cumulant_err,
                         100.0 * result.acceptance_rate);
                
                all_results.push(result);
            }
        }
        
        println!("  Completed N = {} in {:.2} minutes\n", n, start_time.elapsed().as_secs_f64() / 60.0);
    }
    
    // Save detailed results
    save_results(&all_results, "critical_ridge_data.csv").expect("Failed to save results");
    
    // Analyze for critical behavior
    analyze_critical_behavior(&all_results);
    
    // Generate error budget report
    generate_error_report(&all_results);
    
    println!("Total exploration time: {:.2} minutes", start_time.elapsed().as_secs_f64() / 60.0);
    println!("Results saved to: critical_ridge_data.csv");
}

fn save_results(results: &[MeasurementPoint], filename: &str) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    
    // Header
    writeln!(file, "alpha,beta,n,action,action_err,susceptibility,susceptibility_err,specific_heat,specific_heat_err,binder_cumulant,binder_cumulant_err,correlation_length,correlation_length_err,magnetization,magnetization_err,entropy,triangle_sum,n_samples,tau_int,n_eff,acceptance_rate")?;
    
    // Data
    for r in results {
        writeln!(file, "{:.6},{:.6},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6}",
                 r.alpha, r.beta, r.n, 
                 r.mean_action, r.mean_action_err,
                 r.susceptibility, r.susceptibility_err,
                 r.specific_heat, r.specific_heat_err,
                 r.binder_cumulant, r.binder_cumulant_err,
                 r.correlation_length, r.correlation_length_err,
                 r.magnetization, r.magnetization_err,
                 r.entropy, r.triangle_sum,
                 r.n_samples, r.tau_int, r.n_eff, r.acceptance_rate)?;
    }
    
    Ok(())
}

fn analyze_critical_behavior(results: &[MeasurementPoint]) {
    println!("\n=== Critical Behavior Analysis ===");
    
    // Find susceptibility peaks for each system size
    let mut peak_points = Vec::new();
    
    for &n in &[8, 12, 16, 20, 24] {
        let size_results: Vec<_> = results.iter().filter(|r| r.n == n).collect();
        
        if let Some(max_point) = size_results.iter().max_by(|a, b| 
            a.susceptibility.partial_cmp(&b.susceptibility).unwrap()) {
            peak_points.push(*max_point);
            println!("N={}: Peak susceptibility χ={:.4}±{:.4} at α={:.3}, β={:.3}", 
                     n, max_point.susceptibility, max_point.susceptibility_err,
                     max_point.alpha, max_point.beta);
        }
    }
    
    // Analyze the ridge
    if peak_points.len() >= 3 {
        println!("\nCritical Ridge Analysis:");
        println!("Size   α_peak   β_peak   χ_max      U4         ξ");
        println!("----   ------   ------   -----      --         --");
        
        for point in &peak_points {
            println!("{:4}   {:.3}    {:.3}    {:.3}±{:.3}  {:.3}±{:.3}  {:.2}±{:.2}",
                     point.n, point.alpha, point.beta,
                     point.susceptibility, point.susceptibility_err,
                     point.binder_cumulant, point.binder_cumulant_err,
                     point.correlation_length, point.correlation_length_err);
        }
        
        // Fit ridge: α = a*β + b
        let n_points = peak_points.len();
        let sum_beta: f64 = peak_points.iter().map(|p| p.beta).sum();
        let sum_alpha: f64 = peak_points.iter().map(|p| p.alpha).sum();
        let sum_beta2: f64 = peak_points.iter().map(|p| p.beta * p.beta).sum();
        let sum_alpha_beta: f64 = peak_points.iter().map(|p| p.alpha * p.beta).sum();
        
        let slope = (n_points as f64 * sum_alpha_beta - sum_alpha * sum_beta) / 
                   (n_points as f64 * sum_beta2 - sum_beta * sum_beta);
        let intercept = (sum_alpha - slope * sum_beta) / n_points as f64;
        
        println!("\nCritical Ridge Fit: α = {:.4}β + {:.4}", slope, intercept);
        
        // Check for quantum spin liquid signatures
        let avg_binder = peak_points.iter().map(|p| p.binder_cumulant).sum::<f64>() / n_points as f64;
        if avg_binder < 0.5 {
            println!("⚠ Unusual Binder cumulant (<0.5): possible quantum spin liquid behavior");
        }
    }
}

fn generate_error_report(results: &[MeasurementPoint]) {
    println!("\n=== Error Analysis Report ===");
    
    // Statistical quality metrics
    let total_points = results.len();
    let avg_n_eff: f64 = results.iter().map(|r| r.n_eff).sum::<f64>() / total_points as f64;
    let avg_tau: f64 = results.iter().map(|r| r.tau_int).sum::<f64>() / total_points as f64;
    let avg_acceptance: f64 = results.iter().map(|r| r.acceptance_rate).sum::<f64>() / total_points as f64;
    
    println!("Data Quality Summary:");
    println!("  Total measurement points: {}", total_points);
    println!("  Average N_eff: {:.1}", avg_n_eff);
    println!("  Average τ_int: {:.2}", avg_tau);
    println!("  Average acceptance: {:.1}%", 100.0 * avg_acceptance);
    
    // Error magnitude analysis
    let susceptibility_relative_errors: Vec<_> = results.iter()
        .filter(|r| r.susceptibility > 0.0)
        .map(|r| r.susceptibility_err / r.susceptibility)
        .collect();
    
    if !susceptibility_relative_errors.is_empty() {
        let avg_rel_err = susceptibility_relative_errors.iter().sum::<f64>() / susceptibility_relative_errors.len() as f64;
        let max_rel_err = susceptibility_relative_errors.iter().fold(0.0f64, |a, &b| a.max(b));
        
        println!("\nSusceptibility Error Analysis:");
        println!("  Average relative error: {:.1}%", 100.0 * avg_rel_err);
        println!("  Maximum relative error: {:.1}%", 100.0 * max_rel_err);
        
        if avg_rel_err < 0.1 {
            println!("  ✓ Excellent statistical precision");
        } else if avg_rel_err < 0.2 {
            println!("  ✓ Good statistical precision");
        } else {
            println!("  ⚠ Consider increasing measurement statistics");
        }
    }
    
    // Finite-size effects estimate
    println!("\nFinite-Size Effects:");
    for &n in &[8, 12, 16, 20, 24] {
        let size_results: Vec<_> = results.iter().filter(|r| r.n == n).collect();
        if !size_results.is_empty() {
            let max_xi = size_results.iter().map(|r| r.correlation_length).fold(0.0f64, |a, b| a.max(b));
            let finite_size_parameter = max_xi / n as f64;
            
            print!("  N={}: ξ/L ~ {:.3}", n, finite_size_parameter);
            if finite_size_parameter < 0.1 {
                println!(" ✓");
            } else if finite_size_parameter < 0.3 {
                println!(" (moderate)");
            } else {
                println!(" ⚠ (large finite-size effects)");
            }
        }
    }
}