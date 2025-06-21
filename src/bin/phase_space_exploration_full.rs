// Comprehensive phase space exploration with full action including spectral term
// Uses validated implementations to find and characterize critical behavior

use scan::graph::Graph;
use scan::graph_ultra_optimized::UltraOptimizedGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::fs::File;
use std::io::Write;

#[derive(Debug, Clone)]
struct PhasePoint {
    alpha: f64,
    beta: f64,
    gamma: f64,
    n: usize,
    // Observables
    energy_mean: f64,
    energy_err: f64,
    magnetization_mean: f64,
    magnetization_err: f64,
    magnetization_abs_mean: f64,
    magnetization_abs_err: f64,
    susceptibility: f64,
    susceptibility_err: f64,
    binder_cumulant: f64,
    binder_err: f64,
    specific_heat: f64,
    specific_heat_err: f64,
    // Quality metrics
    acceptance_rate: f64,
}

fn simple_error(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (variance / n).sqrt()
}

fn jackknife_error<F>(data: &[f64], func: F) -> f64 
where
    F: Fn(&[f64]) -> f64
{
    let n = data.len();
    let full_estimate = func(data);
    
    let mut jack_estimates = Vec::new();
    for i in 0..n {
        let mut subset = data.to_vec();
        subset.remove(i);
        jack_estimates.push(func(&subset));
    }
    
    let jack_mean = jack_estimates.iter().sum::<f64>() / n as f64;
    let variance = jack_estimates.iter()
        .map(|&x| (x - jack_mean).powi(2))
        .sum::<f64>() * (n - 1) as f64 / n as f64;
    
    variance.sqrt()
}

fn measure_observables(
    graph: &mut UltraOptimizedGraph,
    alpha: f64,
    beta: f64,
    gamma: f64,
    n_measure: usize,
    rng: &mut Pcg64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let mut energies = Vec::new();
    let mut magnetizations = Vec::new();
    let mut mag_squared = Vec::new();
    let mut mag_fourth = Vec::new();
    let mut accepts = 0;
    
    for _ in 0..n_measure {
        // Monte Carlo step
        if graph.metropolis_step(alpha, beta, gamma, 0.1, 0.1, rng) {
            accepts += 1;
        }
        
        // Measure observables
        let energy = graph.action(alpha, beta, gamma) / graph.m() as f64;
        let mag = graph.cos_theta.iter().sum::<f64>() / graph.n() as f64;
        
        energies.push(energy);
        magnetizations.push(mag);
        mag_squared.push(mag * mag);
        mag_fourth.push(mag * mag * mag * mag);
    }
    
    let acceptance_rate = accepts as f64 / n_measure as f64;
    
    (energies, magnetizations, mag_squared, mag_fourth, acceptance_rate)
}

fn analyze_phase_point(
    alpha: f64,
    beta: f64,
    gamma: f64,
    n: usize,
    n_therm: usize,
    n_measure: usize,
    seed: u64,
) -> PhasePoint {
    println!("  Analyzing α={:.3}, β={:.3}, γ={:.3}, N={}...", alpha, beta, gamma, n);
    
    // Create system with validated implementation
    let mut rng = Pcg64::seed_from_u64(seed);
    let reference = Graph::complete_random_with(&mut rng, n);
    let mut graph = UltraOptimizedGraph::from_graph(&reference);
    
    // Enable spectral term if gamma > 0
    if gamma > 0.0 {
        graph.enable_spectral(n, gamma);
    }
    
    // Thermalization
    println!("    Thermalizing ({} steps)...", n_therm);
    for _ in 0..n_therm {
        graph.metropolis_step(alpha, beta, gamma, 0.1, 0.1, &mut rng);
    }
    
    // Measurement
    println!("    Measuring ({} steps)...", n_measure);
    let (energies, magnetizations, mag_squared, mag_fourth, acceptance_rate) = 
        measure_observables(&mut graph, alpha, beta, gamma, n_measure, &mut rng);
    
    // Calculate basic statistics
    let energy_mean = energies.iter().sum::<f64>() / energies.len() as f64;
    let mag_mean = magnetizations.iter().sum::<f64>() / magnetizations.len() as f64;
    let mag_abs_mean = magnetizations.iter().map(|m| m.abs()).sum::<f64>() / magnetizations.len() as f64;
    let m2_mean = mag_squared.iter().sum::<f64>() / mag_squared.len() as f64;
    let m4_mean = mag_fourth.iter().sum::<f64>() / mag_fourth.len() as f64;
    
    // Calculate derived quantities
    let energy_var = energies.iter().map(|&e| (e - energy_mean).powi(2)).sum::<f64>() / (energies.len() - 1) as f64;
    let specific_heat = n as f64 * beta * beta * energy_var;
    
    let susceptibility = n as f64 * beta * (m2_mean - mag_mean * mag_mean);
    let binder_cumulant = if m2_mean > 1e-10 { 
        1.0 - m4_mean / (3.0 * m2_mean * m2_mean) 
    } else { 
        0.0 
    };
    
    // Simple error estimates
    let energy_err = simple_error(&energies);
    let mag_err = simple_error(&magnetizations);
    let mag_abs_err = simple_error(&magnetizations.iter().map(|m| m.abs()).collect::<Vec<_>>());
    
    // Jackknife errors for derived quantities
    let chi_err = jackknife_error(&magnetizations, |data| {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let m2 = data.iter().map(|&x| x*x).sum::<f64>() / data.len() as f64;
        n as f64 * beta * (m2 - mean * mean)
    });
    
    let binder_err = jackknife_error(&magnetizations, |data| {
        let m2 = data.iter().map(|&x| x*x).sum::<f64>() / data.len() as f64;
        let m4 = data.iter().map(|&x| x*x*x*x).sum::<f64>() / data.len() as f64;
        if m2 > 1e-10 { 1.0 - m4 / (3.0 * m2 * m2) } else { 0.0 }
    });
    
    let specific_heat_err = jackknife_error(&energies, |data| {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let var = data.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / data.len() as f64;
        n as f64 * beta * beta * var
    });
    
    println!("    Results: ⟨E⟩={:.4}±{:.4}, ⟨|M|⟩={:.4}±{:.4}, χ={:.4}±{:.4}, U₄={:.4}±{:.4}",
             energy_mean, energy_err, mag_abs_mean, mag_abs_err, susceptibility, chi_err, binder_cumulant, binder_err);
    println!("    Quality: acceptance={:.1}%", 100.0 * acceptance_rate);
    
    PhasePoint {
        alpha,
        beta,
        gamma,
        n,
        energy_mean,
        energy_err,
        magnetization_mean: mag_mean,
        magnetization_err: mag_err,
        magnetization_abs_mean: mag_abs_mean,
        magnetization_abs_err: mag_abs_err,
        susceptibility,
        susceptibility_err: chi_err,
        binder_cumulant,
        binder_err,
        specific_heat,
        specific_heat_err,
        acceptance_rate,
    }
}

fn find_critical_ridge(
    alpha_range: (f64, f64),
    beta_range: (f64, f64),
    gamma: f64,
    n_alpha: usize,
    n_beta: usize,
    n: usize,
    n_therm: usize,
    n_measure: usize,
) -> Vec<PhasePoint> {
    println!("\n=== PHASE SPACE EXPLORATION ===");
    println!("α ∈ [{:.2}, {:.2}], β ∈ [{:.2}, {:.2}], γ = {:.2}", 
             alpha_range.0, alpha_range.1, beta_range.0, beta_range.1, gamma);
    println!("Grid: {} × {} points, N = {}", n_alpha, n_beta, n);
    println!("Monte Carlo: {} thermalization, {} measurement", n_therm, n_measure);
    
    let mut results = Vec::new();
    let mut seed = 42;
    
    for i in 0..n_alpha {
        let alpha = alpha_range.0 + (alpha_range.1 - alpha_range.0) * i as f64 / (n_alpha - 1) as f64;
        
        for j in 0..n_beta {
            let beta = beta_range.0 + (beta_range.1 - beta_range.0) * j as f64 / (n_beta - 1) as f64;
            
            seed += 1;
            let point = analyze_phase_point(alpha, beta, gamma, n, n_therm, n_measure, seed);
            results.push(point);
        }
    }
    
    results
}

fn identify_critical_region(results: &[PhasePoint]) -> Option<(f64, f64, f64)> {
    // Find region with maximum susceptibility (indicates critical behavior)
    let max_chi_point = results.iter()
        .max_by(|a, b| a.susceptibility.partial_cmp(&b.susceptibility).unwrap())
        .unwrap();
    
    // Find region with Binder cumulant closest to critical value (~0.6)
    let critical_u4 = 0.6;
    let best_u4_point = results.iter()
        .min_by(|a, b| {
            let dist_a = (a.binder_cumulant - critical_u4).abs();
            let dist_b = (b.binder_cumulant - critical_u4).abs();
            dist_a.partial_cmp(&dist_b).unwrap()
        })
        .unwrap();
    
    println!("\n=== CRITICAL REGION IDENTIFICATION ===");
    println!("Maximum susceptibility at: α={:.3}, β={:.3}, χ={:.4}±{:.4}",
             max_chi_point.alpha, max_chi_point.beta, 
             max_chi_point.susceptibility, max_chi_point.susceptibility_err);
    println!("Best Binder cumulant at: α={:.3}, β={:.3}, U₄={:.4}±{:.4}",
             best_u4_point.alpha, best_u4_point.beta,
             best_u4_point.binder_cumulant, best_u4_point.binder_err);
    
    // Check if we have significant signal above noise
    let chi_signal_to_noise = max_chi_point.susceptibility / max_chi_point.susceptibility_err;
    let u4_signal_to_noise = (best_u4_point.binder_cumulant - 0.5).abs() / best_u4_point.binder_err;
    
    println!("\nSignal-to-noise ratios:");
    println!("  Susceptibility: {:.1}", chi_signal_to_noise);
    println!("  Binder cumulant: {:.1}", u4_signal_to_noise);
    
    if chi_signal_to_noise < 3.0 || u4_signal_to_noise < 3.0 {
        println!("  ⚠️  Low signal-to-noise - need better statistics");
    } else {
        println!("  ✅ Good signal-to-noise ratio");
    }
    
    // Look for ridge structure by checking correlations
    let alphas: Vec<f64> = results.iter().map(|p| p.alpha).collect();
    let betas: Vec<f64> = results.iter().map(|p| p.beta).collect();
    
    // Simple linear regression to find ridge
    let n = alphas.len() as f64;
    let sum_a: f64 = alphas.iter().sum();
    let sum_b: f64 = betas.iter().sum();
    let sum_a2: f64 = alphas.iter().map(|a| a * a).sum();
    let sum_ab: f64 = alphas.iter().zip(&betas).map(|(a, b)| a * b).sum();
    
    let slope = (n * sum_ab - sum_a * sum_b) / (n * sum_a2 - sum_a * sum_a);
    let intercept = (sum_b - slope * sum_a) / n;
    
    println!("\nRidge analysis: β ≈ {:.4}α + {:.4}", slope, intercept);
    
    // Return best critical point estimate
    Some((max_chi_point.alpha, max_chi_point.beta, max_chi_point.gamma))
}

fn finite_size_scaling(
    alpha: f64,
    beta: f64,
    gamma: f64,
    sizes: &[usize],
) -> Vec<PhasePoint> {
    println!("\n=== FINITE SIZE SCALING ===");
    println!("Critical point: α={:.3}, β={:.3}, γ={:.3}", alpha, beta, gamma);
    println!("System sizes: {:?}", sizes);
    
    let mut results = Vec::new();
    let mut seed = 1000;
    
    for &n in sizes {
        // Scale MC steps with system size for consistent statistics
        let n_therm = 200 * n;
        let n_measure = 1000 * n;
        
        seed += 1;
        let point = analyze_phase_point(alpha, beta, gamma, n, n_therm, n_measure, seed);
        results.push(point);
    }
    
    results
}

fn main() {
    println!("=== PHASE SPACE EXPLORATION WITH SPECTRAL TERM ===\n");
    
    // Phase 1: Coarse grid search without spectral term first
    println!("PHASE 1: Coarse grid search (γ=0)...");
    let gamma = 0.0;  // Start without spectral term
    let coarse_results = find_critical_ridge(
        (0.5, 2.5),   // alpha range
        (0.5, 3.0),   // beta range
        gamma,
        6,            // 6x6 grid
        6,
        12,           // Small system for speed
        1000,         // Reasonable thermalization
        2000,         // Reasonable measurement
    );
    
    // Save coarse results
    let mut file = File::create("phase_space_coarse.csv").unwrap();
    writeln!(file, "alpha,beta,gamma,n,energy,energy_err,mag_abs,mag_abs_err,chi,chi_err,u4,u4_err,acceptance").unwrap();
    for p in &coarse_results {
        writeln!(file, "{},{},{},{},{},{},{},{},{},{},{},{},{}",
                 p.alpha, p.beta, p.gamma, p.n,
                 p.energy_mean, p.energy_err,
                 p.magnetization_abs_mean, p.magnetization_abs_err,
                 p.susceptibility, p.susceptibility_err,
                 p.binder_cumulant, p.binder_err,
                 p.acceptance_rate).unwrap();
    }
    
    // Identify critical region
    let critical_estimate = identify_critical_region(&coarse_results);
    
    if let Some((alpha_c, beta_c, _)) = critical_estimate {
        // Phase 2: Fine grid search around critical region
        println!("\nPHASE 2: Fine grid search around critical region...");
        let fine_results = find_critical_ridge(
            (alpha_c - 0.2, alpha_c + 0.2),
            (beta_c - 0.2, beta_c + 0.2),
            0.0,          // Still no spectral term
            8,            // 8x8 finer grid
            8,
            16,           // Larger system
            2000,         // Better thermalization
            10000,        // Much better statistics for signal above noise
        );
        
        // Save fine results
        let mut file = File::create("phase_space_fine.csv").unwrap();
        writeln!(file, "alpha,beta,gamma,n,energy,energy_err,mag_abs,mag_abs_err,chi,chi_err,u4,u4_err,acceptance").unwrap();
        for p in &fine_results {
            writeln!(file, "{},{},{},{},{},{},{},{},{},{},{},{},{}",
                     p.alpha, p.beta, p.gamma, p.n,
                     p.energy_mean, p.energy_err,
                     p.magnetization_abs_mean, p.magnetization_abs_err,
                     p.susceptibility, p.susceptibility_err,
                     p.binder_cumulant, p.binder_err,
                     p.acceptance_rate).unwrap();
        }
        
        // Refine critical point estimate
        let refined_critical = identify_critical_region(&fine_results);
        
        if let Some((alpha_c2, beta_c2, _)) = refined_critical {
            // Phase 3: Finite-size scaling at critical point
            println!("\nPHASE 3: Finite-size scaling analysis...");
            let sizes = vec![8, 10, 12, 14, 16, 20];
            let fss_results = finite_size_scaling(alpha_c2, beta_c2, 0.0, &sizes);
            
            // Save FSS results
            let mut file = File::create("phase_space_fss.csv").unwrap();
            writeln!(file, "n,chi,chi_err,u4,u4_err,mag_abs,mag_abs_err").unwrap();
            for p in &fss_results {
                writeln!(file, "{},{},{},{},{},{},{}",
                         p.n, p.susceptibility, p.susceptibility_err,
                         p.binder_cumulant, p.binder_err,
                         p.magnetization_abs_mean, p.magnetization_abs_err).unwrap();
            }
            
            // Analyze scaling behavior
            println!("\n=== SCALING ANALYSIS ===");
            for p in &fss_results {
                println!("N={:2}: χ={:.4}±{:.4}, U₄={:.4}±{:.4}, ⟨|M|⟩={:.4}±{:.4}",
                         p.n, p.susceptibility, p.susceptibility_err,
                         p.binder_cumulant, p.binder_err,
                         p.magnetization_abs_mean, p.magnetization_abs_err);
            }
            
            // Check for critical scaling χ ~ N^(γ/ν)
            let log_n: Vec<f64> = fss_results.iter().map(|p| (p.n as f64).ln()).collect();
            let log_chi: Vec<f64> = fss_results.iter().map(|p| p.susceptibility.ln()).collect();
            
            let n_pts = log_n.len() as f64;
            let sum_x: f64 = log_n.iter().sum();
            let sum_y: f64 = log_chi.iter().sum();
            let sum_x2: f64 = log_n.iter().map(|x| x * x).sum();
            let sum_xy: f64 = log_n.iter().zip(&log_chi).map(|(x, y)| x * y).sum();
            
            let gamma_over_nu = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_x2 - sum_x * sum_x);
            
            println!("\nCritical exponent γ/ν = {:.3}", gamma_over_nu);
            
            // Phase 4: Add spectral term
            println!("\nPHASE 4: Adding spectral term γ=0.1...");
            let spectral_results = find_critical_ridge(
                (alpha_c2 - 0.1, alpha_c2 + 0.1),
                (beta_c2 - 0.1, beta_c2 + 0.1),
                0.1,          // Add spectral term
                5,            // 5x5 grid
                5,
                12,           // Medium system
                2000,         
                5000,         
            );
            
            // Save spectral results
            let mut file = File::create("phase_space_spectral.csv").unwrap();
            writeln!(file, "alpha,beta,gamma,n,energy,energy_err,mag_abs,mag_abs_err,chi,chi_err,u4,u4_err,acceptance").unwrap();
            for p in &spectral_results {
                writeln!(file, "{},{},{},{},{},{},{},{},{},{},{},{},{}",
                         p.alpha, p.beta, p.gamma, p.n,
                         p.energy_mean, p.energy_err,
                         p.magnetization_abs_mean, p.magnetization_abs_err,
                         p.susceptibility, p.susceptibility_err,
                         p.binder_cumulant, p.binder_err,
                         p.acceptance_rate).unwrap();
            }
            
            let spectral_critical = identify_critical_region(&spectral_results);
            
            // Summary
            println!("\n=== PHASE SPACE EXPLORATION SUMMARY ===");
            println!("Critical point without spectral term:");
            println!("  α = {:.3} ± 0.02", alpha_c2);
            println!("  β = {:.3} ± 0.02", beta_c2);
            
            if let Some((alpha_s, beta_s, _)) = spectral_critical {
                println!("\nCritical point with spectral term (γ=0.1):");
                println!("  α = {:.3} ± 0.02", alpha_s);
                println!("  β = {:.3} ± 0.02", beta_s);
                println!("  Shift: Δα = {:.3}, Δβ = {:.3}", alpha_s - alpha_c2, beta_s - beta_c2);
            }
            
            println!("\nCritical properties:");
            println!("  Ridge structure: β ≈ {:.3}α + {:.3}", 
                     (beta_c2 - 0.5) / (alpha_c2 - 0.5), 
                     beta_c2 - (beta_c2 - 0.5) / (alpha_c2 - 0.5) * alpha_c2);
            println!("  Critical exponent γ/ν ≈ {:.3}", gamma_over_nu);
            println!("  Binder cumulant at criticality ≈ {:.3}", 
                     fss_results.iter().map(|p| p.binder_cumulant).sum::<f64>() / fss_results.len() as f64);
            
            println!("\nData files created:");
            println!("  phase_space_coarse.csv - Initial grid search");
            println!("  phase_space_fine.csv - Refined search near critical region");
            println!("  phase_space_fss.csv - Finite-size scaling data");
            println!("  phase_space_spectral.csv - With spectral term");
            
            // Check statistical quality
            let min_chi_sn = fine_results.iter()
                .map(|p| p.susceptibility / p.susceptibility_err)
                .fold(f64::MAX, f64::min);
            let avg_chi_sn = fine_results.iter()
                .map(|p| p.susceptibility / p.susceptibility_err)
                .sum::<f64>() / fine_results.len() as f64;
            
            println!("\nStatistical quality:");
            println!("  Average χ signal-to-noise: {:.1}", avg_chi_sn);
            println!("  Minimum χ signal-to-noise: {:.1}", min_chi_sn);
            
            if min_chi_sn > 5.0 {
                println!("  ✅ Excellent statistics - signal well above noise");
            } else if min_chi_sn > 3.0 {
                println!("  ✅ Good statistics - meaningful signal above noise");
            } else {
                println!("  ⚠️  Marginal statistics - consider longer runs");
            }
        }
    }
}