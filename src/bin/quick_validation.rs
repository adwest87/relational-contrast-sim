// Quick validation test for optimized Monte Carlo implementations
// Verifies correctness and performance against known good values

use scan::graph::Graph;
use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::time::Instant;

fn main() {
    println!("=== QUICK VALIDATION TEST ===");
    
    // Test parameters
    let n = 48;
    // Critical parameters from the ridge α ≈ 0.06β + 1.31
    let beta = 2.91;
    let alpha = 0.06 * beta + 1.31;  // Should be ~1.484
    let equilibration_steps = 50_000;  // More equilibration needed
    let production_steps = 50_000;
    let seed = 42;
    
    println!("System: N={}, (β={:.2}, α={:.2})", n, beta, alpha);
    println!("Steps: {} equilibration + {} production", equilibration_steps, production_steps);
    println!();
    
    // Initialize RNG and graph
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut initial_graph = Graph::complete_random_with(&mut rng, n);
    
    // Randomize initial theta values
    use rand::distributions::Uniform;
    use rand::prelude::*;
    let theta_dist = Uniform::new(0.0, std::f64::consts::TAU);
    for link in &mut initial_graph.links {
        link.theta = rng.sample(theta_dist);
    }
    
    let mut graph = FastGraph::from_graph(&initial_graph);
    
    println!("Initial configuration:");
    let initial_mean_cos: f64 = graph.links.iter().map(|l| l.cos_theta).sum::<f64>() / graph.m() as f64;
    let initial_mean_z: f64 = graph.links.iter().map(|l| l.z).sum::<f64>() / graph.m() as f64;
    let initial_mean_w: f64 = graph.links.iter().map(|l| l.exp_neg_z).sum::<f64>() / graph.m() as f64;
    println!("  <cos θ> = {:.3}", initial_mean_cos);
    println!("  <z> = {:.3}", initial_mean_z);
    println!("  <w> = {:.3}", initial_mean_w);
    
    // For critical region, we expect <w> ≈ 0.4-0.5
    // Current <z> ≈ 5 gives <w> ≈ exp(-5) ≈ 0.007, too small
    // We need <z> ≈ 0.7-1.0 for critical region
    
    // Rescale z values to critical region
    for link in &mut graph.links {
        link.update_z(link.z * 0.15);  // Scale down z values
    }
    
    let adjusted_mean_z: f64 = graph.links.iter().map(|l| l.z).sum::<f64>() / graph.m() as f64;
    let adjusted_mean_w: f64 = graph.links.iter().map(|l| l.exp_neg_z).sum::<f64>() / graph.m() as f64;
    println!("\nAdjusted to critical region:");
    println!("  <z> = {:.3}", adjusted_mean_z);
    println!("  <w> = {:.3}", adjusted_mean_w);
    
    // Performance tracking
    let start_time = Instant::now();
    let mut total_accepts = 0;
    let mut nan_count = 0;
    let mut inf_count = 0;
    
    // Equilibration phase with larger step sizes
    print!("Equilibration...");
    let delta_z = 0.2;      // Larger steps to reduce acceptance
    let delta_theta = 0.2;  // Balanced steps
    
    for step in 0..equilibration_steps {
        let info = graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
        if info.accept {
            total_accepts += 1;
        }
        
        // Print progress
        if step > 0 && step % 10000 == 0 {
            let accept_rate = 100.0 * total_accepts as f64 / step as f64;
            print!("\r  Step {}: acceptance {:.1}%", step, accept_rate);
            use std::io::{self, Write};
            io::stdout().flush().unwrap();
        }
    }
    println!("\r  Equilibration done, acceptance: {:.1}%", 
        100.0 * total_accepts as f64 / equilibration_steps as f64);
    
    // Production phase with measurements
    print!("Production run...");
    let mut measurements_first_half = Vec::new();
    let mut measurements_second_half = Vec::new();
    let measure_interval = 100;
    
    for step in 0..production_steps {
        let info = graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
        if info.accept {
            total_accepts += 1;
        }
        
        // Check for NaN/Inf in observables
        if info.delta_w.is_nan() || info.delta_cos.is_nan() {
            nan_count += 1;
        }
        if info.delta_w.is_infinite() || info.delta_cos.is_infinite() {
            inf_count += 1;
        }
        
        // Measure observables
        if step % measure_interval == 0 {
            // Calculate quick observables manually
            let sum_cos: f64 = graph.links.iter().map(|l| l.cos_theta).sum();
            let sum_w: f64 = graph.links.iter().map(|l| l.exp_neg_z).sum();
            let sum_w_cos: f64 = graph.links.iter().map(|l| l.exp_neg_z * l.cos_theta).sum();
            
            // Calculate susceptibility and entropy directly
            let m = graph.m() as f64;
            let mean_w = sum_w / m;
            let mean_cos = sum_cos / m;
            let mean_w_cos = sum_w_cos / m;
            
            // Calculate cos² for susceptibility
            let sum_cos_sq: f64 = graph.links.iter().map(|l| l.cos_theta * l.cos_theta).sum();
            let mean_cos_sq = sum_cos_sq / m;
            
            // Correct susceptibility formula: χ = Nβ(<cos²θ> - <cosθ>²)
            let chi = graph.n() as f64 * beta * (mean_cos_sq - mean_cos * mean_cos);
            
            // Entropy action
            let entropy = graph.entropy_action();
            
            if step < production_steps / 2 {
                measurements_first_half.push((sum_cos, chi, entropy));
            } else {
                measurements_second_half.push((sum_cos, chi, entropy));
            }
        }
    }
    println!(" done");
    
    let total_time = start_time.elapsed();
    let total_steps = equilibration_steps + production_steps;
    let steps_per_sec = total_steps as f64 / total_time.as_secs_f64();
    
    // Calculate statistics
    let acceptance_rate = 100.0 * total_accepts as f64 / total_steps as f64;
    
    // First half statistics
    let mean_cos_first: f64 = measurements_first_half.iter()
        .map(|(sum_cos, _, _)| sum_cos / (n * (n - 1) / 2) as f64)
        .sum::<f64>() / measurements_first_half.len() as f64;
    
    let mean_chi_first: f64 = measurements_first_half.iter()
        .map(|(_, chi, _)| *chi)
        .sum::<f64>() / measurements_first_half.len() as f64;
    
    // Second half statistics
    let mean_cos_second: f64 = measurements_second_half.iter()
        .map(|(sum_cos, _, _)| sum_cos / (n * (n - 1) / 2) as f64)
        .sum::<f64>() / measurements_second_half.len() as f64;
    
    let mean_chi_second: f64 = measurements_second_half.iter()
        .map(|(_, chi, _)| *chi)
        .sum::<f64>() / measurements_second_half.len() as f64;
    
    let mean_entropy_second: f64 = measurements_second_half.iter()
        .map(|(_, _, entropy)| entropy / (n * (n - 1) / 2) as f64)
        .sum::<f64>() / measurements_second_half.len() as f64;
    
    // Combined statistics
    let all_cos: Vec<f64> = measurements_first_half.iter()
        .chain(measurements_second_half.iter())
        .map(|(sum_cos, _, _)| sum_cos / (n * (n - 1) / 2) as f64)
        .collect();
    
    let mean_cos = all_cos.iter().sum::<f64>() / all_cos.len() as f64;
    let std_cos = (all_cos.iter()
        .map(|&x| (x - mean_cos).powi(2))
        .sum::<f64>() / (all_cos.len() - 1) as f64)
        .sqrt();
    let error_cos = std_cos / (all_cos.len() as f64).sqrt();
    
    let all_chi: Vec<f64> = measurements_first_half.iter()
        .chain(measurements_second_half.iter())
        .map(|(_, chi, _)| *chi)
        .collect();
    
    let mean_chi = all_chi.iter().sum::<f64>() / all_chi.len() as f64;
    let std_chi = (all_chi.iter()
        .map(|&x| (x - mean_chi).powi(2))
        .sum::<f64>() / (all_chi.len() - 1) as f64)
        .sqrt();
    let error_chi = std_chi / (all_chi.len() as f64).sqrt();
    
    // Convergence check
    let drift_cos = f64::abs(mean_cos_second - mean_cos_first) / mean_cos_first * 100.0;
    let drift_chi = f64::abs(mean_chi_second - mean_chi_first) / mean_chi_first * 100.0;
    
    // Detailed balance test
    let detailed_balance_pass = test_detailed_balance(&mut graph, alpha, beta, &mut rng);
    
    // Performance metrics
    println!("Performance:");
    let pass_speed = steps_per_sec > 100_000.0;
    println!("  {} Speed: {:.2} M steps/sec", 
        if pass_speed { "✓" } else { "✗" },
        steps_per_sec / 1e6
    );
    println!("  ✓ Total time: {:.1} seconds", total_time.as_secs_f64());
    
    // Observable checks
    println!("\nObservables:");
    let pass_accept = acceptance_rate >= 45.0 && acceptance_rate <= 55.0;
    println!("  {} Acceptance: {:.1}% [{}]", 
        if pass_accept { "✓" } else { "✗" },
        acceptance_rate,
        if pass_accept { "PASS" } else { "FAIL" }
    );
    
    let pass_cos = mean_cos >= 0.15 && mean_cos <= 0.25;
    println!("  {} <cos θ> = {:.3} ± {:.3} [{}]",
        if pass_cos { "✓" } else { "✗" },
        mean_cos, error_cos,
        if pass_cos { "PASS" } else { "FAIL" }
    );
    
    let pass_chi = mean_chi >= 30.0 && mean_chi <= 40.0;
    println!("  {} χ = {:.1} ± {:.1} [{}]",
        if pass_chi { "✓" } else { "✗" },
        mean_chi, error_chi,
        if pass_chi { "PASS" } else { "FAIL" }
    );
    
    let pass_entropy = mean_entropy_second >= -0.5 && mean_entropy_second <= -0.3;
    println!("  {} S_entropy/link = {:.2} [{}]",
        if pass_entropy { "✓" } else { "✗" },
        mean_entropy_second,
        if pass_entropy { "PASS" } else { "FAIL" }
    );
    
    // Convergence checks
    println!("\nConvergence:");
    println!("  ✓ First half: <cos θ> = {:.3}", mean_cos_first);
    println!("  ✓ Second half: <cos θ> = {:.3}", mean_cos_second);
    let pass_drift_cos = drift_cos < 5.0;
    println!("  {} Drift: {:.1}% [{}]",
        if pass_drift_cos { "✓" } else { "✗" },
        drift_cos,
        if pass_drift_cos { "PASS" } else { "FAIL" }
    );
    
    println!("  ✓ First half: χ = {:.1}", mean_chi_first);
    println!("  ✓ Second half: χ = {:.1}", mean_chi_second);
    let pass_drift_chi = drift_chi < 10.0;
    println!("  {} χ drift: {:.1}% [{}]",
        if pass_drift_chi { "✓" } else { "✗" },
        drift_chi,
        if pass_drift_chi { "PASS" } else { "FAIL" }
    );
    
    // Sanity checks
    println!("\nSanity checks:");
    let pass_nan = nan_count == 0;
    let pass_inf = inf_count == 0;
    println!("  {} No NaN/Inf values (found {} NaN, {} Inf)",
        if pass_nan && pass_inf { "✓" } else { "✗" },
        nan_count, inf_count
    );
    
    println!("  {} Detailed balance tested on 1000 random moves",
        if detailed_balance_pass { "✓" } else { "✗" }
    );
    
    // Energy conservation test
    let energy_pass = test_energy_conservation(&mut graph, alpha, beta);
    println!("  {} Energy conservation in microcanonical test",
        if energy_pass { "✓" } else { "✗" }
    );
    
    // Overall result
    println!("\nOVERALL: ");
    let all_pass = pass_speed && pass_accept && pass_cos && pass_chi && 
                   pass_entropy && pass_drift_cos && pass_drift_chi && 
                   pass_nan && pass_inf && detailed_balance_pass && energy_pass;
    
    if all_pass {
        println!("ALL TESTS PASSED ✅");
        println!("Ready for production runs!");
    } else {
        println!("SOME TESTS FAILED ❌");
        println!("Please investigate before running production simulations.");
        std::process::exit(1);
    }
}

// Test detailed balance by checking reversibility
fn test_detailed_balance(graph: &mut FastGraph, alpha: f64, beta: f64, rng: &mut Pcg64) -> bool {
    let n_tests = 1000;
    let mut violations = 0;
    
    for _ in 0..n_tests {
        // Save current state
        let links_before: Vec<_> = graph.links.clone();
        
        // Make a move and record
        let info_forward = graph.metropolis_step(alpha, beta, 0.5, 0.5, rng);
        
        if info_forward.accept {
            // Restore state temporarily to calculate reverse action
            let action_after = graph.action(alpha, beta);
            let links_after = graph.links.clone();
            graph.links = links_before.clone();
            let action_before = graph.action(alpha, beta);
            graph.links = links_after;
            let delta_s_reverse = action_before - action_after;
            
            // Check detailed balance: P(forward) / P(reverse) = exp(-ΔS_forward)
            // For detailed balance, we need to compute delta_s from action change
            let delta_s_forward = action_after - action_before;
            let ratio = f64::exp(-delta_s_forward);
            let reverse_ratio = if delta_s_reverse <= 0.0 { 1.0 } else { f64::exp(-delta_s_reverse) };
            
            // Allow small numerical errors
            if f64::abs(ratio * reverse_ratio - 1.0) > 1e-10 {
                violations += 1;
            }
        }
    }
    
    violations < n_tests / 100  // Allow up to 1% violations due to numerics
}

// Test energy conservation by running with beta = infinity
fn test_energy_conservation(graph: &mut FastGraph, alpha: f64, _beta: f64) -> bool {
    let mut rng = Pcg64::seed_from_u64(12345);
    let beta_large = 1000.0; // Effectively microcanonical
    
    let initial_energy = graph.action(alpha, beta_large);
    let mut max_deviation: f64 = 0.0;
    
    // Run some MC steps
    for _ in 0..1000 {
        let info = graph.metropolis_step(alpha, beta_large, 0.001, 0.001, &mut rng);
        if info.accept {
            let current_energy = graph.action(alpha, beta_large);
            let deviation = f64::abs(current_energy - initial_energy) / f64::abs(initial_energy);
            max_deviation = max_deviation.max(deviation);
        }
    }
    
    max_deviation < 1e-6  // Energy should be conserved to high precision
}