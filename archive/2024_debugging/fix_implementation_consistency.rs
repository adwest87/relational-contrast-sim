// Fix implementation inconsistencies between different graph types
// Comprehensive comparison and debugging to ensure all implementations agree

use scan::graph::Graph;
use scan::graph_fast::FastGraph;
use scan::graph_ultra_optimized::UltraOptimizedGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::fs::File;
use std::io::Write;

#[derive(Debug, Clone)]
struct ImplementationTest {
    implementation: String,
    triangle_sum: f64,
    action: f64,
    magnetization: f64,
    final_state_hash: u64,
    mc_steps_taken: usize,
}

fn hash_state(cos_values: &[f64]) -> u64 {
    // Simple hash of the state for comparison
    let mut hash = 0u64;
    for &val in cos_values {
        let bits = val.to_bits();
        hash = hash.wrapping_mul(31).wrapping_add(bits);
    }
    hash
}

fn run_reference_test(n: usize, alpha: f64, beta: f64, seed: u64, n_steps: usize) -> ImplementationTest {
    println!("Testing Reference implementation...");
    
    let mut rng = Pcg64::seed_from_u64(seed);
    let mut graph = Graph::complete_random_with(&mut rng, n);
    
    // Record initial state
    let initial_triangle = graph.triangle_sum();
    let initial_action = graph.action(alpha, beta);
    
    println!("  Initial triangle sum: {:.6}", initial_triangle);
    println!("  Initial action: {:.6}", initial_action);
    
    // Monte Carlo evolution
    let mut rng = Pcg64::seed_from_u64(seed + 1000);
    for _ in 0..n_steps {
        graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
    }
    
    // Final measurements
    let triangle_sum = graph.triangle_sum();
    let action = graph.action(alpha, beta);
    
    // Calculate magnetization
    let cos_values: Vec<f64> = graph.links.iter().map(|link| link.theta.cos()).collect();
    let magnetization = cos_values.iter().sum::<f64>() / cos_values.len() as f64;
    let state_hash = hash_state(&cos_values);
    
    println!("  Final triangle sum: {:.6}", triangle_sum);
    println!("  Final action: {:.6}", action);
    println!("  Final magnetization: {:.6}", magnetization);
    
    ImplementationTest {
        implementation: "Reference".to_string(),
        triangle_sum,
        action,
        magnetization,
        final_state_hash: state_hash,
        mc_steps_taken: n_steps,
    }
}

fn run_fast_test(n: usize, alpha: f64, beta: f64, seed: u64, n_steps: usize) -> ImplementationTest {
    println!("Testing FastGraph implementation...");
    
    // Create identical initial state as reference
    let mut rng = Pcg64::seed_from_u64(seed);
    let reference = Graph::complete_random_with(&mut rng, n);
    let mut graph = FastGraph::from_graph(&reference);
    
    // Record initial state
    let initial_triangle = graph.triangle_sum();
    let initial_action = graph.action(alpha, beta);
    
    println!("  Initial triangle sum: {:.6}", initial_triangle);
    println!("  Initial action: {:.6}", initial_action);
    
    // Monte Carlo evolution with same RNG sequence
    let mut rng = Pcg64::seed_from_u64(seed + 1000);
    for _ in 0..n_steps {
        graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
    }
    
    // Final measurements
    let triangle_sum = graph.triangle_sum();
    let action = graph.action(alpha, beta);
    
    // Calculate magnetization
    let cos_values: Vec<f64> = graph.links.iter().map(|link| link.cos_theta).collect();
    let magnetization = cos_values.iter().sum::<f64>() / cos_values.len() as f64;
    let state_hash = hash_state(&cos_values);
    
    println!("  Final triangle sum: {:.6}", triangle_sum);
    println!("  Final action: {:.6}", action);
    println!("  Final magnetization: {:.6}", magnetization);
    
    ImplementationTest {
        implementation: "FastGraph".to_string(),
        triangle_sum,
        action,
        magnetization,
        final_state_hash: state_hash,
        mc_steps_taken: n_steps,
    }
}

fn run_ultra_test(n: usize, alpha: f64, beta: f64, seed: u64, n_steps: usize) -> ImplementationTest {
    println!("Testing UltraOptimized implementation...");
    
    let mut graph = UltraOptimizedGraph::new(n, seed);
    
    // Record initial state
    let initial_triangle = graph.triangle_sum();
    let initial_action = graph.action(alpha, beta, 0.0);
    
    println!("  Initial triangle sum: {:.6}", initial_triangle);
    println!("  Initial action: {:.6}", initial_action);
    
    // Monte Carlo evolution
    let mut rng = Pcg64::seed_from_u64(seed + 1000);
    for _ in 0..n_steps {
        graph.metropolis_step(alpha, beta, 0.0, 0.1, 0.1, &mut rng);
    }
    
    // Final measurements
    let triangle_sum = graph.triangle_sum();
    let action = graph.action(alpha, beta, 0.0);
    
    // Calculate magnetization
    let cos_values: Vec<f64> = graph.cos_theta.clone();
    let magnetization = cos_values.iter().sum::<f64>() / cos_values.len() as f64;
    let state_hash = hash_state(&cos_values);
    
    println!("  Final triangle sum: {:.6}", triangle_sum);
    println!("  Final action: {:.6}", action);
    println!("  Final magnetization: {:.6}", magnetization);
    
    ImplementationTest {
        implementation: "UltraOptimized".to_string(),
        triangle_sum,
        action,
        magnetization,
        final_state_hash: state_hash,
        mc_steps_taken: n_steps,
    }
}

fn compare_results(results: &[ImplementationTest]) {
    println!("\n=== IMPLEMENTATION COMPARISON ===");
    
    if results.len() < 2 {
        println!("Need at least 2 implementations to compare");
        return;
    }
    
    let reference = &results[0];
    
    println!("Reference: {}", reference.implementation);
    println!("  Triangle sum: {:.6}", reference.triangle_sum);
    println!("  Action: {:.6}", reference.action);
    println!("  Magnetization: {:.6}", reference.magnetization);
    println!("  State hash: {}", reference.final_state_hash);
    
    for test in &results[1..] {
        println!("\nComparing {} to {}:", test.implementation, reference.implementation);
        
        let triangle_diff = (test.triangle_sum - reference.triangle_sum).abs();
        let action_diff = (test.action - reference.action).abs();
        let mag_diff = (test.magnetization - reference.magnetization).abs();
        let hash_match = test.final_state_hash == reference.final_state_hash;
        
        println!("  Triangle sum: {:.6} (diff: {:.2e})", test.triangle_sum, triangle_diff);
        println!("  Action: {:.6} (diff: {:.2e})", test.action, action_diff);
        println!("  Magnetization: {:.6} (diff: {:.2e})", test.magnetization, mag_diff);
        println!("  State hash: {} (match: {})", test.final_state_hash, hash_match);
        
        // Check tolerances
        let triangle_ok = triangle_diff < 1e-8;
        let action_ok = action_diff < 1e-8;
        let mag_ok = mag_diff < 1e-8;
        
        if triangle_ok && action_ok && mag_ok {
            println!("  ✓ IMPLEMENTATIONS AGREE within tolerance");
        } else {
            println!("  ✗ IMPLEMENTATIONS DISAGREE:");
            if !triangle_ok { println!("    - Triangle sum differs by {:.2e}", triangle_diff); }
            if !action_ok { println!("    - Action differs by {:.2e}", action_diff); }
            if !mag_ok { println!("    - Magnetization differs by {:.2e}", mag_diff); }
        }
    }
}

fn test_binder_cumulant_consistency(n: usize, alpha: f64, beta: f64, seed: u64) {
    println!("\n=== BINDER CUMULANT CONSISTENCY TEST ===");
    
    let n_therm = 1000 * n;
    let n_measure = 2000 * n;
    
    // Test FastGraph
    let mut rng = Pcg64::seed_from_u64(seed);
    let reference = Graph::complete_random_with(&mut rng, n);
    let mut fast_graph = FastGraph::from_graph(&reference);
    let mut rng = Pcg64::seed_from_u64(seed + 1000);
    
    // Thermalization
    for _ in 0..n_therm {
        fast_graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
    }
    
    // Measurement
    let mut fast_mag_samples = Vec::new();
    for _ in 0..n_measure {
        fast_graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
        let mag = fast_graph.links.iter().map(|link| link.cos_theta).sum::<f64>() / fast_graph.links.len() as f64;
        fast_mag_samples.push(mag);
    }
    
    // Test UltraOptimized  
    let mut ultra_graph = UltraOptimizedGraph::new(n, seed);
    let mut rng = Pcg64::seed_from_u64(seed + 1000);
    
    // Thermalization
    for _ in 0..n_therm {
        ultra_graph.metropolis_step(alpha, beta, 0.0, 0.1, 0.1, &mut rng);
    }
    
    // Measurement
    let mut ultra_mag_samples = Vec::new();
    for _ in 0..n_measure {
        ultra_graph.metropolis_step(alpha, beta, 0.0, 0.1, 0.1, &mut rng);
        let mag = ultra_graph.cos_theta.iter().sum::<f64>() / ultra_graph.cos_theta.len() as f64;
        ultra_mag_samples.push(mag);
    }
    
    // Calculate Binder cumulants
    fn calculate_u4(samples: &[f64]) -> f64 {
        let n = samples.len() as f64;
        let m2 = samples.iter().map(|&x| x*x).sum::<f64>() / n;
        let m4 = samples.iter().map(|&x| x.powi(4)).sum::<f64>() / n;
        if m2 > 1e-10 { 1.0 - m4 / (3.0 * m2 * m2) } else { 0.0 }
    }
    
    let fast_u4 = calculate_u4(&fast_mag_samples);
    let ultra_u4 = calculate_u4(&ultra_mag_samples);
    
    println!("FastGraph U₄: {:.6}", fast_u4);
    println!("UltraOptimized U₄: {:.6}", ultra_u4);
    println!("Difference: {:.6} ({:.1}%)", (ultra_u4 - fast_u4).abs(), 
             100.0 * (ultra_u4 - fast_u4).abs() / fast_u4.abs());
    
    if (ultra_u4 - fast_u4).abs() > 0.1 {
        println!("⚠ LARGE BINDER CUMULANT DISCREPANCY DETECTED");
        
        // Detailed analysis
        println!("\nDetailed Analysis:");
        
        let fast_mean = fast_mag_samples.iter().sum::<f64>() / fast_mag_samples.len() as f64;
        let ultra_mean = ultra_mag_samples.iter().sum::<f64>() / ultra_mag_samples.len() as f64;
        
        let fast_std = (fast_mag_samples.iter().map(|&x| (x - fast_mean).powi(2)).sum::<f64>() / (fast_mag_samples.len() - 1) as f64).sqrt();
        let ultra_std = (ultra_mag_samples.iter().map(|&x| (x - ultra_mean).powi(2)).sum::<f64>() / (ultra_mag_samples.len() - 1) as f64).sqrt();
        
        println!("FastGraph: ⟨M⟩ = {:.6} ± {:.6}", fast_mean, fast_std);
        println!("UltraOptimized: ⟨M⟩ = {:.6} ± {:.6}", ultra_mean, ultra_std);
        
        // Check for negative values
        let fast_negative = fast_mag_samples.iter().filter(|&&x| x < 0.0).count();
        let ultra_negative = ultra_mag_samples.iter().filter(|&&x| x < 0.0).count();
        
        println!("FastGraph negative values: {}/{} ({:.1}%)", 
                 fast_negative, fast_mag_samples.len(), 
                 100.0 * fast_negative as f64 / fast_mag_samples.len() as f64);
        println!("UltraOptimized negative values: {}/{} ({:.1}%)", 
                 ultra_negative, ultra_mag_samples.len(), 
                 100.0 * ultra_negative as f64 / ultra_mag_samples.len() as f64);
    } else {
        println!("✓ Binder cumulants agree within tolerance");
    }
}

fn main() {
    println!("=== IMPLEMENTATION CONSISTENCY FIX ===\n");
    
    let n = 8;  // Small system for debugging
    let alpha = 1.37;
    let beta = 2.0;
    let seed = 42;
    let n_steps = 1000;
    
    println!("Test parameters:");
    println!("  N = {}", n);
    println!("  α = {:.3}", alpha);
    println!("  β = {:.3}", beta);
    println!("  Seed = {}", seed);
    println!("  MC steps = {}", n_steps);
    println!();
    
    // Run identical tests on different implementations
    let mut results = Vec::new();
    
    results.push(run_reference_test(n, alpha, beta, seed, n_steps));
    results.push(run_fast_test(n, alpha, beta, seed, n_steps));
    results.push(run_ultra_test(n, alpha, beta, seed, n_steps));
    
    // Compare results
    compare_results(&results);
    
    // Test Binder cumulant specifically
    test_binder_cumulant_consistency(n, alpha, beta, seed);
    
    // Save detailed results
    if let Ok(mut file) = File::create("implementation_consistency_results.csv") {
        writeln!(file, "implementation,triangle_sum,action,magnetization,final_state_hash,mc_steps").unwrap();
        for result in &results {
            writeln!(file, "{},{:.6},{:.6},{:.6},{},{}", 
                     result.implementation, result.triangle_sum, result.action,
                     result.magnetization, result.final_state_hash, result.mc_steps_taken).unwrap();
        }
        println!("\nResults saved to: implementation_consistency_results.csv");
    }
    
    println!("\n=== RECOMMENDATIONS ===");
    println!("1. If implementations disagree, debug the differences step by step");
    println!("2. Ensure identical initial conditions and RNG sequences");
    println!("3. Verify antisymmetry enforcement is consistent");
    println!("4. Check triangle sum calculations match exactly");
    println!("5. Validate magnetization calculations use same definition");
    
    println!("\n=== NEXT STEPS ===");
    println!("1. Run: cargo run --release --bin fix_implementation_consistency");
    println!("2. If discrepancies found, debug individual components");
    println!("3. Re-run alternative explanations study after fixes");
    println!("4. Only make physics claims after implementations agree");
}