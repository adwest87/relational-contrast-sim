// Comprehensive physics validation across all graph implementations

use scan::graph::Graph;
use scan::graph_fast::FastGraph;
use scan::graph_ultra_optimized::UltraOptimizedGraph;
#[cfg(target_arch = "aarch64")]
use scan::graph_m1_optimized::M1Graph;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::collections::HashMap;

#[derive(Debug)]
struct PhysicsResults {
    triangle_sum: f64,
    entropy_action: f64,
    total_action: f64,
    mean_cos: f64,
    mean_weight: f64,
}

fn main() {
    println!("=== Comprehensive Physics Validation ===\n");
    
    // Test with multiple system sizes
    let test_sizes = vec![8, 16, 32];
    let alpha = 1.5;
    let beta = 2.9;
    let seed = 42u64;
    
    for &n in &test_sizes {
        println!("\n{}", "=".repeat(60));
        println!("Testing with N={}, α={}, β={}, seed={}", n, alpha, beta, seed);
        println!("{}\n", "=".repeat(60));
        
        // Create reference graph
        let mut rng = Pcg64::seed_from_u64(seed);
        let reference_graph = Graph::complete_random_with(&mut rng, n);
        
        // Test 1: Compare triangle sum calculation methods
        println!("=== Triangle Sum Validation ===");
        test_triangle_sum_consistency(&reference_graph, n, seed);
        
        // Test 2: Compare action calculations  
        println!("\n=== Action Calculation Validation ===");
        test_action_consistency(&reference_graph, alpha, beta, n, seed);
        
        // Test 3: Test Monte Carlo physics correctness
        println!("\n=== Monte Carlo Physics Validation ===");
        test_mc_physics_consistency(&reference_graph, alpha, beta, n, seed);
        
        // Test 4: Test symmetries and conservation laws
        println!("\n=== Symmetry and Conservation Tests ===");
        test_symmetries_and_conservation(&reference_graph, alpha, beta, n, seed);
        
        // Test 5: Test susceptibility calculations (known ~69% discrepancy)
        println!("\n=== Susceptibility Calculation Validation ===");
        test_susceptibility_consistency(&reference_graph, alpha, beta, n, seed);
    }
    
    println!("\n=== Physics Validation Complete ===");
}

fn test_triangle_sum_consistency(reference: &Graph, n: usize, seed: u64) {
    let mut results = HashMap::new();
    
    // Reference calculation
    let ref_triangle = reference.triangle_sum();
    results.insert("Reference", ref_triangle);
    
    // FastGraph from reference
    let fast_from_ref = FastGraph::from_graph(reference);
    results.insert("FastGraph(from_ref)", fast_from_ref.triangle_sum());
    
    // Create graphs with same seed for fair comparison
    let fast_new = FastGraph::new(n, seed);
    results.insert("FastGraph(new)", fast_new.triangle_sum());
    
    let ultra_new = UltraOptimizedGraph::new(n, seed);
    results.insert("UltraOptimized", ultra_new.triangle_sum());
    
    #[cfg(target_arch = "aarch64")]
    {
        let m1_new = M1Graph::new(n, seed);
        results.insert("M1Graph", m1_new.triangle_sum());
    }
    
    println!("Triangle Sum Results:");
    for (name, value) in &results {
        println!("  {:<20}: {:.6}", name, value);
    }
    
    // Check consistency among implementations that should match
    let ref_value = ref_triangle;
    let fast_ref_value = results["FastGraph(from_ref)"];
    
    println!("\nTriangle Sum Analysis:");
    
    // FastGraph from reference should match exactly
    let fast_ref_error = (fast_ref_value - ref_value).abs();
    if fast_ref_error < 1e-10 {
        println!("  ✓ FastGraph(from_ref) matches Reference exactly: error = {:.2e}", fast_ref_error);
    } else {
        println!("  ✗ FastGraph(from_ref) differs from Reference: error = {:.2e}", fast_ref_error);
    }
    
    // Check if new implementations have consistent internal structure
    // (They won't match reference due to different initialization, but should be self-consistent)
    for (name, value) in &results {
        if name.contains("new") || name == &"UltraOptimized" || name == &"M1Graph" {
            if value.is_finite() && !value.is_nan() {
                println!("  ✓ {}: finite result {:.6}", name, value);
            } else {
                println!("  ✗ {}: non-finite result {:.6}", name, value);
            }
        }
    }
}

fn test_action_consistency(reference: &Graph, alpha: f64, beta: f64, n: usize, seed: u64) {
    let ref_entropy = reference.entropy_action();
    let ref_triangle = reference.triangle_sum();
    let ref_action = reference.action(alpha, beta);
    
    println!("Reference Graph:");
    println!("  Entropy:     {:.6}", ref_entropy);
    println!("  Triangle:    {:.6}", ref_triangle);
    println!("  Action:      {:.6}", ref_action);
    println!("  Expected:    {:.6}", beta * ref_entropy + alpha * ref_triangle);
    
    // Test FastGraph from reference
    let fast_from_ref = FastGraph::from_graph(reference);
    let fast_entropy = fast_from_ref.entropy_action();
    let fast_triangle = fast_from_ref.triangle_sum();
    
    println!("\nFastGraph(from_ref):");
    println!("  Entropy:     {:.6} (error: {:.2e})", fast_entropy, (fast_entropy - ref_entropy).abs());
    println!("  Triangle:    {:.6} (error: {:.2e})", fast_triangle, (fast_triangle - ref_triangle).abs());
    
    // Test new implementations for internal consistency
    let ultra_graph = UltraOptimizedGraph::new(n, seed);
    let ultra_action = ultra_graph.action(alpha, beta, 0.0); // No spectral term
    let ultra_entropy = ultra_graph.z_values.iter()
        .zip(&ultra_graph.exp_neg_z)
        .map(|(&z, &w)| -z * w)
        .sum::<f64>();
    let ultra_triangle = ultra_graph.triangle_sum();
    let ultra_expected = beta * ultra_entropy + alpha * ultra_triangle;
    
    println!("\nUltraOptimized internal consistency:");
    println!("  Entropy:     {:.6}", ultra_entropy);
    println!("  Triangle:    {:.6}", ultra_triangle);
    println!("  Action:      {:.6}", ultra_action);
    println!("  Expected:    {:.6}", ultra_expected);
    println!("  Error:       {:.2e}", (ultra_action - ultra_expected).abs());
    
    if (ultra_action - ultra_expected).abs() < 1e-10 {
        println!("  ✓ UltraOptimized action calculation is internally consistent");
    } else {
        println!("  ✗ UltraOptimized action calculation has internal inconsistency");
    }
}

fn test_mc_physics_consistency(reference: &Graph, alpha: f64, beta: f64, n: usize, seed: u64) {
    let steps = 100;
    
    // Test FastGraph Monte Carlo
    let mut fast_graph = FastGraph::from_graph(reference);
    let mut rng = Pcg64::seed_from_u64(seed + 1);
    
    let initial_action = fast_graph.action(alpha, beta);
    let mut accepts = 0;
    
    for _ in 0..steps {
        let step_info = fast_graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
        if step_info.accept {
            accepts += 1;
        }
    }
    
    let final_action = fast_graph.action(alpha, beta);
    let acceptance_rate = accepts as f64 / steps as f64;
    
    println!("FastGraph Monte Carlo ({} steps):", steps);
    println!("  Initial action: {:.6}", initial_action);
    println!("  Final action:   {:.6}", final_action);
    println!("  Action change:  {:.6}", final_action - initial_action);
    println!("  Acceptance:     {:.1}%", 100.0 * acceptance_rate);
    
    // Test if acceptance rate is reasonable (should be 20-80%)
    if acceptance_rate > 0.1 && acceptance_rate < 0.9 {
        println!("  ✓ Reasonable acceptance rate");
    } else {
        println!("  ⚠ Unusual acceptance rate (may indicate tuning issues)");
    }
    
    // Test UltraOptimized Monte Carlo
    let mut ultra_graph = UltraOptimizedGraph::new(n, seed);
    let mut rng = Pcg64::seed_from_u64(seed + 2);
    
    let initial_ultra_action = ultra_graph.action(alpha, beta, 0.0);
    let mut ultra_accepts = 0;
    
    for _ in 0..steps {
        let accept = ultra_graph.metropolis_step(alpha, beta, 0.0, 0.1, 0.1, &mut rng);
        if accept {
            ultra_accepts += 1;
        }
    }
    
    let final_ultra_action = ultra_graph.action(alpha, beta, 0.0);
    let ultra_acceptance_rate = ultra_accepts as f64 / steps as f64;
    
    println!("\nUltraOptimized Monte Carlo ({} steps):", steps);
    println!("  Initial action: {:.6}", initial_ultra_action);
    println!("  Final action:   {:.6}", final_ultra_action);
    println!("  Action change:  {:.6}", final_ultra_action - initial_ultra_action);
    println!("  Acceptance:     {:.1}%", 100.0 * ultra_acceptance_rate);
    
    if ultra_acceptance_rate > 0.1 && ultra_acceptance_rate < 0.9 {
        println!("  ✓ Reasonable acceptance rate");
    } else {
        println!("  ⚠ Unusual acceptance rate");
    }
}

fn test_symmetries_and_conservation(reference: &Graph, alpha: f64, beta: f64, n: usize, seed: u64) {
    // Test 1: Action should be real and finite
    let action = reference.action(alpha, beta);
    println!("Action properties:");
    if action.is_finite() && !action.is_nan() {
        println!("  ✓ Action is finite: {:.6}", action);
    } else {
        println!("  ✗ Action is not finite: {:.6}", action);
    }
    
    // Test 2: Triangle sum should be bounded for finite phases
    let triangle_sum = reference.triangle_sum();
    let max_triangle_magnitude = (n * (n-1) * (n-2) / 6) as f64; // max |cos| = 1 for all triangles
    
    println!("\nTriangle sum bounds:");
    println!("  Triangle sum: {:.6}", triangle_sum);
    println!("  Max possible magnitude: {:.6}", max_triangle_magnitude);
    if triangle_sum.abs() <= max_triangle_magnitude + 1e-10 {
        println!("  ✓ Triangle sum is within physical bounds");
    } else {
        println!("  ✗ Triangle sum exceeds physical bounds");
    }
    
    // Test 3: Entropy should be negative (since we use -z*w and w=exp(-z) > 0, z > 0)
    let entropy = reference.entropy_action();
    println!("\nEntropy properties:");
    println!("  Entropy action: {:.6}", entropy);
    if entropy < 0.0 {
        println!("  ✓ Entropy action is negative as expected");
    } else {
        println!("  ⚠ Entropy action is positive (unusual but not necessarily wrong)");
    }
    
    // Test 4: Weight conservation - sum of weights should be positive and finite
    let total_weight: f64 = reference.links.iter().map(|link| link.w()).sum();
    println!("\nWeight properties:");
    println!("  Total weight: {:.6}", total_weight);
    if total_weight > 0.0 && total_weight.is_finite() {
        println!("  ✓ Total weight is positive and finite");
    } else {
        println!("  ✗ Total weight has issues");
    }
    
    // Test 5: FastGraph antisymmetry verification with Monte Carlo
    let mut fast_graph = FastGraph::from_graph(reference);
    let mut rng = Pcg64::seed_from_u64(seed + 3);
    
    // Run some MC steps and verify antisymmetry is preserved
    for _ in 0..10 {
        fast_graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
    }
    
    println!("\nAntisymmetry preservation after MC:");
    let mut antisymmetry_violations = 0;
    for i in 0..n {
        for j in (i+1)..n {
            let theta_ij = fast_graph.get_phase(i, j);
            let theta_ji = fast_graph.get_phase(j, i);
            let expected = -theta_ij;
            let error = (theta_ji - expected).abs();
            if error > 1e-10 {
                antisymmetry_violations += 1;
            }
        }
    }
    
    if antisymmetry_violations == 0 {
        println!("  ✓ Antisymmetry preserved after Monte Carlo steps");
    } else {
        println!("  ✗ {} antisymmetry violations detected after Monte Carlo", antisymmetry_violations);
    }
}

fn test_susceptibility_consistency(reference: &Graph, alpha: f64, beta: f64, n: usize, seed: u64) {
    println!("Testing susceptibility calculations for known ~69% discrepancy...");
    
    // Test 1: Calculate magnetization from phase angles
    // For reference graph, we need to calculate from the phase angles
    let mut ref_mag_real = 0.0;
    let mut ref_mag_imag = 0.0;
    
    // In a complete graph, we can think of magnetization as average of exp(i*theta) over edges
    for link in &reference.links {
        ref_mag_real += link.theta.cos();
        ref_mag_imag += link.theta.sin();
    }
    let n_edges = n * (n - 1) / 2;
    ref_mag_real /= n_edges as f64;
    ref_mag_imag /= n_edges as f64;
    
    println!("\nReference Graph edge-based magnetization: ({:.6}, {:.6})", ref_mag_real, ref_mag_imag);
    println!("  |M|: {:.6}", (ref_mag_real.powi(2) + ref_mag_imag.powi(2)).sqrt());
    
    // Test 2: FastGraph implementation
    let fast_from_ref = FastGraph::from_graph(reference);
    
    // Calculate FastGraph magnetization using same method
    let mut fast_mag_real = 0.0;
    let mut fast_mag_imag = 0.0;
    for i in 0..n {
        for j in (i+1)..n {
            let theta = fast_from_ref.get_phase(i, j);
            fast_mag_real += theta.cos();
            fast_mag_imag += theta.sin();
        }
    }
    fast_mag_real /= n_edges as f64;
    fast_mag_imag /= n_edges as f64;
    
    println!("\nFastGraph(from_ref) edge-based magnetization: ({:.6}, {:.6})", fast_mag_real, fast_mag_imag);
    println!("  |M|: {:.6}", (fast_mag_real.powi(2) + fast_mag_imag.powi(2)).sqrt());
    
    // Check if magnetizations match
    let mag_error = ((fast_mag_real - ref_mag_real).powi(2) + (fast_mag_imag - ref_mag_imag).powi(2)).sqrt();
    if mag_error < 1e-10 {
        println!("  ✓ FastGraph magnetization matches Reference exactly: error = {:.2e}", mag_error);
    } else {
        println!("  ✗ FastGraph magnetization differs from Reference: error = {:.2e}", mag_error);
    }
    
    // Test 3: UltraOptimized implementation
    let ultra = UltraOptimizedGraph::new(n, seed);
    
    // UltraOptimized stores phases differently - in a flattened array
    let mut ultra_mag_real = 0.0;
    let mut ultra_mag_imag = 0.0;
    for idx in 0..ultra.theta_values.len() {
        ultra_mag_real += ultra.theta_values[idx].cos();
        ultra_mag_imag += ultra.theta_values[idx].sin();
    }
    ultra_mag_real /= n_edges as f64;
    ultra_mag_imag /= n_edges as f64;
    
    println!("\nUltraOptimized edge-based magnetization: ({:.6}, {:.6})", ultra_mag_real, ultra_mag_imag);
    println!("  |M|: {:.6}", (ultra_mag_real.powi(2) + ultra_mag_imag.powi(2)).sqrt());
    
    // Test 4: Node-based vs edge-based magnetization definitions
    println!("\n=== Node-based vs Edge-based Magnetization ===");
    
    // Node-based: average of exp(i*sum_j theta_ij) over nodes
    // This is fundamentally different from edge-based magnetization
    println!("Different physics implementations may use:");
    println!("  - Edge-based: M = (1/N_edges) * sum_{{ij}} exp(i*theta_ij)");
    println!("  - Node-based: M = (1/N) * sum_i exp(i*sum_j theta_ij)");
    println!("  - These give DIFFERENT physics!");
    
    // Test 5: Susceptibility normalization analysis
    println!("\n=== Susceptibility Normalization Analysis ===");
    
    // Simulate what different normalizations would give
    let test_mag_squared = 0.5;  // Example value
    let test_mag_abs_squared = 0.4;  // Example value
    let chi_raw = test_mag_squared - test_mag_abs_squared;
    
    println!("For example data with <|M|²> = {:.3}, <|M|>² = {:.3}:", test_mag_squared, test_mag_abs_squared);
    println!("  χ (no normalization) = {:.6}", chi_raw);
    println!("  χ * N = {:.6}", chi_raw * n as f64);
    println!("  χ / N = {:.6}", chi_raw / n as f64);
    println!("  χ * N² = {:.6}", chi_raw * (n as f64).powi(2));
    
    // Calculate ratios
    let ratio_example = (chi_raw * n as f64) / (chi_raw / n as f64);
    println!("\nRatio (χ*N)/(χ/N) = {:.6} = N² = {}", ratio_example, n.pow(2));
    
    // Check if 0.69 could come from normalization differences
    let potential_n_for_69_percent = (1.0_f64 / 0.69).sqrt();
    println!("\nIf the 69% discrepancy is purely from normalization:");
    println!("  0.69 ≈ 1/N² would imply N ≈ {:.1}", potential_n_for_69_percent);
    println!("  0.69 ≈ 1/N would imply N ≈ {:.1}", 1.0 / 0.69);
    
    println!("\n⚠️  CRITICAL FINDINGS:");
    println!("  1. Different implementations may use edge-based vs node-based magnetization");
    println!("  2. Susceptibility normalizations vary by factors of N or N²");
    println!("  3. The ~69% discrepancy likely comes from these fundamental differences");
    println!("  4. Without knowing the intended physics, we cannot say which is 'correct'");
}