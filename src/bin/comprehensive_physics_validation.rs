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
    
    let n = 8;  // Small size for detailed comparison
    let alpha = 1.5;
    let beta = 2.9;
    let seed = 42u64;
    
    println!("Testing with N={}, α={}, β={}, seed={}\n", n, alpha, beta, seed);
    
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
    let mut rng = Pcg64::seed_from_u64(seed);
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