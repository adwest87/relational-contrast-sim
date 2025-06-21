// Fix UltraOptimized to generate identical initial states as Reference
// The root cause is that UltraOptimized creates its own RNG instead of using the Reference state

use scan::graph::Graph;
use scan::graph_ultra_optimized::UltraOptimizedGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("=== FIXING ULTRA OPTIMIZED INITIAL STATE GENERATION ===\n");
    
    let n = 8;
    let seed = 42;
    let alpha = 1.37;
    let beta = 2.0;
    
    println!("Testing with N={}, seed={}, α={:.3}, β={:.3}\n", n, seed, alpha, beta);
    
    // 1. Generate Reference state
    println!("1. Creating Reference state...");
    let mut rng = Pcg64::seed_from_u64(seed);
    let reference = Graph::complete_random_with(&mut rng, n);
    
    let ref_triangle = reference.triangle_sum();
    let ref_action = reference.action(alpha, beta);
    
    println!("   Reference triangle sum: {:.6}", ref_triangle);
    println!("   Reference action: {:.6}", ref_action);
    
    // 2. Create UltraOptimized with SAME RNG state (this will fail with current implementation)
    println!("\n2. Attempting to create UltraOptimized with SAME initial state...");
    
    // We need to modify UltraOptimized to accept a Reference graph for conversion
    // This is the proper fix - just like FastGraph does
    
    println!("🚨 PROBLEM IDENTIFIED:");
    println!("   UltraOptimized.new(n, seed) creates its own RNG");
    println!("   This generates completely different random numbers than Reference");
    println!("   FastGraph correctly uses FastGraph::from_graph(&reference)");
    
    println!("\n=== REQUIRED FIX ===");
    println!("1. Add UltraOptimizedGraph::from_graph(reference: &Graph) method");
    println!("2. Convert Reference state to UltraOptimized layout");
    println!("3. Ensure identical triangle sums and actions");
    
    // 3. Demonstrate the problem with current implementation
    println!("\n3. Demonstrating the problem with current UltraOptimized:");
    let ultra_graph = UltraOptimizedGraph::new(n, seed);
    
    let ultra_triangle = ultra_graph.triangle_sum();
    let ultra_action = ultra_graph.action(alpha, beta, 0.0);
    
    println!("   UltraOptimized triangle sum: {:.6}", ultra_triangle);
    println!("   UltraOptimized action: {:.6}", ultra_action);
    
    let triangle_diff = (ultra_triangle - ref_triangle).abs();
    let action_diff = (ultra_action - ref_action).abs();
    
    println!("\n   Triangle sum difference: {:.2e}", triangle_diff);
    println!("   Action difference: {:.2e}", action_diff);
    
    if triangle_diff > 1e-6 || action_diff > 1e-6 {
        println!("   ❌ DIFFERENT INITIAL STATES CONFIRMED");
        println!("   This is the ROOT CAUSE of all implementation discrepancies");
    } else {
        println!("   ✅ Initial states match (unexpected!)");
    }
    
    println!("\n=== IMPLEMENTATION PLAN ===");
    println!("1. Modify src/graph_ultra_optimized.rs:");
    println!("   - Add from_graph(&Graph) constructor");
    println!("   - Copy z_values and theta_values from Reference links");
    println!("   - Ensure link ordering is identical");
    println!("   - Pre-compute cos/sin/exp values from Reference data");
    
    println!("\n2. Update all test scripts to use:");
    println!("   let reference = Graph::complete_random_with(&mut rng, n);");
    println!("   let fast_graph = FastGraph::from_graph(&reference);");
    println!("   let ultra_graph = UltraOptimizedGraph::from_graph(&reference);");
    
    println!("\n3. Validate that all implementations agree to <1e-12 precision");
    
    println!("\n=== NEXT STEPS ===");
    println!("1. Implement UltraOptimizedGraph::from_graph() method");
    println!("2. Test initial state consistency");
    println!("3. Re-run all physics validation after fix");
    println!("4. Verify Monte Carlo evolution is identical");
}