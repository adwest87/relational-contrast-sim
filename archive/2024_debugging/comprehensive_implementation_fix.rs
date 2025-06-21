// Comprehensive fix for all implementation discrepancies
// Now that initial states match, we need to ensure MC algorithms are identical

use scan::graph::Graph;
use scan::graph_fast::FastGraph;
use scan::graph_ultra_optimized::UltraOptimizedGraph;
use rand::SeedableRng;
use rand::Rng;
use rand_pcg::Pcg64;

fn main() {
    println!("=== COMPREHENSIVE IMPLEMENTATION FIX ===\n");
    
    let n = 6; // Small system for detailed debugging
    let seed = 42;
    let alpha = 1.37;
    let beta = 2.0;
    
    println!("Test parameters: N={}, seed={}, α={:.3}, β={:.3}\n", n, seed, alpha, beta);
    
    // 1. Verify initial state consistency (should be fixed now)
    println!("1. Verifying initial state consistency...");
    let mut rng = Pcg64::seed_from_u64(seed);
    let reference = Graph::complete_random_with(&mut rng, n);
    let fast_graph = FastGraph::from_graph(&reference);
    let ultra_graph = UltraOptimizedGraph::from_graph(&reference);
    
    let ref_triangle = reference.triangle_sum();
    let fast_triangle = fast_graph.triangle_sum();
    let ultra_triangle = ultra_graph.triangle_sum();
    
    println!("   Reference triangle sum: {:.12}", ref_triangle);
    println!("   FastGraph triangle sum: {:.12}", fast_triangle);
    println!("   UltraOptimized triangle sum: {:.12}", ultra_triangle);
    
    let fast_error = (fast_triangle - ref_triangle).abs();
    let ultra_error = (ultra_triangle - ref_triangle).abs();
    
    if fast_error < 1e-12 && ultra_error < 1e-12 {
        println!("   ✅ All initial states match exactly");
    } else {
        println!("   ❌ Initial state discrepancy detected");
        println!("   FastGraph error: {:.2e}", fast_error);
        println!("   UltraOptimized error: {:.2e}", ultra_error);
        return;
    }
    
    // 2. Debug Monte Carlo algorithm differences step by step
    println!("\n2. Debugging Monte Carlo algorithm differences...");
    
    // Create fresh copies for detailed step-by-step analysis
    let mut fast_mc = FastGraph::from_graph(&reference);
    let mut ultra_mc = UltraOptimizedGraph::from_graph(&reference);
    
    println!("   Analyzing single Monte Carlo step in detail...");
    
    // Use same RNG state for both
    let mut rng1 = Pcg64::seed_from_u64(seed + 1000);
    let mut rng2 = Pcg64::seed_from_u64(seed + 1000);
    
    // Record pre-step state
    let fast_pre_triangle = fast_mc.triangle_sum();
    let ultra_pre_triangle = ultra_mc.triangle_sum();
    
    println!("   Pre-step triangle sums:");
    println!("     FastGraph: {:.12}", fast_pre_triangle);
    println!("     UltraOptimized: {:.12}", ultra_pre_triangle);
    
    // Perform one step
    let fast_step_info = fast_mc.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng1);
    let ultra_accept = ultra_mc.metropolis_step(alpha, beta, 0.0, 0.1, 0.1, &mut rng2);
    
    let fast_post_triangle = fast_mc.triangle_sum();
    let ultra_post_triangle = ultra_mc.triangle_sum();
    
    println!("   Post-step triangle sums:");
    println!("     FastGraph: {:.12} (accepted: {})", fast_post_triangle, fast_step_info.accept);
    println!("     UltraOptimized: {:.12} (accepted: {})", ultra_post_triangle, ultra_accept);
    
    let post_step_diff = (fast_post_triangle - ultra_post_triangle).abs();
    println!("   Post-step difference: {:.2e}", post_step_diff);
    
    if post_step_diff > 1e-10 {
        println!("   🚨 MONTE CARLO ALGORITHMS DIFFER!");
        
        // 3. Identify specific differences
        println!("\n3. Analyzing algorithmic differences...");
        
        println!("   Possible causes:");
        println!("   a) Different random number consumption");
        println!("   b) Different proposal mechanisms");
        println!("   c) Different acceptance criteria");
        println!("   d) Different update ordering");
        println!("   e) Numerical precision differences");
        
        // Test random number consumption
        println!("\n   Testing random number consumption:");
        let mut test_rng1 = Pcg64::seed_from_u64(12345);
        let mut test_rng2 = Pcg64::seed_from_u64(12345);
        
        println!("   First 5 random numbers:");
        for i in 0..5 {
            let r1: f64 = test_rng1.gen();
            let r2: f64 = test_rng2.gen();
            println!("     {}: {:.12} vs {:.12} (diff: {:.2e})", i+1, r1, r2, (r1-r2).abs());
        }
        
        // Test specific implementation signatures
        println!("\n   Implementation signatures:");
        println!("   FastGraph.metropolis_step(α, β, Δz, Δθ, rng) -> StepInfo");
        println!("   UltraOptimized.metropolis_step(α, β, γ, Δz, Δθ, rng) -> bool");
        println!("   🚨 DIFFERENT SIGNATURES! UltraOptimized has extra γ parameter");
        
    } else {
        println!("   ✅ Monte Carlo steps are identical");
    }
    
    // 4. Test with many steps to see evolution
    println!("\n4. Testing evolution over many steps...");
    
    let mut fast_evolution = FastGraph::from_graph(&reference);
    let mut ultra_evolution = UltraOptimizedGraph::from_graph(&reference);
    
    let mut rng_fast = Pcg64::seed_from_u64(seed + 2000);
    let mut rng_ultra = Pcg64::seed_from_u64(seed + 2000);
    
    let checkpoints = [1, 5, 10, 20, 50];
    
    for &checkpoint in &checkpoints {
        // Evolve to checkpoint
        while fast_evolution.links.len() < checkpoint {
            // This is wrong - we need step counter, not links.len()
            break;
        }
        
        // Let's just do a fixed number of steps
        for _ in 0..checkpoint {
            fast_evolution.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng_fast);
            ultra_evolution.metropolis_step(alpha, beta, 0.0, 0.1, 0.1, &mut rng_ultra);
        }
        
        let fast_tri = fast_evolution.triangle_sum();
        let ultra_tri = ultra_evolution.triangle_sum();
        let diff = (fast_tri - ultra_tri).abs();
        
        println!("   After {} steps: diff = {:.2e}", checkpoint, diff);
        
        if diff > 0.1 {
            println!("     🚨 Large divergence detected");
            break;
        }
    }
    
    // 5. Summary and recommendations
    println!("\n=== SUMMARY AND RECOMMENDATIONS ===");
    
    println!("✅ FIXED: Initial state generation");
    println!("   UltraOptimized.from_graph() now creates identical states");
    
    if post_step_diff > 1e-10 {
        println!("❌ REMAINING: Monte Carlo algorithm discrepancies");
        println!("\n📋 Required fixes:");
        println!("1. Unify Metropolis method signatures:");
        println!("   - FastGraph: metropolis_step(α, β, Δz, Δθ, rng) -> StepInfo");
        println!("   - UltraOptimized: metropolis_step(α, β, γ, Δz, Δθ, rng) -> bool");
        println!("   Fix: Make γ=0 the default in UltraOptimized when not needed");
        
        println!("\n2. Ensure identical random number consumption:");
        println!("   - Both should draw same sequence for identical decisions");
        println!("   - Check link selection order");
        println!("   - Verify proposal generation");
        
        println!("\n3. Standardize acceptance criteria:");
        println!("   - Use identical Metropolis formula");
        println!("   - Same numerical precision");
        println!("   - Same handling of edge cases (ΔS = 0, etc.)");
        
        println!("\n4. Validate triangle sum updates:");
        println!("   - FastGraph: recalculates from scratch?");
        println!("   - UltraOptimized: uses incremental cache");
        println!("   - Ensure both give identical results");
        
    } else {
        println!("✅ ALL IMPLEMENTATIONS NOW CONSISTENT!");
        println!("   Ready for physics validation");
    }
    
    println!("\n=== NEXT STEPS ===");
    println!("1. Fix remaining MC discrepancies identified above");
    println!("2. Re-run implementation consistency tests");
    println!("3. Validate all physics calculations are identical");
    println!("4. Re-run critical ridge exploration with fixed implementations");
    println!("5. Determine if 'exotic physics' was purely due to bugs");
}