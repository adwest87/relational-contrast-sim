// Test that UltraOptimized fix produces identical initial states as Reference
// This validates the from_graph() method implementation

use scan::graph::Graph;
use scan::graph_fast::FastGraph;
use scan::graph_ultra_optimized::UltraOptimizedGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("=== TESTING ULTRA OPTIMIZED FIX ===\n");
    
    let n = 8;
    let seed = 42;
    let alpha = 1.37;
    let beta = 2.0;
    
    println!("Test parameters: N={}, seed={}, α={:.3}, β={:.3}\n", n, seed, alpha, beta);
    
    // 1. Create Reference state
    println!("1. Creating Reference state...");
    let mut rng = Pcg64::seed_from_u64(seed);
    let reference = Graph::complete_random_with(&mut rng, n);
    
    let ref_triangle = reference.triangle_sum();
    let ref_action = reference.action(alpha, beta);
    
    println!("   Reference triangle sum: {:.12}", ref_triangle);
    println!("   Reference action: {:.12}", ref_action);
    
    // 2. Create FastGraph from Reference (should be identical)
    println!("\n2. Creating FastGraph from Reference...");
    let fast_graph = FastGraph::from_graph(&reference);
    
    let fast_triangle = fast_graph.triangle_sum();
    let fast_action = fast_graph.action(alpha, beta);
    
    println!("   FastGraph triangle sum: {:.12}", fast_triangle);
    println!("   FastGraph action: {:.12}", fast_action);
    
    let fast_triangle_error = (fast_triangle - ref_triangle).abs();
    let fast_action_error = (fast_action - ref_action).abs();
    
    println!("   FastGraph triangle error: {:.2e}", fast_triangle_error);
    println!("   FastGraph action error: {:.2e}", fast_action_error);
    
    if fast_triangle_error < 1e-10 && fast_action_error < 1e-10 {
        println!("   ✅ FastGraph matches Reference exactly");
    } else {
        println!("   ❌ FastGraph conversion failed");
        return;
    }
    
    // 3. Create UltraOptimized from Reference (NEW FIXED VERSION)
    println!("\n3. Creating UltraOptimized from Reference (FIXED)...");
    let ultra_graph = UltraOptimizedGraph::from_graph(&reference);
    
    let ultra_triangle = ultra_graph.triangle_sum();
    let ultra_action = ultra_graph.action(alpha, beta, 0.0);
    
    println!("   UltraOptimized triangle sum: {:.12}", ultra_triangle);
    println!("   UltraOptimized action: {:.12}", ultra_action);
    
    let ultra_triangle_error = (ultra_triangle - ref_triangle).abs();
    let ultra_action_error = (ultra_action - ref_action).abs();
    
    println!("   UltraOptimized triangle error: {:.2e}", ultra_triangle_error);
    println!("   UltraOptimized action error: {:.2e}", ultra_action_error);
    
    if ultra_triangle_error < 1e-10 && ultra_action_error < 1e-10 {
        println!("   ✅ UltraOptimized matches Reference exactly");
    } else {
        println!("   ❌ UltraOptimized conversion failed");
        println!("   This indicates a bug in the from_graph() implementation");
        return;
    }
    
    // 4. Test magnetization consistency
    println!("\n4. Testing magnetization consistency...");
    
    let ref_mag: f64 = reference.links.iter().map(|link| link.theta.cos()).sum::<f64>() / reference.links.len() as f64;
    let fast_mag: f64 = fast_graph.links.iter().map(|link| link.cos_theta).sum::<f64>() / fast_graph.links.len() as f64;
    let ultra_mag: f64 = ultra_graph.cos_theta.iter().sum::<f64>() / ultra_graph.cos_theta.len() as f64;
    
    println!("   Reference magnetization: {:.12}", ref_mag);
    println!("   FastGraph magnetization: {:.12}", fast_mag);
    println!("   UltraOptimized magnetization: {:.12}", ultra_mag);
    
    let fast_mag_error = (fast_mag - ref_mag).abs();
    let ultra_mag_error = (ultra_mag - ref_mag).abs();
    
    println!("   FastGraph magnetization error: {:.2e}", fast_mag_error);
    println!("   UltraOptimized magnetization error: {:.2e}", ultra_mag_error);
    
    if fast_mag_error < 1e-10 && ultra_mag_error < 1e-10 {
        println!("   ✅ All magnetizations match exactly");
    } else {
        println!("   ❌ Magnetization discrepancy detected");
    }
    
    // 5. Test consistency after a few Monte Carlo steps
    println!("\n5. Testing Monte Carlo consistency...");
    
    // Create identical copies for MC evolution
    let mut fast_mc = FastGraph::from_graph(&reference);
    let mut ultra_mc = UltraOptimizedGraph::from_graph(&reference);
    
    // Use IDENTICAL RNG sequence
    let mut rng1 = Pcg64::seed_from_u64(seed + 1000);
    let mut rng2 = Pcg64::seed_from_u64(seed + 1000);
    
    println!("   Performing 10 MC steps with identical RNG sequences...");
    
    let mut fast_accepts = 0;
    let mut ultra_accepts = 0;
    
    for step in 0..10 {
        let fast_step_info = fast_mc.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng1);
        let ultra_accept = ultra_mc.metropolis_step(alpha, beta, 0.0, 0.1, 0.1, &mut rng2);
        
        if fast_step_info.accept { fast_accepts += 1; }
        if ultra_accept { ultra_accepts += 1; }
        
        let fast_t = fast_mc.triangle_sum();
        let ultra_t = ultra_mc.triangle_sum();
        let triangle_diff = (fast_t - ultra_t).abs();
        
        println!("   Step {}: FastGraph triangle={:.6}, UltraOptimized triangle={:.6}, diff={:.2e}", 
                 step+1, fast_t, ultra_t, triangle_diff);
    }
    
    println!("   FastGraph accepted: {}/10", fast_accepts);
    println!("   UltraOptimized accepted: {}/10", ultra_accepts);
    
    // Final state comparison
    let final_fast_triangle = fast_mc.triangle_sum();
    let final_ultra_triangle = ultra_mc.triangle_sum();
    let final_triangle_diff = (final_fast_triangle - final_ultra_triangle).abs();
    
    let final_fast_action = fast_mc.action(alpha, beta);
    let final_ultra_action = ultra_mc.action(alpha, beta, 0.0);
    let final_action_diff = (final_fast_action - final_ultra_action).abs();
    
    println!("\n   Final state comparison:");
    println!("   FastGraph final triangle: {:.12}", final_fast_triangle);
    println!("   UltraOptimized final triangle: {:.12}", final_ultra_triangle);
    println!("   Triangle difference: {:.2e}", final_triangle_diff);
    println!("   Action difference: {:.2e}", final_action_diff);
    
    // 6. Summary
    println!("\n=== FIX VALIDATION SUMMARY ===");
    
    let initial_fix_success = ultra_triangle_error < 1e-10 && ultra_action_error < 1e-10;
    let mc_consistency = final_triangle_diff < 1e-6; // Allow small MC differences
    
    if initial_fix_success {
        println!("✅ INITIAL STATE FIX: SUCCESS");
        println!("   UltraOptimized now generates identical initial states as Reference");
    } else {
        println!("❌ INITIAL STATE FIX: FAILED");
        println!("   from_graph() method has bugs");
    }
    
    if mc_consistency {
        println!("✅ MONTE CARLO CONSISTENCY: ACCEPTABLE");
        println!("   MC evolution remains reasonably consistent");
    } else {
        println!("❌ MONTE CARLO CONSISTENCY: FAILED");
        println!("   MC evolution diverges too quickly");
    }
    
    if initial_fix_success && mc_consistency {
        println!("\n🎉 IMPLEMENTATION FIX SUCCESSFUL!");
        println!("   Ready to re-run all physics validation tests");
        println!("   Previous 'exotic physics' was due to implementation bugs");
    } else {
        println!("\n⚠️  ADDITIONAL FIXES NEEDED");
        println!("   Implementation discrepancies remain");
    }
}