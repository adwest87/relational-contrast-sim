// Critical diagnostic for stuck anti-ordered system
use scan::graph_fast::{FastGraph, BatchedObservables};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::f64::consts::PI;

fn main() {
    println!("🚨 CRITICAL DIAGNOSTIC: STUCK ANTI-ORDERED SYSTEM");
    println!("=================================================");
    
    let n = 32;
    let alpha = 1.5;
    let temperature = 10.0;
    let beta = 1.0 / temperature;
    
    println!("System: N={}, T={}, α={}, β={}", n, temperature, alpha, beta);
    
    // Initialize anti-ordered system (all θ=π)
    let mut graph = FastGraph::new(n, 12345);
    for i in 0..graph.links.len() {
        graph.links[i].update_theta(PI);
    }
    let mut rng = Pcg64::seed_from_u64(42);
    
    println!("\n📊 PART 1: MOVE LOGGING (1000 steps with δθ=0.3)");
    println!("================================================");
    
    let mut sign_flip_attempts = 0;
    let mut sign_flip_accepts = 0;
    let mut total_theta_moves = 0;
    let mut accepted_theta_moves = 0;
    let mut energy_changes = Vec::new();
    
    // Log first few moves in detail
    for step in 0..1000 {
        let energy_before = graph.action(alpha, beta);
        
        // Get link to be updated
        let link_idx = rng.gen_range(0..graph.links.len());
        let do_z_update = 0.3 > 0.0 && rng.gen_bool(0.5);
        
        if !do_z_update { // Theta update
            total_theta_moves += 1;
            
            let old_theta = graph.links[link_idx].theta;
            let old_cos = old_theta.cos();
            let d_theta = rng.gen_range(-0.3..=0.3);
            let new_theta = old_theta + d_theta;
            let new_cos = new_theta.cos();
            
            // Check if this would flip the sign of cos θ
            let would_flip_sign = old_cos * new_cos < 0.0;
            if would_flip_sign {
                sign_flip_attempts += 1;
            }
            
            // Perform the move
            let info = graph.metropolis_step(alpha, beta, 0.3, 0.3, &mut rng);
            
            if info.accept && !do_z_update {
                accepted_theta_moves += 1;
                if would_flip_sign {
                    sign_flip_accepts += 1;
                }
            }
            
            let energy_after = graph.action(alpha, beta);
            let delta_e = energy_after - energy_before;
            
            if delta_e.abs() > 1e-10 {
                energy_changes.push(delta_e);
            }
            
            // Log first 10 theta moves in detail
            if step < 10 && !do_z_update {
                println!("\nStep {}: Link {} theta move", step, link_idx);
                println!("  Old θ: {:.4} (cos={:.4})", old_theta, old_cos);
                println!("  Δθ: {:.4}, New θ: {:.4} (cos={:.4})", d_theta, new_theta, new_cos);
                println!("  Would flip sign: {}", would_flip_sign);
                println!("  Energy: {:.6} → {:.6} (ΔE={:.6})", energy_before, energy_after, delta_e);
                println!("  Accepted: {}", info.accept);
            }
        }
    }
    
    println!("\n📈 MOVE STATISTICS:");
    println!("Total theta moves: {}", total_theta_moves);
    println!("Accepted theta moves: {} ({:.1}%)", accepted_theta_moves, 
        100.0 * accepted_theta_moves as f64 / total_theta_moves as f64);
    println!("Sign flip attempts: {}", sign_flip_attempts);
    println!("Sign flip accepts: {} ({:.1}%)", sign_flip_accepts,
        if sign_flip_attempts > 0 { 100.0 * sign_flip_accepts as f64 / sign_flip_attempts as f64 } else { 0.0 });
    
    if !energy_changes.is_empty() {
        let mean_delta_e = energy_changes.iter().sum::<f64>() / energy_changes.len() as f64;
        let max_delta_e = energy_changes.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        println!("Mean |ΔE|: {:.6}", mean_delta_e.abs());
        println!("Max |ΔE|: {:.6}", max_delta_e.abs());
    }
    
    println!("\n🔄 PART 2: TEST DIFFERENT MOVE SIZES");
    println!("====================================");
    
    let move_sizes = vec![
        ("Small (δθ=0.3)", 0.3),
        ("Medium (δθ=π/2)", PI/2.0),
        ("Large (δθ=π)", PI),
        ("Full rotation (δθ=2π)", 2.0*PI),
    ];
    
    for (name, delta_theta) in move_sizes {
        // Reset to anti-ordered
        for i in 0..graph.links.len() {
            graph.links[i].update_theta(PI);
        }
        
        let mut accepts = 0;
        let mut theta_moves = 0;
        let mut final_mean_cos = 0.0;
        
        // Run 10000 steps with this move size
        for _ in 0..10000 {
            let info = graph.metropolis_step(alpha, beta, 0.3, delta_theta, &mut rng);
            if info.accept {
                accepts += 1;
            }
            
            // Count theta moves (approximately half)
            if rng.gen_bool(0.5) {
                theta_moves += 1;
            }
        }
        
        // Measure final state
        let mut obs_calc = BatchedObservables::new();
        let obs = obs_calc.measure(&graph, alpha, beta);
        final_mean_cos = obs.mean_cos;
        
        println!("\n{}: {} moves", name, theta_moves);
        println!("  Acceptance rate: {:.1}%", 100.0 * accepts as f64 / 10000.0);
        println!("  Final <cos θ>: {:.4}", final_mean_cos);
        println!("  Escaped from -1? {}", final_mean_cos > -0.9);
    }
    
    println!("\n🏔️ PART 3: ENERGY LANDSCAPE ANALYSIS");
    println!("=====================================");
    
    // Test ordered state (all θ=0)
    for i in 0..graph.links.len() {
        graph.links[i].update_theta(0.0);
    }
    let e_ordered = graph.action(alpha, beta);
    let entropy_ordered = graph.entropy_action();
    let triangle_ordered = graph.triangle_sum();
    
    // Test anti-ordered state (all θ=π)
    for i in 0..graph.links.len() {
        graph.links[i].update_theta(PI);
    }
    let e_anti_ordered = graph.action(alpha, beta);
    let entropy_anti = graph.entropy_action();
    let triangle_anti = graph.triangle_sum();
    
    // Test random state
    let mut rng_fresh = Pcg64::seed_from_u64(999);
    for i in 0..graph.links.len() {
        let theta = rng_fresh.gen_range(0.0..2.0*PI);
        graph.links[i].update_theta(theta);
    }
    let e_random = graph.action(alpha, beta);
    let entropy_random = graph.entropy_action();
    let triangle_random = graph.triangle_sum();
    
    println!("Energy landscape:");
    println!("  E(θ=0):     {:.6} (S_entropy={:.3}, S_triangle={:.3})", 
        e_ordered, entropy_ordered, triangle_ordered);
    println!("  E(θ=π):     {:.6} (S_entropy={:.3}, S_triangle={:.3})", 
        e_anti_ordered, entropy_anti, triangle_anti);
    println!("  E(random):  {:.6} (S_entropy={:.3}, S_triangle={:.3})", 
        e_random, entropy_random, triangle_random);
    println!("  ΔE(π→0):    {:.6}", e_ordered - e_anti_ordered);
    println!("  ΔE(π→rand): {:.6}", e_random - e_anti_ordered);
    
    // Check if anti-ordered is a deep minimum
    if e_anti_ordered < e_ordered && e_anti_ordered < e_random {
        println!("\n⚠️ WARNING: Anti-ordered state (θ=π) is the GLOBAL MINIMUM!");
    } else if e_anti_ordered < e_random {
        println!("\n⚠️ Anti-ordered state is lower energy than random!");
    }
    
    println!("\n🔬 PART 4: SINGLE SPIN FLIP ENERGY BARRIER");
    println!("==========================================");
    
    // Reset to anti-ordered
    for i in 0..graph.links.len() {
        graph.links[i].update_theta(PI);
    }
    
    // Try flipping a single link from π to 0
    let test_link = 0;
    let e_before_flip = graph.action(alpha, beta);
    graph.links[test_link].update_theta(0.0);
    let e_after_flip = graph.action(alpha, beta);
    graph.links[test_link].update_theta(PI); // restore
    
    let barrier = e_after_flip - e_before_flip;
    let boltzmann_factor = (-barrier * beta).exp();
    
    println!("Single link flip (π→0):");
    println!("  Energy barrier: ΔE = {:.6}", barrier);
    println!("  Boltzmann factor at T={}: exp(-βΔE) = {:.6e}", temperature, boltzmann_factor);
    println!("  Acceptance probability: {:.2}%", 100.0 * boltzmann_factor.min(1.0));
    
    // Try intermediate angles
    println!("\nIntermediate angle barriers:");
    let test_angles = vec![3.0*PI/4.0, PI/2.0, PI/4.0, 0.0];
    for &angle in &test_angles {
        graph.links[test_link].update_theta(angle);
        let e_angle = graph.action(alpha, beta);
        let barrier_angle = e_angle - e_before_flip;
        println!("  θ={:.2} (cos={:.3}): ΔE={:.6}, P(accept)={:.2}%", 
            angle, angle.cos(), barrier_angle, 
            100.0 * (-barrier_angle * beta).exp().min(1.0));
        graph.links[test_link].update_theta(PI); // restore
    }
    
    println!("\n🎯 PART 5: GLOBAL MOVE TEST");
    println!("===========================");
    
    // Test global flip: all θ → -θ
    for i in 0..graph.links.len() {
        graph.links[i].update_theta(PI);
    }
    let e_before_global = graph.action(alpha, beta);
    
    // Flip all angles
    for i in 0..graph.links.len() {
        graph.links[i].update_theta(0.0);
    }
    let e_after_global = graph.action(alpha, beta);
    
    let global_barrier = e_after_global - e_before_global;
    println!("Global flip (all π→0):");
    println!("  Energy change: ΔE = {:.6}", global_barrier);
    println!("  Per link: ΔE/N_links = {:.6}", global_barrier / graph.links.len() as f64);
    
    // Test if there's a symmetry
    println!("\n🔍 SYMMETRY CHECK:");
    
    // Compare cos distributions
    let mut cos_values_0 = Vec::new();
    let mut cos_values_pi = Vec::new();
    
    for i in 0..graph.links.len() {
        graph.links[i].update_theta(0.0);
    }
    for link in &graph.links {
        cos_values_0.push(link.cos_theta);
    }
    
    for i in 0..graph.links.len() {
        graph.links[i].update_theta(PI);
    }
    for link in &graph.links {
        cos_values_pi.push(link.cos_theta);
    }
    
    println!("All θ=0:  All cos θ = {:.3}", cos_values_0[0]);
    println!("All θ=π:  All cos θ = {:.3}", cos_values_pi[0]);
    println!("Triangle sum invariant under θ→θ+π? {}", 
        (triangle_ordered - triangle_anti).abs() < 1e-10);
    
    println!("\n📋 DIAGNOSIS SUMMARY:");
    println!("====================");
    if sign_flip_accepts == 0 && sign_flip_attempts > 0 {
        println!("❌ NO sign flips accepted with small moves!");
    }
    if barrier > 5.0 / beta {
        println!("❌ Energy barrier ({:.2}) >> kT ({:.2}) - system is FROZEN!", barrier, 1.0/beta);
    }
    if e_anti_ordered < e_random {
        println!("❌ Anti-ordered state is energetically favored over disorder!");
    }
}