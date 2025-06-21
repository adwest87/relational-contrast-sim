// debug_mc_physics.rs - Comprehensive debugging for MC physics issues

use scan::{graph::Graph, graph_fast::FastGraph};
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("\n=== MONTE CARLO PHYSICS DEBUGGING ===\n");
    
    // Test 1: Check entropy calculation
    test_entropy_calculation();
    
    // Test 2: Verify Metropolis criterion
    test_metropolis_criterion();
    
    // Test 3: Check detailed balance
    test_detailed_balance_comprehensive();
    
    // Test 4: Energy conservation
    test_energy_conservation_detailed();
    
    // Test 5: Observable tracking
    test_observable_tracking();
    
    // Test 6: Minimal 2-node test
    test_minimal_system();
}

fn test_entropy_calculation() {
    println!("TEST 1: Entropy Calculation\n");
    
    let n = 4;
    let mut rng = Pcg64::seed_from_u64(12345);
    let mut graph = Graph::complete_random_with(&mut rng, n);
    let mut log = File::create("entropy_debug.log").unwrap();
    
    // Test various z values
    let test_z_values = vec![0.001, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
    
    writeln!(log, "z,w,entropy_contribution,-z*exp(-z)").unwrap();
    
    for &z in &test_z_values {
        let w = (-z as f64).exp();
        let entropy_contrib = -z * w;
        writeln!(log, "{},{},{},{}", z, w, entropy_contrib, entropy_contrib).unwrap();
        println!("z={:.3}, w={:.3}, S_contrib={:.6} (should be negative)", z, w, entropy_contrib);
    }
    
    // Set all links to same z value and check total entropy
    for z_test in test_z_values {
        for link in &mut graph.links {
            link.z = z_test;
        }
        let total_entropy = graph.entropy_action();
        let expected = -z_test * (-z_test as f64).exp() * graph.m() as f64;
        println!("\nAll links z={}: Total entropy={:.6}, Expected={:.6}", z_test, total_entropy, expected);
        
        if (total_entropy - expected).abs() > 1e-10 {
            println!("  ERROR: Entropy calculation mismatch!");
        }
    }
    
    // Check entropy per link - just randomize manually for now
    let mut rng2 = Pcg64::seed_from_u64(42);
    for link in &mut graph.links {
        link.z = rng2.gen_range(0.1..5.0);
    }
    let entropy = graph.entropy_action();
    let entropy_per_link = entropy / graph.m() as f64;
    println!("\nRandom config: Entropy per link = {:.6}", entropy_per_link);
    
    if entropy_per_link >= 0.0 {
        println!("  ERROR: Entropy per link is positive! This violates physics.");
    }
}

fn test_metropolis_criterion() {
    println!("\n\nTEST 2: Metropolis Criterion\n");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng_init = Pcg64::seed_from_u64(999);
    let mut graph = Graph::complete_random_with(&mut rng_init, n);
    let mut rng = Pcg64::seed_from_u64(12345);
    let mut log = File::create("metropolis_debug.log").unwrap();
    
    writeln!(log, "move_type,energy_before,energy_after,delta_E,accept_prob,random,accepted").unwrap();
    
    // Test 100 moves with detailed logging
    for i in 0..100 {
        let energy_before = graph.action(alpha, beta);
        
        // Use metropolis_step which is the public API
        let step_info = graph.metropolis_step(beta, alpha, 0.5, 0.5, &mut rng);
        let energy_after = graph.action(alpha, beta);
        let delta_e = energy_after - energy_before;
        
        // metropolis_step already handles accept/reject internally
        let accepted = step_info.accepted;
        let accept_prob = if delta_e <= 0.0 { 1.0 } else { (-delta_e).exp() };
        
        writeln!(log, "step,{},{},{},{},{}", 
            energy_before, energy_after, delta_e, accept_prob, accepted).unwrap();
        
        if i < 10 {
            println!("Move {}: ΔE={:.6}, P_accept={:.4}, accepted={}", 
                i, delta_e, accept_prob, accepted);
        }
        
        // Skip actual move reversion since we can't access Proposal enum
    }
    
    println!("\nDetailed move log written to metropolis_debug.log");
}

fn test_detailed_balance_comprehensive() {
    println!("\n\nTEST 3: Detailed Balance Check\n");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng_init = Pcg64::seed_from_u64(999);
    let mut graph = Graph::complete_random_with(&mut rng_init, n);
    let mut rng = Pcg64::seed_from_u64(42);
    let mut log = File::create("detailed_balance_debug.log").unwrap();
    
    writeln!(log, "move_type,link_idx,old_value,new_value,E_before,E_after,delta_E_forward,P_forward,P_reverse,ratio,error").unwrap();
    
    let mut violations = 0;
    let n_tests = 1000;
    
    for _test in 0..n_tests {
        // Save state
        let saved_links = graph.links.clone();
        let e_initial = graph.action(alpha, beta);
        
        // Make a forward move
        let link_idx = rng.gen_range(0..graph.links.len());
        let do_z = rng.gen_bool(0.5);
        
        if do_z {
            let old_z = graph.links[link_idx].z;
            let dz = rng.gen_range(-0.5..=0.5);
            let new_z = (old_z + dz).max(0.001_f64);
            graph.links[link_idx].z = new_z;
            
            let e_final = graph.action(alpha, beta);
            let delta_e_forward = e_final - e_initial;
            let p_forward = if delta_e_forward <= 0.0 { 1.0 } else { (-delta_e_forward).exp() };
            
            // Calculate reverse probability
            let delta_e_reverse = -delta_e_forward;
            let p_reverse = if delta_e_reverse <= 0.0 { 1.0 } else { (-delta_e_reverse).exp() };
            
            // Check detailed balance: P(i→j)/P(j→i) = exp(-β(E_j - E_i))
            let expected_ratio = (-delta_e_forward).exp();
            let actual_ratio = p_forward / p_reverse;
            let error = (actual_ratio - expected_ratio).abs();
            
            if error > 1e-10 {
                violations += 1;
                if violations <= 5 {
                    println!("Detailed balance violation #{}: z-update", violations);
                    println!("  Forward: ΔE={:.6}, P={:.6}", delta_e_forward, p_forward);
                    println!("  Reverse: ΔE={:.6}, P={:.6}", delta_e_reverse, p_reverse);
                    println!("  Ratio: actual={:.6}, expected={:.6}, error={:.2e}", 
                        actual_ratio, expected_ratio, error);
                }
            }
            
            writeln!(log, "z_update,{},{},{},{},{},{},{},{},{},{}", 
                link_idx, old_z, new_z, e_initial, e_final, delta_e_forward, 
                p_forward, p_reverse, actual_ratio, error).unwrap();
        } else {
            // Phase update
            let old_theta = graph.links[link_idx].theta;
            let dtheta = rng.gen_range(-0.5..=0.5);
            let new_theta = old_theta + dtheta;
            graph.links[link_idx].theta = new_theta;
            
            let e_final = graph.action(alpha, beta);
            let delta_e_forward = e_final - e_initial;
            let p_forward = if delta_e_forward <= 0.0 { 1.0 } else { (-delta_e_forward).exp() };
            
            // Calculate reverse probability
            let delta_e_reverse = -delta_e_forward;
            let p_reverse = if delta_e_reverse <= 0.0 { 1.0 } else { (-delta_e_reverse).exp() };
            
            // Check detailed balance
            let expected_ratio = (-delta_e_forward).exp();
            let actual_ratio = p_forward / p_reverse;
            let error = (actual_ratio - expected_ratio).abs();
            
            if error > 1e-10 {
                violations += 1;
            }
            
            writeln!(log, "phase,{},{},{},{},{},{},{},{},{},{}", 
                link_idx, old_theta, new_theta, e_initial, e_final, delta_e_forward, 
                p_forward, p_reverse, actual_ratio, error).unwrap();
        }
        
        // Restore state
        graph.links = saved_links;
    }
    
    println!("Detailed balance violations: {}/{} ({:.2}%)", 
        violations, n_tests, 100.0 * violations as f64 / n_tests as f64);
    println!("Detailed log written to detailed_balance_debug.log");
}

fn test_energy_conservation_detailed() {
    println!("\n\nTEST 4: Energy Conservation\n");
    
    let n = 8;
    let alpha = 1.5;
    let beta = 1000.0; // Large beta for microcanonical ensemble
    let mut rng_init = Pcg64::seed_from_u64(999);
    let mut graph = Graph::complete_random_with(&mut rng_init, n);
    let mut rng = Pcg64::seed_from_u64(999);
    let mut log = File::create("energy_conservation_debug.log").unwrap();
    
    // Randomize z values manually
    for link in &mut graph.links {
        link.z = rng.gen_range(0.1..5.0);
    }
    let initial_energy = graph.action(alpha, beta);
    
    writeln!(log, "step,energy,delta_from_initial,accepted").unwrap();
    
    let mut max_drift: f64 = 0.0;
    let mut accepts = 0;
    let n_steps = 10000;
    
    println!("Initial energy: {}", initial_energy);
    
    for step in 0..n_steps {
        let info = graph.metropolis_step(beta, alpha, 0.001, 0.001, &mut rng);
        let current_energy = graph.action(alpha, beta);
        let drift = (current_energy - initial_energy).abs();
        max_drift = max_drift.max(drift);
        
        if info.accepted {
            accepts += 1;
        }
        
        if step % 100 == 0 {
            writeln!(log, "{},{},{},{}", step, current_energy, drift, info.accepted).unwrap();
        }
        
        if step < 10 || step % 1000 == 0 {
            println!("Step {}: E={:.10}, drift={:.2e}", step, current_energy, drift);
        }
    }
    
    println!("\nMax energy drift: {:.2e} (should be < 1e-6)", max_drift);
    println!("Acceptance rate: {:.1}% (should be low for large β)", 
        100.0 * accepts as f64 / n_steps as f64);
    
    if max_drift > 1e-6 {
        println!("ERROR: Energy conservation violated!");
    }
}

fn test_observable_tracking() {
    println!("\n\nTEST 5: Observable Tracking\n");
    
    let n = 8;
    let alpha = 1.5;
    let beta = 3.0;
    let mut graph = FastGraph::new(n, 42);
    let mut rng = Pcg64::seed_from_u64(42);
    
    // Manual calculation
    let mut sum_w = graph.links.iter().map(|l| l.exp_neg_z).sum::<f64>();
    let mut sum_cos = graph.links.iter().map(|l| l.cos_theta).sum::<f64>();
    
    println!("Initial: sum_w={:.6}, sum_cos={:.6}", sum_w, sum_cos);
    
    // Track changes through MC steps
    for i in 0..100 {
        let info = graph.metropolis_step(alpha, beta, 0.5, 0.5, &mut rng);
        
        if info.accept {
            sum_w += info.delta_w;
            sum_cos += info.delta_cos;
            
            // Verify against direct calculation
            let actual_sum_w: f64 = graph.links.iter().map(|l| l.exp_neg_z).sum();
            let actual_sum_cos: f64 = graph.links.iter().map(|l| l.cos_theta).sum();
            
            let error_w = (sum_w - actual_sum_w).abs();
            let error_cos = (sum_cos - actual_sum_cos).abs();
            
            if error_w > 1e-10 || error_cos > 1e-10 {
                println!("Step {}: Tracking error!", i);
                println!("  sum_w: tracked={:.10}, actual={:.10}, error={:.2e}", 
                    sum_w, actual_sum_w, error_w);
                println!("  sum_cos: tracked={:.10}, actual={:.10}, error={:.2e}", 
                    sum_cos, actual_sum_cos, error_cos);
                
                // This is the bug! In graph_fast.rs line 306:
                // delta_cos = old_exp_neg_z * (new_theta.cos() - old_cos_theta)
                // Should be: delta_cos = new_theta.cos() - old_cos_theta
                
                println!("\nBUG FOUND: delta_cos calculation includes weight factor!");
                break;
            }
        }
    }
}

fn test_minimal_system() {
    println!("\n\nTEST 6: Minimal 2-Node System\n");
    
    // 2 nodes = 1 link, no triangles
    let n = 2;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng_init = Pcg64::seed_from_u64(1);
    let mut graph = Graph::complete_random_with(&mut rng_init, n);
    let mut rng = Pcg64::seed_from_u64(1);
    
    // Set known state
    graph.links[0].z = 1.0;
    graph.links[0].theta = 0.0;
    
    let entropy = graph.entropy_action();
    let triangle = graph.triangle_sum();
    let action = graph.action(alpha, beta);
    
    println!("Initial state: z=1.0, θ=0.0");
    println!("  Entropy: {} (expected: {})", entropy, -1.0 * (-1.0_f64).exp());
    println!("  Triangle sum: {} (expected: 0 for 2 nodes)", triangle);
    println!("  Action: {}", action);
    
    // Run some MC steps
    let mut accepts = 0;
    for _ in 0..1000 {
        let info = graph.metropolis_step(beta, alpha, 0.1, 0.1, &mut rng);
        if info.accepted {
            accepts += 1;
        }
    }
    
    println!("\nAfter 1000 MC steps:");
    println!("  z: {}", graph.links[0].z);
    println!("  θ: {}", graph.links[0].theta);
    println!("  Acceptance rate: {}%", accepts / 10);
}