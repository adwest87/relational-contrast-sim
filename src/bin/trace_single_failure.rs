// CRITICAL: Trace a single ΔE=0 rejection to find the exact bug
use scan::graph_fast::FastGraph;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn main() {
    println!("🔍 TRACING SINGLE ΔE=0 REJECTION");
    println!("=================================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(42);
    let mut graph = FastGraph::new(n, 123);
    
    // Find the first ΔE=0 rejection
    for step in 0..10000 {
        let energy_before = graph.action(alpha, beta);
        let state_before = graph.links.clone();
        let hash_before = hash_state(&state_before);
        
        // Perform move
        let info = graph.metropolis_step(alpha, beta, 0.3, 0.3, &mut rng);
        
        let energy_after = graph.action(alpha, beta);
        let state_after = graph.links.clone();
        let hash_after = hash_state(&state_after);
        let delta_e = energy_after - energy_before;
        
        // Found our bug case!
        if delta_e == 0.0 && !info.accept {
            println!("\n🚨 FOUND ΔE=0 REJECTION at step {}", step);
            println!("==========================================");
            
            // 1. Complete state comparison
            println!("\n1. STATE COMPARISON:");
            println!("Hash before: 0x{:016x}", hash_before);
            println!("Hash after:  0x{:016x}", hash_after);
            println!("States identical: {}", hash_before == hash_after);
            
            if hash_before != hash_after {
                println!("\n🔍 STATE DIFFERENCES:");
                for (i, (before, after)) in state_before.iter().zip(state_after.iter()).enumerate() {
                    if before.z != after.z || before.theta != after.theta {
                        println!("Link {}: z {:.17e} -> {:.17e}, theta {:.17e} -> {:.17e}",
                            i, before.z, after.z, before.theta, after.theta);
                    }
                }
            }
            
            // 2. Energy calculation trace
            println!("\n2. ENERGY CALCULATION TRACE:");
            println!("External energy before: {:.17e}", energy_before);
            println!("External energy after:  {:.17e}", energy_after);
            println!("External ΔE:            {:.17e}", delta_e);
            
            // 3. Step-by-step internal energy calculation
            println!("\n3. INTERNAL CALCULATION RECONSTRUCTION:");
            
            // Restore state and manually trace the metropolis step
            graph.links = state_before.clone();
            trace_metropolis_step(&mut graph, alpha, beta, &mut rng, step);
            
            break;
        }
    }
}

fn hash_state(links: &[scan::graph_fast::FastLink]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for link in links {
        link.z.to_bits().hash(&mut hasher);
        link.theta.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

fn trace_metropolis_step(graph: &mut FastGraph, alpha: f64, beta: f64, rng: &mut Pcg64, original_step: usize) {
    println!("Reconstructing metropolis step {}...", original_step);
    
    // Use the same RNG sequence to reproduce exact same move
    let mut temp_rng = Pcg64::seed_from_u64(42);
    for _ in 0..original_step {
        // Skip the exact same RNG calls that happened before
        let _ = temp_rng.gen_range(0..graph.links.len());
        if 0.3 > 0.0 && temp_rng.gen_bool(0.5) {
            // Z update path
            let _ = temp_rng.gen_range(-0.3..=0.3);
        } else {
            // Theta update path  
            let _ = temp_rng.gen_range(-0.3..=0.3);
        }
    }
    
    // Now perform the exact same move with tracing
    let energy_before = graph.action(alpha, beta);
    println!("Energy before move: {:.17e}", energy_before);
    
    let link_idx = temp_rng.gen_range(0..graph.links.len());
    let do_z_update = 0.3 > 0.0 && temp_rng.gen_bool(0.5);
    
    println!("Move: link_idx={}, do_z_update={}", link_idx, do_z_update);
    
    if do_z_update {
        trace_z_update(graph, link_idx, beta, &mut temp_rng, alpha);
    } else {
        trace_theta_update(graph, link_idx, alpha, &mut temp_rng, beta);
    }
    
    let energy_after = graph.action(alpha, beta);
    println!("Energy after move: {:.17e}", energy_after);
    println!("External ΔE: {:.17e}", energy_after - energy_before);
}

fn trace_z_update(graph: &mut FastGraph, link_idx: usize, beta: f64, rng: &mut Pcg64, alpha: f64) {
    println!("\n🔍 TRACING Z-UPDATE:");
    
    let link = &graph.links[link_idx];
    let old_z = link.z;
    let old_exp_neg_z = link.exp_neg_z;
    
    let d_z = rng.gen_range(-0.3..=0.3);
    let new_z = (old_z + d_z).max(0.001);
    
    println!("old_z: {:.17e}", old_z);
    println!("d_z: {:.17e}", d_z);
    println!("new_z (before clamp): {:.17e}", old_z + d_z);
    println!("new_z (after clamp): {:.17e}", new_z);
    println!("z_diff: {:.17e}", (new_z - old_z).abs());
    
    // Check no-op threshold
    if (new_z - old_z).abs() < 1e-15 {
        println!("✓ No-op detected (z_diff < 1e-15) - should accept");
        return;
    }
    
    let new_exp_neg_z = (-new_z).exp();
    println!("old_exp_neg_z: {:.17e}", old_exp_neg_z);
    println!("new_exp_neg_z: {:.17e}", new_exp_neg_z);
    
    let delta_entropy = (-new_z * new_exp_neg_z) - (-old_z * old_exp_neg_z);
    let delta_s = beta * delta_entropy;
    
    println!("delta_entropy: {:.17e}", delta_entropy);
    println!("delta_s: {:.17e}", delta_s);
    println!("|delta_s|: {:.17e}", delta_s.abs());
    println!("EPSILON: {:.17e}", 1e-6);
    println!("|delta_s| <= EPSILON: {}", delta_s.abs() <= 1e-6);
    
    if delta_s.abs() <= 1e-6 {
        println!("✓ Should accept due to epsilon threshold");
        graph.links[link_idx].update_z(new_z);
    } else {
        let prob = (-delta_s).exp();
        let rand_val = rng.gen_range(0.0..1.0);
        println!("exp(-delta_s): {:.17e}", prob);
        println!("random value: {:.17e}", rand_val);
        println!("accept: {}", rand_val < prob);
        
        if rand_val < prob {
            graph.links[link_idx].update_z(new_z);
        }
    }
}

fn trace_theta_update(graph: &mut FastGraph, link_idx: usize, alpha: f64, rng: &mut Pcg64, beta: f64) {
    println!("\n🔍 TRACING THETA-UPDATE:");
    
    let link = &graph.links[link_idx];
    let old_theta = link.theta;
    
    let d_theta = rng.gen_range(-0.3..=0.3);
    let new_theta = old_theta + d_theta;
    
    println!("old_theta: {:.17e}", old_theta);
    println!("d_theta: {:.17e}", d_theta);
    println!("new_theta: {:.17e}", new_theta);
    println!("theta_diff: {:.17e}", (new_theta - old_theta).abs());
    
    // Check no-op threshold
    if (new_theta - old_theta).abs() < 1e-15 {
        println!("✓ No-op detected (theta_diff < 1e-15) - should accept");
        return;
    }
    
    // Calculate triangle delta step by step
    println!("\n🔺 TRIANGLE CALCULATION:");
    
    // Manual triangle calculation to trace
    let before_triangle = manual_triangle_sum(graph);
    
    // Update theta temporarily to calculate after
    graph.links[link_idx].update_theta(new_theta);
    let after_triangle = manual_triangle_sum(graph);
    
    // Restore old theta
    graph.links[link_idx].update_theta(old_theta);
    
    let manual_delta = after_triangle - before_triangle;
    
    // Also calculate using internal method
    let internal_delta = calculate_triangle_delta(graph, link_idx, new_theta);
    
    println!("Triangle before: {:.17e}", before_triangle);
    println!("Triangle after:  {:.17e}", after_triangle);
    println!("Manual delta:    {:.17e}", manual_delta);
    println!("Internal delta:  {:.17e}", internal_delta);
    println!("Delta diff:      {:.17e}", (manual_delta - internal_delta).abs());
    
    let delta_s = alpha * internal_delta;
    
    println!("delta_s: {:.17e}", delta_s);
    println!("|delta_s|: {:.17e}", delta_s.abs());
    println!("EPSILON: {:.17e}", 1e-6);
    println!("|delta_s| <= EPSILON: {}", delta_s.abs() <= 1e-6);
    
    if delta_s.abs() <= 1e-6 {
        println!("✓ Should accept due to epsilon threshold");
        graph.links[link_idx].update_theta(new_theta);
    } else {
        let prob = (-delta_s).exp();
        let rand_val = rng.gen_range(0.0..1.0);
        println!("exp(-delta_s): {:.17e}", prob);
        println!("random value: {:.17e}", rand_val);
        println!("accept: {}", rand_val < prob);
        
        if rand_val < prob {
            graph.links[link_idx].update_theta(new_theta);
        }
    }
}

fn manual_triangle_sum(graph: &FastGraph) -> f64 {
    let mut sum = 0.0;
    let n = graph.n();
    
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let idx_ij = graph.link_index(i, j);
                let idx_jk = graph.link_index(j, k);
                let idx_ik = graph.link_index(i, k);
                
                let theta_sum = graph.links[idx_ij].theta + 
                               graph.links[idx_jk].theta + 
                               graph.links[idx_ik].theta;
                
                sum += 3.0 * theta_sum.cos();
            }
        }
    }
    
    sum
}

fn calculate_triangle_delta(graph: &FastGraph, link_idx: usize, new_theta: f64) -> f64 {
    let link = &graph.links[link_idx];
    let (i, j) = (link.i as usize, link.j as usize);
    let old_theta = link.theta;
    let delta_theta = new_theta - old_theta;
    
    const SMALL_DELTA_THRESHOLD: f64 = 1e-8;
    
    let mut contributions = Vec::new();
    
    for k in 0..graph.n() {
        if k != i && k != j {
            let idx_ik = if i < k { 
                graph.link_index(i, k) 
            } else { 
                graph.link_index(k, i) 
            };
            let idx_jk = if j < k { 
                graph.link_index(j, k) 
            } else { 
                graph.link_index(k, j) 
            };
            
            let other_sum = graph.links[idx_ik].theta + graph.links[idx_jk].theta;
            let old_total = old_theta + other_sum;
            
            let contribution = if delta_theta.abs() < SMALL_DELTA_THRESHOLD {
                -3.0 * old_total.sin() * delta_theta
            } else {
                let new_total = new_theta + other_sum;
                3.0 * (new_total.cos() - old_total.cos())
            };
            
            contributions.push(contribution);
        }
    }
    
    // Kahan summation
    let mut sum = 0.0;
    let mut c = 0.0;
    for &val in &contributions {
        let y = val - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}