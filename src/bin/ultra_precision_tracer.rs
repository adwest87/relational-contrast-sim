// Ultra-precision tracer to find exact floating point differences
use scan::graph_fast::FastGraph;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

fn main() {
    println!("🔬 ULTRA-PRECISION TRACER");
    println!("=========================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(12345);
    let mut graph = FastGraph::new(n, 999);
    
    for step in 0..1000 {
        let energy_before = graph.action(alpha, beta);
        let links_before = graph.links.clone();
        
        // Perform move
        let info = graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
        
        let energy_after = graph.action(alpha, beta);
        let delta_energy = energy_after - energy_before;
        
        // Check for ΔE=0 rejections
        if delta_energy == 0.0 && !info.accept {
            println!("\n❌ REJECTION at step {}", step);
            
            // Ultra-detailed state analysis
            let mut max_z_diff = 0.0f64;
            let mut max_theta_diff = 0.0f64;
            let mut max_cos_diff = 0.0f64;
            let mut max_sin_diff = 0.0f64;
            let mut max_w_diff = 0.0f64;
            
            for (i, (before, after)) in links_before.iter().zip(graph.links.iter()).enumerate() {
                let z_diff = (before.z - after.z).abs();
                let theta_diff = (before.theta - after.theta).abs();
                let cos_diff = (before.cos_theta - after.cos_theta).abs();
                let sin_diff = (before.sin_theta - after.sin_theta).abs();
                let w_diff = (before.exp_neg_z - after.exp_neg_z).abs();
                
                max_z_diff = max_z_diff.max(z_diff);
                max_theta_diff = max_theta_diff.max(theta_diff);
                max_cos_diff = max_cos_diff.max(cos_diff);
                max_sin_diff = max_sin_diff.max(sin_diff);
                max_w_diff = max_w_diff.max(w_diff);
                
                if z_diff > 0.0 || theta_diff > 0.0 || cos_diff > 0.0 || sin_diff > 0.0 || w_diff > 0.0 {
                    println!("  Link {}: z_diff={:.2e}, theta_diff={:.2e}, cos_diff={:.2e}, sin_diff={:.2e}, w_diff={:.2e}", 
                        i, z_diff, theta_diff, cos_diff, sin_diff, w_diff);
                }
            }
            
            println!("Maximum differences:");
            println!("  max_z_diff: {:.2e} (< 1e-15: {})", max_z_diff, max_z_diff < 1e-15);
            println!("  max_theta_diff: {:.2e} (< 1e-15: {})", max_theta_diff, max_theta_diff < 1e-15);
            println!("  max_cos_diff: {:.2e} (< 1e-15: {})", max_cos_diff, max_cos_diff < 1e-15);
            println!("  max_sin_diff: {:.2e} (< 1e-15: {})", max_sin_diff, max_sin_diff < 1e-15);
            println!("  max_w_diff: {:.2e} (< 1e-15: {})", max_w_diff, max_w_diff < 1e-15);
            
            // Check if ALL differences are below various thresholds
            let total_diff = max_z_diff + max_theta_diff + max_cos_diff + max_sin_diff + max_w_diff;
            println!("  total_diff: {:.2e}", total_diff);
            
            if total_diff == 0.0 {
                println!("  🚨 PERFECT NO-OP: All differences are exactly 0.0");
            } else if total_diff < 1e-15 {
                println!("  ⚠️ TINY CHANGES: All differences < 1e-15");
            } else if total_diff < f64::EPSILON {
                println!("  ⚠️ EPSILON LEVEL: All differences < f64::EPSILON ({:.2e})", f64::EPSILON);
            }
            
            // Manual verification using the ACTUAL metropolis_step logic
            println!("\n🔍 MANUAL VERIFICATION:");
            
            // Reset to before state and manually trace
            for (i, link_before) in links_before.iter().enumerate() {
                graph.links[i] = *link_before;
            }
            
            // Now manually call the exact same sequence
            let mut manual_rng = Pcg64::seed_from_u64(12345);
            
            // Skip to the same RNG state (this is hacky but works for debugging)
            for _ in 0..step {
                let _: f64 = manual_rng.gen_range(0.0..1.0);
                let _: f64 = manual_rng.gen_range(0.0..1.0);
                let _: f64 = manual_rng.gen_range(0.0..1.0);
            }
            
            manual_metropolis_trace(&mut graph, alpha, beta, 0.1, 0.1, &mut manual_rng);
            
            break;
        }
        
        if step % 100 == 0 {
            println!("Step {}...", step);
        }
    }
}

fn manual_metropolis_trace(graph: &mut FastGraph, alpha: f64, beta: f64, delta_z: f64, delta_theta: f64, rng: &mut Pcg64) {
    let link_idx = rng.gen_range(0..graph.links.len());
    let do_z_update = delta_z > 0.0 && rng.gen_bool(0.5);
    
    println!("Manual trace: link_idx={}, do_z_update={}", link_idx, do_z_update);
    
    if do_z_update {
        println!("=== Z-UPDATE PATH ===");
        let link = &graph.links[link_idx];
        let old_z = link.z;
        let random_delta = rng.gen_range(-delta_z..=delta_z);
        let new_z = (old_z + random_delta).max(0.001);
        
        println!("  old_z: {:.17e}", old_z);
        println!("  random_delta: {:.17e}", random_delta);
        println!("  old_z + random_delta: {:.17e}", old_z + random_delta);
        println!("  new_z (after max): {:.17e}", new_z);
        println!("  difference: {:.17e}", (new_z - old_z).abs());
        println!("  difference < 1e-15: {}", (new_z - old_z).abs() < 1e-15);
        
        if (new_z - old_z).abs() < 1e-15 {
            println!("  ✅ NO-OP DETECTED: Should always accept");
        } else {
            println!("  🔄 REAL MOVE: Proceed with Metropolis");
        }
    } else {
        println!("=== THETA-UPDATE PATH ===");
        let link = &graph.links[link_idx];
        let old_theta = link.theta;
        let d_theta = rng.gen_range(-delta_theta..=delta_theta);
        let new_theta = old_theta + d_theta;
        
        println!("  old_theta: {:.17e}", old_theta);
        println!("  d_theta: {:.17e}", d_theta);
        println!("  new_theta: {:.17e}", new_theta);
        println!("  difference: {:.17e}", (new_theta - old_theta).abs());
        println!("  difference < 1e-15: {}", (new_theta - old_theta).abs() < 1e-15);
        
        if (new_theta - old_theta).abs() < 1e-15 {
            println!("  ✅ NO-OP DETECTED: Should always accept");
        } else {
            println!("  🔄 REAL MOVE: Proceed with Metropolis");
        }
    }
}