// CRITICAL: Trace exact location where ΔE=0 moves are rejected
use scan::graph_fast::FastGraph;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

fn main() {
    println!("🔍 TRACING NO-OP REJECTION BUG");
    println!("==============================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(12345);
    let mut graph = FastGraph::new(n, 999);
    
    // Force boundary conditions to maximize no-op probability
    for link in &mut graph.links {
        link.update_z(0.001);
    }
    
    println!("Searching for no-op rejection...");
    
    for attempt in 0..10000 {
        let energy_before = graph.action(alpha, beta);
        let links_before = graph.links.clone();
        
        // Attempt move
        let info = graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
        
        let energy_after = graph.action(alpha, beta);
        let delta_e = energy_after - energy_before;
        
        // Check if this is a no-op
        let is_noop = links_before.iter().zip(graph.links.iter())
            .all(|(before, after)| {
                (before.z - after.z).abs() < 1e-15 && 
                (before.theta - after.theta).abs() < 1e-15
            });
        
        // Found a rejected no-op!
        if is_noop && !info.accept {
            println!("\n❌ FOUND REJECTED NO-OP AT ATTEMPT {}", attempt);
            println!("Energy before: {:.16}", energy_before);
            println!("Energy after:  {:.16}", energy_after);
            println!("ΔE = {:.16}", delta_e);
            println!("ΔE in hex: 0x{:016x}", delta_e.to_bits());
            println!("ΔE == 0.0: {}", delta_e == 0.0);
            println!("ΔE <= 0.0: {}", delta_e <= 0.0);
            println!("ΔE abs(): {:.2e}", delta_e.abs());
            println!("info.accept: {}", info.accept);
            println!("info.delta_w: {}", info.delta_w);
            println!("info.delta_cos: {}", info.delta_cos);
            
            // Check which type of move this was
            let z_changed = links_before.iter().zip(graph.links.iter())
                .any(|(before, after)| (before.z - after.z).abs() > 1e-15);
            let theta_changed = links_before.iter().zip(graph.links.iter())
                .any(|(before, after)| (before.theta - after.theta).abs() > 1e-15);
            
            println!("Z changed: {}", z_changed);
            println!("Theta changed: {}", theta_changed);
            
            if !z_changed && !theta_changed {
                println!("✅ Confirmed: True no-op move (no state change)");
                println!("BUG: This should ALWAYS be accepted!");
                break;
            } else {
                println!("❓ State changed but ΔE=0 - check energy calculation");
            }
        }
        
        if attempt % 1000 == 0 && attempt > 0 {
            println!("Checked {} attempts...", attempt);
        }
    }
    
    println!("\nNow let's manually trace the FastGraph metropolis_step function...");
    trace_metropolis_step_manually();
}

fn trace_metropolis_step_manually() {
    println!("\n🔬 MANUAL TRACE OF METROPOLIS STEP");
    println!("==================================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(42);
    let mut graph = FastGraph::new(n, 123);
    
    // Force to boundary
    for link in &mut graph.links {
        link.update_z(0.001);
    }
    
    println!("Initial state set. Starting trace...");
    
    // Manually replicate metropolis_step logic with extensive logging
    let link_idx = rng.gen_range(0..graph.links.len());
    println!("Selected link_idx: {}", link_idx);
    
    let delta_z = 0.1;
    let delta_theta = 0.1;
    let do_z_update = delta_z > 0.0 && rng.gen_bool(0.5);
    
    println!("do_z_update: {}", do_z_update);
    
    if do_z_update {
        println!("=== Z-UPDATE PATH ===");
        let link = &graph.links[link_idx];
        let old_z = link.z;
        let old_exp_neg_z = link.exp_neg_z;
        
        println!("old_z: {:.16}", old_z);
        println!("old_exp_neg_z: {:.16}", old_exp_neg_z);
        
        let random_delta = rng.gen_range(-delta_z..=delta_z);
        println!("random_delta: {:.16}", random_delta);
        
        let new_z_raw = old_z + random_delta;
        let new_z = new_z_raw.max(0.001);
        
        println!("new_z_raw: {:.16}", new_z_raw);
        println!("new_z (clamped): {:.16}", new_z);
        println!("z_changed: {}", (new_z - old_z).abs() >= 1e-15);
        
        if (new_z - old_z).abs() < 1e-15 {
            println!("✅ No-op detected in z-update - should accept");
        } else {
            let new_exp_neg_z = (-new_z).exp();
            let delta_entropy = (-new_z * new_exp_neg_z) - (-old_z * old_exp_neg_z);
            let delta_s = beta * delta_entropy;
            
            println!("new_exp_neg_z: {:.16}", new_exp_neg_z);
            println!("delta_entropy: {:.16}", delta_entropy);
            println!("delta_s: {:.16}", delta_s);
            println!("delta_s <= 0.0: {}", delta_s <= 0.0);
            
            if delta_s <= 0.0 {
                println!("✅ Should accept (delta_s <= 0)");
            } else {
                let random_accept = rng.gen_range(0.0..1.0);
                let exp_term = { -delta_s }.exp();
                let should_accept = random_accept < exp_term;
                
                println!("random_accept: {:.16}", random_accept);
                println!("exp(-delta_s): {:.16}", exp_term);
                println!("should_accept: {}", should_accept);
            }
        }
    } else {
        println!("=== THETA-UPDATE PATH ===");
        let link = &graph.links[link_idx];
        let old_theta = link.theta;
        let old_cos_theta = link.cos_theta;
        
        println!("old_theta: {:.16}", old_theta);
        println!("old_cos_theta: {:.16}", old_cos_theta);
        
        let d_theta = rng.gen_range(-delta_theta..=delta_theta);
        let new_theta = old_theta + d_theta;
        
        println!("d_theta: {:.16}", d_theta);
        println!("new_theta: {:.16}", new_theta);
        
        // This is where the bug might be - check the state change detection
        println!("(new_theta - old_theta).abs(): {:.2e}", (new_theta - old_theta).abs());
        println!("State change detected: {}", (new_theta - old_theta).abs() >= 1e-15);
        
        if (new_theta - old_theta).abs() < 1e-15 {
            println!("✅ No-op detected in theta-update - should accept");
        } else {
            // Apply the move first (this is what the current code does)
            println!("Applying theta move...");
            
            // Calculate energy difference
            let energy_before = graph.action(alpha, beta);
            
            // Temporarily update to calculate delta
            let original_theta = graph.links[link_idx].theta;
            let original_cos_theta = graph.links[link_idx].cos_theta;
            graph.links[link_idx].update_theta(new_theta);
            
            let energy_after = graph.action(alpha, beta);
            let delta_energy = energy_after - energy_before;
            
            println!("energy_before: {:.16}", energy_before);
            println!("energy_after:  {:.16}", energy_after);
            println!("delta_energy:  {:.16}", delta_energy);
            println!("delta_energy <= 0.0: {}", delta_energy <= 0.0);
            
            // Restore for now
            graph.links[link_idx].update_theta(original_theta);
            
            if delta_energy <= 0.0 {
                println!("✅ Should accept (ΔE <= 0)");
            } else {
                let random_accept = rng.gen_range(0.0..1.0);
                let exp_term = (-beta * delta_energy).exp();  // Note: using beta here!
                let should_accept = random_accept < exp_term;
                
                println!("random_accept: {:.16}", random_accept);
                println!("exp(-β*ΔE): {:.16}", exp_term);
                println!("should_accept: {}", should_accept);
            }
        }
    }
}