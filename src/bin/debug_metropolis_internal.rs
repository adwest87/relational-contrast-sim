// Debug the internal Metropolis logic step by step
use scan::graph_fast::FastGraph;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

fn main() {
    println!("🔍 DEBUGGING INTERNAL METROPOLIS LOGIC");
    println!("======================================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(12345);
    let mut graph = FastGraph::new(n, 999);
    
    // Look for the first no-op rejection
    for step in 0..1000 {
        let energy_before = graph.action(alpha, beta);
        let links_before = graph.links.clone();
        
        // MANUALLY replicate metropolis_step with debug output
        let link_idx = rng.gen_range(0..graph.links.len());
        let delta_z = 0.1;
        let delta_theta = 0.1;
        let do_z_update = delta_z > 0.0 && rng.gen_bool(0.5);
        
        println!("\n=== STEP {} ===", step);
        println!("link_idx: {}, do_z_update: {}", link_idx, do_z_update);
        
        let accept = if do_z_update {
            println!("Z-UPDATE PATH");
            let link = &graph.links[link_idx];
            let old_z = link.z;
            let old_exp_neg_z = link.exp_neg_z;
            let new_z = (old_z + rng.gen_range(-delta_z..=delta_z)).max(0.001);
            
            println!("  old_z: {:.17e}", old_z);
            println!("  new_z: {:.17e}", new_z);
            
            let new_exp_neg_z = (-new_z).exp();
            let delta_entropy = (-new_z * new_exp_neg_z) - (-old_z * old_exp_neg_z);
            let delta_s = beta * delta_entropy;
            
            println!("  delta_entropy: {:.17e}", delta_entropy);
            println!("  delta_s: {:.17e}", delta_s);
            println!("  delta_s <= 0.0: {}", delta_s <= 0.0);
            
            let accept_decision = if delta_s <= 0.0 { 
                println!("  🟢 SHOULD ACCEPT (delta_s <= 0)");
                true 
            } else { 
                let rnd = rng.gen_range(0.0..1.0);
                let exp_term = (-delta_s).exp();
                let should_accept = rnd < exp_term;
                println!("  🎲 RNG decision: {} < {} = {}", rnd, exp_term, should_accept);
                should_accept
            };
            
            if accept_decision {
                graph.links[link_idx].update_z(new_z);
            }
            
            accept_decision
        } else {
            println!("THETA-UPDATE PATH");
            let link = &graph.links[link_idx];
            let old_theta = link.theta;
            let d_theta = rng.gen_range(-delta_theta..=delta_theta);
            let new_theta = old_theta + d_theta;
            
            println!("  old_theta: {:.17e}", old_theta);
            println!("  d_theta: {:.17e}", d_theta);
            println!("  new_theta: {:.17e}", new_theta);
            
            let delta_triangle = calculate_triangle_delta(&graph, link_idx, new_theta);
            let delta_s = alpha * delta_triangle;
            
            println!("  delta_triangle: {:.17e}", delta_triangle);
            println!("  delta_s: {:.17e}", delta_s);
            println!("  delta_s <= 0.0: {}", delta_s <= 0.0);
            
            let accept_decision = if delta_s <= 0.0 { 
                println!("  🟢 SHOULD ACCEPT (delta_s <= 0)");
                true 
            } else { 
                let rnd = rng.gen_range(0.0..1.0);
                let exp_term = (-delta_s).exp();
                let should_accept = rnd < exp_term;
                println!("  🎲 RNG decision: {} < {} = {}", rnd, exp_term, should_accept);
                should_accept
            };
            
            if accept_decision {
                graph.links[link_idx].update_theta(new_theta);
            }
            
            accept_decision
        };
        
        let energy_after = graph.action(alpha, beta);
        let delta_energy = energy_after - energy_before;
        
        // Check if state actually changed
        let state_changed = links_before.iter().zip(graph.links.iter())
            .any(|(before, after)| {
                (before.z - after.z).abs() > 1e-15 || 
                (before.theta - after.theta).abs() > 1e-15
            });
        
        println!("Manual accept: {}", accept);
        println!("State changed: {}", state_changed);
        println!("Delta energy: {:.17e}", delta_energy);
        
        // Check for violation
        if delta_energy == 0.0 && !accept {
            println!("❌ FOUND VIOLATION!");
            println!("  ΔE is exactly 0 but move was rejected");
            println!("  This should be impossible with fixed Metropolis criterion");
            break;
        }
        
        if delta_energy == 0.0 && accept {
            println!("✅ ΔE=0 move correctly accepted");
        }
        
        if step >= 50 {
            println!("No violations found in first 50 steps");
            break;
        }
    }
}

// Manual triangle delta calculation
fn calculate_triangle_delta(graph: &FastGraph, link_idx: usize, new_theta: f64) -> f64 {
    let link = &graph.links[link_idx];
    let (i, j) = (link.i as usize, link.j as usize);
    let old_theta = link.theta;
    let n = graph.n();
    
    let mut delta = 0.0;
    
    for k in 0..n {
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
            
            let old_sum = old_theta + graph.links[idx_ik].theta + graph.links[idx_jk].theta;
            let new_sum = new_theta + graph.links[idx_ik].theta + graph.links[idx_jk].theta;
            
            delta += 3.0 * (new_sum.cos() - old_sum.cos());
        }
    }
    
    delta
}