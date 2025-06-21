// Debug the exact cases where ΔE=0 but move is rejected
use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("🔍 DEBUGGING EXACT ΔE=0 REJECTIONS");
    println!("===================================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(42);
    let mut graph = FastGraph::new(n, 123);
    
    let mut violations_found = 0;
    
    for step in 0..1000 {
        let energy_before = graph.action(alpha, beta);
        let links_before = graph.links.clone();
        
        // Perform move
        let info = graph.metropolis_step(alpha, beta, 0.3, 0.3, &mut rng);
        
        let energy_after = graph.action(alpha, beta);
        let delta_e = energy_after - energy_before;
        
        // Check for exact violation case
        if delta_e == 0.0 && !info.accept {
            violations_found += 1;
            
            if violations_found <= 3 {
                println!("\n❌ VIOLATION #{} at step {}", violations_found, step);
                println!("External ΔE: {:.17e}", delta_e);
                println!("External ΔE hex: 0x{:016x}", delta_e.to_bits());
                
                // Check what type of move was attempted
                let mut z_changed = false;
                let mut theta_changed = false;
                let mut which_link = None;
                
                for (i, (before, after)) in links_before.iter().zip(graph.links.iter()).enumerate() {
                    let z_diff = (before.z - after.z).abs();
                    let theta_diff = (before.theta - after.theta).abs();
                    
                    if z_diff > 1e-15 {
                        z_changed = true;
                        which_link = Some(i);
                        println!("Z changed on link {}: {:.17e} -> {:.17e} (diff: {:.2e})", 
                            i, before.z, after.z, z_diff);
                    }
                    if theta_diff > 1e-15 {
                        theta_changed = true;
                        which_link = Some(i);
                        println!("Theta changed on link {}: {:.17e} -> {:.17e} (diff: {:.2e})", 
                            i, before.theta, after.theta, theta_diff);
                    }
                }
                
                if !z_changed && !theta_changed {
                    println!("🤔 No state change detected - this should be a no-op!");
                } else {
                    println!("Move type: z_changed={}, theta_changed={}", z_changed, theta_changed);
                    
                    // If we know which link changed, calculate internal energy change
                    if let Some(link_idx) = which_link {
                        if theta_changed {
                            let old_theta = links_before[link_idx].theta;
                            let new_theta = graph.links[link_idx].theta;
                            
                            // Restore old state temporarily
                            graph.links = links_before.clone();
                            let internal_delta = calculate_triangle_delta(&graph, link_idx, new_theta);
                            let internal_delta_s = alpha * internal_delta;
                            
                            println!("Internal triangle delta: {:.17e}", internal_delta);
                            println!("Internal ΔS: {:.17e}", internal_delta_s);
                            println!("Internal |ΔS|: {:.17e}", internal_delta_s.abs());
                            println!("EPSILON threshold: 1e-6 = {:.17e}", 1e-6);
                            println!("Should accept: {}", internal_delta_s.abs() <= 1e-6);
                            
                            // Restore new state
                            graph.links[link_idx].update_theta(new_theta);
                        }
                        
                        if z_changed {
                            let old_z = links_before[link_idx].z;
                            let new_z = graph.links[link_idx].z;
                            let old_exp_neg_z = links_before[link_idx].exp_neg_z;
                            let new_exp_neg_z = graph.links[link_idx].exp_neg_z;
                            
                            let delta_entropy = (-new_z * new_exp_neg_z) - (-old_z * old_exp_neg_z);
                            let internal_delta_s = beta * delta_entropy;
                            
                            println!("Internal entropy delta: {:.17e}", delta_entropy);
                            println!("Internal ΔS: {:.17e}", internal_delta_s);
                            println!("Internal |ΔS|: {:.17e}", internal_delta_s.abs());
                            println!("EPSILON threshold: 1e-6 = {:.17e}", 1e-6);
                            println!("Should accept: {}", internal_delta_s.abs() <= 1e-6);
                        }
                    }
                }
            }
            
            if violations_found >= 5 {
                break;
            }
        }
    }
    
    println!("\n📊 SUMMARY: Found {} violations in 1000 steps", violations_found);
}

// Calculate triangle sum delta using the same logic as in FastGraph
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