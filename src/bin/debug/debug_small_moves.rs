// Debug small moves that fall between no-op and epsilon thresholds
use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("🔍 DEBUGGING SMALL MOVES");
    println!("========================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(42);
    let mut graph = FastGraph::new(n, 123);
    
    for step in 0..1000 {
        let energy_before = graph.action(alpha, beta);
        let links_before = graph.links.clone();
        
        // Perform move
        let info = graph.metropolis_step(alpha, beta, 0.3, 0.3, &mut rng);
        
        let energy_after = graph.action(alpha, beta);
        let delta_e = energy_after - energy_before;
        
        // Look for moves with very small external energy change
        if delta_e.abs() < 1e-12 && !info.accept {
            println!("\n🔍 SMALL MOVE REJECTION at step {}", step);
            println!("External ΔE: {:.17e}", delta_e);
            
            // Find what changed (copy links to avoid borrowing issues)
            let links_after = graph.links.clone();
            for (i, (before, after)) in links_before.iter().zip(links_after.iter()).enumerate() {
                let z_diff = (before.z - after.z).abs();
                let theta_diff = (before.theta - after.theta).abs();
                
                if z_diff > 0.0 {
                    println!("Z change on link {}: diff={:.17e}", i, z_diff);
                    println!("  no-op threshold (1e-15): {}", z_diff < 1e-15);
                    println!("  epsilon threshold (1e-6): {}", z_diff < 1e-6);
                    
                    // Calculate internal energy change
                    let old_exp_neg_z = before.exp_neg_z;
                    let new_exp_neg_z = after.exp_neg_z;
                    let delta_entropy = (-after.z * new_exp_neg_z) - (-before.z * old_exp_neg_z);
                    let internal_delta_s = beta * delta_entropy;
                    
                    println!("  internal entropy delta: {:.17e}", delta_entropy);
                    println!("  internal ΔS: {:.17e}", internal_delta_s);
                    println!("  |internal ΔS|: {:.17e}", internal_delta_s.abs());
                    println!("  should accept (|ΔS| <= 1e-6): {}", internal_delta_s.abs() <= 1e-6);
                }
                
                if theta_diff > 0.0 {
                    println!("Theta change on link {}: diff={:.17e}", i, theta_diff);
                    println!("  no-op threshold (1e-15): {}", theta_diff < 1e-15);
                    println!("  epsilon threshold (1e-6): {}", theta_diff < 1e-6);
                    
                    // Calculate triangle delta using old state
                    graph.links = links_before.clone();
                    let internal_delta = calculate_triangle_delta(&graph, i, after.theta);
                    let internal_delta_s = alpha * internal_delta;
                    graph.links = links_after.clone(); // restore new state
                    
                    println!("  internal triangle delta: {:.17e}", internal_delta);
                    println!("  internal ΔS: {:.17e}", internal_delta_s);
                    println!("  |internal ΔS|: {:.17e}", internal_delta_s.abs());
                    println!("  should accept (|ΔS| <= 1e-6): {}", internal_delta_s.abs() <= 1e-6);
                }
            }
            
            // Only show first few cases
            if step > 50 { break; }
        }
    }
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