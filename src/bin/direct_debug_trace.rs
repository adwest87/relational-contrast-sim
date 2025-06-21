// Direct debug trace by modifying metropolis_step behavior
use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("🎯 DIRECT DEBUG TRACE");
    println!("====================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(12345);
    let mut graph = FastGraph::new(n, 999);
    
    // Run until we find a rejected ΔE=0 move, then analyze
    for step in 0..1000 {
        let energy_before = graph.action(alpha, beta);
        let links_before = graph.links.clone();
        
        // Call metropolis_step 
        let info = graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
        
        let energy_after = graph.action(alpha, beta);
        let delta_energy = energy_after - energy_before;
        
        // Check for the problematic case
        if delta_energy == 0.0 && !info.accept {
            println!("\n❌ FOUND PROBLEMATIC REJECTION at step {}", step);
            println!("ΔE: {:.17e}", delta_energy);
            println!("info.accept: {}", info.accept);
            
            // Check exact state differences
            let mut any_diff = false;
            for (i, (before, after)) in links_before.iter().zip(graph.links.iter()).enumerate() {
                let z_diff = before.z - after.z;
                let theta_diff = before.theta - after.theta;
                
                if z_diff != 0.0 || theta_diff != 0.0 {
                    println!("Link {} changed: z_diff={:.17e}, theta_diff={:.17e}", i, z_diff, theta_diff);
                    any_diff = true;
                }
            }
            
            if !any_diff {
                println!("🚨 NO STATE CHANGE: This is a true no-op that was rejected!");
                println!("This should be IMPOSSIBLE with correct no-op detection.");
                
                // The issue must be that the no-op detection threshold is wrong
                // or the logic has a bug. Let's examine what the thresholds should be.
                
                println!("\n🔍 THRESHOLD ANALYSIS:");
                println!("Current threshold: 1e-15");
                println!("Machine epsilon: {:.2e}", f64::EPSILON);
                println!("Smallest representable difference: {:.2e}", f64::MIN_POSITIVE);
                
                // Test different thresholds
                let test_values = [0.0, 1e-16, 1e-15, 1e-14, 1e-13, f64::EPSILON, 1e-10, 1e-8];
                
                for &threshold in &test_values {
                    let z_close = links_before.iter().zip(graph.links.iter())
                        .all(|(before, after)| (before.z - after.z).abs() <= threshold);
                    let theta_close = links_before.iter().zip(graph.links.iter())
                        .all(|(before, after)| (before.theta - after.theta).abs() <= threshold);
                    
                    println!("  Threshold {:.1e}: z_close={}, theta_close={}", threshold, z_close, theta_close);
                }
                
                break;
            } else {
                println!("🤔 STATE DID CHANGE: This might be a precision issue in energy calculation");
            }
        }
        
        if step % 100 == 0 {
            println!("Step {}...", step);
        }
    }
    
    println!("\nTrace completed.");
}