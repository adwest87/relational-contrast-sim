// Debug why moves with ΔE=0 are being rejected
// This should NEVER happen in correct Metropolis algorithm

use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("🔍 DEBUGGING ZERO-ENERGY MOVES");
    println!("==============================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(12345);
    let mut graph = FastGraph::new(n, 999);
    
    println!("Testing moves with ΔE=0...");
    
    let mut zero_energy_moves = 0;
    let mut zero_energy_rejects = 0;
    
    for step in 0..10000 {
        let energy_before = graph.action(alpha, beta);
        
        // Save exact state
        let links_before = graph.links.clone();
        
        // Attempt move
        let info = graph.metropolis_step(alpha, beta, 0.3, 0.3, &mut rng);
        
        let energy_after = graph.action(alpha, beta);
        let delta_e = energy_after - energy_before;
        
        // Check for exactly zero energy change
        if delta_e.abs() < 1e-15 {
            zero_energy_moves += 1;
            
            if !info.accept {
                zero_energy_rejects += 1;
                
                if zero_energy_rejects <= 5 {
                    println!("\n❌ CRITICAL BUG #{}: Zero-energy move rejected!", zero_energy_rejects);
                    println!("  Step: {}", step);
                    println!("  Energy before: {:.16}", energy_before);
                    println!("  Energy after:  {:.16}", energy_after);
                    println!("  Delta E: {:.2e}", delta_e);
                    println!("  Move accepted: {}", info.accept);
                    
                    // Check if state actually changed
                    let state_changed = links_before.iter().zip(graph.links.iter())
                        .any(|(before, after)| {
                            (before.z - after.z).abs() > 1e-15 || 
                            (before.theta - after.theta).abs() > 1e-15
                        });
                    
                    println!("  State changed: {}", state_changed);
                    
                    if !state_changed {
                        println!("  DIAGNOSIS: Move was a no-op (no actual state change)");
                    } else {
                        println!("  DIAGNOSIS: State changed but energy stayed same");
                    }
                }
            }
        }
        
        // Also check if proposed move was actually a no-op
        let is_noop = links_before.iter().zip(graph.links.iter())
            .all(|(before, after)| {
                (before.z - after.z).abs() < 1e-15 && 
                (before.theta - after.theta).abs() < 1e-15
            });
        
        if is_noop && step < 10 {
            println!("Step {}: No-op move detected (state unchanged)", step);
        }
    }
    
    println!("\n📊 ZERO-ENERGY MOVE STATISTICS:");
    println!("Total zero-energy moves: {}", zero_energy_moves);
    println!("Zero-energy moves rejected: {}", zero_energy_rejects);
    println!("Zero-energy rejection rate: {:.2}%", 
        if zero_energy_moves > 0 { 
            100.0 * zero_energy_rejects as f64 / zero_energy_moves as f64 
        } else { 
            0.0 
        });
    
    if zero_energy_rejects > 0 {
        println!("\n❌ FUNDAMENTAL METROPOLIS VIOLATION DETECTED!");
        println!("   Moves with ΔE=0 must ALWAYS be accepted.");
        println!("   This indicates a critical bug in the acceptance criterion.");
    } else {
        println!("\n✅ Zero-energy moves correctly accepted.");
    }
}