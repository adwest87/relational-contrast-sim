// Detailed diagnosis of no-op move handling
use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("🔍 DETAILED NO-OP MOVE DIAGNOSIS");
    println!("=================================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(42);
    let mut graph = FastGraph::new(n, 123);
    
    // Force system into boundary condition where no-ops are likely
    for link in &mut graph.links {
        link.update_z(0.001); // Force to minimum boundary
    }
    
    println!("Testing no-op moves at z boundary...");
    
    let mut total_moves = 0;
    let mut noop_by_z = 0;
    let mut noop_by_theta = 0;
    let mut noop_accepts = 0;
    let mut noop_rejects = 0;
    
    for step in 0..1000 {
        let links_before = graph.links.clone();
        let energy_before = graph.action(alpha, beta);
        
        // Attempt move with small delta to force no-ops
        let info = graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
        
        let energy_after = graph.action(alpha, beta);
        let is_noop = links_before.iter().zip(graph.links.iter())
            .all(|(before, after)| {
                (before.z - after.z).abs() < 1e-14 && 
                (before.theta - after.theta).abs() < 1e-14
            });
        
        total_moves += 1;
        
        if is_noop {
            // Determine which type of no-op this was
            let z_diff_avg = links_before.iter().zip(graph.links.iter())
                .map(|(before, after)| (before.z - after.z).abs())
                .fold(0.0, f64::max);
            let theta_diff_avg = links_before.iter().zip(graph.links.iter())
                .map(|(before, after)| (before.theta - after.theta).abs())
                .fold(0.0, f64::max);
            
            if z_diff_avg < theta_diff_avg {
                noop_by_z += 1;
            } else {
                noop_by_theta += 1;
            }
            
            if info.accept {
                noop_accepts += 1;
            } else {
                noop_rejects += 1;
                
                if noop_rejects <= 3 {
                    println!("\n❌ No-op rejection #{}", noop_rejects);
                    println!("  Step: {}", step);
                    println!("  Energy change: {:.2e}", energy_after - energy_before);
                    println!("  Max z difference: {:.2e}", z_diff_avg);
                    println!("  Max θ difference: {:.2e}", theta_diff_avg);
                    println!("  info.accept: {}", info.accept);
                    println!("  info.delta_w: {}", info.delta_w);
                    println!("  info.delta_cos: {}", info.delta_cos);
                }
            }
        }
        
        if step % 200 == 0 && step > 0 {
            println!("Step {}: No-ops by z: {}, by θ: {}, accepts: {}, rejects: {}", 
                step, noop_by_z, noop_by_theta, noop_accepts, noop_rejects);
        }
    }
    
    println!("\n📊 FINAL STATISTICS:");
    println!("Total moves: {}", total_moves);
    println!("No-ops by z boundary: {}", noop_by_z);
    println!("No-ops by θ: {}", noop_by_theta);
    println!("No-op accepts: {}", noop_accepts);
    println!("No-op rejects: {}", noop_rejects);
    println!("No-op rejection rate: {:.2}%", 
        if (noop_by_z + noop_by_theta) > 0 {
            100.0 * noop_rejects as f64 / (noop_by_z + noop_by_theta) as f64
        } else {
            0.0
        });
    
    if noop_rejects > 0 {
        println!("\n❌ NO-OP REJECTION BUG CONFIRMED!");
        println!("Fix not working correctly in FastGraph implementation.");
    } else {
        println!("\n✅ No-op moves correctly accepted.");
    }
}