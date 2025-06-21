// CRITICAL: Trace exact floating point values causing ΔE=0 rejections
use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("🔬 PRECISION TRACER: Finding exact cause of ΔE=0 rejections");
    println!("===========================================================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(12345);
    let mut graph = FastGraph::new(n, 999);
    
    let mut rejection_count = 0;
    let mut total_moves = 0;
    
    for step in 0..50000 {
        // Calculate energy with FULL precision
        let energy_before = graph.action(alpha, beta);
        
        // Store complete state before move
        let links_before = graph.links.clone();
        
        // Perform move
        let info = graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
        
        // Calculate energy with FULL precision  
        let energy_after = graph.action(alpha, beta);
        let delta_energy = energy_after - energy_before;
        
        total_moves += 1;
        
        // Check for ΔE=0 rejections with FULL PRECISION analysis
        if delta_energy == 0.0 && !info.accept {
            rejection_count += 1;
            
            println!("\n❌ REJECTION #{} at step {}", rejection_count, step);
            println!("Energy before: {:.17e}", energy_before);
            println!("Energy after:  {:.17e}", energy_after);
            println!("Delta energy:  {:.17e}", delta_energy);
            println!("Delta energy hex: 0x{:016x}", delta_energy.to_bits());
            println!("Delta == 0.0: {}", delta_energy == 0.0);
            println!("Delta <= 0.0: {}", delta_energy <= 0.0);
            println!("Info.accept: {}", info.accept);
            
            // Detailed state analysis
            let state_changed = links_before.iter().zip(graph.links.iter())
                .any(|(before, after)| {
                    (before.z - after.z).abs() > 1e-15 || 
                    (before.theta - after.theta).abs() > 1e-15
                });
            
            println!("State actually changed: {}", state_changed);
            
            if !state_changed {
                println!("🚨 TRUE NO-OP: State unchanged but ΔE=0 rejected!");
                
                // Check which variables have changed at all
                for (i, (before, after)) in links_before.iter().zip(graph.links.iter()).enumerate() {
                    let z_diff = (before.z - after.z).abs();
                    let theta_diff = (before.theta - after.theta).abs();
                    let cos_diff = (before.cos_theta - after.cos_theta).abs();
                    let sin_diff = (before.sin_theta - after.sin_theta).abs();
                    let w_diff = (before.exp_neg_z - after.exp_neg_z).abs();
                    
                    if z_diff > 0.0 || theta_diff > 0.0 || cos_diff > 0.0 || sin_diff > 0.0 || w_diff > 0.0 {
                        println!("  Link {}: z_diff={:.2e}, theta_diff={:.2e}, cos_diff={:.2e}, sin_diff={:.2e}, w_diff={:.2e}", 
                            i, z_diff, theta_diff, cos_diff, sin_diff, w_diff);
                    }
                }
            } else {
                println!("🔍 STATE CHANGED: Investigating energy calculation...");
                
                // Recalculate energy step by step
                let entropy_before = links_before.iter().map(|l| -l.z * l.exp_neg_z).sum::<f64>();
                let entropy_after = graph.links.iter().map(|l| -l.z * l.exp_neg_z).sum::<f64>();
                let triangle_before = calculate_triangle_sum(&links_before, n);
                let triangle_after = graph.triangle_sum();
                
                println!("  Entropy before: {:.17e}", entropy_before);
                println!("  Entropy after:  {:.17e}", entropy_after);
                println!("  Entropy delta:  {:.17e}", entropy_after - entropy_before);
                println!("  Triangle before: {:.17e}", triangle_before);
                println!("  Triangle after:  {:.17e}", triangle_after);
                println!("  Triangle delta:  {:.17e}", triangle_after - triangle_before);
                
                let action_before_recalc = beta * entropy_before + alpha * triangle_before;
                let action_after_recalc = beta * entropy_after + alpha * triangle_after;
                let delta_recalc = action_after_recalc - action_before_recalc;
                
                println!("  Action before (recalc): {:.17e}", action_before_recalc);
                println!("  Action after (recalc):  {:.17e}", action_after_recalc);
                println!("  Delta (recalc):         {:.17e}", delta_recalc);
                println!("  Recalc == original:     {}", (delta_recalc - delta_energy).abs() < 1e-15);
            }
            
            if rejection_count >= 5 {
                println!("\n🛑 Stopping after 5 rejections for detailed analysis");
                break;
            }
        }
        
        if step % 10000 == 0 && step > 0 {
            println!("Step {}: {} ΔE=0 rejections out of {} moves ({:.2}%)", 
                step, rejection_count, total_moves, 100.0 * rejection_count as f64 / total_moves as f64);
        }
    }
    
    println!("\n📊 FINAL STATISTICS:");
    println!("Total moves: {}", total_moves);
    println!("ΔE=0 rejections: {}", rejection_count);
    println!("Rejection rate: {:.3}%", 100.0 * rejection_count as f64 / total_moves as f64);
}

// Manual triangle sum calculation for verification
fn calculate_triangle_sum(links: &[scan::graph_fast::FastLink], n: usize) -> f64 {
    let mut sum = 0.0;
    
    // Generate all triangles
    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let idx_ij = link_index(i, j, n);
                let idx_jk = link_index(j, k, n);
                let idx_ik = link_index(i, k, n);
                
                let theta_sum = links[idx_ij].theta + links[idx_jk].theta + links[idx_ik].theta;
                sum += 3.0 * theta_sum.cos();
            }
        }
    }
    
    sum
}

// Manual link index calculation
fn link_index(i: usize, j: usize, n: usize) -> usize {
    let (i, j) = if i < j { (i, j) } else { (j, i) };
    i * n - i * (i + 1) / 2 + j - i - 1
}