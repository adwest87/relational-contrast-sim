// Test the fundamental Metropolis criterion directly
use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("🧪 TESTING METROPOLIS CRITERION");
    println!("===============================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(42);
    let mut graph = FastGraph::new(n, 123);
    
    let mut delta_e_zero_rejects = 0;
    let mut delta_e_negative_rejects = 0;
    let mut total_moves = 0;
    
    for _ in 0..10000 {
        let energy_before = graph.action(alpha, beta);
        
        // Perform move
        let info = graph.metropolis_step(alpha, beta, 0.3, 0.3, &mut rng);
        
        let energy_after = graph.action(alpha, beta);
        let delta_e = energy_after - energy_before;
        
        total_moves += 1;
        
        // Check Metropolis violations
        if delta_e <= 0.0 && !info.accept {
            if delta_e == 0.0 {
                delta_e_zero_rejects += 1;
                if delta_e_zero_rejects <= 3 {
                    println!("❌ VIOLATION #{}: ΔE=0 rejected", delta_e_zero_rejects);
                    println!("   ΔE = {:.16}", delta_e);
                    println!("   ΔE hex: 0x{:016x}", delta_e.to_bits());
                }
            } else {
                delta_e_negative_rejects += 1;
                if delta_e_negative_rejects <= 3 {
                    println!("❌ VIOLATION #{}: ΔE<0 rejected", delta_e_negative_rejects);
                    println!("   ΔE = {:.16}", delta_e);
                }
            }
        }
    }
    
    println!("\n📊 METROPOLIS CRITERION VIOLATIONS:");
    println!("ΔE=0 moves rejected: {}", delta_e_zero_rejects);
    println!("ΔE<0 moves rejected: {}", delta_e_negative_rejects);
    println!("Total moves: {}", total_moves);
    println!("ΔE≤0 rejection rate: {:.2}%", 
        100.0 * (delta_e_zero_rejects + delta_e_negative_rejects) as f64 / total_moves as f64);
    
    if delta_e_zero_rejects + delta_e_negative_rejects == 0 {
        println!("✅ Metropolis criterion is correct!");
    } else {
        println!("❌ FUNDAMENTAL METROPOLIS VIOLATION!");
        println!("   This violates the basic requirement that ΔE≤0 moves must always be accepted.");
    }
}