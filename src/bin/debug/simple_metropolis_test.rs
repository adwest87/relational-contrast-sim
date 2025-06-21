// Simple test to verify Metropolis criterion fix
use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("🧪 SIMPLE METROPOLIS TEST");
    println!("========================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(42);
    let mut graph = FastGraph::new(n, 123);
    
    // Test a specific case that should always accept
    for _ in 0..100 {
        let energy_before = graph.action(alpha, beta);
        let info = graph.metropolis_step(alpha, beta, 0.001, 0.001, &mut rng);
        let energy_after = graph.action(alpha, beta);
        let delta_e = energy_after - energy_before;
        
        if delta_e == 0.0 {
            println!("Found ΔE=0 move: accepted={}", info.accept);
            if !info.accept {
                println!("❌ BUG: ΔE=0 move rejected!");
                break;
            } else {
                println!("✅ ΔE=0 move correctly accepted");
            }
        }
        
        if delta_e < 0.0 && !info.accept {
            println!("❌ BUG: ΔE<0 move rejected! ΔE={}", delta_e);
            break;
        }
    }
    
    println!("Test completed.");
}