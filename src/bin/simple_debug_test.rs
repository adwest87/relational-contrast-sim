// Simple test to see debug output
use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    let mut rng = Pcg64::seed_from_u64(12345);
    let mut graph = FastGraph::new(4, 999);
    
    // Run just a few steps to see debug output
    for i in 0..10 {
        println!("=== STEP {} ===", i);
        let info = graph.metropolis_step(1.5, 3.0, 0.1, 0.1, &mut rng);
        println!("Result: accept={}", info.accept);
    }
}