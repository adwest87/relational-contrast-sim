// Focused debug on the specific problematic case
use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    let mut rng = Pcg64::seed_from_u64(12345);
    let mut graph = FastGraph::new(4, 999);
    
    // Run exactly to step 75 where we know the problem occurs
    for step in 0..80 {
        let energy_before = graph.action(1.5, 3.0);
        let links_before = graph.links.clone();
        
        eprintln!("=== STEP {} ===", step);
        let info = graph.metropolis_step(1.5, 3.0, 0.1, 0.1, &mut rng);
        
        let energy_after = graph.action(1.5, 3.0);
        let delta_energy = energy_after - energy_before;
        
        if step == 75 {
            eprintln!("STEP 75 ANALYSIS:");
            eprintln!("  ΔE: {:.17e}", delta_energy);
            eprintln!("  accept: {}", info.accept);
            
            let state_changed = links_before.iter().zip(graph.links.iter())
                .any(|(before, after)| {
                    (before.z - after.z).abs() > 0.0 || 
                    (before.theta - after.theta).abs() > 0.0
                });
            eprintln!("  state_changed: {}", state_changed);
            
            if delta_energy == 0.0 && !info.accept {
                eprintln!("  🚨 FOUND THE BUG!");
            }
        }
        
        if step >= 76 {
            break;
        }
    }
}