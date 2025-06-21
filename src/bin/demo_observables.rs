use scan::graph_fast::{FastGraph, BatchedObservables};
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("=== Advanced Observables Demo ===\n");
    
    // Parameters
    let n = 20;
    let alpha = 1.5;
    let beta_values = vec![0.5, 1.0, 2.0, 3.0, 5.0];
    let equilibration_steps = 5000;
    let measurement_steps = 10000;
    let measure_interval = 10;
    
    println!("System size: N = {}", n);
    println!("Triangle coupling: α = {}", alpha);
    println!("Equilibration: {} steps", equilibration_steps);
    println!("Measurements: {} steps\n", measurement_steps);
    
    for &beta in &beta_values {
        println!("β = {}:", beta);
        println!("---------");
        
        // Initialize system
        let mut graph = FastGraph::new(n, 42 + (beta * 1000.0) as u64);
        let mut rng = Pcg64::seed_from_u64(42 + (beta * 1000.0) as u64);
        let mut observables = BatchedObservables::new();
        
        // Equilibrate
        for _ in 0..equilibration_steps {
            graph.metropolis_step(alpha, beta, 0.2, 0.5, &mut rng);
        }
        
        // Reset accumulators after equilibration
        observables.reset_accumulators();
        
        // Measure
        for step in 0..measurement_steps {
            graph.metropolis_step(alpha, beta, 0.2, 0.5, &mut rng);
            
            if step % measure_interval == 0 {
                let obs = observables.measure(&graph, alpha, beta);
                
                // Print progress and key observables every 1000 steps
                if step % 1000 == 0 && step > 0 {
                    println!("  Step {}: χ={:.3}, C={:.3}, U4={:.3}, ξ={:.3}",
                        step,
                        obs.susceptibility,
                        obs.specific_heat,
                        obs.binder_cumulant,
                        obs.correlation_length
                    );
                }
            }
        }
        
        // Final measurement
        let final_obs = observables.measure(&graph, alpha, beta);
        
        println!("\nFinal results:");
        println!("  <w> = {:.4} ± {:.4}", final_obs.mean_w, final_obs.var_w.sqrt());
        println!("  <cos θ> = {:.4}", final_obs.mean_cos);
        println!("  χ (susceptibility) = {:.4}", final_obs.susceptibility);
        println!("  C (specific heat) = {:.4}", final_obs.specific_heat);
        println!("  U₄ (Binder cumulant) = {:.4}", final_obs.binder_cumulant);
        println!("  ξ (correlation length) = {:.4}", final_obs.correlation_length);
        println!("  Samples accumulated: {}", observables.sample_count());
        
        // Show correlation function
        let (g0, g1) = graph.correlation_function();
        let xi_calc = graph.calculate_correlation_length();
        println!("\n  Correlation function:");
        println!("    G(0) = {:.6}", g0);
        println!("    G(1) = {:.6}", g1);
        println!("    ξ (from C(r)) = {:.4}", xi_calc);
        println!();
    }
    
    println!("\n=== Summary ===");
    println!("All observables implemented with correct normalizations:");
    println!("✓ Specific heat: C = (1/N) * (<E²> - <E>²)");
    println!("✓ Binder cumulant: U₄ = 1 - <m⁴>/(3<m²>²)");
    println!("✓ Susceptibility: χ = N * (<m²> - <m>²) where m = (1/N)∑cos(θ)");
    println!("✓ Correlation length: ξ = sqrt(<r²·C(r)> / <C(r)>)");
    println!("✓ Jackknife error estimation available via JackknifeEstimator");
    println!("\nNote: For complete graphs, correlation length calculation");
    println!("uses simplified distance metric (all nodes at distance 1).");
}