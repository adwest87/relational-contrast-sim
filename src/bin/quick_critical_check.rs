// Quick check at various parameter points
use scan::graph_fast::{FastGraph, BatchedObservables};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::io::Write;

fn main() {
    println!("🔍 QUICK CRITICAL REGION CHECK");
    println!("==============================");
    
    let n = 48;
    let test_points = vec![
        (2.88, 1.48, "Paper"),
        (1.0, 1.5, "Low β"),
        (10.0, 1.5, "High β"),
        (2.88, 0.5, "Low α"),
        (2.88, 3.0, "High α"),
    ];
    
    println!("System: N={}", n);
    println!("Running 20k equilibration + 20k production");
    
    for (beta, alpha, label) in test_points {
        print!("\n{} (β={:.2}, α={:.2}): ", label, beta, alpha);
        std::io::stdout().flush().unwrap();
        
        let mut rng = Pcg64::seed_from_u64(42);
        let mut graph = FastGraph::new(n, 12345);
        
        // Equilibration
        for _ in 0..20_000 {
            graph.metropolis_step(alpha, beta, 0.5, 0.5, &mut rng);
        }
        
        // Production
        let mut chi_values = Vec::new();
        let mut cos_values = Vec::new();
        let mut observable_calc = BatchedObservables::new();
        
        for step in 0..20_000 {
            graph.metropolis_step(alpha, beta, 0.5, 0.5, &mut rng);
            
            if step % 100 == 0 {
                let obs = observable_calc.measure(&graph, alpha, beta);
                chi_values.push(obs.susceptibility);
                cos_values.push(obs.mean_cos);
            }
        }
        
        let mean_chi = chi_values.iter().sum::<f64>() / chi_values.len() as f64;
        let mean_cos = cos_values.iter().sum::<f64>() / cos_values.len() as f64;
        
        // Calculate Binder
        let m2_values: Vec<f64> = cos_values.iter().map(|&m| m * m).collect();
        let m4_values: Vec<f64> = cos_values.iter().map(|&m| m.powi(4)).collect();
        let mean_m2 = m2_values.iter().sum::<f64>() / m2_values.len() as f64;
        let mean_m4 = m4_values.iter().sum::<f64>() / m4_values.len() as f64;
        let binder = 1.0 - mean_m4 / (3.0 * mean_m2.powi(2));
        
        // Check energy scales
        let entropy = graph.entropy_action();
        let triangle = graph.triangle_sum();
        let total_action = beta * entropy + alpha * triangle;
        
        println!("\n  χ = {:.1}, U₄ = {:.3}, |M| = {:.3}", mean_chi, binder, mean_cos.abs());
        println!("  S_entropy = {:.1}, S_triangle = {:.1}, S_total = {:.1}", 
            entropy, triangle, total_action);
        println!("  Energy contributions: β*S_e = {:.1}, α*S_t = {:.1}",
            beta * entropy, alpha * triangle);
    }
    
    println!("\n\n📊 ANALYSIS:");
    println!("If χ ~ 2-3 everywhere, the system may not have a phase transition");
    println!("or we need much larger system sizes to see it.");
}