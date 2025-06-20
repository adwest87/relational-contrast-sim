// Example of using Metal GPU acceleration for Monte Carlo simulations

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("This example requires macOS with Metal support");
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use scan::graph::Graph;
    use scan::graph_metal::MetalGraph;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use std::time::Instant;
    
    println!("Metal GPU Monte Carlo Example");
    println!("=============================");
    
    // Parameters
    let n = 72;  // Larger system size to benefit from GPU
    let equilibration_steps = 5_000;
    let production_steps = 20_000;
    let alpha = 1.5;
    let beta = 2.9;
    let delta_z = 0.1;
    let delta_theta = 0.1;
    
    println!("\nSimulation parameters:");
    println!("  System size: N = {} ({} links)", n, n * (n - 1) / 2);
    println!("  Equilibration: {} steps", equilibration_steps);
    println!("  Production: {} steps", production_steps);
    println!("  α = {}, β = {}", alpha, beta);
    
    // Create initial graph on CPU
    println!("\nInitializing graph...");
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let cpu_graph = Graph::complete_random_with(&mut rng, n);
    
    // Transfer to GPU
    println!("Creating GPU graph...");
    let mut gpu_graph = MetalGraph::from_graph(&cpu_graph)?;
    
    // Equilibration phase
    println!("\nEquilibration phase...");
    let start = Instant::now();
    let mut eq_accepts = 0u32;
    
    for step in 0..equilibration_steps {
        let accepts = gpu_graph.metropolis_step_gpu(alpha, beta, delta_z, delta_theta);
        eq_accepts += accepts;
        
        if step > 0 && step % 1000 == 0 {
            let (mean_cos, mean_w, _) = gpu_graph.compute_observables_gpu();
            println!("  Step {}: <cos θ> = {:.4}, <w> = {:.4}", step, mean_cos, mean_w);
        }
    }
    
    let eq_time = start.elapsed();
    let n_links = n * (n - 1) / 2;
    let eq_rate = (equilibration_steps * n_links) as f64 / eq_time.as_secs_f64();
    println!("Equilibration complete:");
    println!("  Time: {:.2} s", eq_time.as_secs_f64());
    println!("  Accept rate: {:.1}%", 100.0 * eq_accepts as f64 / (equilibration_steps * n_links) as f64);
    println!("  Link updates/sec: {:.2} million", eq_rate / 1e6);
    
    // Production phase with measurements
    println!("\nProduction phase...");
    let start = Instant::now();
    let mut prod_accepts = 0u32;
    let mut measurements = Vec::new();
    let measure_interval = 100;
    
    for step in 0..production_steps {
        let accepts = gpu_graph.metropolis_step_gpu(alpha, beta, delta_z, delta_theta);
        prod_accepts += accepts;
        
        // Measure observables periodically
        if step % measure_interval == 0 {
            let (mean_cos, mean_w, mean_w_cos) = gpu_graph.compute_observables_gpu();
            let entropy = gpu_graph.entropy_action_gpu();
            let triangle_sum = gpu_graph.triangle_sum_gpu() as f64;
            
            measurements.push((mean_cos, mean_w, entropy, triangle_sum));
            
            if step % 5000 == 0 && step > 0 {
                println!("  Step {}: <cos θ> = {:.4}, <w> = {:.4}, S = {:.2}", 
                    step, mean_cos, mean_w, entropy);
            }
        }
    }
    
    let prod_time = start.elapsed();
    let prod_rate = (production_steps * n_links) as f64 / prod_time.as_secs_f64();
    
    println!("\nProduction complete:");
    println!("  Time: {:.2} s", prod_time.as_secs_f64());
    println!("  Accept rate: {:.1}%", 100.0 * prod_accepts as f64 / (production_steps * n_links) as f64);
    println!("  Link updates/sec: {:.2} million", prod_rate / 1e6);
    
    // Analyze measurements
    let n_meas = measurements.len() as f64;
    let avg_cos = measurements.iter().map(|(c, _, _, _)| c).sum::<f64>() / n_meas;
    let avg_w = measurements.iter().map(|(_, w, _, _)| w).sum::<f64>() / n_meas;
    let avg_entropy = measurements.iter().map(|(_, _, s, _)| s).sum::<f64>() / n_meas;
    let avg_triangles = measurements.iter().map(|(_, _, _, t)| t).sum::<f64>() / n_meas;
    
    println!("\nFinal results ({} measurements):", measurements.len());
    println!("  <cos θ> = {:.6} ± {:.6}", avg_cos, calculate_error(&measurements.iter().map(|(c, _, _, _)| *c).collect()));
    println!("  <w> = {:.6} ± {:.6}", avg_w, calculate_error(&measurements.iter().map(|(_, w, _, _)| *w).collect()));
    println!("  <S> = {:.2} ± {:.2}", avg_entropy, calculate_error(&measurements.iter().map(|(_, _, s, _)| *s).collect()));
    println!("  <Σ△> = {:.2} ± {:.2}", avg_triangles, calculate_error(&measurements.iter().map(|(_, _, _, t)| *t).collect()));
    
    // Performance summary
    let total_steps = equilibration_steps + production_steps;
    let total_time = eq_time + prod_time;
    let total_link_updates = (total_steps * n_links) as f64;
    
    println!("\nPerformance summary:");
    println!("  Total MC steps: {}", total_steps);
    println!("  Total link updates: {:.2} million", total_link_updates / 1e6);
    println!("  Total time: {:.2} s", total_time.as_secs_f64());
    println!("  Average rate: {:.2} million link updates/sec", total_link_updates / total_time.as_secs_f64() / 1e6);
    
    // GPU utilization estimate
    let theoretical_flops = 8e9; // M1 GPU: ~8 TFLOPS
    let flops_per_update = 50.0; // Rough estimate: exp, cos, sin, etc.
    let achieved_flops = total_link_updates * flops_per_update / total_time.as_secs_f64();
    let utilization = achieved_flops / theoretical_flops * 100.0;
    
    println!("\nGPU utilization estimate:");
    println!("  Theoretical peak: 8 TFLOPS");
    println!("  Estimated FLOPS: {:.2} GFLOPS", achieved_flops / 1e9);
    println!("  Utilization: {:.1}%", utilization);
    
    Ok(())
}

fn calculate_error(data: &Vec<f64>) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (variance / n).sqrt()
}