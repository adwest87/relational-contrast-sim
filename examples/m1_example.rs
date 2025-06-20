// Example of using M1-optimized Monte Carlo simulation

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    println!("This example requires Apple Silicon. Using fast implementation instead.");
    fast_example();
}

#[cfg(target_arch = "aarch64")]
fn main() {
    use scan::graph::Graph;
    use scan::graph_m1_optimized::M1Graph;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_pcg::Pcg64;
    use std::time::Instant;
    
    println!("M1-Optimized Monte Carlo Example");
    println!("================================");
    
    // Parameters
    let n = 36;
    let equilibration_steps = 10_000;
    let production_steps = 50_000;
    let alpha = 1.5;
    let beta = 2.9;
    let delta_z = 0.1;
    let delta_theta = 0.1;
    
    // Create initial graph
    let mut init_rng = ChaCha20Rng::seed_from_u64(42);
    let init_graph = Graph::complete_random_with(&mut init_rng, n);
    
    // Create M1-optimized graph from initial state
    let mut graph = M1Graph::from_graph(&init_graph);
    let mut rng = Pcg64::seed_from_u64(42);
    
    // Equilibration
    println!("\nEquilibrating for {} steps...", equilibration_steps);
    let start = Instant::now();
    
    for _ in 0..equilibration_steps {
        graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
    }
    
    let eq_time = start.elapsed();
    println!("Equilibration complete in {:.2} s ({:.0} steps/sec)", 
        eq_time.as_secs_f64(),
        equilibration_steps as f64 / eq_time.as_secs_f64()
    );
    
    // Production run with measurements
    println!("\nProduction run for {} steps...", production_steps);
    let start = Instant::now();
    let mut measurements = Vec::new();
    let mut accepts = 0;
    
    for step in 0..production_steps {
        let info = graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
        if info.accept {
            accepts += 1;
        }
        
        // Measure every 100 steps
        if step % 100 == 0 {
            let (mean_cos, mean_w, _) = graph.calculate_observables_simd();
            measurements.push((mean_cos, mean_w));
        }
    }
    
    let prod_time = start.elapsed();
    println!("Production complete in {:.2} s ({:.0} steps/sec)", 
        prod_time.as_secs_f64(),
        production_steps as f64 / prod_time.as_secs_f64()
    );
    
    // Analyze results
    let acceptance_rate = accepts as f64 / production_steps as f64;
    let avg_cos: f64 = measurements.iter().map(|(c, _)| c).sum::<f64>() / measurements.len() as f64;
    let avg_w: f64 = measurements.iter().map(|(_, w)| w).sum::<f64>() / measurements.len() as f64;
    
    println!("\nResults:");
    println!("  Acceptance rate: {:.1}%", acceptance_rate * 100.0);
    println!("  <cos θ> = {:.6} ± {:.6}", avg_cos, calculate_error(&measurements.iter().map(|(c, _)| *c).collect()));
    println!("  <w> = {:.6} ± {:.6}", avg_w, calculate_error(&measurements.iter().map(|(_, w)| *w).collect()));
    
    // Performance summary
    let total_steps = equilibration_steps + production_steps;
    let total_time = eq_time + prod_time;
    println!("\nPerformance Summary:");
    println!("  Total steps: {}", total_steps);
    println!("  Total time: {:.2} s", total_time.as_secs_f64());
    println!("  Average rate: {:.0} steps/sec", total_steps as f64 / total_time.as_secs_f64());
    println!("  Using: NEON SIMD + {} threads", rayon::current_num_threads());
}

fn calculate_error(data: &Vec<f64>) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (variance / n).sqrt()
}

// Fallback for non-ARM architectures
fn fast_example() {
    use scan::graph_fast::FastGraph;
    use rand_pcg::Pcg64;
    use rand::SeedableRng;
    use std::time::Instant;
    
    let n = 36;
    let steps = 50_000;
    let alpha = 1.5;
    let beta = 2.9;
    
    let mut graph = FastGraph::new(n, 42);
    let mut rng = Pcg64::seed_from_u64(42);
    
    let start = Instant::now();
    let mut accepts = 0;
    
    for _ in 0..steps {
        let info = graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
        if info.accept {
            accepts += 1;
        }
    }
    
    let elapsed = start.elapsed();
    println!("Fast implementation (non-M1):");
    println!("  {} steps in {:.2} s", steps, elapsed.as_secs_f64());
    println!("  Rate: {:.0} steps/sec", steps as f64 / elapsed.as_secs_f64());
    println!("  Acceptance: {:.1}%", 100.0 * accepts as f64 / steps as f64);
}