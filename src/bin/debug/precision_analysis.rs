// Analyze numerical precision issues in triangle_sum_delta
use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("🔬 NUMERICAL PRECISION ANALYSIS");
    println!("===============================");
    
    // Test energy scale analysis
    energy_scale_analysis();
    
    // Test triangle calculation precision
    triangle_precision_test();
    
    // Test trigonometric precision
    trig_precision_test();
}

fn energy_scale_analysis() {
    println!("\n📊 ENERGY SCALE ANALYSIS");
    println!("========================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(42);
    let mut graph = FastGraph::new(n, 123);
    
    let mut energies = Vec::new();
    let mut deltas = Vec::new();
    
    for _ in 0..1000 {
        let energy_before = graph.action(alpha, beta);
        energies.push(energy_before);
        
        let _ = graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
        
        let energy_after = graph.action(alpha, beta);
        let delta = (energy_after - energy_before).abs();
        if delta > 0.0 {
            deltas.push(delta);
        }
    }
    
    let mean_energy = energies.iter().sum::<f64>() / energies.len() as f64;
    let min_energy = energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_energy = energies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    let mean_delta = if !deltas.is_empty() { deltas.iter().sum::<f64>() / deltas.len() as f64 } else { 0.0 };
    let min_delta = if !deltas.is_empty() { deltas.iter().fold(f64::INFINITY, |a, &b| a.min(b)) } else { 0.0 };
    let max_delta = if !deltas.is_empty() { deltas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) } else { 0.0 };
    
    println!("Energy statistics:");
    println!("  Mean energy: {:.6e}", mean_energy);
    println!("  Min energy:  {:.6e}", min_energy);
    println!("  Max energy:  {:.6e}", max_energy);
    println!("  Range:       {:.6e}", max_energy - min_energy);
    
    println!("ΔE statistics ({} non-zero deltas):", deltas.len());
    println!("  Mean |ΔE|:   {:.6e}", mean_delta);
    println!("  Min |ΔE|:    {:.6e}", min_delta);
    println!("  Max |ΔE|:    {:.6e}", max_delta);
    
    // Calculate relative tolerances
    let relative_to_energy = mean_delta / mean_energy.abs();
    println!("  Relative to energy: {:.6e}", relative_to_energy);
    
    // Suggest appropriate epsilon
    let suggested_epsilon = mean_energy.abs() * 1e-12;
    println!("  Suggested absolute ε: {:.6e}", suggested_epsilon);
    println!("  Suggested relative ε: 1e-12");
}

fn triangle_precision_test() {
    println!("\n🔺 TRIANGLE CALCULATION PRECISION TEST");
    println!("=====================================");
    
    let n = 4;
    let mut graph = FastGraph::new(n, 123);
    
    // Test identical calculation 1000 times
    let link_idx = 0;
    let new_theta = 0.1;
    
    let mut results = Vec::new();
    for i in 0..1000 {
        // Create identical state each time
        if i > 0 {
            graph = FastGraph::new(n, 123);
        }
        
        let result = calculate_triangle_sum_delta_reference(&graph, link_idx, new_theta);
        results.push(result);
    }
    
    let first = results[0];
    let variance = results.iter().map(|&x| (x - first).powi(2)).sum::<f64>() / results.len() as f64;
    let max_diff = results.iter().map(|&x| (x - first).abs()).fold(0.0, f64::max);
    
    println!("Triangle calculation stability:");
    println!("  First result: {:.17e}", first);
    println!("  Variance:     {:.6e}", variance);
    println!("  Max diff:     {:.6e}", max_diff);
    println!("  Stable:       {}", variance < 1e-14);
}

fn trig_precision_test() {
    println!("\n📐 TRIGONOMETRIC PRECISION TEST");
    println!("===============================");
    
    // Test cos(x+δ) - cos(x) precision for small δ
    let base_angles: [f64; 6] = [0.0, 0.1, 0.5, 1.0, 1.5, 3.0];
    let deltas: [f64; 5] = [1e-16, 1e-12, 1e-8, 1e-4, 1e-2];
    
    println!("Testing cos(x+δ) - cos(x) vs analytical approximation:");
    println!("Format: angle, delta, numerical, analytical, rel_error");
    
    for &angle in &base_angles {
        for &delta in &deltas {
            let numerical = (angle + delta).cos() - angle.cos();
            let analytical = -angle.sin() * delta - 0.5 * angle.cos() * delta * delta;
            let rel_error = if analytical != 0.0 { 
                ((numerical - analytical) / analytical).abs() 
            } else { 
                numerical.abs() 
            };
            
            println!("  {:.1}, {:.0e}, {:.6e}, {:.6e}, {:.2e}", 
                angle, delta, numerical, analytical, rel_error);
        }
    }
}

// Reference implementation for comparison
fn calculate_triangle_sum_delta_reference(graph: &FastGraph, link_idx: usize, new_theta: f64) -> f64 {
    let link = &graph.links[link_idx];
    let (i, j) = (link.i as usize, link.j as usize);
    let old_theta = link.theta;
    
    let mut delta = 0.0;
    
    for k in 0..graph.n() {
        if k != i && k != j {
            let idx_ik = if i < k { 
                graph.link_index(i, k) 
            } else { 
                graph.link_index(k, i) 
            };
            let idx_jk = if j < k { 
                graph.link_index(j, k) 
            } else { 
                graph.link_index(k, j) 
            };
            
            let old_sum = old_theta + graph.links[idx_ik].theta + graph.links[idx_jk].theta;
            let new_sum = new_theta + graph.links[idx_ik].theta + graph.links[idx_jk].theta;
            
            delta += 3.0 * (new_sum.cos() - old_sum.cos());
        }
    }
    
    delta
}