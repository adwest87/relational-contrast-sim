// Test O(2) symmetry properties
use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::f64::consts::TAU;

fn main() {
    println!("🔄 SYMMETRY TEST");
    println!("================");
    println!("Testing O(2) symmetry properties\n");
    
    let n = 48;
    let beta = 10.0; // Low temperature to see ordering
    let alpha = 1.5;
    
    println!("System: N={}, β={:.1}, α={:.1}", n, beta, alpha);
    
    // Test 1: Check that we're measuring per-link averages correctly
    println!("\n📊 TEST 1: Order parameter calculation");
    println!("=====================================");
    
    let mut graph = FastGraph::new(n, 12345);
    
    // Set all angles to 0 (perfect alignment)
    for i in 0..graph.links.len() {
        graph.links[i].update_theta(0.0);
    }
    
    let cos_sum: f64 = graph.links.iter().map(|l| l.cos_theta).sum();
    let sin_sum: f64 = graph.links.iter().map(|l| l.sin_theta).sum();
    let m = graph.links.len() as f64;
    
    println!("All θ=0:");
    println!("  Sum cos θ = {:.3}", cos_sum);
    println!("  Sum sin θ = {:.3}", sin_sum);
    println!("  <cos θ> = {:.3}", cos_sum / m);
    println!("  <sin θ> = {:.3}", sin_sum / m);
    println!("  Expected: <cos θ> = 1.0, <sin θ> = 0.0");
    
    // Set all angles to π/2
    for i in 0..graph.links.len() {
        graph.links[i].update_theta(TAU / 4.0);
    }
    
    let cos_sum: f64 = graph.links.iter().map(|l| l.cos_theta).sum();
    let sin_sum: f64 = graph.links.iter().map(|l| l.sin_theta).sum();
    
    println!("\nAll θ=π/2:");
    println!("  <cos θ> = {:.3}", cos_sum / m);
    println!("  <sin θ> = {:.3}", sin_sum / m);
    println!("  Expected: <cos θ> = 0.0, <sin θ> = 1.0");
    
    // Test 2: Evolution from different initial conditions
    println!("\n📊 TEST 2: Evolution from ordered states");
    println!("=======================================");
    
    let equilibration_steps = 50_000;
    let production_steps = 50_000;
    let initial_conditions = vec![
        (0.0, "θ=0 (aligned +x)"),
        (TAU/4.0, "θ=π/2 (aligned +y)"),
        (TAU/2.0, "θ=π (aligned -x)"),
        (3.0*TAU/4.0, "θ=3π/2 (aligned -y)"),
    ];
    
    for (initial_theta, label) in initial_conditions {
        print!("\n{}: ", label);
        
        let mut rng = Pcg64::seed_from_u64(42);
        let mut graph = FastGraph::new(n, 12345);
        
        // Initialize all angles to specific value
        for i in 0..graph.links.len() {
            graph.links[i].update_theta(initial_theta);
        }
        
        // Equilibrate
        print!("equilibrating...");
        for _ in 0..equilibration_steps {
            graph.metropolis_step(alpha, beta, 0.3, 0.3, &mut rng);
        }
        
        // Measure
        print!(" measuring...");
        let mut cos_values = Vec::new();
        let mut sin_values = Vec::new();
        
        for step in 0..production_steps {
            graph.metropolis_step(alpha, beta, 0.3, 0.3, &mut rng);
            
            if step % 100 == 0 {
                let cos_sum: f64 = graph.links.iter().map(|l| l.cos_theta).sum();
                let sin_sum: f64 = graph.links.iter().map(|l| l.sin_theta).sum();
                cos_values.push(cos_sum / m);
                sin_values.push(sin_sum / m);
            }
        }
        
        let mean_cos = cos_values.iter().sum::<f64>() / cos_values.len() as f64;
        let mean_sin = sin_values.iter().sum::<f64>() / sin_values.len() as f64;
        let magnitude = (mean_cos * mean_cos + mean_sin * mean_sin).sqrt();
        
        println!("\n  <cos θ> = {:.3}, <sin θ> = {:.3}", mean_cos, mean_sin);
        println!("  |M| = {:.3}", magnitude);
        
        // Check angle distribution
        let angles: Vec<f64> = graph.links.iter().map(|l| l.theta).collect();
        let mean_angle = angles.iter().sum::<f64>() / angles.len() as f64;
        let angle_std = (angles.iter()
            .map(|&a| {
                let diff = (a - mean_angle).rem_euclid(TAU);
                let diff = if diff > TAU/2.0 { diff - TAU } else { diff };
                diff * diff
            })
            .sum::<f64>() / angles.len() as f64).sqrt();
        
        println!("  Mean angle: {:.3} ({:.1}°)", mean_angle, mean_angle.to_degrees());
        println!("  Angle std: {:.3} ({:.1}°)", angle_std, angle_std.to_degrees());
    }
    
    // Test 3: Check for spontaneous symmetry breaking
    println!("\n📊 TEST 3: Spontaneous symmetry breaking");
    println!("======================================");
    
    let n_replicas = 10;
    let mut magnitudes = Vec::new();
    let mut angles = Vec::new();
    
    print!("Running {} replicas from random initial conditions", n_replicas);
    
    for replica in 0..n_replicas {
        if replica % 2 == 0 {
            print!(".");
        }
        
        let mut rng = Pcg64::seed_from_u64(100 + replica as u64);
        let mut graph = FastGraph::new(n, 200 + replica as u64);
        
        // Equilibrate
        for _ in 0..100_000 {
            graph.metropolis_step(alpha, beta, 0.3, 0.3, &mut rng);
        }
        
        // Measure final state
        let cos_sum: f64 = graph.links.iter().map(|l| l.cos_theta).sum();
        let sin_sum: f64 = graph.links.iter().map(|l| l.sin_theta).sum();
        let mean_cos = cos_sum / m;
        let mean_sin = sin_sum / m;
        
        let magnitude = (mean_cos * mean_cos + mean_sin * mean_sin).sqrt();
        let angle = mean_sin.atan2(mean_cos);
        
        magnitudes.push(magnitude);
        angles.push(angle);
    }
    
    println!("\n\nResults:");
    println!("  Magnitudes |M|: {:?}", magnitudes.iter()
        .map(|&x| format!("{:.3}", x))
        .collect::<Vec<_>>());
    println!("  Angles (degrees): {:?}", angles.iter()
        .map(|&x| format!("{:.0}", x.to_degrees()))
        .collect::<Vec<_>>());
    
    let mean_magnitude = magnitudes.iter().sum::<f64>() / n_replicas as f64;
    println!("\n  Mean |M| = {:.3}", mean_magnitude);
    
    if mean_magnitude > 0.5 {
        println!("  ✅ System shows ordering at low temperature");
        
        // Check if angles are uniformly distributed
        let angle_variance = angles.iter()
            .map(|&a| a * a)
            .sum::<f64>() / n_replicas as f64;
        
        if angle_variance > 1.0 {
            println!("  ✅ Spontaneous symmetry breaking: different replicas choose different directions");
        } else {
            println!("  ❌ No spontaneous symmetry breaking: all replicas align similarly");
        }
    } else {
        println!("  ❌ System does not order even at low temperature!");
        println!("  This suggests a fundamental issue with the model");
    }
}