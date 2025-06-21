// Test if order parameter is working correctly
use scan::graph_fast::{FastGraph, BatchedObservables};
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("🧭 ORDER PARAMETER TEST");
    println!("======================");
    
    // Test 1: Initialize all angles to 0 (perfect alignment)
    println!("\n📐 TEST 1: PERFECT ALIGNMENT (θ=0 for all links)");
    test_aligned_state();
    
    // Test 2: Initialize all angles randomly
    println!("\n🎲 TEST 2: RANDOM CONFIGURATION");
    test_random_state();
    
    // Test 3: Force specific angles to test cos calculation
    println!("\n🔧 TEST 3: SPECIFIC ANGLE TESTS");
    test_specific_angles();
    
    // Test 4: Very low temperature evolution from aligned state
    println!("\n❄️ TEST 4: LOW TEMPERATURE EVOLUTION");
    test_low_temperature_evolution();
}

fn test_aligned_state() {
    let n = 16;
    let mut graph = FastGraph::new(n, 12345);
    
    // Force all links to have θ = 0 (perfect alignment)
    for i in 0..graph.links.len() {
        graph.links[i].update_theta(0.0);
    }
    
    let mut observable_calc = BatchedObservables::new();
    let obs = observable_calc.measure(&graph, 1.5, 3.0);
    
    println!("Perfect alignment (all θ=0):");
    println!("  <cos θ> = {:.6}", obs.mean_cos);
    println!("  Expected: ~1.0");
    println!("  |M| = {:.6}", obs.mean_cos.abs());
    
    // Check individual link angles
    println!("  First 5 link angles: {:?}", 
        graph.links.iter().take(5).map(|l| l.theta).collect::<Vec<_>>());
    
    // Verify all cosines are 1.0
    let all_cos_one = graph.links.iter().all(|l| (l.cos_theta - 1.0).abs() < 1e-10);
    println!("  All cos(θ) ≈ 1.0: {}", all_cos_one);
}

fn test_random_state() {
    let n = 16;
    let graph = FastGraph::new(n, 999); // Random initialization
    
    let mut observable_calc = BatchedObservables::new();
    let obs = observable_calc.measure(&graph, 1.5, 3.0);
    
    println!("Random initialization:");
    println!("  <cos θ> = {:.6}", obs.mean_cos);
    println!("  Expected: close to 0.0");
    println!("  |M| = {:.6}", obs.mean_cos.abs());
    
    // Check angle distribution
    let angles: Vec<f64> = graph.links.iter().map(|l| l.theta).collect();
    let mean_angle = angles.iter().sum::<f64>() / angles.len() as f64;
    let angle_variance = angles.iter().map(|&a| (a - mean_angle).powi(2)).sum::<f64>() / angles.len() as f64;
    
    println!("  Mean angle: {:.6}", mean_angle);
    println!("  Angle variance: {:.6}", angle_variance);
    println!("  First 5 angles: {:?}", &angles[0..5]);
}

fn test_specific_angles() {
    let n = 4; // Small system for manual verification
    let mut graph = FastGraph::new(n, 12345);
    
    // Test specific angle values
    let test_angles = [0.0, std::f64::consts::PI/4.0, std::f64::consts::PI/2.0, std::f64::consts::PI];
    
    for &angle in &test_angles {
        // Set all links to this angle
        for i in 0..graph.links.len() {
            graph.links[i].update_theta(angle);
        }
        
        let mut observable_calc = BatchedObservables::new();
        let obs = observable_calc.measure(&graph, 1.5, 3.0);
        
        println!("θ = {:.4} ({:.1}°):", angle, angle.to_degrees());
        println!("  <cos θ> = {:.6} (expected: {:.6})", obs.mean_cos, angle.cos());
        println!("  Difference: {:.2e}", (obs.mean_cos - angle.cos()).abs());
    }
}

fn test_low_temperature_evolution() {
    let n = 16;
    let mut graph = FastGraph::new(n, 12345);
    let mut rng = Pcg64::seed_from_u64(42);
    
    // Start with aligned state (all θ = 0)
    for i in 0..graph.links.len() {
        graph.links[i].update_theta(0.0);
    }
    
    println!("Starting from perfect alignment, evolving at very low T...");
    
    let alpha = 1.5;
    let beta = 100.0; // Very low temperature T = 0.01
    let delta_z = 0.01; // Very small moves
    let delta_theta = 0.01;
    
    let mut observable_calc = BatchedObservables::new();
    
    // Initial state
    let obs_initial = observable_calc.measure(&graph, alpha, beta);
    println!("Initial: <cos θ> = {:.6}", obs_initial.mean_cos);
    
    // Evolve for several steps
    let steps = [100, 1000, 10000];
    for &step_count in &steps {
        for _ in 0..step_count {
            graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
        }
        
        let obs = observable_calc.measure(&graph, alpha, beta);
        println!("After {} steps: <cos θ> = {:.6}", step_count, obs.mean_cos);
        
        if step_count == 100 {
            // Check if any angles changed significantly
            let max_angle = graph.links.iter().map(|l| l.theta.abs()).fold(0.0, f64::max);
            println!("  Max |θ|: {:.6}", max_angle);
        }
    }
    
    // Final check: should still be highly ordered at low T
    let obs_final = observable_calc.measure(&graph, alpha, beta);
    println!("Final: <cos θ> = {:.6}", obs_final.mean_cos);
    
    if obs_final.mean_cos.abs() < 0.8 {
        println!("⚠️ WARNING: Order parameter degraded at low temperature!");
    } else {
        println!("✓ Order parameter maintained at low temperature");
    }
}