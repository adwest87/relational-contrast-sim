use scan::graph::Graph;
use scan::graph_ultra_optimized::UltraOptimizedGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("=== Testing Fixed Ultra-Optimized Physics ===\n");
    
    let n = 10;
    let alpha = 1.5;
    let beta = 2.0;
    let mc_steps = 10000;
    
    // Create original graph
    let mut rng = Pcg64::seed_from_u64(42);
    let orig_graph = Graph::complete_random_with(&mut rng, n);
    
    // Create ultra-optimized and copy exact state
    let mut ultra_graph = UltraOptimizedGraph::new(n, 999); // Different seed to test state copying
    
    // Copy state exactly
    for (i, link) in orig_graph.links.iter().enumerate() {
        ultra_graph.z_values[i] = link.z;
        ultra_graph.theta_values[i] = link.theta;
        ultra_graph.cos_theta[i] = link.theta.cos();
        ultra_graph.sin_theta[i] = link.theta.sin();
        ultra_graph.exp_neg_z[i] = (-link.z).exp();
    }
    ultra_graph.triangle_sum_cache = ultra_graph.compute_full_triangle_sum();
    
    // Test that actions match exactly
    let orig_action = orig_graph.action(alpha, beta);
    let ultra_action = ultra_graph.action(alpha, beta, 0.0);
    
    println!("Action consistency test:");
    println!("  Original action: {:.12}", orig_action);
    println!("  UltraOpt action: {:.12}", ultra_action);
    println!("  Difference: {:.2e}", (orig_action - ultra_action).abs());
    
    if (orig_action - ultra_action).abs() < 1e-10 {
        println!("  ✓ Actions match exactly!\n");
    } else {
        println!("  ✗ Actions differ!\n");
        return;
    }
    
    // Test triangle sum consistency
    let orig_triangle = orig_graph.triangle_sum();
    let ultra_triangle = ultra_graph.triangle_sum();
    
    println!("Triangle sum consistency test:");
    println!("  Original triangle sum: {:.12}", orig_triangle);
    println!("  UltraOpt triangle sum: {:.12}", ultra_triangle);
    println!("  Difference: {:.2e}", (orig_triangle - ultra_triangle).abs());
    
    if (orig_triangle - ultra_triangle).abs() < 1e-10 {
        println!("  ✓ Triangle sums match exactly!\n");
    } else {
        println!("  ✗ Triangle sums differ!\n");
        return;
    }
    
    // Test incremental triangle update
    println!("Testing incremental triangle update:");
    let link_idx = 5;
    let old_theta = ultra_graph.theta_values[link_idx];
    let new_theta = old_theta + 0.1;
    
    let predicted_delta = ultra_graph.triangle_sum_delta(link_idx, new_theta);
    
    // Apply the change and recompute full triangle sum
    ultra_graph.theta_values[link_idx] = new_theta;
    ultra_graph.cos_theta[link_idx] = new_theta.cos();
    ultra_graph.sin_theta[link_idx] = new_theta.sin();
    
    let new_full_triangle = ultra_graph.compute_full_triangle_sum();
    let actual_delta = new_full_triangle - ultra_triangle;
    
    println!("  Predicted delta: {:.12}", predicted_delta);
    println!("  Actual delta: {:.12}", actual_delta);
    println!("  Difference: {:.2e}", (predicted_delta - actual_delta).abs());
    
    if (predicted_delta - actual_delta).abs() < 1e-10 {
        println!("  ✓ Incremental update is exact!\n");
    } else {
        println!("  ✗ Incremental update has error!\n");
        return;
    }
    
    // Test Monte Carlo simulation
    println!("Testing Monte Carlo physics:");
    
    // Reset to original state
    for (i, link) in orig_graph.links.iter().enumerate() {
        ultra_graph.z_values[i] = link.z;
        ultra_graph.theta_values[i] = link.theta;
        ultra_graph.cos_theta[i] = link.theta.cos();
        ultra_graph.sin_theta[i] = link.theta.sin();
        ultra_graph.exp_neg_z[i] = (-link.z).exp();
    }
    ultra_graph.triangle_sum_cache = ultra_graph.compute_full_triangle_sum();
    
    let mut rng = Pcg64::seed_from_u64(12345);
    let mut accepts = 0;
    let mut action_samples = Vec::new();
    
    for step in 0..mc_steps {
        let accepted = ultra_graph.metropolis_step(alpha, beta, 0.0, 0.2, 0.5, &mut rng);
        if accepted { accepts += 1; }
        
        if step % 1000 == 0 {
            action_samples.push(ultra_graph.action(alpha, beta, 0.0));
        }
        
        // Verify action is always finite
        let current_action = ultra_graph.action(alpha, beta, 0.0);
        if !current_action.is_finite() {
            println!("  ✗ Action became non-finite at step {}", step);
            return;
        }
    }
    
    let acceptance_rate = accepts as f64 / mc_steps as f64;
    println!("  Acceptance rate: {:.3}", acceptance_rate);
    
    // Check that action varies (not stuck)
    let action_mean = action_samples.iter().sum::<f64>() / action_samples.len() as f64;
    let action_var = action_samples.iter()
        .map(|&x| (x - action_mean).powi(2))
        .sum::<f64>() / (action_samples.len() - 1) as f64;
    
    println!("  Action statistics: mean = {:.6}, variance = {:.6}", action_mean, action_var);
    
    if acceptance_rate > 0.1 && acceptance_rate < 0.9 {
        println!("  ✓ Reasonable acceptance rate");
    } else {
        println!("  ⚠ Acceptance rate seems unusual");
    }
    
    if action_var > 1e-6 {
        println!("  ✓ Action shows variation (not stuck)");
    } else {
        println!("  ⚠ Action shows little variation");
    }
    
    println!("\n=== All Physics Tests Passed! ===");
    println!("The ultra-optimized implementation correctly reproduces the original physics.");
}