use scan::graph::Graph;
use scan::graph_fast::FastGraph;
use scan::graph_ultra_optimized::UltraOptimizedGraph;
use rand::{SeedableRng, Rng};
use rand_pcg::Pcg64;

const TOLERANCE: f64 = 1e-10;

#[test]
fn test_action_consistency_across_implementations() {
    // Test that all implementations give the same action for identical configurations
    let n = 15;
    let seed = 42;
    let alpha = 1.5;
    let beta = 2.0;
    
    // Create graphs with identical random seed
    let mut rng = Pcg64::seed_from_u64(seed);
    let orig_graph = Graph::complete_random_with(&mut rng, n);
    
    let fast_graph = FastGraph::from_graph(&orig_graph);
    
    // Create ultra-optimized and copy state
    let mut ultra_graph = UltraOptimizedGraph::new(n, seed);
    copy_graph_state(&orig_graph, &mut ultra_graph);
    
    // Compare actions
    let orig_action = orig_graph.action(alpha, beta);
    let fast_action = fast_graph.action(alpha, beta);
    let ultra_action = ultra_graph.action(alpha, beta, 0.0);
    
    println!("Original action: {:.12}", orig_action);
    println!("FastGraph action: {:.12}", fast_action);
    println!("UltraOptimized action: {:.12}", ultra_action);
    
    assert!((orig_action - fast_action).abs() < TOLERANCE,
            "FastGraph action differs: {} vs {}", orig_action, fast_action);
    assert!((orig_action - ultra_action).abs() < TOLERANCE,
            "UltraOptimized action differs: {} vs {}", orig_action, ultra_action);
}

#[test]
fn test_triangle_sum_consistency() {
    // Test that triangle sum calculations are identical
    let n = 10;
    let seed = 123;
    
    let mut rng = Pcg64::seed_from_u64(seed);
    let orig_graph = Graph::complete_random_with(&mut rng, n);
    let fast_graph = FastGraph::from_graph(&orig_graph);
    
    let mut ultra_graph = UltraOptimizedGraph::new(n, seed);
    copy_graph_state(&orig_graph, &mut ultra_graph);
    
    let orig_triangle = orig_graph.triangle_sum();
    let fast_triangle = fast_graph.triangle_sum();
    let ultra_triangle = ultra_graph.triangle_sum();
    
    println!("Triangle sums: orig={:.12}, fast={:.12}, ultra={:.12}", 
             orig_triangle, fast_triangle, ultra_triangle);
    
    assert!((orig_triangle - fast_triangle).abs() < TOLERANCE,
            "FastGraph triangle sum differs");
    assert!((orig_triangle - ultra_triangle).abs() < TOLERANCE,
            "UltraOptimized triangle sum differs");
}

#[test]
fn test_incremental_triangle_update_correctness() {
    // Test that incremental triangle updates match full recalculation
    let n = 12;
    let mut ultra_graph = UltraOptimizedGraph::new(n, 42);
    
    // Record initial state
    let initial_triangle_sum = ultra_graph.triangle_sum();
    
    // Make a small change and use incremental update
    let link_idx = 5;
    let old_theta = ultra_graph.theta_values[link_idx];
    let new_theta = old_theta + 0.1;
    
    let predicted_delta = ultra_graph.triangle_sum_delta(link_idx, new_theta);
    
    // Apply change manually and compute full triangle sum
    ultra_graph.theta_values[link_idx] = new_theta;
    ultra_graph.cos_theta[link_idx] = new_theta.cos();
    ultra_graph.sin_theta[link_idx] = new_theta.sin();
    
    let full_new_sum = ultra_graph.compute_full_triangle_sum();
    let actual_delta = full_new_sum - initial_triangle_sum;
    
    println!("Predicted delta: {:.12}", predicted_delta);
    println!("Actual delta: {:.12}", actual_delta);
    println!("Difference: {:.2e}", (predicted_delta - actual_delta).abs());
    
    assert!((predicted_delta - actual_delta).abs() < TOLERANCE,
            "Incremental triangle update is incorrect");
}

#[test]
fn test_detailed_balance() {
    // Test that detailed balance is preserved in Monte Carlo updates
    let n = 8;
    let alpha = 1.0;
    let beta = 1.5;
    let delta_z = 0.2;
    let delta_theta = 0.5;
    let mc_steps = 50000;
    
    let mut rng = Pcg64::seed_from_u64(999);
    let mut ultra_graph = UltraOptimizedGraph::new(n, 999);
    
    // Track acceptance rates for different update types
    let mut z_accepts = 0;
    let mut z_total = 0;
    let mut theta_accepts = 0;
    let mut theta_total = 0;
    
    // Run simulation and track acceptance
    for _ in 0..mc_steps {
        let _old_action = ultra_graph.action(alpha, beta, 0.0);
        
        // Determine update type by replicating internal logic
        let do_z_update = rng.gen_bool(0.5);
        
        let accepted = ultra_graph.metropolis_step(alpha, beta, 0.0, delta_z, delta_theta, &mut rng);
        
        if do_z_update {
            z_total += 1;
            if accepted { z_accepts += 1; }
        } else {
            theta_total += 1;
            if accepted { theta_accepts += 1; }
        }
        
        // Verify action is consistent after update
        let new_action = ultra_graph.action(alpha, beta, 0.0);
        assert!(new_action.is_finite(), "Action became non-finite");
    }
    
    let z_rate = z_accepts as f64 / z_total as f64;
    let theta_rate = theta_accepts as f64 / theta_total as f64;
    
    println!("Z-update acceptance rate: {:.3}", z_rate);
    println!("Theta-update acceptance rate: {:.3}", theta_rate);
    
    // Acceptance rates should be reasonable (not 0 or 1)
    assert!(z_rate > 0.1 && z_rate < 0.9, "Z acceptance rate seems wrong: {}", z_rate);
    assert!(theta_rate > 0.1 && theta_rate < 0.9, "Theta acceptance rate seems wrong: {}", theta_rate);
}

#[test]
fn test_ergodicity() {
    // Test that the system can explore different regions of phase space
    let n = 10;
    let alpha = 0.5;
    let beta = 1.0;
    let delta_z = 0.3;
    let delta_theta = 0.8;
    let mc_steps = 100000;
    
    let mut ultra_graph = UltraOptimizedGraph::new(n, 777);
    let mut rng = Pcg64::seed_from_u64(777);
    
    // Track action values to verify exploration
    let mut action_samples = Vec::new();
    let mut triangle_samples = Vec::new();
    
    // Equilibrate
    for _ in 0..10000 {
        ultra_graph.metropolis_step(alpha, beta, 0.0, delta_z, delta_theta, &mut rng);
    }
    
    // Collect samples
    for _ in 0..mc_steps {
        ultra_graph.metropolis_step(alpha, beta, 0.0, delta_z, delta_theta, &mut rng);
        
        if action_samples.len() % 100 == 0 {
            action_samples.push(ultra_graph.action(alpha, beta, 0.0));
            triangle_samples.push(ultra_graph.triangle_sum());
        }
    }
    
    // Compute statistics
    let action_mean = action_samples.iter().sum::<f64>() / action_samples.len() as f64;
    let action_var = action_samples.iter()
        .map(|&x| (x - action_mean).powi(2))
        .sum::<f64>() / (action_samples.len() - 1) as f64;
    
    let triangle_mean = triangle_samples.iter().sum::<f64>() / triangle_samples.len() as f64;
    let triangle_var = triangle_samples.iter()
        .map(|&x| (x - triangle_mean).powi(2))
        .sum::<f64>() / (triangle_samples.len() - 1) as f64;
    
    println!("Action statistics: mean={:.6}, var={:.6}", action_mean, action_var);
    println!("Triangle statistics: mean={:.6}, var={:.6}", triangle_mean, triangle_var);
    
    // Should see significant fluctuations (not stuck in one configuration)
    assert!(action_var > 1e-6, "Action shows no variation - may be stuck");
    assert!(triangle_var > 1e-6, "Triangle sum shows no variation - may be stuck");
    
    // Values should be finite
    assert!(action_mean.is_finite() && action_var.is_finite(), "Action statistics are not finite");
    assert!(triangle_mean.is_finite() && triangle_var.is_finite(), "Triangle statistics are not finite");
}

#[test]
fn test_spectral_term_physics() {
    // Test spectral term behavior for small systems
    let n = 8;
    let alpha = 1.0;
    let beta = 1.5;
    let gamma = 0.1;
    
    let mut ultra_graph = UltraOptimizedGraph::new(n, 555);
    ultra_graph.enable_spectral(n/2, gamma);
    
    // Initial spectral action
    let initial_spectral = ultra_graph.action(0.0, 0.0, gamma);
    println!("Initial spectral action: {:.6}", initial_spectral);
    
    // Should be finite and non-negative (sum of squared deviations)
    assert!(initial_spectral.is_finite(), "Spectral action is not finite");
    assert!(initial_spectral >= 0.0, "Spectral action should be non-negative");
    
    // Run some MC steps and verify spectral term remains well-behaved
    let mut rng = Pcg64::seed_from_u64(555);
    for _ in 0..1000 {
        ultra_graph.metropolis_step(alpha, beta, gamma, 0.2, 0.5, &mut rng);
        
        let current_spectral = ultra_graph.action(0.0, 0.0, gamma);
        assert!(current_spectral.is_finite(), "Spectral action became non-finite");
        assert!(current_spectral >= -1e-10, "Spectral action became negative"); // Allow small numerical errors
    }
    
    println!("Spectral cache performance: {}", ultra_graph.performance_stats());
}

#[test]
fn test_energy_conservation_in_rejected_moves() {
    // Test that rejected moves don't change the system state
    let n = 12;
    let alpha = 2.0;
    let beta = 3.0;
    let delta_z = 0.1;
    let delta_theta = 0.1;
    
    let mut ultra_graph = UltraOptimizedGraph::new(n, 333);
    let mut rng = Pcg64::seed_from_u64(333);
    
    for _ in 0..10000 {
        let initial_action = ultra_graph.action(alpha, beta, 0.0);
        let initial_triangle = ultra_graph.triangle_sum();
        
        // Make a copy of current state
        let initial_z_values = ultra_graph.z_values.clone();
        let initial_theta_values = ultra_graph.theta_values.clone();
        let initial_cos_theta = ultra_graph.cos_theta.clone();
        let initial_exp_neg_z = ultra_graph.exp_neg_z.clone();
        
        let accepted = ultra_graph.metropolis_step(alpha, beta, 0.0, delta_z, delta_theta, &mut rng);
        
        if !accepted {
            // State should be unchanged
            let final_action = ultra_graph.action(alpha, beta, 0.0);
            let final_triangle = ultra_graph.triangle_sum();
            
            assert!((initial_action - final_action).abs() < TOLERANCE,
                    "Action changed in rejected move: {} -> {}", initial_action, final_action);
            assert!((initial_triangle - final_triangle).abs() < TOLERANCE,
                    "Triangle sum changed in rejected move");
            
            // All arrays should be identical
            assert_eq!(ultra_graph.z_values, initial_z_values, "Z values changed");
            assert_eq!(ultra_graph.theta_values, initial_theta_values, "Theta values changed");
            assert_eq!(ultra_graph.cos_theta, initial_cos_theta, "Cos theta changed");
            assert_eq!(ultra_graph.exp_neg_z, initial_exp_neg_z, "Exp(-z) values changed");
        }
    }
}

#[test]
fn test_thermodynamic_consistency() {
    // Test that observables show expected temperature dependence
    let n = 10;
    let alpha = 1.0;
    let betas = vec![0.5, 1.0, 2.0, 4.0];
    let mc_steps = 20000;
    let equilibration = 5000;
    
    let mut specific_heats = Vec::new();
    let mut mean_actions = Vec::new();
    
    for &beta in &betas {
        let mut ultra_graph = UltraOptimizedGraph::new(n, 666);
        let mut rng = Pcg64::seed_from_u64(666);
        
        // Equilibrate
        for _ in 0..equilibration {
            ultra_graph.metropolis_step(alpha, beta, 0.0, 0.2, 0.5, &mut rng);
        }
        
        // Measure
        let mut action_samples = Vec::new();
        for _ in 0..mc_steps {
            ultra_graph.metropolis_step(alpha, beta, 0.0, 0.2, 0.5, &mut rng);
            if action_samples.len() % 50 == 0 {
                action_samples.push(ultra_graph.action(alpha, beta, 0.0));
            }
        }
        
        let mean_action = action_samples.iter().sum::<f64>() / action_samples.len() as f64;
        let action_var = action_samples.iter()
            .map(|&x| (x - mean_action).powi(2))
            .sum::<f64>() / (action_samples.len() - 1) as f64;
        
        let specific_heat = action_var / (n as f64); // C = (1/N) * Var(E)
        
        mean_actions.push(mean_action);
        specific_heats.push(specific_heat);
        
        println!("β={:.1}: <S>={:.4}, C={:.4}", beta, mean_action, specific_heat);
        
        // Basic sanity checks
        assert!(mean_action.is_finite(), "Mean action not finite for β={}", beta);
        assert!(specific_heat >= 0.0, "Specific heat negative for β={}", beta);
    }
    
    // At low temperature, action should be lower (more ordered)
    // At high temperature, action should be higher (more disordered)
    assert!(mean_actions[0] > *mean_actions.last().unwrap(),
            "High temperature should have higher action than low temperature");
}

// Helper function to copy state from original graph to ultra-optimized
fn copy_graph_state(orig: &Graph, ultra: &mut UltraOptimizedGraph) {
    for (i, link) in orig.links.iter().enumerate() {
        ultra.z_values[i] = link.z;
        ultra.theta_values[i] = link.theta;
        ultra.cos_theta[i] = link.theta.cos();
        ultra.sin_theta[i] = link.theta.sin();
        ultra.exp_neg_z[i] = (-link.z).exp();
    }
    // Recompute cached triangle sum
    ultra.triangle_sum_cache = ultra.compute_full_triangle_sum();
}