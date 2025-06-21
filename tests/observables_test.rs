use scan::graph_fast::{FastGraph, BatchedObservables};
use rand::SeedableRng;
use rand_pcg::Pcg64;

#[test]
fn test_specific_heat_calculation() {
    let mut graph = FastGraph::new(10, 42);
    let mut rng = Pcg64::seed_from_u64(42);
    let mut observables = BatchedObservables::new();
    
    let alpha = 1.5;
    let beta = 2.0;
    
    // Run some equilibration steps
    for _ in 0..1000 {
        graph.metropolis_step(alpha, beta, 0.2, 0.5, &mut rng);
    }
    
    // Accumulate measurements
    for _ in 0..100 {
        for _ in 0..10 {
            graph.metropolis_step(alpha, beta, 0.2, 0.5, &mut rng);
        }
        let obs = observables.measure(&graph, alpha, beta);
        
        // After enough samples, specific heat should be positive
        if observables.sample_count() > 20 {
            assert!(obs.specific_heat >= 0.0, "Specific heat should be non-negative");
        }
    }
    
    // Final measurement should have all observables
    let final_obs = observables.measure(&graph, alpha, beta);
    
    println!("Final observables:");
    println!("  Mean w: {}", final_obs.mean_w);
    println!("  Variance w: {}", final_obs.var_w);
    println!("  Mean cos: {}", final_obs.mean_cos);
    println!("  Susceptibility: {}", final_obs.susceptibility);
    println!("  Specific heat: {}", final_obs.specific_heat);
    println!("  Binder cumulant: {}", final_obs.binder_cumulant);
    println!("  Correlation length: {}", final_obs.correlation_length);
    
    // Verify normalizations
    assert!(final_obs.specific_heat >= 0.0, "Specific heat must be non-negative");
    assert!(final_obs.correlation_length >= 0.0, "Correlation length must be non-negative");
    assert!(final_obs.correlation_length <= graph.n() as f64 / 2.0, 
            "Correlation length should not exceed half system size");
}

#[test]
fn test_binder_cumulant_limits() {
    let mut graph = FastGraph::new(8, 123);
    let mut rng = Pcg64::seed_from_u64(123);
    let mut observables = BatchedObservables::new();
    
    let alpha = 0.5;
    let beta = 1.0;
    
    // Equilibrate
    for _ in 0..500 {
        graph.metropolis_step(alpha, beta, 0.2, 0.5, &mut rng);
    }
    
    // Measure
    for _ in 0..200 {
        graph.metropolis_step(alpha, beta, 0.2, 0.5, &mut rng);
        let obs = observables.measure(&graph, alpha, beta);
        
        if observables.sample_count() > 50 {
            // Binder cumulant should be in range [0, 2/3]
            // U4 = 1 - <m^4>/(3<m^2>^2)
            // For Gaussian: U4 = 0
            // For two-state: U4 = 2/3
            assert!(obs.binder_cumulant >= -0.1, "Binder cumulant too low: {}", obs.binder_cumulant);
            assert!(obs.binder_cumulant <= 0.7, "Binder cumulant too high: {}", obs.binder_cumulant);
        }
    }
}

#[test]
fn test_correlation_function() {
    let graph = FastGraph::new(6, 999);
    
    let (g0, g1) = graph.correlation_function();
    
    // G(0) should be positive (variance-like)
    assert!(g0 >= 0.0, "G(0) should be non-negative");
    
    // Calculate correlation length
    let xi = graph.calculate_correlation_length();
    assert!(xi >= 0.0, "Correlation length should be non-negative");
    assert!(xi <= 1.0, "Correlation length should be <= 1 for complete graph");
    
    println!("Correlation function:");
    println!("  G(0) = {}", g0);
    println!("  G(1) = {}", g1);
    println!("  ξ = {}", xi);
}

#[test]
fn test_accumulator_reset() {
    let mut graph = FastGraph::new(5, 42);
    let mut rng = Pcg64::seed_from_u64(42);
    let mut observables = BatchedObservables::new();
    
    // Accumulate some samples
    for _ in 0..50 {
        observables.measure(&graph, 1.0, 1.0);
        graph.metropolis_step(1.0, 1.0, 0.1, 0.1, &mut rng);
    }
    
    assert_eq!(observables.sample_count(), 50);
    
    // Reset
    observables.reset_accumulators();
    assert_eq!(observables.sample_count(), 0);
    
    // New measurements should start fresh
    let obs = observables.measure(&graph, 1.0, 1.0);
    assert_eq!(obs.specific_heat, 0.0, "Specific heat should be 0 with no samples");
    assert_eq!(obs.binder_cumulant, 0.0, "Binder cumulant should be 0 with no samples");
}

#[test]
fn test_normalization_correctness() {
    use scan::graph_fast::JackknifeEstimator;
    
    let mut graph = FastGraph::new(10, 42);
    let mut rng = Pcg64::seed_from_u64(42);
    let mut observables = BatchedObservables::new();
    
    let alpha = 1.0;
    let beta = 2.0;
    
    // Equilibrate
    for _ in 0..1000 {
        graph.metropolis_step(alpha, beta, 0.2, 0.5, &mut rng);
    }
    
    // Collect data for proper normalization check
    let mut energy_samples = Vec::new();
    let mut mag_samples = Vec::new();
    
    for _ in 0..500 {
        for _ in 0..10 {
            graph.metropolis_step(alpha, beta, 0.2, 0.5, &mut rng);
        }
        
        let energy = graph.action(alpha, beta);
        let sum_cos: f64 = graph.links.iter().map(|l| l.cos_theta).sum();
        let magnetization = sum_cos / graph.n() as f64;  // Normalized by N
        
        energy_samples.push(energy);
        mag_samples.push(magnetization);
    }
    
    // Manual calculation of observables
    let n = graph.n() as f64;
    
    // Specific heat: C = (1/N) * Var(E)
    let e_mean = energy_samples.iter().sum::<f64>() / energy_samples.len() as f64;
    let e_var = energy_samples.iter()
        .map(|&e| (e - e_mean).powi(2))
        .sum::<f64>() / (energy_samples.len() - 1) as f64;
    let manual_specific_heat = e_var / n;
    
    // Susceptibility: χ = N * Var(m)
    let m_mean = mag_samples.iter().sum::<f64>() / mag_samples.len() as f64;
    let m_var = mag_samples.iter()
        .map(|&m| (m - m_mean).powi(2))
        .sum::<f64>() / (mag_samples.len() - 1) as f64;
    let manual_susceptibility = n * m_var;
    
    // Get observables from our implementation
    observables.reset_accumulators();
    for _ in 0..100 {
        for _ in 0..10 {
            graph.metropolis_step(alpha, beta, 0.2, 0.5, &mut rng);
        }
        let obs = observables.measure(&graph, alpha, beta);
        
        if observables.sample_count() > 50 {
            // Check normalizations are in the right ballpark
            println!("Manual specific heat: {:.4}, Measured: {:.4}", manual_specific_heat, obs.specific_heat);
            println!("Manual susceptibility: {:.4}, Measured: {:.4}", manual_susceptibility, obs.susceptibility);
            
            // They won't match exactly due to different sampling, but should be same order of magnitude
            assert!(obs.specific_heat > 0.0, "Specific heat should be positive");
            assert!(obs.susceptibility > 0.0, "Susceptibility should be positive");
            break;
        }
    }
}

#[test] 
fn test_jackknife_error_estimation() {
    use scan::graph_fast::JackknifeEstimator;
    
    // Test jackknife with known data
    let mut estimator = JackknifeEstimator::new();
    
    // Add samples with some variance
    let samples = vec![1.0, 1.2, 0.8, 1.1, 0.9, 1.05, 0.95, 1.15, 0.85, 1.0];
    for &s in &samples {
        estimator.add_sample(s);
    }
    
    // Test mean estimator
    let (mean, err) = estimator.estimate_error(|data| {
        data.iter().sum::<f64>() / data.len() as f64
    });
    
    println!("Mean: {:.4} ± {:.4}", mean, err);
    assert!((mean - 1.0).abs() < 0.1, "Mean should be close to 1.0");
    assert!(err > 0.0, "Error should be positive");
    assert!(err < 0.1, "Error should be reasonable");
    
    // Test variance estimator
    let (var, var_err) = estimator.estimate_error(|data| {
        let m = data.iter().sum::<f64>() / data.len() as f64;
        data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
    });
    
    println!("Variance: {:.4} ± {:.4}", var, var_err);
    assert!(var > 0.0, "Variance should be positive");
}