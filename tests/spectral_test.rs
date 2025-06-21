use scan::graph::Graph;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[test]
fn test_laplacian_properties() {
    // Test that Laplacian has correct properties
    let mut rng = ChaCha20Rng::seed_from_u64(12345);
    let g = Graph::complete_random_with(&mut rng, 5);
    let laplacian = g.laplacian_matrix();
    let n = g.n();
    
    // Check symmetry
    for i in 0..n {
        for j in 0..n {
            assert!(
                (laplacian[(i, j)] - laplacian[(j, i)]).abs() < 1e-10,
                "Laplacian not symmetric at ({}, {})", i, j
            );
        }
    }
    
    // Check row sums are zero (within numerical tolerance)
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| laplacian[(i, j)]).sum();
        assert!(
            row_sum.abs() < 1e-10,
            "Row {} sum is {} (should be 0)", i, row_sum
        );
    }
    
    // Check diagonal entries are positive
    for i in 0..n {
        assert!(
            laplacian[(i, i)] > 0.0,
            "Diagonal entry {} is not positive", i
        );
    }
}

#[test]
fn test_laplacian_eigenvalues() {
    // Test eigenvalue properties
    let mut rng = ChaCha20Rng::seed_from_u64(12345);
    let g = Graph::complete_random_with(&mut rng, 6);
    let eigenvalues = g.laplacian_eigenvalues();
    
    // First eigenvalue should be zero (connected graph)
    assert!(
        eigenvalues[0].abs() < 1e-10,
        "First eigenvalue is {} (should be 0)", eigenvalues[0]
    );
    
    // All eigenvalues should be non-negative
    for (i, &lambda) in eigenvalues.iter().enumerate() {
        assert!(
            lambda >= -1e-10,
            "Eigenvalue {} is negative: {}", i, lambda
        );
    }
    
    // Eigenvalues should be sorted
    for i in 1..eigenvalues.len() {
        assert!(
            eigenvalues[i] >= eigenvalues[i-1],
            "Eigenvalues not sorted: {} > {}", eigenvalues[i-1], eigenvalues[i]
        );
    }
}

#[test]
fn test_spectral_action_zero_gamma() {
    // Test that spectral action is zero when gamma = 0
    let g = Graph::complete_random(5);
    let spectral = g.spectral_action(0.0, 3);
    assert_eq!(spectral, 0.0, "Spectral action should be 0 when gamma = 0");
}

#[test]
fn test_spectral_action_scaling() {
    // Test that spectral action scales linearly with gamma
    let g = Graph::complete_random(5);
    let gamma1 = 0.5;
    let gamma2 = 1.0;
    let n_cut = 3;
    
    let spectral1 = g.spectral_action(gamma1, n_cut);
    let spectral2 = g.spectral_action(gamma2, n_cut);
    
    assert!(
        (spectral2 - 2.0 * spectral1).abs() < 1e-10,
        "Spectral action does not scale linearly with gamma"
    );
}

#[test]
fn test_spectral_action_invariance() {
    // Test that spectral action is invariant under weight rescaling
    let mut g = Graph::complete_random(5);
    let gamma = 0.3;
    let n_cut = 4;
    
    let spectral_before = g.spectral_action(gamma, n_cut);
    
    // Rescale all weights by a constant factor
    let lambda = 2.5;
    g.rescale(lambda);
    
    // Laplacian eigenvalues scale with lambda
    let spectral_after = g.spectral_action(gamma, n_cut);
    
    // The rescaled spectral action should scale quadratically with lambda
    let expected = spectral_before * lambda * lambda;
    assert!(
        (spectral_after - expected).abs() / expected < 1e-10,
        "Spectral action scaling incorrect: expected {}, got {}", 
        expected, spectral_after
    );
}

#[test]
fn test_full_action_includes_spectral() {
    // Test that full action includes the spectral term
    let g = Graph::complete_random(5);
    let alpha = 1.5;
    let beta = 2.0;
    let gamma = 0.4;
    let n_cut = 3;
    
    let basic_action = g.action(alpha, beta);
    let spectral_action = g.spectral_action(gamma, n_cut);
    let full_action = g.full_action(alpha, beta, gamma, n_cut);
    
    assert!(
        (full_action - (basic_action + spectral_action)).abs() < 1e-10,
        "Full action does not equal sum of components"
    );
}

#[test]
fn test_metropolis_with_spectral() {
    // Test that Metropolis with spectral term runs without error
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let mut g = Graph::complete_random_with(&mut rng, 5);
    
    let alpha = 1.0;
    let beta = 2.0;
    let gamma = 0.5;  // Stronger spectral term
    let n_cut = 3;
    let delta_z = 0.3;  // Larger perturbations
    let delta_theta = 0.5;
    
    // Run some Metropolis steps
    let mut accepted = 0;
    for _ in 0..100 {
        let info = g.metropolis_step_full(
            beta, alpha, gamma, n_cut, delta_z, delta_theta, &mut rng
        );
        if info.accepted {
            accepted += 1;
        }
    }
    
    // Just check that we get some accepts and rejects
    assert!(
        accepted > 0,
        "No acceptances in 100 steps"
    );
    
    // Print for debugging (spectral term impact on acceptance)
    println!("Acceptance rate with spectral term: {}/100", accepted);
}

#[test]
fn test_spectral_regularization_effect() {
    // Test that spectral term affects eigenvalue distribution
    let mut rng = ChaCha20Rng::seed_from_u64(1234);
    let mut g1 = Graph::complete_random_with(&mut rng, 8);
    let mut g2 = g1.clone();
    
    let alpha = 1.0;
    let beta = 2.0;
    let delta_z = 0.1;
    let delta_theta = 0.2;
    
    // Run without spectral term
    for _ in 0..5000 {
        g1.metropolis_step(beta, alpha, delta_z, delta_theta, &mut rng);
    }
    
    // Run with spectral term
    let gamma = 1.0;  // Stronger regularization
    let n_cut = 5;
    for _ in 0..5000 {
        g2.metropolis_step_full(beta, alpha, gamma, n_cut, delta_z, delta_theta, &mut rng);
    }
    
    // Compare eigenvalue spreads
    let eigs1 = g1.laplacian_eigenvalues();
    let eigs2 = g2.laplacian_eigenvalues();
    
    // Compute variance of first n_cut eigenvalues (excluding zero eigenvalue)
    let var1: f64 = eigs1[1..=n_cut].iter()
        .map(|&x| x * x)
        .sum::<f64>() / n_cut as f64
        - (eigs1[1..=n_cut].iter().sum::<f64>() / n_cut as f64).powi(2);
        
    let var2: f64 = eigs2[1..=n_cut].iter()
        .map(|&x| x * x)
        .sum::<f64>() / n_cut as f64
        - (eigs2[1..=n_cut].iter().sum::<f64>() / n_cut as f64).powi(2);
    
    // With regularization, variance should be smaller
    assert!(
        var2 < var1,
        "Spectral regularization did not reduce eigenvalue variance: {} vs {}",
        var1, var2
    );
}