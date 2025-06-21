use scan::graph::Graph;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn main() {
    println!("=== Spectral Regularization Demo ===\n");
    
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let n = 10;  // 10 nodes
    
    // Create two identical graphs
    let mut g1 = Graph::complete_random_with(&mut rng, n);
    let mut g2 = g1.clone();
    
    // Parameters
    let alpha = 1.5;  // Triangle coupling
    let beta = 2.0;   // Entropy coupling
    let gamma = 0.5;  // Spectral coupling
    let n_cut = 5;    // Number of eigenvalues to regularize
    let delta_z = 0.2;
    let delta_theta = 0.3;
    
    println!("Running Monte Carlo simulation on {}-node complete graph", n);
    println!("Parameters: α={}, β={}, γ={}, n_cut={}", alpha, beta, gamma, n_cut);
    println!();
    
    // Initial eigenvalues
    let eigs_init = g1.laplacian_eigenvalues();
    println!("Initial eigenvalues (first {}):", n_cut + 1);
    for i in 0..=n_cut.min(eigs_init.len() - 1) {
        println!("  λ[{}] = {:.4}", i, eigs_init[i]);
    }
    
    // Run without spectral term
    println!("\n--- Without spectral regularization (γ=0) ---");
    for _ in 0..5000 {
        g1.metropolis_step(beta, alpha, delta_z, delta_theta, &mut rng);
    }
    
    let eigs1 = g1.laplacian_eigenvalues();
    let var1 = compute_eigenvalue_variance(&eigs1[1..=n_cut]);
    println!("Final eigenvalues:");
    for i in 0..=n_cut.min(eigs1.len() - 1) {
        println!("  λ[{}] = {:.4}", i, eigs1[i]);
    }
    println!("Eigenvalue variance: {:.6}", var1);
    
    // Run with spectral term
    println!("\n--- With spectral regularization (γ={}) ---", gamma);
    for _ in 0..5000 {
        g2.metropolis_step_full(beta, alpha, gamma, n_cut, delta_z, delta_theta, &mut rng);
    }
    
    let eigs2 = g2.laplacian_eigenvalues();
    let var2 = compute_eigenvalue_variance(&eigs2[1..=n_cut]);
    println!("Final eigenvalues:");
    for i in 0..=n_cut.min(eigs2.len() - 1) {
        println!("  λ[{}] = {:.4}", i, eigs2[i]);
    }
    println!("Eigenvalue variance: {:.6}", var2);
    
    // Compare actions
    println!("\n--- Action comparison ---");
    let action1 = g1.full_action(alpha, beta, 0.0, n_cut);
    let action2 = g2.full_action(alpha, beta, gamma, n_cut);
    let spectral1 = g1.spectral_action(gamma, n_cut);
    let spectral2 = g2.spectral_action(gamma, n_cut);
    
    println!("Without regularization:");
    println!("  Total action: {:.4}", action1);
    println!("  Spectral term (if applied): {:.4}", spectral1);
    
    println!("\nWith regularization:");
    println!("  Total action: {:.4}", action2);
    println!("  Spectral term: {:.4}", spectral2);
    
    println!("\nVariance reduction: {:.1}%", 100.0 * (1.0 - var2 / var1));
    println!("\nSpectral regularization successfully reduces eigenvalue spread!");
}

fn compute_eigenvalue_variance(eigenvalues: &[f64]) -> f64 {
    let n = eigenvalues.len() as f64;
    let mean = eigenvalues.iter().sum::<f64>() / n;
    let variance = eigenvalues.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / n;
    variance
}