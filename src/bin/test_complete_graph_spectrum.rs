// Test spectral properties of weighted complete graphs
// Verifies mathematical derivations and explores non-uniform weights

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rand::Rng;
use rand_pcg::Pcg64;
use rand::SeedableRng;
use std::f64::consts::PI;

/// Weight function type
type WeightFn = Box<dyn Fn(usize, usize) -> f64>;

/// Create weighted Laplacian for complete graph
fn complete_graph_laplacian(n: usize, weight_fn: &WeightFn) -> DMatrix<f64> {
    let mut laplacian = DMatrix::zeros(n, n);
    
    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            if i != j {
                let w = weight_fn(i, j);
                laplacian[(i, j)] = -w;
                row_sum += w;
            }
        }
        laplacian[(i, i)] = row_sum;
    }
    
    laplacian
}

/// Analyze eigenvalue spectrum
fn analyze_spectrum(laplacian: &DMatrix<f64>) -> (Vec<f64>, f64, f64) {
    let eigen = SymmetricEigen::new(laplacian.clone());
    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let gap = if eigenvalues.len() >= 2 {
        eigenvalues[1] - eigenvalues[0]
    } else {
        0.0
    };
    
    let d_eff = if gap > 1e-10 {
        -2.0 * (laplacian.nrows() as f64).ln() / gap.ln()
    } else {
        f64::INFINITY
    };
    
    (eigenvalues, gap, d_eff)
}

/// Test uniform weights - should give degenerate spectrum
fn test_uniform_weights() {
    println!("=== Test 1: Uniform Weights ===");
    println!("Theory: λ₁ = 0, λ₂ = ... = λ_N = Nw\n");
    
    let w = 0.5;
    let weight_fn: WeightFn = Box::new(move |_, _| w);
    
    for &n in &[5, 10, 20] {
        let laplacian = complete_graph_laplacian(n, &weight_fn);
        let (eigenvalues, gap, d_eff) = analyze_spectrum(&laplacian);
        
        println!("N = {}:", n);
        println!("  First 5 eigenvalues: {:?}", 
                 eigenvalues.iter().take(5).map(|&x| format!("{:.4}", x)).collect::<Vec<_>>());
        
        // Check degeneracy
        let expected = n as f64 * w;
        let degenerate = eigenvalues[1..].iter()
            .all(|&lambda| (lambda - expected).abs() < 1e-10);
        
        println!("  λ₂ = {:.4}, Expected = {:.4}", eigenvalues[1], expected);
        println!("  Degenerate: {}", if degenerate { "YES ✓" } else { "NO ✗" });
        println!("  Spectral gap: {:.4}", gap);
        println!("  Effective dimension: {:.2}", d_eff);
        println!();
    }
}

/// Test random weights - should break degeneracy
fn test_random_weights() {
    println!("\n=== Test 2: Random Weights ===");
    println!("Breaking symmetry with random weights\n");
    
    let mut rng = Pcg64::seed_from_u64(42);
    
    for &n in &[5, 10, 20] {
        // Create random weight matrix
        let mut weights = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in i+1..n {
                let w = rng.gen_range(0.1..1.0);
                weights[(i, j)] = w;
                weights[(j, i)] = w;
            }
        }
        
        let weight_fn: WeightFn = Box::new(move |i, j| weights[(i, j)]);
        let laplacian = complete_graph_laplacian(n, &weight_fn);
        let (eigenvalues, gap, d_eff) = analyze_spectrum(&laplacian);
        
        println!("N = {}:", n);
        println!("  First 5 eigenvalues: {:?}", 
                 eigenvalues.iter().take(5).map(|&x| format!("{:.4}", x)).collect::<Vec<_>>());
        
        // Check if degeneracy is lifted
        let unique_count = eigenvalues[1..].windows(2)
            .filter(|w| (w[1] - w[0]).abs() > 1e-6)
            .count() + 1;
        
        println!("  Unique eigenvalues (excluding λ₁): {}/{}", unique_count, n-1);
        println!("  Spectral gap: {:.4}", gap);
        println!("  Effective dimension: {:.2}", d_eff);
        println!("  Eigenvalue spread: [{:.4}, {:.4}]", eigenvalues[1], eigenvalues[n-1]);
        println!();
    }
}

/// Test geometric weights - nodes on a circle
fn test_geometric_weights() {
    println!("\n=== Test 3: Geometric Weights ===");
    println!("Nodes embedded on unit circle\n");
    
    let n = 10;
    
    for &sigma in &[0.5, 1.0, 2.0] {
        let weight_fn: WeightFn = Box::new(move |i, j| {
            let theta_i = 2.0 * PI * i as f64 / n as f64;
            let theta_j = 2.0 * PI * j as f64 / n as f64;
            let d_ij = 2.0 * ((theta_i - theta_j).abs() / 2.0).sin();
            (-d_ij * d_ij / (sigma * sigma)).exp()
        });
        
        let laplacian = complete_graph_laplacian(n, &weight_fn);
        let (eigenvalues, gap, d_eff) = analyze_spectrum(&laplacian);
        
        println!("σ = {}:", sigma);
        println!("  First 5 eigenvalues: {:?}", 
                 eigenvalues.iter().take(5).map(|&x| format!("{:.4}", x)).collect::<Vec<_>>());
        
        if eigenvalues.len() >= 3 && eigenvalues[2] > 1e-10 {
            let ratio = eigenvalues[1] / eigenvalues[2];
            println!("  λ₂/λ₃ = {:.4} (scale separation)", ratio);
        }
        
        println!("  Spectral gap: {:.4}", gap);
        println!("  Effective dimension: {:.2}", d_eff);
        println!();
    }
}

/// Test power-law weights
fn test_power_law_weights() {
    println!("\n=== Test 4: Power-Law Weights ===");
    println!("w_ij = 1 / (1 + |i-j|)^α\n");
    
    let n = 20;
    
    for &alpha in &[1.0, 1.5, 2.0] {
        let weight_fn: WeightFn = Box::new(move |i, j| {
            let dist = (i as f64 - j as f64).abs();
            1.0 / (1.0 + dist).powf(alpha)
        });
        
        let laplacian = complete_graph_laplacian(n, &weight_fn);
        let (eigenvalues, gap, d_eff) = analyze_spectrum(&laplacian);
        
        println!("α = {}:", alpha);
        println!("  First 5 eigenvalues: {:?}", 
                 eigenvalues.iter().take(5).map(|&x| format!("{:.4}", x)).collect::<Vec<_>>());
        println!("  Last 3 eigenvalues: {:?}", 
                 eigenvalues.iter().rev().take(3).map(|&x| format!("{:.4}", x)).collect::<Vec<_>>());
        println!("  Spectral gap: {:.4}", gap);
        println!("  Effective dimension: {:.2}", d_eff);
        
        // Check for power-law scaling in spectrum
        let log_ratios: Vec<f64> = eigenvalues[1..5].windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        let avg_log_ratio = log_ratios.iter().sum::<f64>() / log_ratios.len() as f64;
        println!("  Average eigenvalue growth: {:.4}", avg_log_ratio.exp());
        println!();
    }
}

/// Search for weights that give d_eff ≈ 4
fn search_for_4d() {
    println!("\n=== Test 5: Search for 4D Emergence ===");
    println!("Finding weight distributions that give d_eff ≈ 4\n");
    
    let n = 20;
    let target_d = 4.0;
    
    // From theory: for uniform weights, need w = N^(-3/2)
    let theoretical_w = (n as f64).powf(-1.5);
    println!("Theoretical uniform weight for d=4: w = {:.6}", theoretical_w);
    
    // Test theoretical prediction
    let weight_fn: WeightFn = Box::new(move |_, _| theoretical_w);
    let laplacian = complete_graph_laplacian(n, &weight_fn);
    let (_, gap, d_eff) = analyze_spectrum(&laplacian);
    
    println!("  Uniform w = {:.6}: gap = {:.6}, d_eff = {:.2}", theoretical_w, gap, d_eff);
    
    // Try mixed weights
    println!("\nMixed weight strategies:");
    
    // Strategy 1: Two-scale weights
    let w_near = 0.1;
    let w_far = 0.001;
    let cutoff = n / 4;
    
    let weight_fn: WeightFn = Box::new(move |i, j| {
        if (i as i32 - j as i32).abs() < cutoff as i32 {
            w_near
        } else {
            w_far
        }
    });
    
    let laplacian = complete_graph_laplacian(n, &weight_fn);
    let (eigenvalues, gap, d_eff) = analyze_spectrum(&laplacian);
    
    println!("  Two-scale (near={}, far={}): d_eff = {:.2}", w_near, w_far, d_eff);
    
    // Check spectral structure
    let gaps: Vec<f64> = eigenvalues.windows(2)
        .map(|w| w[1] - w[0])
        .take(5)
        .collect();
    println!("  First 5 gaps: {:?}", gaps.iter().map(|&x| format!("{:.4}", x)).collect::<Vec<_>>());
}

fn main() {
    println!("=== Complete Graph Spectral Analysis ===\n");
    
    test_uniform_weights();
    test_random_weights();
    test_geometric_weights();
    test_power_law_weights();
    search_for_4d();
    
    println!("\n=== Key Insights ===");
    println!("1. Uniform complete graphs have (N-1)-fold degenerate spectra");
    println!("2. Any weight variation breaks degeneracy");
    println!("3. Geometric/power-law weights create multi-scale structure");
    println!("4. Getting d_eff = 4 requires N^(-3/2) scaling for uniform weights");
    println!("5. Complete graphs may be too symmetric for realistic spacetime");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_uniform_spectrum_theory() {
        // Test that uniform weights give λ₂ = Nw
        let n = 10;
        let w = 0.3;
        let weight_fn: WeightFn = Box::new(move |_, _| w);
        
        let laplacian = complete_graph_laplacian(n, &weight_fn);
        let (eigenvalues, gap, _) = analyze_spectrum(&laplacian);
        
        assert!((eigenvalues[0]).abs() < 1e-10);
        assert!((eigenvalues[1] - n as f64 * w).abs() < 1e-10);
        assert!((gap - n as f64 * w).abs() < 1e-10);
        
        // Check all non-zero eigenvalues are equal
        for i in 2..n {
            assert!((eigenvalues[i] - eigenvalues[1]).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_laplacian_properties() {
        // Test that Laplacian rows sum to zero
        let n = 5;
        let weight_fn: WeightFn = Box::new(|i, j| (i + j + 1) as f64 * 0.1);
        let laplacian = complete_graph_laplacian(n, &weight_fn);
        
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| laplacian[(i, j)]).sum();
            assert!(row_sum.abs() < 1e-10);
        }
    }
}