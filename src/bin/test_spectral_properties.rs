// Test spectral properties of weighted complete graphs
// Key question: Does the spectral gap lead to 4D spacetime?

use nalgebra::{DMatrix, SymmetricEigen};
use std::fs::File;
use std::io::Write;

/// Create weighted Laplacian matrix for complete graph
/// L_ij = δ_ij * Σ_k w_ik - w_ij
fn weighted_laplacian(n: usize, weight: f64) -> DMatrix<f64> {
    let mut laplacian = DMatrix::zeros(n, n);
    
    // For complete graph with uniform weights
    for i in 0..n {
        for j in 0..n {
            if i == j {
                // Diagonal: sum of all weights connected to node i
                laplacian[(i, j)] = (n - 1) as f64 * weight;
            } else {
                // Off-diagonal: -w_ij
                laplacian[(i, j)] = -weight;
            }
        }
    }
    
    laplacian
}

/// Calculate spectral properties
fn analyze_spectrum(n: usize, weight: f64) -> (Vec<f64>, f64, f64) {
    let laplacian = weighted_laplacian(n, weight);
    
    // Compute eigenvalues
    let eigen = SymmetricEigen::new(laplacian);
    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    // Spectral gap
    let spectral_gap = if eigenvalues.len() >= 2 {
        eigenvalues[1] - eigenvalues[0]
    } else {
        0.0
    };
    
    // Effective dimension (if spectral gap > 0)
    let d_eff = if spectral_gap > 1e-10 {
        -2.0 * (n as f64).ln() / spectral_gap.ln()
    } else {
        f64::INFINITY
    };
    
    (eigenvalues, spectral_gap, d_eff)
}

fn main() {
    println!("=== Spectral Properties of Complete Graphs ===\n");
    
    let test_sizes = vec![5, 10, 20, 40];
    let weight = 0.5;
    
    println!("Testing with uniform weight w = {}\n", weight);
    
    // Store results for plotting
    let mut results = Vec::new();
    
    for &n in &test_sizes {
        println!("N = {} nodes:", n);
        
        let (eigenvalues, gap, d_eff) = analyze_spectrum(n, weight);
        
        // Print first few eigenvalues
        println!("  First 5 eigenvalues:");
        for (i, &lambda) in eigenvalues.iter().take(5).enumerate() {
            println!("    λ_{{{:}}} = {:.6}", i, lambda);
        }
        
        println!("  Spectral gap (λ_2 - λ_1) = {:.6}", gap);
        println!("  Effective dimension = {:.3}", d_eff);
        
        // Theoretical analysis for complete graph
        let theoretical_gap = n as f64 * weight;  // For complete graph with uniform weights
        println!("  Theoretical gap = {:.6}", theoretical_gap);
        println!("  Ratio actual/theoretical = {:.6}", gap / theoretical_gap);
        
        results.push((n, gap, d_eff));
        println!();
    }
    
    // Analysis
    println!("\n=== Analysis ===");
    println!("\nFor a complete graph with uniform weights:");
    println!("- All eigenvalues except λ_0 = 0 are degenerate");
    println!("- λ_1 = λ_2 = ... = λ_{{N-1}} = N * w");
    println!("- Spectral gap = N * w");
    println!("- Effective dimension d_eff = -2 ln(N) / ln(N*w)");
    
    println!("\nDoes this give 4D spacetime?");
    println!("For d_eff = 4, we need: -2 ln(N) / ln(gap) = 4");
    println!("This gives: gap = N^(-1/2)");
    println!("For uniform weights: w = N^(-3/2)");
    
    println!("\nTesting the 4D condition:");
    for &n in &test_sizes {
        let w_for_4d = (n as f64).powf(-1.5);
        let (_, gap_4d, d_eff_4d) = analyze_spectrum(n, w_for_4d);
        println!("  N = {}: w = {:.6} gives d_eff = {:.3}", n, w_for_4d, d_eff_4d);
    }
    
    // Write results to file
    let mut file = File::create("spectral_results.csv").unwrap();
    writeln!(file, "N,weight,spectral_gap,effective_dimension").unwrap();
    
    // Test various weights
    println!("\n\nDetailed weight scan for N = 20:");
    let n = 20;
    for i in 1..=50 {
        let w = 0.01 * i as f64;
        let (_, gap, d_eff) = analyze_spectrum(n, w);
        writeln!(file, "{},{},{},{}", n, w, gap, d_eff).unwrap();
        
        if (d_eff - 4.0).abs() < 0.1 {
            println!("  w = {:.3} gives d_eff = {:.3} (close to 4D!)", w, d_eff);
        }
    }
    
    println!("\nResults written to spectral_results.csv");
    
    // Final insights
    println!("\n=== Key Insights ===");
    println!("1. Complete graph with uniform weights has highly degenerate spectrum");
    println!("2. Effective dimension depends on weight strength");
    println!("3. For 4D emergence, weights must scale as w ~ N^(-3/2)");
    println!("4. This suggests weights must weaken as system size grows");
    println!("5. Non-uniform weights needed for realistic spacetime structure");
    
    println!("\n⚠️  CRITICAL: Uniform complete graph is too symmetric!");
    println!("Real spacetime emergence likely requires:");
    println!("- Non-uniform weight distribution");
    println!("- Weights that break the complete graph symmetry");
    println!("- Competition between entropy and geometric terms");
}

#[test]
fn test_laplacian_properties() {
    // Test that Laplacian has correct properties
    let n = 5;
    let w = 0.5;
    let lap = weighted_laplacian(n, w);
    
    // Row sums should be zero
    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| lap[(i, j)]).sum();
        assert!(row_sum.abs() < 1e-10, "Row {} sum = {}", i, row_sum);
    }
    
    // Should be symmetric
    for i in 0..n {
        for j in 0..n {
            assert!((lap[(i, j)] - lap[(j, i)]).abs() < 1e-10);
        }
    }
}

#[test] 
fn test_spectral_gap_formula() {
    // For complete graph with uniform weights
    // Theoretical gap = N * w
    let n = 10;
    let w = 0.3;
    let (_, gap, _) = analyze_spectrum(n, w);
    let theoretical = n as f64 * w;
    assert!((gap - theoretical).abs() < 1e-10);
}