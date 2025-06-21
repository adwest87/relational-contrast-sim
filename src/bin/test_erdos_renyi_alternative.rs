// Quick test of Erdős-Rényi graphs as alternative to complete graphs
// Tests spectral properties and 4D emergence potential

use nalgebra::{DMatrix, SymmetricEigen};
use rand::Rng;
use rand_pcg::Pcg64;
use rand::SeedableRng;
use std::time::Instant;

/// Create Erdős-Rényi graph with connection probability p
fn create_erdos_renyi_graph(n: usize, p: f64, rng: &mut impl Rng) -> Vec<(usize, usize, f64)> {
    let mut edges = Vec::new();
    
    for i in 0..n {
        for j in (i+1)..n {
            if rng.gen::<f64>() < p {
                // Connected - assign random weight
                let weight = rng.gen_range(0.1..1.0);
                edges.push((i, j, weight));
            }
        }
    }
    
    edges
}

/// Create complete graph for comparison
fn create_complete_graph(n: usize, rng: &mut impl Rng) -> Vec<(usize, usize, f64)> {
    let mut edges = Vec::new();
    
    for i in 0..n {
        for j in (i+1)..n {
            let weight = rng.gen_range(0.1..1.0);
            edges.push((i, j, weight));
        }
    }
    
    edges
}

/// Create weighted Laplacian from edge list
fn create_laplacian(n: usize, edges: &[(usize, usize, f64)]) -> DMatrix<f64> {
    let mut laplacian = DMatrix::zeros(n, n);
    
    for &(i, j, weight) in edges {
        laplacian[(i, j)] = -weight;
        laplacian[(j, i)] = -weight;
        laplacian[(i, i)] += weight;
        laplacian[(j, j)] += weight;
    }
    
    laplacian
}

/// Analyze spectral properties
fn analyze_spectrum(laplacian: &DMatrix<f64>) -> (Vec<f64>, f64, f64, bool) {
    let eigen = SymmetricEigen::new(laplacian.clone());
    
    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let gap = if eigenvalues.len() >= 2 && eigenvalues[0].abs() < 1e-10 {
        eigenvalues[1] - eigenvalues[0]
    } else {
        0.0
    };
    
    let d_eff = if gap > 1e-10 {
        let n = laplacian.nrows() as f64;
        -2.0 * n.ln() / gap.ln()
    } else {
        f64::INFINITY
    };
    
    let connected = eigenvalues[0].abs() < 1e-10;
    
    (eigenvalues, gap, d_eff, connected)
}

/// Count unique eigenvalues (spectral richness)
fn count_unique_eigenvalues(eigenvalues: &[f64], tolerance: f64) -> usize {
    if eigenvalues.is_empty() {
        return 0;
    }
    
    let mut unique_count = 1;
    for i in 1..eigenvalues.len() {
        if (eigenvalues[i] - eigenvalues[i-1]).abs() > tolerance {
            unique_count += 1;
        }
    }
    unique_count
}

/// Calculate spectral gap ratio (measures eigenvalue spacing)
fn spectral_gap_ratio(eigenvalues: &[f64]) -> f64 {
    if eigenvalues.len() < 3 {
        return 1.0;
    }
    
    let gap1 = eigenvalues[2] - eigenvalues[1];
    let gap2 = eigenvalues[1] - eigenvalues[0];
    
    if gap2 > 1e-10 {
        gap1 / gap2
    } else {
        f64::INFINITY
    }
}

fn main() {
    println!("=== Erdős-Rényi vs Complete Graph Spectral Comparison ===\n");
    
    let mut rng = Pcg64::seed_from_u64(42);
    
    // Test different sizes
    for &n in &[20, 50, 100] {
        println!("=== N = {} ===", n);
        
        // Erdős-Rényi connection probability: p = 2*ln(N)/N
        let p_er = 2.0 * (n as f64).ln() / (n as f64);
        println!("Erdős-Rényi probability p = {:.4}", p_er);
        
        // Create graphs
        let er_edges = create_erdos_renyi_graph(n, p_er, &mut rng);
        let complete_edges = create_complete_graph(n, &mut rng);
        
        println!("Erdős-Rényi edges: {} (density: {:.3})", 
                 er_edges.len(), 
                 2.0 * er_edges.len() as f64 / (n * (n-1)) as f64);
        println!("Complete graph edges: {} (density: 1.000)", complete_edges.len());
        
        // Create Laplacians and analyze
        let start = Instant::now();
        let er_laplacian = create_laplacian(n, &er_edges);
        let er_construction_time = start.elapsed().as_secs_f64();
        
        let start = Instant::now();
        let complete_laplacian = create_laplacian(n, &complete_edges);
        let complete_construction_time = start.elapsed().as_secs_f64();
        
        // Analyze spectra
        let (er_eigenvalues, er_gap, er_d_eff, er_connected) = analyze_spectrum(&er_laplacian);
        let (complete_eigenvalues, complete_gap, complete_d_eff, complete_connected) = analyze_spectrum(&complete_laplacian);
        
        // Spectral richness analysis
        let er_unique = count_unique_eigenvalues(&er_eigenvalues, 1e-8);
        let complete_unique = count_unique_eigenvalues(&complete_eigenvalues, 1e-8);
        
        let er_gap_ratio = spectral_gap_ratio(&er_eigenvalues);
        let complete_gap_ratio = spectral_gap_ratio(&complete_eigenvalues);
        
        println!("\n--- Connectivity ---");
        println!("Erdős-Rényi connected: {}", if er_connected { "YES" } else { "NO" });
        println!("Complete connected: {}", if complete_connected { "YES" } else { "NO" });
        
        if er_connected && complete_connected {
            println!("\n--- Spectral Properties ---");
            println!("Erdős-Rényi:");
            println!("  Spectral gap: {:.6}", er_gap);
            println!("  Effective dimension: {:.3}", er_d_eff);
            println!("  Unique eigenvalues: {}/{} ({:.1}%)", 
                     er_unique, n, 100.0 * er_unique as f64 / n as f64);
            println!("  Gap ratio λ₃-λ₂/λ₂-λ₁: {:.3}", er_gap_ratio);
            
            println!("Complete:");
            println!("  Spectral gap: {:.6}", complete_gap);
            println!("  Effective dimension: {:.3}", complete_d_eff);
            println!("  Unique eigenvalues: {}/{} ({:.1}%)", 
                     complete_unique, n, 100.0 * complete_unique as f64 / n as f64);
            println!("  Gap ratio λ₃-λ₂/λ₂-λ₁: {:.3}", complete_gap_ratio);
            
            println!("\n--- 4D Emergence Assessment ---");
            let er_4d_distance = (er_d_eff - 4.0).abs();
            let complete_4d_distance = (complete_d_eff - 4.0).abs();
            
            println!("Erdős-Rényi |d_eff - 4|: {:.3}", er_4d_distance);
            println!("Complete |d_eff - 4|: {:.3}", complete_4d_distance);
            
            if er_4d_distance < complete_4d_distance {
                println!("🎯 Erdős-Rényi closer to 4D!");
            } else if complete_4d_distance < er_4d_distance {
                println!("🎯 Complete graph closer to 4D");
            } else {
                println!("🎯 Similar distance to 4D");
            }
            
            println!("\n--- Spectral Richness ---");
            if er_unique > complete_unique {
                println!("✅ Erdős-Rényi has richer spectrum ({} vs {} unique)", er_unique, complete_unique);
            } else if complete_unique > er_unique {
                println!("❌ Complete graph has more unique eigenvalues ({} vs {})", complete_unique, er_unique);
            } else {
                println!("⚖️  Similar spectral richness");
            }
            
            // Print first few eigenvalues for comparison
            println!("\nFirst 5 eigenvalues:");
            print!("Erdős-Rényi: ");
            for i in 0..5.min(er_eigenvalues.len()) {
                print!("{:.4} ", er_eigenvalues[i]);
            }
            println!();
            print!("Complete:     ");
            for i in 0..5.min(complete_eigenvalues.len()) {
                print!("{:.4} ", complete_eigenvalues[i]);
            }
            println!();
            
        } else {
            println!("⚠️  Disconnected graph(s) - skipping spectral analysis");
        }
        
        println!("\n--- Computational Cost ---");
        println!("Erdős-Rényi construction: {:.6}s", er_construction_time);
        println!("Complete construction: {:.6}s", complete_construction_time);
        println!("Speedup factor: {:.2}x", complete_construction_time / er_construction_time.max(1e-9));
        
        println!("\n{}", "=".repeat(60));
    }
    
    println!("\n=== Summary ===");
    println!("Test completed. See ALTERNATIVE_GRAPH_QUICK_TEST.md for analysis.");
}