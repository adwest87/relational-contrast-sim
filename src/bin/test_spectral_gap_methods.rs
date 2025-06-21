// Test the new spectral gap and effective dimension methods
use scan::graph::Graph;
use rand_pcg::Pcg64;
use rand::SeedableRng;

fn main() {
    println!("=== Testing New Spectral Gap Methods ===\n");
    
    // Test with small graphs first
    let mut rng = Pcg64::seed_from_u64(12345);
    
    for &n in &[3, 5, 10] {
        println!("=== N = {} ===", n);
        let graph = Graph::complete_random_with(&mut rng, n);
        
        // Test spectral gap calculation
        match graph.spectral_gap() {
            Ok(gap) => {
                println!("Spectral gap: {:.6}", gap);
                
                // Test effective dimension
                match graph.effective_dimension() {
                    Ok(d_eff) => {
                        println!("Effective dimension: {:.2}", d_eff);
                    }
                    Err(e) => {
                        println!("Effective dimension error: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("Spectral gap error: {}", e);
            }
        }
        
        // Print some eigenvalues for inspection
        let eigenvalues = graph.laplacian_eigenvalues();
        println!("First 5 eigenvalues: {:?}", 
                 eigenvalues.iter().take(5).map(|&x| format!("{:.6}", x)).collect::<Vec<_>>());
        println!();
    }
    
    // Test uniform complete graph (should match theory)
    println!("=== Uniform Weight Test (Verification) ===");
    let n = 10;
    let mut uniform_graph = Graph::complete_random_with(&mut rng, n);
    
    // Set all weights to the same value
    let uniform_w = 0.5;
    for link in &mut uniform_graph.links {
        link.set_w(uniform_w);
    }
    
    match uniform_graph.spectral_gap() {
        Ok(gap) => {
            let expected_gap = n as f64 * uniform_w;
            println!("Uniform N={}, w={}: gap = {:.6}, expected = {:.6}", 
                     n, uniform_w, gap, expected_gap);
            println!("Matches theory: {}", (gap - expected_gap).abs() < 1e-10);
            
            match uniform_graph.effective_dimension() {
                Ok(d_eff) => {
                    println!("Effective dimension: {:.2}", d_eff);
                }
                Err(e) => {
                    println!("Effective dimension error: {}", e);
                }
            }
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
    
    // Test 4D emergence condition
    println!("\n=== 4D Emergence Test ===");
    let n = 20;
    let mut graph_4d = Graph::complete_random_with(&mut rng, n);
    
    // Set weight for d_eff ≈ 4 according to theory: w = N^(-3/2)
    let w_for_4d = (n as f64).powf(-1.5);
    for link in &mut graph_4d.links {
        link.set_w(w_for_4d);
    }
    
    match graph_4d.effective_dimension() {
        Ok(d_eff) => {
            println!("N={}, w={:.6}: d_eff = {:.3}", n, w_for_4d, d_eff);
            println!("4D emergence: {}", if (d_eff - 4.0).abs() < 0.1 { "YES ✓" } else { "NO" });
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
    
    println!("\n=== Key Physics Observables Added ===");
    println!("✓ spectral_gap() - The fundamental observable for emergent spacetime");
    println!("✓ effective_dimension() - Target: d_eff ≈ 4 for realistic spacetime");
    println!("✓ These replace magnetic observables (magnetization, susceptibility)");
    println!("✓ Ready for correct Relational Contrast Framework physics");
}