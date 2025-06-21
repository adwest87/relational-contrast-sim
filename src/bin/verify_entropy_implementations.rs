// Verify and compare different entropy implementations
// Determine if they represent different physics or are related by constants

use scan::graph::Graph;
use scan::minimal_correct_physics::MinimalGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::fs::File;
use std::io::Write;

/// Calculate entropy using original formula: -z * exp(-z)
fn entropy_original(z: f64) -> f64 {
    -z * (-z).exp()
}

/// Calculate entropy using correct formula: w * ln(w) where w = exp(-z)
fn entropy_correct(z: f64) -> f64 {
    let w = (-z).exp();
    if w > 0.0 {
        w * w.ln()
    } else {
        0.0
    }
}

/// Alternative form: Can we express correct entropy in terms of z?
/// w * ln(w) = exp(-z) * ln(exp(-z)) = exp(-z) * (-z) = -z * exp(-z)
/// Wait... these ARE the same!
fn verify_mathematical_equivalence(z: f64) -> (f64, f64, f64) {
    let w = (-z).exp();
    
    // Method 1: w * ln(w)
    let method1 = if w > 0.0 { w * w.ln() } else { 0.0 };
    
    // Method 2: -z * exp(-z)
    let method2 = -z * (-z).exp();
    
    // Method 3: Expand ln(exp(-z)) = -z
    let method3 = w * (-z);  // This should equal method2
    
    (method1, method2, method3)
}

fn main() {
    println!("=== Entropy Implementation Verification ===\n");
    
    // Test 1: Mathematical equivalence
    println!("Test 1: Mathematical Equivalence");
    println!("w * ln(w) vs -z * exp(-z) where w = exp(-z)");
    println!("{:>10} {:>10} {:>15} {:>15} {:>15} {:>15}", 
             "z", "w", "w*ln(w)", "-z*exp(-z)", "Difference", "Status");
    println!("{}", "-".repeat(90));
    
    for &z in &[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0] {
        let (m1, m2, m3) = verify_mathematical_equivalence(z);
        let w = (-z).exp();
        let diff = (m1 - m2).abs();
        let status = if diff < 1e-10 { "SAME" } else { "DIFFERENT" };
        
        println!("{:10.6} {:10.6} {:15.10} {:15.10} {:15.2e} {:>15}", 
                 z, w, m1, m2, diff, status);
    }
    
    // Test 2: Compare implementations on small graphs
    println!("\n\nTest 2: Compare Graph Implementations");
    
    for &n in &[3, 4, 5] {
        println!("\nN = {} nodes:", n);
        
        // Create a graph with original implementation
        let mut rng = Pcg64::seed_from_u64(42);
        let graph = Graph::complete_random_with(&mut rng, n);
        
        // Calculate using original method
        let entropy_orig = graph.entropy_action();
        
        // Calculate using correct formula
        let entropy_corr: f64 = graph.links.iter().map(|link| {
            let w = link.w();
            if w > 0.0 { w * w.ln() } else { 0.0 }
        }).sum();
        
        println!("  Original (-z*w):     {:.10}", entropy_orig);
        println!("  Correct (w*ln(w)):   {:.10}", entropy_corr);
        println!("  Difference:          {:.2e}", (entropy_orig - entropy_corr).abs());
        println!("  Status:              {}", 
                 if (entropy_orig - entropy_corr).abs() < 1e-10 { "SAME" } else { "DIFFERENT" });
    }
    
    // Test 3: Detailed analysis for a single link
    println!("\n\nTest 3: Single Link Detailed Analysis");
    let z = 0.693147_f64;  // -ln(0.5)
    let w = (-z).exp();
    
    println!("For z = {:.6} (corresponding to w = {:.6}):", z, w);
    println!("\nMethod 1: w * ln(w)");
    println!("  = {:.6} * ln({:.6})", w, w);
    println!("  = {:.6} * {:.6}", w, w.ln());
    println!("  = {:.6}", w * w.ln());
    
    println!("\nMethod 2: -z * exp(-z)");
    println!("  = -{:.6} * exp(-{:.6})", z, z);
    println!("  = -{:.6} * {:.6}", z, (-z).exp());
    println!("  = {:.6}", -z * (-z).exp());
    
    println!("\nMethod 3: Using ln(exp(-z)) = -z");
    println!("  w * ln(w) = exp(-z) * ln(exp(-z))");
    println!("  = exp(-z) * (-z)");
    println!("  = {:.6} * (-{:.6})", w, z);
    println!("  = {:.6}", w * (-z));
    
    // Test 4: Plot both formulas
    println!("\n\nTest 4: Plotting entropy vs z");
    
    let mut file = File::create("entropy_comparison.csv").unwrap();
    writeln!(file, "z,w,w_ln_w,neg_z_exp_neg_z,difference").unwrap();
    
    for i in 0..=100 {
        let z = i as f64 * 0.1;  // z from 0 to 10
        let w = (-z).exp();
        let wlnw = if w > 0.0 { w * w.ln() } else { 0.0 };
        let zexpz = -z * (-z).exp();
        let diff = (wlnw - zexpz).abs();
        
        writeln!(file, "{},{},{},{},{}", z, w, wlnw, zexpz, diff).unwrap();
    }
    
    println!("Data written to entropy_comparison.csv");
    
    // Final verdict
    println!("\n=== FINAL VERDICT ===");
    println!("Mathematical proof:");
    println!("  w * ln(w) where w = exp(-z)");
    println!("  = exp(-z) * ln(exp(-z))");
    println!("  = exp(-z) * (-z)");
    println!("  = -z * exp(-z)");
    println!("\nThe formulas are MATHEMATICALLY IDENTICAL!");
    println!("\nThe 'discrepancy' was a misunderstanding.");
    println!("Both implementations calculate the same physics.");
    println!("\nHowever, the interpretation matters:");
    println!("- We should think in terms of weights w ∈ (0,1]");
    println!("- The entropy w*ln(w) is negative for w < 1");
    println!("- This drives weights toward zero (disconnection)");
}

#[test]
fn test_entropy_equivalence() {
    // Rigorous test of mathematical equivalence
    for i in 0..1000 {
        let z = i as f64 * 0.01;  // Test z from 0 to 10
        let (m1, m2, _) = verify_mathematical_equivalence(z);
        assert!((m1 - m2).abs() < 1e-14, 
                "Methods differ at z={}: {} vs {}", z, m1, m2);
    }
}