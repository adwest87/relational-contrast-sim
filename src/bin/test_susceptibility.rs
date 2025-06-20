// Test different susceptibility formulas to find the correct one

use scan::graph::Graph;
use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("=== SUSCEPTIBILITY FORMULA TEST ===\n");
    
    let n = 48;
    let beta = 2.91;
    let alpha = 1.50;
    
    // Create test configurations
    let mut rng = Pcg64::seed_from_u64(42);
    let mut graph = Graph::complete_random_with(&mut rng, n);
    
    // Set to near-critical configuration
    use rand::distributions::Uniform;
    use rand::prelude::*;
    let theta_dist = Uniform::new(-0.5, 0.5);
    for link in &mut graph.links {
        link.theta = rng.sample(theta_dist);
        link.z = 0.8 + rng.gen_range(-0.1..0.1);
    }
    
    let mut fast_graph = FastGraph::from_graph(&graph);
    
    // Equilibrate
    println!("Equilibrating...");
    for _ in 0..50_000 {
        fast_graph.metropolis_step(alpha, beta, 0.3, 0.3, &mut rng);
    }
    
    // Measure various susceptibilities
    println!("Measuring susceptibilities after equilibration:\n");
    
    let m = fast_graph.m() as f64;
    let n_float = n as f64;
    
    // Collect measurements
    let mut measurements = Vec::new();
    for _ in 0..1000 {
        for _ in 0..100 {
            fast_graph.metropolis_step(alpha, beta, 0.3, 0.3, &mut rng);
        }
        
        let sum_cos: f64 = fast_graph.links.iter().map(|l| l.cos_theta).sum();
        let sum_sin: f64 = fast_graph.links.iter().map(|l| l.sin_theta).sum();
        let sum_cos_sq: f64 = fast_graph.links.iter().map(|l| l.cos_theta * l.cos_theta).sum();
        let sum_sin_sq: f64 = fast_graph.links.iter().map(|l| l.sin_theta * l.sin_theta).sum();
        let sum_w: f64 = fast_graph.links.iter().map(|l| l.exp_neg_z).sum();
        let sum_w_cos: f64 = fast_graph.links.iter().map(|l| l.exp_neg_z * l.cos_theta).sum();
        let sum_w_sin: f64 = fast_graph.links.iter().map(|l| l.exp_neg_z * l.sin_theta).sum();
        let sum_z: f64 = fast_graph.links.iter().map(|l| l.z).sum();
        let sum_z_sq: f64 = fast_graph.links.iter().map(|l| l.z * l.z).sum();
        
        measurements.push((
            sum_cos / m, sum_sin / m, sum_cos_sq / m, sum_sin_sq / m,
            sum_w / m, sum_w_cos / m, sum_w_sin / m,
            sum_z / m, sum_z_sq / m
        ));
    }
    
    // Calculate averages
    let n_meas = measurements.len() as f64;
    let mean_cos = measurements.iter().map(|m| m.0).sum::<f64>() / n_meas;
    let mean_sin = measurements.iter().map(|m| m.1).sum::<f64>() / n_meas;
    let mean_cos_sq = measurements.iter().map(|m| m.2).sum::<f64>() / n_meas;
    let mean_sin_sq = measurements.iter().map(|m| m.3).sum::<f64>() / n_meas;
    let mean_w = measurements.iter().map(|m| m.4).sum::<f64>() / n_meas;
    let mean_w_cos = measurements.iter().map(|m| m.5).sum::<f64>() / n_meas;
    let mean_w_sin = measurements.iter().map(|m| m.6).sum::<f64>() / n_meas;
    let mean_z = measurements.iter().map(|m| m.7).sum::<f64>() / n_meas;
    let mean_z_sq = measurements.iter().map(|m| m.8).sum::<f64>() / n_meas;
    
    println!("Basic observables:");
    println!("  <cos θ> = {:.4}", mean_cos);
    println!("  <sin θ> = {:.4}", mean_sin);
    println!("  <cos² θ> = {:.4}", mean_cos_sq);
    println!("  <sin² θ> = {:.4}", mean_sin_sq);
    println!("  <w> = {:.4}", mean_w);
    println!("  <w cos θ> = {:.4}", mean_w_cos);
    println!("  <w sin θ> = {:.4}", mean_w_sin);
    println!("  <z> = {:.4}", mean_z);
    println!("  <z²> = {:.4}", mean_z_sq);
    
    println!("\nDifferent susceptibility formulas:");
    
    // 1. Standard magnetic susceptibility
    let chi_magnetic = n_float * (mean_cos_sq - mean_cos * mean_cos);
    println!("\n1. Magnetic susceptibility χ_m = N(<cos² θ> - <cos θ>²):");
    println!("   χ_m = {} * ({:.4} - {:.4}²) = {:.2}", n, mean_cos_sq, mean_cos, chi_magnetic);
    
    // 2. Weighted susceptibility (original attempt)
    let chi_weighted = n_float * beta * (mean_w_cos - mean_w * mean_cos);
    println!("\n2. Weighted susceptibility χ_w = Nβ(<w cos θ> - <w><cos θ>):");
    println!("   χ_w = {} * {:.2} * ({:.4} - {:.4} * {:.4}) = {:.2}", 
        n, beta, mean_w_cos, mean_w, mean_cos, chi_weighted);
    
    // 3. Full angular susceptibility
    let chi_angular = n_float * ((mean_cos_sq - mean_cos * mean_cos) + (mean_sin_sq - mean_sin * mean_sin));
    println!("\n3. Full angular susceptibility χ_θ = N(<cos² θ> - <cos θ>² + <sin² θ> - <sin θ>²):");
    println!("   χ_θ = {} * ({:.4} + {:.4}) = {:.2}", n, 
        mean_cos_sq - mean_cos * mean_cos,
        mean_sin_sq - mean_sin * mean_sin,
        chi_angular);
    
    // 4. Link weight susceptibility
    let chi_link = n_float * (mean_z_sq - mean_z * mean_z);
    println!("\n4. Link weight susceptibility χ_z = N(<z²> - <z>²):");
    println!("   χ_z = {} * ({:.4} - {:.4}²) = {:.2}", n, mean_z_sq, mean_z, chi_link);
    
    // 5. Proper relational susceptibility (coupling between θ and z)
    // This comes from ∂²F/∂h² where h couples to cos(θ)
    let chi_relational = n_float * beta * (mean_cos_sq - mean_cos * mean_cos);
    println!("\n5. Relational susceptibility χ_r = Nβ(<cos² θ> - <cos θ>²):");
    println!("   χ_r = {} * {:.2} * ({:.4} - {:.4}²) = {:.2}", 
        n, beta, mean_cos_sq, mean_cos, chi_relational);
    
    // Calculate correlation functions
    println!("\n\nCorrelation analysis:");
    
    // θ-z correlation
    let sum_z_cos: f64 = fast_graph.links.iter().map(|l| l.z * l.cos_theta).sum::<f64>() / m;
    let corr_z_cos = sum_z_cos - mean_z * mean_cos;
    println!("  <z cos θ> - <z><cos θ> = {:.4}", corr_z_cos);
    
    // w-cos correlation
    let corr_w_cos = mean_w_cos - mean_w * mean_cos;
    println!("  <w cos θ> - <w><cos θ> = {:.4}", corr_w_cos);
    
    println!("\nExpected behavior at critical point:");
    println!("  - χ should diverge as N^(γ/ν) with γ/ν ≈ 1.75");
    println!("  - For N=48, expect χ ≈ 30-40");
    println!("  - The correct formula should show large values");
    
    // Test which formula gives reasonable values
    println!("\nFormula assessment:");
    if chi_magnetic > 10.0 {
        println!("  ✓ Magnetic susceptibility shows critical behavior");
    }
    if chi_relational > 10.0 {
        println!("  ✓ Relational susceptibility shows critical behavior");
    }
    if chi_weighted < 1.0 {
        println!("  ✗ Weighted susceptibility too small - likely wrong formula");
    }
}