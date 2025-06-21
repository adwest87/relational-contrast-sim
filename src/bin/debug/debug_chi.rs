// Debug negative susceptibility issue

use scan::graph::Graph;
use scan::graph_fast::FastGraph;
use rand_pcg::Pcg64;

fn main() {
    let n = 24;
    let beta = 2.91;
    let alpha = 1.48;
    
    // Create a simple test graph
    let mut rng = Pcg64::seed_from_u64(42);
    let mut graph = Graph::complete_random_with(&mut rng, n);
    
    // Set all theta to 0 initially for debugging
    for link in &mut graph.links {
        link.theta = 0.0;
        link.z = 0.8; // Critical region
    }
    
    let fast_graph = FastGraph::from_graph(&graph);
    
    // Calculate observables manually
    let m = fast_graph.m() as f64;
    let sum_cos: f64 = fast_graph.links.iter().map(|l| l.cos_theta).sum();
    let sum_w: f64 = fast_graph.links.iter().map(|l| l.exp_neg_z).sum();
    let sum_w_cos: f64 = fast_graph.links.iter().map(|l| l.exp_neg_z * l.cos_theta).sum();
    
    let mean_cos = sum_cos / m;
    let mean_w = sum_w / m;
    let mean_w_cos = sum_w_cos / m;
    
    println!("Debug susceptibility calculation:");
    println!("  N = {}", n);
    println!("  M = {}", m as i32);
    println!("  β = {}", beta);
    println!();
    println!("  sum_cos = {:.6}", sum_cos);
    println!("  sum_w = {:.6}", sum_w);
    println!("  sum_w_cos = {:.6}", sum_w_cos);
    println!();
    println!("  <cos θ> = {:.6}", mean_cos);
    println!("  <w> = {:.6}", mean_w);
    println!("  <w cos θ> = {:.6}", mean_w_cos);
    println!();
    
    // Susceptibility χ = N * β * (<w cos θ> - <w><cos θ>)
    let chi_term = mean_w_cos - mean_w * mean_cos;
    let chi = n as f64 * beta * chi_term;
    
    println!("  <w cos θ> - <w><cos θ> = {:.6} - {:.6} * {:.6} = {:.6}", 
        mean_w_cos, mean_w, mean_cos, chi_term);
    println!("  χ = N * β * ({:.6}) = {} * {:.2} * {:.6} = {:.6}", 
        chi_term, n, beta, chi_term, chi);
    println!();
    
    // Expected values for all theta = 0
    println!("Expected values (all θ = 0, z = 0.8):");
    println!("  <cos θ> = 1.0");
    println!("  <w> = exp(-0.8) = {:.6}", (-0.8f64).exp());
    println!("  <w cos θ> = <w> * 1.0 = {:.6}", (-0.8f64).exp());
    println!("  χ term = <w> - <w> = 0");
    println!("  χ = 0 (as expected for aligned state)");
    
    // Now randomize theta
    println!("\nAfter randomizing theta:");
    use rand::distributions::Uniform;
    use rand::prelude::*;
    let theta_dist = Uniform::new(0.0, std::f64::consts::TAU);
    
    let mut graph2 = graph.clone();
    for link in &mut graph2.links {
        link.theta = rng.sample(theta_dist);
    }
    
    let fast_graph2 = FastGraph::from_graph(&graph2);
    
    let sum_cos2: f64 = fast_graph2.links.iter().map(|l| l.cos_theta).sum();
    let sum_w2: f64 = fast_graph2.links.iter().map(|l| l.exp_neg_z).sum();
    let sum_w_cos2: f64 = fast_graph2.links.iter().map(|l| l.exp_neg_z * l.cos_theta).sum();
    
    let mean_cos2 = sum_cos2 / m;
    let mean_w2 = sum_w2 / m;
    let mean_w_cos2 = sum_w_cos2 / m;
    
    let chi_term2 = mean_w_cos2 - mean_w2 * mean_cos2;
    let chi2 = n as f64 * beta * chi_term2;
    
    println!("  <cos θ> = {:.6}", mean_cos2);
    println!("  <w> = {:.6}", mean_w2);
    println!("  <w cos θ> = {:.6}", mean_w_cos2);
    println!("  χ term = {:.6} - {:.6} * {:.6} = {:.6}", 
        mean_w_cos2, mean_w2, mean_cos2, chi_term2);
    println!("  χ = {} * {:.2} * {:.6} = {:.6}", n, beta, chi_term2, chi2);
    
    if chi2 < 0.0 {
        println!("\nWARNING: χ is negative! This suggests <w cos θ> < <w><cos θ>");
        println!("This can happen if w and cos θ are anti-correlated.");
    }
}