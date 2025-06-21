// Check energy landscape for different configurations
use scan::graph_fast::FastGraph;
use std::f64::consts::PI;

fn main() {
    println!("🏔️ ENERGY LANDSCAPE CHECK");
    println!("========================");
    
    let n = 8; // Small system for clarity
    let beta = 3.0;
    let alpha = 1.5;
    
    println!("System: N={}, β={:.1}, α={:.1}", n, beta, alpha);
    println!("Number of links: {}", n*(n-1)/2);
    println!("Number of triangles: {}\n", n*(n-1)*(n-2)/6);
    
    let mut graph = FastGraph::new(n, 12345);
    
    // Test different uniform configurations
    let test_configs = vec![
        (0.0, "All θ=0"),
        (PI/4.0, "All θ=π/4"),
        (PI/2.0, "All θ=π/2"),
        (3.0*PI/4.0, "All θ=3π/4"),
        (PI, "All θ=π"),
    ];
    
    println!("📊 UNIFORM CONFIGURATIONS:");
    println!("=========================");
    
    for (theta, label) in &test_configs {
        // Set all links to same angle
        for i in 0..graph.links.len() {
            graph.links[i].update_theta(*theta);
        }
        
        let entropy = graph.entropy_action();
        let triangle_sum = graph.triangle_sum();
        let total_action = beta * entropy + alpha * triangle_sum;
        
        // For uniform configuration, all triangles have sum 3θ
        let triangle_angle_sum = 3.0 * theta;
        let cos_triangle = triangle_angle_sum.cos();
        
        println!("{} ({:.3} rad):", label, theta);
        println!("  Triangle angle sum: 3θ = {:.3} ({:.1}°)", 
            triangle_angle_sum, triangle_angle_sum.to_degrees());
        println!("  cos(3θ) = {:.3}", cos_triangle);
        println!("  S_entropy = {:.3}", entropy);
        println!("  S_triangle = {:.3} (per triangle: {:.3})", 
            triangle_sum, triangle_sum / (n*(n-1)*(n-2)/6) as f64);
        println!("  Total action = {:.3}", total_action);
        println!();
    }
    
    // Find minimum energy configuration
    let mut min_action = f64::INFINITY;
    let mut min_theta = 0.0;
    
    println!("🔍 SEARCHING FOR ENERGY MINIMUM:");
    println!("================================");
    
    for i in 0..100 {
        let theta = i as f64 * 2.0 * PI / 100.0;
        
        for j in 0..graph.links.len() {
            graph.links[j].update_theta(theta);
        }
        
        let action = graph.action(alpha, beta);
        
        if action < min_action {
            min_action = action;
            min_theta = theta;
        }
    }
    
    println!("Minimum at θ = {:.3} ({:.1}°)", min_theta, min_theta.to_degrees());
    println!("Minimum action = {:.3}", min_action);
    
    // Check mixed configurations
    println!("\n📊 MIXED CONFIGURATIONS:");
    println!("=======================");
    
    // Half 0, half π
    for i in 0..graph.links.len() {
        if i % 2 == 0 {
            graph.links[i].update_theta(0.0);
        } else {
            graph.links[i].update_theta(PI);
        }
    }
    
    let mixed_action = graph.action(alpha, beta);
    println!("Half θ=0, half θ=π: action = {:.3}", mixed_action);
    
    // Random configuration
    use rand::{SeedableRng, Rng};
    use rand_pcg::Pcg64;
    let mut rng = Pcg64::seed_from_u64(42);
    
    for i in 0..graph.links.len() {
        let theta = rng.gen_range(0.0..2.0*PI);
        graph.links[i].update_theta(theta);
    }
    
    let random_action = graph.action(alpha, beta);
    println!("Random angles: action = {:.3}", random_action);
    
    // Analysis
    println!("\n📋 ANALYSIS:");
    println!("===========");
    
    // The cosine of 3θ has minima at θ = π/3, π, 5π/3
    // cos(3θ) = -1 at these points
    println!("For cos(3θ):");
    println!("  Minima at: θ = π/3 (60°), π (180°), 5π/3 (300°)");
    println!("  Maxima at: θ = 0 (0°), 2π/3 (120°), 4π/3 (240°)");
    
    println!("\nThis creates a 3-fold degenerate ground state!");
    println!("The system has Z₃ symmetry, not continuous O(2) symmetry.");
    
    // Check if this matches our minimum
    let expected_minima = [PI/3.0, PI, 5.0*PI/3.0];
    let closest_minimum = expected_minima.iter()
        .min_by(|&&a, &&b| {
            let diff_a = (a - min_theta).abs();
            let diff_b = (b - min_theta).abs();
            diff_a.partial_cmp(&diff_b).unwrap()
        })
        .unwrap();
    
    if (min_theta - closest_minimum).abs() < 0.1 {
        println!("\n✅ Found minimum matches expected Z₃ ground state");
    } else {
        println!("\n❌ Found minimum does not match expected values");
    }
}