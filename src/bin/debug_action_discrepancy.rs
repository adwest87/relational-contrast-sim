use scan::graph::Graph;
use scan::graph_fast::FastGraph;
use scan::graph_ultra_optimized::UltraOptimizedGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("=== Action Discrepancy Debug ===\n");
    
    let n = 8;
    let seed = 42;
    let alpha = 1.5;
    let beta = 2.0;
    
    // Create original graph
    let mut rng = Pcg64::seed_from_u64(seed);
    let orig_graph = Graph::complete_random_with(&mut rng, n);
    
    // Create fast graph from original
    let fast_graph = FastGraph::from_graph(&orig_graph);
    
    // Create ultra-optimized with same RNG state
    let mut rng2 = Pcg64::seed_from_u64(seed);
    let mut ultra_graph = UltraOptimizedGraph::new_with_rng(&mut rng2, n);
    
    println!("System size: N = {}", n);
    println!("Parameters: α = {}, β = {}\n", alpha, beta);
    
    // Compare basic properties
    println!("=== Basic Properties ===");
    println!("Original links: {}", orig_graph.links.len());
    println!("FastGraph links: {}", fast_graph.m());
    println!("UltraOptimized links: {}", ultra_graph.m());
    
    // Compare individual components
    println!("\n=== Action Components ===");
    
    // Original components
    let orig_entropy = orig_graph.entropy_action();
    let orig_triangle = orig_graph.triangle_sum();
    let orig_action = beta * orig_entropy + alpha * orig_triangle;
    
    println!("Original:");
    println!("  Entropy: {:.12}", orig_entropy);
    println!("  Triangle sum: {:.12}", orig_triangle);
    println!("  Action: {:.12}", orig_action);
    println!("  Direct action: {:.12}", orig_graph.action(alpha, beta));
    
    // FastGraph components
    let fast_entropy = fast_graph.entropy_action();
    let fast_triangle = fast_graph.triangle_sum();
    let fast_action = beta * fast_entropy + alpha * fast_triangle;
    
    println!("\nFastGraph:");
    println!("  Entropy: {:.12}", fast_entropy);
    println!("  Triangle sum: {:.12}", fast_triangle);
    println!("  Action: {:.12}", fast_action);
    println!("  Direct action: {:.12}", fast_graph.action(alpha, beta));
    
    // UltraOptimized components
    let ultra_entropy: f64 = ultra_graph.z_values.iter()
        .zip(&ultra_graph.exp_neg_z)
        .map(|(&z, &w)| -z * w)
        .sum();
    let ultra_triangle = ultra_graph.triangle_sum();
    let ultra_action = beta * ultra_entropy + alpha * ultra_triangle;
    
    println!("\nUltraOptimized:");
    println!("  Entropy: {:.12}", ultra_entropy);
    println!("  Triangle sum: {:.12}", ultra_triangle);
    println!("  Action: {:.12}", ultra_action);
    println!("  Direct action: {:.12}", ultra_graph.action(alpha, beta, 0.0));
    
    // Compare individual links
    println!("\n=== First 5 Links Comparison ===");
    for i in 0..5.min(orig_graph.links.len()) {
        let orig_link = &orig_graph.links[i];
        let fast_link = &fast_graph.links[i];
        
        println!("Link {}:", i);
        println!("  Original: z={:.6}, θ={:.6}, w={:.6}", 
                 orig_link.z, orig_link.theta, orig_link.w());
        println!("  FastGraph: z={:.6}, θ={:.6}, w={:.6}", 
                 fast_link.z, fast_link.theta, fast_link.w());
        println!("  UltraOpt: z={:.6}, θ={:.6}, w={:.6}", 
                 ultra_graph.z_values[i], ultra_graph.theta_values[i], ultra_graph.exp_neg_z[i]);
        
        if (orig_link.z - fast_link.z).abs() > 1e-10 {
            println!("  ⚠ Z values differ!");
        }
        if (orig_link.theta - fast_link.theta).abs() > 1e-10 {
            println!("  ⚠ Theta values differ!");
        }
    }
    
    // Check if the issue is in initialization
    println!("\n=== Initialization Check ===");
    
    // Copy state from original to ultra-optimized manually
    copy_graph_state(&orig_graph, &mut ultra_graph);
    
    let ultra_action_after_copy = ultra_graph.action(alpha, beta, 0.0);
    println!("UltraOptimized action after copying state: {:.12}", ultra_action_after_copy);
    
    if (orig_action - ultra_action_after_copy).abs() < 1e-10 {
        println!("✓ Actions match after copying state - issue is in initialization");
    } else {
        println!("✗ Actions still don't match - issue is in action calculation");
    }
    
    // Compare triangle sums after copy
    let ultra_triangle_after_copy = ultra_graph.triangle_sum();
    println!("Triangle sum after copy: {:.12}", ultra_triangle_after_copy);
    
    if (orig_triangle - ultra_triangle_after_copy).abs() > 1e-10 {
        println!("⚠ Triangle sums still differ after copy");
    }
}

// Helper function to copy state from original graph to ultra-optimized
fn copy_graph_state(orig: &Graph, ultra: &mut UltraOptimizedGraph) {
    for (i, link) in orig.links.iter().enumerate() {
        ultra.z_values[i] = link.z;
        ultra.theta_values[i] = link.theta;
        ultra.cos_theta[i] = link.theta.cos();
        ultra.sin_theta[i] = link.theta.sin();
        ultra.exp_neg_z[i] = (-link.z).exp();
    }
    // Recompute cached triangle sum
    ultra.triangle_sum_cache = ultra.compute_full_triangle_sum();
}