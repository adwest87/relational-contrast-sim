// Quick test to check if the antisymmetry bug is fixed

use scan::graph::Graph;
use scan::graph_fast::FastGraph;
use scan::graph_ultra_optimized::UltraOptimizedGraph;
use scan::graph_m1_optimized::M1Graph;

fn main() {
    let n = 8;  // Small size for quick testing
    
    // Create all graphs from the same initial state for fair comparison
    let orig_graph = Graph::complete_random_with(&mut rand::thread_rng(), n);
    let fast_graph = FastGraph::from_graph(&orig_graph);
    
    // For UltraOptimized and M1, we need to manually copy the state from orig_graph
    // since they don't have from_graph() methods, so let's just test what we can
    println!("Testing with different initializations (not directly comparable):");
    
    println!("Triangle sum comparison for N={}:", n);
    println!("Original:      {:.6}", orig_graph.triangle_sum());
    println!("FastGraph:     {:.6}", fast_graph.triangle_sum());
    println!("UltraOpt:      {:.6}", ultra_graph.triangle_sum());
    println!("M1Opt:         {:.6}", m1_graph.triangle_sum());
    
    // Check if values match within tolerance
    let orig_sum = orig_graph.triangle_sum();
    let fast_sum = fast_graph.triangle_sum();
    let ultra_sum = ultra_graph.triangle_sum();
    let m1_sum = m1_graph.triangle_sum();
    
    println!("\nDifferences from original:");
    println!("FastGraph:     {:.6} ({:.1}% error)", 
             fast_sum - orig_sum, 
             100.0 * (fast_sum - orig_sum).abs() / orig_sum.abs());
    println!("UltraOpt:      {:.6} ({:.1}% error)", 
             ultra_sum - orig_sum, 
             100.0 * (ultra_sum - orig_sum).abs() / orig_sum.abs());
    println!("M1Opt:         {:.6} ({:.1}% error)", 
             m1_sum - orig_sum, 
             100.0 * (m1_sum - orig_sum).abs() / orig_sum.abs());
    
    // Test if bug is fixed (errors should be small now)
    let tolerance = 1e-10;
    if (fast_sum - ultra_sum).abs() < tolerance {
        println!("\n✓ FastGraph and UltraOpt now match!");
    } else {
        println!("\n✗ FastGraph and UltraOpt still differ by {:.2e}", 
                 (fast_sum - ultra_sum).abs());
    }
}