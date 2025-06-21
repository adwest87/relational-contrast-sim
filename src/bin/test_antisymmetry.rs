// Test antisymmetry property: θ_ji = -θ_ij

use scan::graph_fast::FastGraph;
use scan::graph_ultra_optimized::UltraOptimizedGraph;
use scan::graph_m1_optimized::M1Graph;

fn main() {
    let n = 6;
    
    println!("Testing antisymmetry property θ_ji = -θ_ij:");
    
    // Test FastGraph
    let fast_graph = FastGraph::new(n, 42);
    println!("\nFastGraph:");
    for i in 0..n {
        for j in (i+1)..n {
            let theta_ij = fast_graph.get_phase(i, j);
            let theta_ji = fast_graph.get_phase(j, i);
            let expected = -theta_ij;
            let error = (theta_ji - expected).abs();
            println!("  θ_{}{} = {:.6}, θ_{}{} = {:.6}, -θ_{}{} = {:.6}, error = {:.2e}", 
                     i, j, theta_ij, j, i, theta_ji, i, j, expected, error);
            if error > 1e-10 {
                println!("    ✗ ANTISYMMETRY VIOLATION!");
            }
        }
    }
    
    // Test UltraOptimized
    let ultra_graph = UltraOptimizedGraph::new(n, 42);
    println!("\nUltraOptimized:");
    for i in 0..n {
        for j in (i+1)..n {
            let theta_ij = ultra_graph.get_phase(i, j);
            let theta_ji = ultra_graph.get_phase(j, i);
            let expected = -theta_ij;
            let error = (theta_ji - expected).abs();
            println!("  θ_{}{} = {:.6}, θ_{}{} = {:.6}, -θ_{}{} = {:.6}, error = {:.2e}", 
                     i, j, theta_ij, j, i, theta_ji, i, j, expected, error);
            if error > 1e-10 {
                println!("    ✗ ANTISYMMETRY VIOLATION!");
            }
        }
    }
    
    // Test M1Graph
    let m1_graph = M1Graph::new(n, 42);
    println!("\nM1Graph:");
    for i in 0..n {
        for j in (i+1)..n {
            let theta_ij = m1_graph.get_phase(i, j);
            let theta_ji = m1_graph.get_phase(j, i);
            let expected = -theta_ij;
            let error = (theta_ji - expected).abs();
            println!("  θ_{}{} = {:.6}, θ_{}{} = {:.6}, -θ_{}{} = {:.6}, error = {:.2e}", 
                     i, j, theta_ij, j, i, theta_ji, i, j, expected, error);
            if error > 1e-10 {
                println!("    ✗ ANTISYMMETRY VIOLATION!");
            }
        }
    }
    
    println!("\nTesting triangle sum internal consistency:");
    
    // For FastGraph, test if triangle_sum matches get_phase-based calculation
    let fast_triangle_direct = fast_graph.triangle_sum();
    let mut fast_triangle_manual = 0.0;
    for i in 0..n {
        for j in (i+1)..n {
            for k in (j+1)..n {
                let t_ij = fast_graph.get_phase(i, j);
                let t_jk = fast_graph.get_phase(j, k);
                let t_ki = fast_graph.get_phase(k, i);
                fast_triangle_manual += (t_ij + t_jk + t_ki).cos();
            }
        }
    }
    
    println!("FastGraph triangle sum:");
    println!("  Direct method:  {:.6}", fast_triangle_direct);
    println!("  Manual calc:    {:.6}", fast_triangle_manual);
    println!("  Difference:     {:.2e}", (fast_triangle_direct - fast_triangle_manual).abs());
    
    if (fast_triangle_direct - fast_triangle_manual).abs() < 1e-10 {
        println!("  ✓ FastGraph triangle sum is internally consistent");
    } else {
        println!("  ✗ FastGraph triangle sum has internal inconsistency");
    }
}