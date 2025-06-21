// Final physics check to verify fixes work across different system sizes and parameters

use scan::graph::Graph;
use scan::graph_fast::FastGraph;
use scan::graph_ultra_optimized::UltraOptimizedGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("=== Final Physics Correctness Check ===\n");
    
    // Test multiple system sizes
    let sizes = vec![4, 6, 8, 12];
    let parameters = vec![
        (0.5, 1.0),   // Low alpha, low beta
        (1.5, 2.9),   // Critical region parameters  
        (3.0, 0.5),   // High alpha, low beta
    ];
    
    for &n in &sizes {
        println!("System size N = {}:", n);
        
        for &(alpha, beta) in &parameters {
            println!("  Parameters α={}, β={}:", alpha, beta);
            
            // Create reference graph
            let mut rng = Pcg64::seed_from_u64(42);
            let reference = Graph::complete_random_with(&mut rng, n);
            
            // Test FastGraph consistency
            let fast_graph = FastGraph::from_graph(&reference);
            
            let ref_triangle = reference.triangle_sum();
            let fast_triangle = fast_graph.triangle_sum();
            let triangle_error = (fast_triangle - ref_triangle).abs();
            
            let ref_action = reference.action(alpha, beta);
            let fast_action = fast_graph.action(alpha, beta);
            let action_error = (fast_action - ref_action).abs();
            
            if triangle_error < 1e-10 && action_error < 1e-10 {
                println!("    ✓ FastGraph matches Reference (triangle: {:.2e}, action: {:.2e})", 
                         triangle_error, action_error);
            } else {
                println!("    ✗ FastGraph differs (triangle: {:.2e}, action: {:.2e})", 
                         triangle_error, action_error);
            }
            
            // Test UltraOptimized internal consistency
            let ultra_graph = UltraOptimizedGraph::new(n, 42);
            let ultra_action = ultra_graph.action(alpha, beta, 0.0);
            let ultra_entropy: f64 = ultra_graph.z_values.iter()
                .zip(&ultra_graph.exp_neg_z)
                .map(|(&z, &w)| -z * w)
                .sum();
            let ultra_triangle = ultra_graph.triangle_sum();
            let ultra_expected = beta * ultra_entropy + alpha * ultra_triangle;
            let ultra_error = (ultra_action - ultra_expected).abs();
            
            if ultra_error < 1e-10 {
                println!("    ✓ UltraOptimized internally consistent (error: {:.2e})", ultra_error);
            } else {
                println!("    ✗ UltraOptimized inconsistent (error: {:.2e})", ultra_error);
            }
        }
        println!();
    }
    
    // Test Monte Carlo detailed balance for a specific case
    println!("=== Monte Carlo Detailed Balance Test ===");
    let n = 8;
    let alpha = 1.5;
    let beta = 2.9;
    
    let mut rng = Pcg64::seed_from_u64(123);
    let reference = Graph::complete_random_with(&mut rng, n);
    let mut fast_graph = FastGraph::from_graph(&reference);
    
    let initial_action = fast_graph.action(alpha, beta);
    println!("Initial action: {:.6}", initial_action);
    
    // Run many MC steps and track acceptance
    let steps = 1000;
    let mut accepts = 0;
    let mut energy_changes = Vec::new();
    let mut large_changes = 0;
    
    let mut rng = Pcg64::seed_from_u64(456);
    
    for i in 0..steps {
        let pre_action = fast_graph.action(alpha, beta);
        let step_info = fast_graph.metropolis_step(alpha, beta, 0.05, 0.05, &mut rng);
        let post_action = fast_graph.action(alpha, beta);
        
        let delta_action = post_action - pre_action;
        energy_changes.push(delta_action);
        
        if step_info.accept {
            accepts += 1;
            if delta_action.abs() > 0.1 {
                large_changes += 1;
            }
        }
        
        // Check for any NaN or infinite values
        if !post_action.is_finite() {
            println!("  ✗ Non-finite action at step {}: {}", i, post_action);
            break;
        }
    }
    
    let final_action = fast_graph.action(alpha, beta);
    let acceptance_rate = accepts as f64 / steps as f64;
    
    println!("Final action: {:.6}", final_action);
    println!("Action change: {:.6}", final_action - initial_action);
    println!("Acceptance rate: {:.1}%", 100.0 * acceptance_rate);
    println!("Large changes (|ΔS| > 0.1): {}", large_changes);
    
    // Check if action is conserved on rejected moves
    let mut conserved_rejections = 0;
    let mut total_rejections = 0;
    
    for i in 0..100 {  // Test 100 moves specifically for conservation
        let pre_action = fast_graph.action(alpha, beta);
        let step_info = fast_graph.metropolis_step(alpha, beta, 0.05, 0.05, &mut rng);
        let post_action = fast_graph.action(alpha, beta);
        
        if !step_info.accept {
            total_rejections += 1;
            if (post_action - pre_action).abs() < 1e-12 {
                conserved_rejections += 1;
            }
        }
    }
    
    if total_rejections > 0 {
        let conservation_rate = conserved_rejections as f64 / total_rejections as f64;
        println!("Action conservation on rejections: {:.1}% ({}/{})", 
                 100.0 * conservation_rate, conserved_rejections, total_rejections);
        
        if conservation_rate > 0.95 {
            println!("✓ Excellent action conservation");
        } else {
            println!("⚠ Poor action conservation - may indicate numerical issues");
        }
    }
    
    // Final antisymmetry check
    println!("\n=== Final Antisymmetry Check ===");
    let mut violations = 0;
    let mut max_violation = 0.0f64;
    
    for i in 0..n {
        for j in (i+1)..n {
            let theta_ij = fast_graph.get_phase(i, j);
            let theta_ji = fast_graph.get_phase(j, i);
            let violation = (theta_ji + theta_ij).abs();
            
            if violation > 1e-10 {
                violations += 1;
                max_violation = max_violation.max(violation);
            }
        }
    }
    
    if violations == 0 {
        println!("✓ Perfect antisymmetry: θ_ji = -θ_ij for all pairs");
    } else {
        println!("✗ {} antisymmetry violations, max = {:.2e}", violations, max_violation);
    }
    
    println!("\n=== Physics Check Complete ===");
}