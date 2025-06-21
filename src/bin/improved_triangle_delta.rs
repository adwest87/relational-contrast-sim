// Test improved triangle_sum_delta with proper numerical precision
use scan::graph_fast::FastGraph;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

// Kahan summation for numerical stability
fn kahan_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut c = 0.0;
    for &val in values {
        let y = val - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

// Improved triangle sum delta using analytical derivatives where possible
fn triangle_sum_delta_improved(graph: &FastGraph, link_idx: usize, new_theta: f64) -> f64 {
    let link = &graph.links[link_idx];
    let (i, j) = (link.i as usize, link.j as usize);
    let old_theta = link.theta;
    let delta_theta = new_theta - old_theta;
    
    // If delta is very small, use analytical approximation: -sin(x) * δx
    const SMALL_DELTA_THRESHOLD: f64 = 1e-8;
    
    let mut contributions = Vec::new();
    
    for k in 0..graph.n() {
        if k != i && k != j {
            let idx_ik = if i < k { 
                graph.link_index(i, k) 
            } else { 
                graph.link_index(k, i) 
            };
            let idx_jk = if j < k { 
                graph.link_index(j, k) 
            } else { 
                graph.link_index(k, j) 
            };
            
            let other_sum = graph.links[idx_ik].theta + graph.links[idx_jk].theta;
            let old_total = old_theta + other_sum;
            
            let contribution = if delta_theta.abs() < SMALL_DELTA_THRESHOLD {
                // Use analytical derivative: d/dx[3*cos(x)] = -3*sin(x)
                -3.0 * old_total.sin() * delta_theta
            } else {
                // Use direct calculation for larger changes
                let new_total = new_theta + other_sum;
                3.0 * (new_total.cos() - old_total.cos())
            };
            
            contributions.push(contribution);
        }
    }
    
    // Use Kahan summation for numerical stability
    kahan_sum(&contributions)
}

fn main() {
    println!("🔧 TESTING IMPROVED TRIANGLE DELTA CALCULATION");
    println!("===============================================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(12345);
    let mut graph = FastGraph::new(n, 999);
    
    let mut precision_violations = 0;
    let mut large_discrepancies = Vec::new();
    
    // Test 1000 moves and compare old vs improved calculation
    for step in 0..1000 {
        let energy_before = graph.action(alpha, beta);
        let links_before = graph.links.clone();
        
        // Simulate a theta move manually
        let link_idx = rng.gen_range(0..graph.links.len());
        let old_theta = graph.links[link_idx].theta;
        let d_theta = rng.gen_range(-0.1..=0.1);
        let new_theta = old_theta + d_theta;
        
        // Calculate using original method (reference)
        let delta_original = triangle_sum_delta_reference(&graph, link_idx, new_theta);
        
        // Calculate using improved method
        let delta_improved = triangle_sum_delta_improved(&graph, link_idx, new_theta);
        
        // Calculate true energy difference
        let original_state = graph.links[link_idx].theta;
        graph.links[link_idx].update_theta(new_theta);
        let energy_after = graph.action(alpha, beta);
        let true_delta_energy = energy_after - energy_before;
        let true_delta_triangle = true_delta_energy / alpha; // Remove alpha scaling
        graph.links[link_idx].update_theta(original_state); // Restore
        
        // Compare methods
        let diff_methods = (delta_improved - delta_original).abs();
        let error_original = (delta_original - true_delta_triangle).abs();
        let error_improved = (delta_improved - true_delta_triangle).abs();
        
        // Check for precision violations (external ΔE ≈ 0 but internal calculation large)
        if true_delta_energy.abs() < 1e-12 && (alpha * delta_original).abs() > 1e-10 {
            precision_violations += 1;
            if precision_violations <= 5 {
                println!("❌ Precision violation #{}: true_ΔE={:.2e}, calc_ΔS={:.2e}", 
                    precision_violations, true_delta_energy, alpha * delta_original);
            }
        }
        
        // Track large discrepancies
        if diff_methods > 1e-10 || error_original > 1e-8 {
            large_discrepancies.push((step, diff_methods, error_original, error_improved, true_delta_energy));
        }
        
        // Progress reporting
        if step % 200 == 0 {
            println!("Step {}: precision_violations={}, large_discrepancies={}", 
                step, precision_violations, large_discrepancies.len());
        }
    }
    
    println!("\n📊 RESULTS SUMMARY:");
    println!("Precision violations: {} / 1000", precision_violations);
    println!("Large discrepancies: {} / 1000", large_discrepancies.len());
    
    if !large_discrepancies.is_empty() {
        println!("\nWorst discrepancies:");
        large_discrepancies.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (i, &(step, diff, err_orig, err_imp, true_de)) in large_discrepancies.iter().take(5).enumerate() {
            println!("  #{}: step={}, method_diff={:.2e}, orig_err={:.2e}, imp_err={:.2e}, true_ΔE={:.2e}", 
                i+1, step, diff, err_orig, err_imp, true_de);
        }
    }
    
    // Determine appropriate epsilon based on findings
    let energy_scale = 11.0; // From previous analysis
    let relative_precision = 1e-12;
    let suggested_epsilon = energy_scale * relative_precision;
    
    println!("\n🎯 EPSILON RECOMMENDATIONS:");
    println!("Based on energy scale ~{:.1} and relative precision ~{:.0e}:", energy_scale, relative_precision);
    println!("  Suggested absolute epsilon: {:.2e}", suggested_epsilon);
    println!("  Current EPSILON=0.1 is off by factor: {:.0e}", 0.1 / suggested_epsilon);
}

// Reference implementation matching current code
fn triangle_sum_delta_reference(graph: &FastGraph, link_idx: usize, new_theta: f64) -> f64 {
    let link = &graph.links[link_idx];
    let (i, j) = (link.i as usize, link.j as usize);
    let old_theta = link.theta;
    
    let mut delta = 0.0;
    
    for k in 0..graph.n() {
        if k != i && k != j {
            let idx_ik = if i < k { 
                graph.link_index(i, k) 
            } else { 
                graph.link_index(k, i) 
            };
            let idx_jk = if j < k { 
                graph.link_index(j, k) 
            } else { 
                graph.link_index(k, j) 
            };
            
            let old_sum = old_theta + graph.links[idx_ik].theta + graph.links[idx_jk].theta;
            let new_sum = new_theta + graph.links[idx_ik].theta + graph.links[idx_jk].theta;
            
            delta += 3.0 * (new_sum.cos() - old_sum.cos());
        }
    }
    
    delta
}