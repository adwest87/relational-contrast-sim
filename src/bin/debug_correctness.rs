// Debug correctness issues between original and optimized implementations

use scan::graph::Graph;
use scan::graph_fast::FastGraph;
use scan::observables::Observables;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_pcg::Pcg64;
use std::collections::HashMap;

/// Detailed state snapshot for debugging
#[derive(Debug, Clone)]
struct StateSnapshot {
    step: usize,
    action_before: f64,
    action_after: f64,
    accepted: bool,
    link_idx: Option<usize>,
    link_state: Option<(f64, f64)>, // (z, theta) or (w, theta)
    entropy: f64,
    triangle_sum: f64,
    mean_w: f64,
    mean_cos: f64,
}

/// Compare single MC step between implementations
fn trace_single_step(step_num: usize) {
    println!("\n=== Tracing MC Step {} ===", step_num);
    
    let n = 24;
    let alpha = 1.5;
    let beta = 2.9;
    let delta_z = 0.1;
    let delta_theta = 0.1;
    let seed = 12345;
    
    // Initialize both with same seed
    let mut orig_rng = ChaCha20Rng::seed_from_u64(seed);
    let mut opt_rng = Pcg64::seed_from_u64(seed);
    
    let mut orig_graph = Graph::complete_random_with(&mut orig_rng, n);
    let mut opt_graph = FastGraph::new(n, seed);
    
    // Run steps up to the target
    for i in 0..step_num {
        orig_graph.metropolis_step(beta, alpha, delta_z, delta_theta, &mut orig_rng);
        opt_graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut opt_rng);
    }
    
    // Capture state before step
    println!("\nBefore step {}:", step_num);
    println!("  Original:");
    println!("    Entropy: {:.6}", orig_graph.entropy_action());
    println!("    Triangle sum: {:.6}", orig_graph.triangle_sum());
    println!("    Action: {:.6}", orig_graph.action(alpha, beta));
    
    println!("  Optimized:");
    println!("    Entropy: {:.6}", opt_graph.entropy_action());
    println!("    Triangle sum: {:.6}", opt_graph.triangle_sum());
    println!("    Action: {:.6}", opt_graph.action(alpha, beta));
    
    // Perform the step with detailed tracking
    println!("\nPerforming step {}:", step_num);
    let orig_info = orig_graph.metropolis_step(beta, alpha, delta_z, delta_theta, &mut orig_rng);
    let opt_info = opt_graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut opt_rng);
    
    println!("  Original accepted: {}", orig_info.accepted);
    println!("  Optimized accepted: {}", opt_info.accept);
    
    // Compare final state
    println!("\nAfter step {}:", step_num);
    println!("  Entropy diff: {:.6}", 
        (orig_graph.entropy_action() - opt_graph.entropy_action()).abs());
    println!("  Triangle diff: {:.6}", 
        (orig_graph.triangle_sum() - opt_graph.triangle_sum()).abs());
}

/// Check numerical precision issues
fn check_numerical_precision() {
    println!("\n=== Checking Numerical Precision ===");
    
    let test_values = vec![
        0.1, 0.01, 0.001, 1.0, 10.0, 100.0,
        std::f64::consts::PI, std::f64::consts::E
    ];
    
    for &val in &test_values {
        let f64_exp = (-val).exp();
        let f32_exp = (-(val as f32)).exp() as f64;
        let diff = (f64_exp - f32_exp).abs();
        let rel_err = diff / f64_exp;
        
        println!("  exp(-{:.3}): f64={:.6}, f32={:.6}, rel_err={:.2e}", 
            val, f64_exp, f32_exp, rel_err);
    }
    
    // Check trigonometric precision
    println!("\nTrigonometric precision:");
    for i in 0..8 {
        let angle = i as f64 * std::f64::consts::PI / 4.0;
        let f64_cos = angle.cos();
        let f32_cos = (angle as f32).cos() as f64;
        let diff = (f64_cos - f32_cos).abs();
        
        println!("  cos({:.3}π): f64={:.6}, f32={:.6}, diff={:.2e}", 
            angle / std::f64::consts::PI, f64_cos, f32_cos, diff);
    }
}

/// Compare RNG sequences
fn compare_rng_sequences() {
    println!("\n=== Comparing RNG Sequences ===");
    
    let seed = 12345;
    let mut chacha = ChaCha20Rng::seed_from_u64(seed);
    let mut pcg = Pcg64::seed_from_u64(seed);
    
    println!("First 10 uniform samples:");
    for i in 0..10 {
        use rand::Rng;
        let chacha_val: f64 = chacha.gen();
        let pcg_val: f64 = pcg.gen();
        println!("  {}: ChaCha={:.6}, PCG={:.6}, diff={:.6}", 
            i, chacha_val, pcg_val, (chacha_val - pcg_val).abs());
    }
    
    // Reset and check gen_range
    let mut chacha = ChaCha20Rng::seed_from_u64(seed);
    let mut pcg = Pcg64::seed_from_u64(seed);
    
    println!("\nFirst 10 gen_range(0..n) samples:");
    let n = 100;
    for i in 0..10 {
        use rand::Rng;
        let chacha_val = chacha.gen_range(0..n);
        let pcg_val = pcg.gen_range(0..n);
        println!("  {}: ChaCha={}, PCG={}, diff={}", 
            i, chacha_val, pcg_val, (chacha_val as i32 - pcg_val as i32).abs());
    }
}

/// Check initialization differences
fn check_initialization() {
    println!("\n=== Checking Initialization ===");
    
    let n = 10;
    let seed = 12345;
    
    let mut orig_rng = ChaCha20Rng::seed_from_u64(seed);
    let orig_graph = Graph::complete_random_with(&mut orig_rng, n);
    
    let opt_graph = FastGraph::new(n, seed);
    
    println!("Number of links:");
    println!("  Original: {}", orig_graph.m());
    println!("  Optimized: {}", opt_graph.m());
    
    println!("\nFirst 5 link weights:");
    for i in 0..5.min(orig_graph.links.len()) {
        let orig_w = orig_graph.links[i].w();
        let opt_w = opt_graph.links[i].exp_neg_z as f64;
        println!("  Link {}: orig={:.6}, opt={:.6}, diff={:.6}", 
            i, orig_w, opt_w, (orig_w - opt_w).abs());
    }
    
    println!("\nInitial sums:");
    println!("  Original sum_w: {:.6}", orig_graph.sum_weights());
    println!("  Optimized sum_w: {:.6}", 
        opt_graph.links.iter().map(|l| l.exp_neg_z as f64).sum::<f64>());
}

/// Detailed step-by-step comparison
fn detailed_comparison(n_steps: usize) {
    println!("\n=== Detailed Step-by-Step Comparison ===");
    
    let n = 24;
    let alpha = 1.5;
    let beta = 2.9;
    let delta_z = 0.1;
    let delta_theta = 0.1;
    let seed = 12345;
    
    // Create same initial graph for both
    let mut init_rng = ChaCha20Rng::seed_from_u64(seed);
    let init_graph = Graph::complete_random_with(&mut init_rng, n);
    
    let mut orig_rng = ChaCha20Rng::seed_from_u64(seed);
    let mut opt_rng = Pcg64::seed_from_u64(seed);
    
    let mut orig_graph = init_graph.clone();
    let mut opt_graph = FastGraph::from_graph(&init_graph);
    
    let mut orig_accepts = 0;
    let mut opt_accepts = 0;
    
    for step in 0..n_steps {
        // Capture before state
        let orig_action_before = orig_graph.action(alpha, beta);
        let opt_action_before = opt_graph.action(alpha, beta);
        
        // Perform steps
        let orig_info = orig_graph.metropolis_step(beta, alpha, delta_z, delta_theta, &mut orig_rng);
        let opt_info = opt_graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut opt_rng);
        
        if orig_info.accepted { orig_accepts += 1; }
        if opt_info.accept { opt_accepts += 1; }
        
        // Check divergence
        let action_diff = (orig_graph.action(alpha, beta) - opt_graph.action(alpha, beta)).abs();
        
        if action_diff > 1e-3 || step < 10 || step % 1000 == 0 {
            println!("\nStep {}: action_diff={:.6}", step, action_diff);
            println!("  Accept: orig={}, opt={}", orig_info.accepted, opt_info.accept);
            println!("  Action before: orig={:.6}, opt={:.6}", orig_action_before, opt_action_before);
            println!("  Action after: orig={:.6}, opt={:.6}", 
                orig_graph.action(alpha, beta), opt_graph.action(alpha, beta));
            
            if action_diff > 0.1 {
                println!("  LARGE DIVERGENCE DETECTED!");
                break;
            }
        }
    }
    
    println!("\nFinal statistics after {} steps:", n_steps);
    println!("  Original accepts: {} ({:.1}%)", orig_accepts, 100.0 * orig_accepts as f64 / n_steps as f64);
    println!("  Optimized accepts: {} ({:.1}%)", opt_accepts, 100.0 * opt_accepts as f64 / n_steps as f64);
    
    // Final observables
    let orig_obs = Observables::measure(&orig_graph, beta, alpha);
    let opt_mean_cos: f64 = opt_graph.links.iter().map(|l| l.cos_theta as f64).sum::<f64>() / opt_graph.m() as f64;
    
    println!("\nFinal observables:");
    println!("  <cos θ>: orig={:.6}, opt={:.6}", orig_obs.mean_cos, opt_mean_cos);
    println!("  <w>: orig={:.6}, opt={:.6}", orig_obs.mean_w, 
        opt_graph.links.iter().map(|l| l.exp_neg_z as f64).sum::<f64>() / opt_graph.m() as f64);
}

/// Check for parameter order issues
fn check_parameter_order() {
    println!("\n=== Checking Parameter Order ===");
    
    let n = 10;
    let alpha = 1.5;
    let beta = 2.9;
    let seed = 12345;
    
    let mut rng1 = ChaCha20Rng::seed_from_u64(seed);
    let mut rng2 = ChaCha20Rng::seed_from_u64(seed);
    
    let mut graph1 = Graph::complete_random_with(&mut rng1, n);
    let mut graph2 = Graph::complete_random_with(&mut rng2, n);
    
    println!("Original implementation:");
    println!("  action(alpha={}, beta={}) = {:.6}", alpha, beta, graph1.action(alpha, beta));
    println!("  β*entropy + α*triangles = {:.6}", 
        beta * graph1.entropy_action() + alpha * graph1.triangle_sum());
    
    // The optimized implementation might have swapped parameter order
    println!("\nIf parameters were swapped:");
    println!("  action(alpha={}, beta={}) = {:.6}", beta, alpha, graph1.action(beta, alpha));
    
    // Test metropolis_step parameter order
    let info1 = graph1.metropolis_step(beta, alpha, 0.1, 0.1, &mut rng1);
    let info2 = graph2.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng2); // swapped
    
    println!("\nMetropolis step results:");
    println!("  (β, α) order: accepted={}", info1.accepted);
    println!("  (α, β) order: accepted={}", info2.accepted);
}

/// Main debugging function
fn main() {
    println!("Monte Carlo Optimization Debugging");
    println!("==================================");
    
    // Check basic numerical issues
    check_numerical_precision();
    
    // Check RNG differences
    compare_rng_sequences();
    
    // Check initialization
    check_initialization();
    
    // Check parameter order
    check_parameter_order();
    
    // Detailed comparison
    detailed_comparison(100);
    
    // Trace specific steps
    trace_single_step(1);
    trace_single_step(10);
    
    println!("\n=== Debugging Complete ===");
}