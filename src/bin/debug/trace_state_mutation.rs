// Trace exact state mutations during metropolis step
use scan::graph_fast::FastGraph;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

fn main() {
    println!("🔍 TRACING STATE MUTATIONS");
    println!("==========================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(42);
    let mut graph = FastGraph::new(n, 123);
    
    // Go directly to step 6 where we know the bug occurs
    for _ in 0..6 {
        let _ = graph.metropolis_step(alpha, beta, 0.3, 0.3, &mut rng);
    }
    
    // Now trace step 6 in detail
    println!("About to perform step 6...");
    
    let energy_before = graph.action(alpha, beta);
    let state_before = graph.links.clone();
    
    println!("Energy before: {:.17e}", energy_before);
    println!("State before (first 3 links):");
    for i in 0..3 {
        println!("  Link {}: z={:.17e}, theta={:.17e}", 
            i, state_before[i].z, state_before[i].theta);
    }
    
    // Manually perform the metropolis step with full tracing
    let link_idx = rng.gen_range(0..graph.links.len());
    let do_z_update = 0.3 > 0.0 && rng.gen_bool(0.5);
    
    println!("\nMove details:");
    println!("link_idx: {}", link_idx);
    println!("do_z_update: {}", do_z_update);
    
    let info = if do_z_update {
        trace_z_step(&mut graph, link_idx, beta, &mut rng)
    } else {
        trace_theta_step(&mut graph, link_idx, alpha, &mut rng)
    };
    
    let energy_after = graph.action(alpha, beta);
    let state_after = graph.links.clone();
    
    println!("\nResults:");
    println!("StepInfo.accept: {}", info.accept);
    println!("Energy after: {:.17e}", energy_after);
    println!("External ΔE: {:.17e}", energy_after - energy_before);
    
    println!("\nState after (first 3 links):");
    for i in 0..3 {
        println!("  Link {}: z={:.17e}, theta={:.17e}", 
            i, state_after[i].z, state_after[i].theta);
    }
    
    println!("\nState changes:");
    let mut any_changes = false;
    for i in 0..state_before.len() {
        if state_before[i].z != state_after[i].z || state_before[i].theta != state_after[i].theta {
            println!("  Link {} changed: z {:.17e}->{:.17e}, theta {:.17e}->{:.17e}",
                i, state_before[i].z, state_after[i].z, 
                state_before[i].theta, state_after[i].theta);
            any_changes = true;
        }
    }
    if !any_changes {
        println!("  NO STATE CHANGES");
    }
    
    // The critical bug check
    if (energy_after - energy_before) == 0.0 && !info.accept {
        println!("\n🚨 BUG CONFIRMED: ΔE=0 but move rejected!");
    } else if (energy_after - energy_before) == 0.0 && info.accept {
        println!("\n🤔 STRANGE: ΔE=0 and move accepted, but external calc shows no change");
    } else {
        println!("\n✓ Normal case: ΔE≠0");
    }
}

fn trace_z_step(graph: &mut FastGraph, link_idx: usize, beta: f64, rng: &mut Pcg64) -> scan::graph_fast::StepInfo {
    println!("\n🔍 Z-UPDATE TRACE:");
    
    let link = &graph.links[link_idx];
    let old_z = link.z;
    let old_exp_neg_z = link.exp_neg_z;
    
    let d_z = rng.gen_range(-0.3..=0.3);
    let new_z = (old_z + d_z).max(0.001);
    
    println!("old_z: {:.17e}", old_z);
    println!("d_z: {:.17e}", d_z);
    println!("new_z: {:.17e}", new_z);
    println!("z_diff: {:.17e}", (new_z - old_z).abs());
    
    // Check for no-op due to boundary clamping
    if (new_z - old_z).abs() < 1e-15 {
        println!("→ NO-OP: returning accept=true");
        scan::graph_fast::StepInfo {
            accept: true,
            delta_w: 0.0,
            delta_cos: 0.0,
        }
    } else {
        let new_exp_neg_z = (-new_z).exp();
        
        // Fast entropy change calculation
        let delta_entropy = (-new_z * new_exp_neg_z) - (-old_z * old_exp_neg_z);
        let delta_s = beta * delta_entropy;
        
        println!("delta_entropy: {:.17e}", delta_entropy);
        println!("delta_s: {:.17e}", delta_s);
        println!("|delta_s|: {:.17e}", delta_s.abs());
        
        const EPSILON: f64 = 1e-6;
        let accept = if delta_s.abs() <= EPSILON { 
            println!("→ ACCEPT: |delta_s| <= epsilon");
            true 
        } else { 
            let prob = (-delta_s).exp();
            let rand_val = rng.gen_range(0.0..1.0);
            let accept_prob = rand_val < prob;
            println!("→ PROBABILISTIC: exp(-delta_s)={:.6e}, rand={:.6e}, accept={}", 
                prob, rand_val, accept_prob);
            accept_prob
        };
        
        if accept {
            println!("→ UPDATING STATE: new_z={:.17e}", new_z);
            graph.links[link_idx].update_z(new_z);
            scan::graph_fast::StepInfo {
                accept: true,
                delta_w: new_exp_neg_z - old_exp_neg_z,
                delta_cos: 0.0,
            }
        } else {
            println!("→ REJECTING: no state change");
            scan::graph_fast::StepInfo {
                accept: false,
                delta_w: 0.0,
                delta_cos: 0.0,
            }
        }
    }
}

fn trace_theta_step(graph: &mut FastGraph, link_idx: usize, alpha: f64, rng: &mut Pcg64) -> scan::graph_fast::StepInfo {
    println!("\n🔍 THETA-UPDATE TRACE:");
    
    let link = &graph.links[link_idx];
    let old_theta = link.theta;
    let old_cos_theta = link.cos_theta;
    
    let d_theta = rng.gen_range(-0.3..=0.3);
    let new_theta = old_theta + d_theta;
    
    println!("old_theta: {:.17e}", old_theta);
    println!("d_theta: {:.17e}", d_theta);
    println!("new_theta: {:.17e}", new_theta);
    println!("theta_diff: {:.17e}", (new_theta - old_theta).abs());
    
    // Check for no-op due to very small theta change
    if (new_theta - old_theta).abs() < 1e-15 {
        println!("→ NO-OP: returning accept=true");
        scan::graph_fast::StepInfo {
            accept: true,
            delta_w: 0.0,
            delta_cos: 0.0,
        }
    } else {
        // Calculate triangle sum change BEFORE applying the move
        let delta_triangle = calculate_triangle_delta(graph, link_idx, new_theta);
        let delta_s = alpha * delta_triangle;
        
        println!("delta_triangle: {:.17e}", delta_triangle);
        println!("delta_s: {:.17e}", delta_s);
        println!("|delta_s|: {:.17e}", delta_s.abs());
        
        const EPSILON: f64 = 1e-6;
        let accept = if delta_s.abs() <= EPSILON { 
            println!("→ ACCEPT: |delta_s| <= epsilon");
            true 
        } else { 
            let prob = (-delta_s).exp();
            let rand_val = rng.gen_range(0.0..1.0);
            let accept_prob = rand_val < prob;
            println!("→ PROBABILISTIC: exp(-delta_s)={:.6e}, rand={:.6e}, accept={}", 
                prob, rand_val, accept_prob);
            accept_prob
        };
        
        if accept {
            println!("→ UPDATING STATE: new_theta={:.17e}", new_theta);
            graph.links[link_idx].update_theta(new_theta);
            let delta_cos = new_theta.cos() - old_cos_theta;
            
            scan::graph_fast::StepInfo {
                accept: true,
                delta_w: 0.0,
                delta_cos,
            }
        } else {
            println!("→ REJECTING: no state change");
            scan::graph_fast::StepInfo {
                accept: false,
                delta_w: 0.0,
                delta_cos: 0.0,
            }
        }
    }
}

fn calculate_triangle_delta(graph: &FastGraph, link_idx: usize, new_theta: f64) -> f64 {
    let link = &graph.links[link_idx];
    let (i, j) = (link.i as usize, link.j as usize);
    let old_theta = link.theta;
    let delta_theta = new_theta - old_theta;
    
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
                -3.0 * old_total.sin() * delta_theta
            } else {
                let new_total = new_theta + other_sum;
                3.0 * (new_total.cos() - old_total.cos())
            };
            
            contributions.push(contribution);
        }
    }
    
    // Kahan summation
    let mut sum = 0.0;
    let mut c = 0.0;
    for &val in &contributions {
        let y = val - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}