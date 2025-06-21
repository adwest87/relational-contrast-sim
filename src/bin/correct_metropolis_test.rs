// CORRECT test of Metropolis criterion using proposed energy changes
use scan::graph_fast::FastGraph;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

fn main() {
    println!("🧪 CORRECT METROPOLIS CRITERION TEST");
    println!("====================================");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(42);
    let mut graph = FastGraph::new(n, 123);
    
    let mut proposed_zero_rejects = 0;
    let mut proposed_negative_rejects = 0;
    let mut total_moves = 0;
    
    for _ in 0..10000 {
        // Calculate proposed energy change BEFORE applying move
        let (proposed_delta_e, info) = metropolis_step_with_proposed_energy(&mut graph, alpha, beta, &mut rng);
        
        total_moves += 1;
        
        // Check CORRECT Metropolis criterion:
        // If PROPOSED ΔE ≤ 0, the move must be accepted
        if proposed_delta_e <= 0.0 && !info.accept {
            if proposed_delta_e == 0.0 {
                proposed_zero_rejects += 1;
                if proposed_zero_rejects <= 3 {
                    println!("❌ VIOLATION #{}: Proposed ΔE=0 rejected", proposed_zero_rejects);
                    println!("   Proposed ΔE = {:.16}", proposed_delta_e);
                }
            } else {
                proposed_negative_rejects += 1;
                if proposed_negative_rejects <= 3 {
                    println!("❌ VIOLATION #{}: Proposed ΔE<0 rejected", proposed_negative_rejects);
                    println!("   Proposed ΔE = {:.16}", proposed_delta_e);
                }
            }
        }
    }
    
    println!("\n📊 CORRECT METROPOLIS CRITERION RESULTS:");
    println!("Proposed ΔE=0 moves rejected: {}", proposed_zero_rejects);
    println!("Proposed ΔE<0 moves rejected: {}", proposed_negative_rejects);
    println!("Total moves: {}", total_moves);
    println!("Violation rate: {:.2}%", 
        100.0 * (proposed_zero_rejects + proposed_negative_rejects) as f64 / total_moves as f64);
    
    if proposed_zero_rejects + proposed_negative_rejects == 0 {
        println!("✅ Metropolis criterion is CORRECT!");
    } else {
        println!("❌ Metropolis criterion violations found!");
    }
}

// Modified metropolis step that returns the proposed energy change
fn metropolis_step_with_proposed_energy(
    graph: &mut FastGraph, 
    alpha: f64, 
    beta: f64, 
    rng: &mut Pcg64
) -> (f64, scan::graph_fast::StepInfo) {
    
    let link_idx = rng.gen_range(0..graph.links.len());
    let do_z_update = 0.3 > 0.0 && rng.gen_bool(0.5);
    
    if do_z_update {
        // Z-update
        let link = &graph.links[link_idx];
        let old_z = link.z;
        let old_exp_neg_z = link.exp_neg_z;
        let d_z = rng.gen_range(-0.3..=0.3);
        let new_z = (old_z + d_z).max(0.001);
        
        // Check for no-op due to boundary clamping
        if (new_z - old_z).abs() < 1e-15 {
            // True no-op: proposed ΔE = 0
            let info = scan::graph_fast::StepInfo {
                accept: true,
                delta_w: 0.0,
                delta_cos: 0.0,
            };
            (0.0, info)
        } else {
            let new_exp_neg_z = (-new_z).exp();
            
            // Calculate PROPOSED energy change
            let delta_entropy = (-new_z * new_exp_neg_z) - (-old_z * old_exp_neg_z);
            let proposed_delta_e = beta * delta_entropy;
            
            const EPSILON: f64 = 1e-6;
            let accept = if proposed_delta_e.abs() <= EPSILON { 
                true 
            } else { 
                rng.gen_range(0.0..1.0) < (-proposed_delta_e).exp() 
            };
            
            if accept {
                graph.links[link_idx].update_z(new_z);
                let info = scan::graph_fast::StepInfo {
                    accept: true,
                    delta_w: new_exp_neg_z - old_exp_neg_z,
                    delta_cos: 0.0,
                };
                (proposed_delta_e, info)
            } else {
                let info = scan::graph_fast::StepInfo {
                    accept: false,
                    delta_w: 0.0,
                    delta_cos: 0.0,
                };
                (proposed_delta_e, info)
            }
        }
    } else {
        // Phase update
        let link = &graph.links[link_idx];
        let old_theta = link.theta;
        let old_cos_theta = link.cos_theta;
        let d_theta = rng.gen_range(-0.3..=0.3);
        let new_theta = old_theta + d_theta;
        
        // Check for no-op due to very small theta change
        if (new_theta - old_theta).abs() < 1e-15 {
            // True no-op: proposed ΔE = 0
            let info = scan::graph_fast::StepInfo {
                accept: true,
                delta_w: 0.0,
                delta_cos: 0.0,
            };
            (0.0, info)
        } else {
            // Calculate PROPOSED triangle sum change
            let delta_triangle = calculate_triangle_delta(graph, link_idx, new_theta);
            let proposed_delta_e = alpha * delta_triangle;
            
            const EPSILON: f64 = 1e-6;
            let accept = if proposed_delta_e.abs() <= EPSILON { 
                true 
            } else { 
                rng.gen_range(0.0..1.0) < (-proposed_delta_e).exp() 
            };
            
            if accept {
                graph.links[link_idx].update_theta(new_theta);
                let delta_cos = new_theta.cos() - old_cos_theta;
                
                let info = scan::graph_fast::StepInfo {
                    accept: true,
                    delta_w: 0.0,
                    delta_cos,
                };
                (proposed_delta_e, info)
            } else {
                let info = scan::graph_fast::StepInfo {
                    accept: false,
                    delta_w: 0.0,
                    delta_cos: 0.0,
                };
                (proposed_delta_e, info)
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