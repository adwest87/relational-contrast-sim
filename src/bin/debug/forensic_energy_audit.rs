// Forensic energy audit for Monte Carlo detailed balance violations
// This will instrument every energy calculation and track discrepancies

use scan::{graph::Graph, graph_fast::FastGraph};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("🔍 FORENSIC ENERGY AUDIT");
    println!("========================");
    
    // Test 1: Energy calculation consistency
    test_energy_calculation_consistency();
    
    // Test 2: Metropolis acceptance audit
    test_metropolis_acceptance_audit();
    
    // Test 3: Energy conservation during moves
    test_energy_conservation_moves();
    
    // Test 4: Detailed balance verification
    test_detailed_balance_rigorous();
}

fn test_energy_calculation_consistency() {
    println!("\n🧮 TEST 1: Energy Calculation Consistency");
    
    let n = 6; // Small system for exact verification
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(12345);
    
    // Create identical initial states
    let graph_basic = Graph::complete_random_with(&mut rng, n);
    let graph_fast = FastGraph::from_graph(&graph_basic);
    
    // Test energy consistency
    let energy_basic = graph_basic.action(alpha, beta);
    let energy_fast = graph_fast.action(alpha, beta);
    
    println!("Basic graph energy: {:.12}", energy_basic);
    println!("Fast graph energy:  {:.12}", energy_fast);
    println!("Difference: {:.2e}", (energy_basic - energy_fast).abs());
    
    if (energy_basic - energy_fast).abs() > 1e-10 {
        println!("❌ CRITICAL: Energy calculation inconsistency!");
    } else {
        println!("✅ Energy calculations consistent");
    }
    
    // Test individual components
    let entropy_basic = graph_basic.entropy_action();
    let entropy_fast = graph_fast.entropy_action();
    let triangle_basic = graph_basic.triangle_sum();
    let triangle_fast = graph_fast.triangle_sum();
    
    println!("\nComponent comparison:");
    println!("Entropy - Basic: {:.12}, Fast: {:.12}, Diff: {:.2e}", 
        entropy_basic, entropy_fast, (entropy_basic - entropy_fast).abs());
    println!("Triangle - Basic: {:.12}, Fast: {:.12}, Diff: {:.2e}", 
        triangle_basic, triangle_fast, (triangle_basic - triangle_fast).abs());
}

fn test_metropolis_acceptance_audit() {
    println!("\n⚖️ TEST 2: Metropolis Acceptance Audit");
    
    let n = 4;
    let alpha = 1.5;
    let beta = 3.0;
    let mut rng = Pcg64::seed_from_u64(99999);
    let mut graph = FastGraph::new(n, 12345);
    
    let mut audit_log = File::create("metropolis_audit.log").unwrap();
    writeln!(audit_log, "step,move_type,energy_before,energy_after,delta_energy,expected_prob,random,accepted,correct").unwrap();
    
    let mut errors = 0;
    let n_tests = 1000;
    
    for step in 0..n_tests {
        // Save state
        let links_before = graph.links.clone();
        let energy_before = graph.action(alpha, beta);
        
        // Attempt move
        let info = graph.metropolis_step(alpha, beta, 0.3, 0.3, &mut rng);
        
        // Calculate actual energy after
        let energy_after = graph.action(alpha, beta);
        let delta_energy = energy_after - energy_before;
        
        // Calculate expected acceptance probability
        let expected_prob = if delta_energy <= 0.0 {
            1.0
        } else {
            (-beta * delta_energy).exp()
        };
        
        // Check if acceptance decision was correct
        // We can't know the exact random number used, but we can verify logic
        let move_type = if info.accept { "accepted" } else { "rejected" };
        let correct = if delta_energy <= 0.0 {
            info.accept // Should always accept
        } else {
            true // Can't verify random decision, but log for analysis
        };
        
        if !correct {
            errors += 1;
            println!("❌ Error at step {}: ΔE={:.6}, should accept but didn't", step, delta_energy);
        }
        
        writeln!(audit_log, "{},{},{:.12},{:.12},{:.12},{:.6},unknown,{},{}", 
            step, move_type, energy_before, energy_after, delta_energy, expected_prob, info.accept, correct).unwrap();
        
        // Verify energy conservation if move rejected
        if !info.accept {
            let energy_after_reject = graph.action(alpha, beta);
            if (energy_after_reject - energy_before).abs() > 1e-12 {
                println!("❌ Energy not conserved on rejection! ΔE={:.2e}", 
                    energy_after_reject - energy_before);
                errors += 1;
            }
        }
    }
    
    println!("Metropolis audit: {}/{} errors ({:.2}%)", errors, n_tests, 100.0 * errors as f64 / n_tests as f64);
    println!("Detailed log written to metropolis_audit.log");
}

fn test_energy_conservation_moves() {
    println!("\n🔋 TEST 3: Energy Conservation During Moves");
    
    let n = 6;
    let alpha = 1.5;
    let beta = 100.0; // High beta for mostly rejected moves
    let mut rng = Pcg64::seed_from_u64(777);
    let mut graph = FastGraph::new(n, 555);
    
    let mut energy_log = File::create("energy_conservation.log").unwrap();
    writeln!(energy_log, "step,energy_before,energy_after,delta_computed,delta_actual,drift,accepted").unwrap();
    
    let initial_energy = graph.action(alpha, beta);
    let mut cumulative_drift = 0.0;
    let mut max_drift: f64 = 0.0;
    let mut total_accepts = 0;
    
    println!("Initial energy: {:.12}", initial_energy);
    
    for step in 0..10000 {
        let energy_before = graph.action(alpha, beta);
        
        // Store state for delta calculation
        let links_before = graph.links.clone();
        
        // Make move
        let info = graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
        
        let energy_after = graph.action(alpha, beta);
        let delta_actual = energy_after - energy_before;
        
        // Calculate expected delta from first principles if accepted
        let delta_computed = if info.accept {
            graph.links = links_before.clone();
            let energy_restored = graph.action(alpha, beta);
            let delta_expected = energy_after - energy_restored;
            graph.links = links_before; // Restore state for next iteration
            delta_expected
        } else {
            0.0 // Should be zero for rejected moves
        };
        
        let drift_from_initial = energy_after - initial_energy;
        cumulative_drift += delta_actual;
        max_drift = max_drift.max(drift_from_initial.abs());
        
        if info.accept {
            total_accepts += 1;
        }
        
        // Check for violations
        if !info.accept && delta_actual.abs() > 1e-12 {
            println!("❌ Energy changed on REJECTED move at step {}: ΔE={:.2e}", step, delta_actual);
        }
        
        if step % 1000 == 0 {
            writeln!(energy_log, "{},{:.12},{:.12},{:.12},{:.12},{:.12},{}", 
                step, energy_before, energy_after, delta_computed, delta_actual, drift_from_initial, info.accept).unwrap();
        }
        
        if step % 2000 == 0 && step > 0 {
            println!("Step {}: Energy={:.8}, Drift={:.2e}, Max_drift={:.2e}, Accept_rate={:.1}%", 
                step, energy_after, cumulative_drift, max_drift, 100.0 * total_accepts as f64 / step as f64);
        }
    }
    
    println!("Final energy drift: {:.2e}", cumulative_drift);
    println!("Maximum absolute drift: {:.2e}", max_drift);
    println!("Acceptance rate: {:.1}%", 100.0 * total_accepts as f64 / 10000.0);
}

fn test_detailed_balance_rigorous() {
    println!("\n⚖️ TEST 4: Rigorous Detailed Balance Verification");
    
    let n = 4; // Small system for exhaustive testing
    let alpha = 1.0;
    let beta = 2.0;
    let mut rng = Pcg64::seed_from_u64(2024);
    let mut graph = FastGraph::new(n, 1111);
    
    // Equilibrate first
    for _ in 0..5000 {
        graph.metropolis_step(alpha, beta, 0.3, 0.3, &mut rng);
    }
    
    let mut balance_log = File::create("detailed_balance.log").unwrap();
    writeln!(balance_log, "test,config_i,config_j,energy_i,energy_j,delta_E,attempts_ij,accepts_ij,attempts_ji,accepts_ji,ratio_measured,ratio_expected,error").unwrap();
    
    let mut violations = 0;
    let n_tests = 100;
    
    for test in 0..n_tests {
        // Get current configuration i
        let config_i = graph.links.clone();
        let energy_i = graph.action(alpha, beta);
        
        // Try to make a move to get configuration j
        let mut attempts_ij = 0;
        let mut accepts_ij = 0;
        let mut config_j = None;
        let mut energy_j = 0.0;
        
        // Attempt moves until we get an accepted one
        for _ in 0..100 {
            graph.links = config_i.clone();
            let info = graph.metropolis_step(alpha, beta, 0.5, 0.5, &mut rng);
            attempts_ij += 1;
            
            if info.accept {
                config_j = Some(graph.links.clone());
                energy_j = graph.action(alpha, beta);
                accepts_ij += 1;
                break;
            }
        }
        
        if let Some(config_j_state) = config_j {
            // Now test reverse move j -> i
            let mut attempts_ji = 0;
            let mut accepts_ji = 0;
            
            for _ in 0..1000 {
                graph.links = config_j_state.clone();
                let info = graph.metropolis_step(alpha, beta, 0.5, 0.5, &mut rng);
                attempts_ji += 1;
                
                // Check if we got back to configuration i (or close enough)
                if info.accept {
                    let current_energy = graph.action(alpha, beta);
                    if (current_energy - energy_i).abs() < 1e-10 {
                        accepts_ji += 1;
                    }
                }
            }
            
            if attempts_ji > 0 && accepts_ij > 0 {
                let ratio_measured = (accepts_ij as f64 / attempts_ij as f64) / (accepts_ji as f64 / attempts_ji as f64);
                let delta_E = energy_j - energy_i;
                let ratio_expected = (-beta * delta_E).exp();
                let error = (ratio_measured - ratio_expected).abs();
                
                writeln!(balance_log, "{},config_i,config_j,{:.12},{:.12},{:.12},{},{},{},{},{:.6},{:.6},{:.2e}", 
                    test, energy_i, energy_j, delta_E, 
                    attempts_ij, accepts_ij, attempts_ji, accepts_ji, ratio_measured, ratio_expected, error).unwrap();
                
                if error > 0.1 {
                    violations += 1;
                    println!("❌ Detailed balance violation #{}: ratio={:.3}, expected={:.3}, error={:.2e}", 
                        violations, ratio_measured, ratio_expected, error);
                }
            }
        }
    }
    
    println!("Detailed balance violations: {}/{} ({:.1}%)", violations, n_tests, 100.0 * violations as f64 / n_tests as f64);
    println!("Detailed analysis written to detailed_balance.log");
}