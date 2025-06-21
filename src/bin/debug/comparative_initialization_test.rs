// Comparative initialization test to verify ergodicity
use scan::graph_fast::{FastGraph, BatchedObservables};
use rand::SeedableRng;
use std::fs::File;
use std::io::Write;
use std::f64::consts::PI;

fn main() {
    println!("🔄 COMPARATIVE INITIALIZATION TEST");
    println!("==================================");
    println!("Testing ergodicity at high temperature T=10");
    println!("Three different initial conditions should converge to same <cos θ>");
    
    let n = 32;
    let alpha = 1.5;
    let temperature = 10.0;
    let beta = 1.0 / temperature;
    let total_steps = 1_000_000;
    let measure_interval = 1000; // Measure every 1000 steps
    let delta_z = 0.3;
    let delta_theta = 0.3;
    
    println!("System: N={}, T={:.1}, α={:.1}", n, temperature, alpha);
    println!("Total steps: {}, measuring every {} steps", total_steps, measure_interval);
    
    // Initialize three systems with different starting conditions
    let mut systems = vec![
        ("Ordered (θ=0)", create_ordered_system(n, 0.0)),
        ("Anti-ordered (θ=π)", create_ordered_system(n, PI)),
        ("Random", create_random_system(n)),
    ];
    
    // Prepare CSV file for plotting
    let mut csv_file = File::create("initialization_convergence.csv").expect("Failed to create CSV file");
    writeln!(csv_file, "step,ordered_cos_theta,anti_ordered_cos_theta,random_cos_theta").unwrap();
    
    // Storage for final analysis
    let mut trajectories = vec![Vec::new(); 3];
    
    println!("\nInitial states:");
    for (i, (name, (graph, _))) in systems.iter().enumerate() {
        let mut obs_calc = BatchedObservables::new();
        let obs = obs_calc.measure(graph, alpha, beta);
        println!("  {}: <cos θ> = {:.6}", name, obs.mean_cos);
        trajectories[i].push((0, obs.mean_cos));
    }
    
    println!("\nRunning simulation...");
    
    // Run simulation for all systems simultaneously
    for step in 1..=total_steps {
        // Evolve all systems one step
        for (_, (graph, rng)) in systems.iter_mut() {
            graph.metropolis_step(alpha, beta, delta_z, delta_theta, rng);
        }
        
        // Measure periodically
        if step % measure_interval == 0 {
            let mut cos_theta_values = Vec::new();
            
            for (i, (name, (graph, _))) in systems.iter().enumerate() {
                let mut obs_calc = BatchedObservables::new();
                let obs = obs_calc.measure(graph, alpha, beta);
                cos_theta_values.push(obs.mean_cos);
                trajectories[i].push((step, obs.mean_cos));
            }
            
            // Write to CSV
            writeln!(csv_file, "{},{:.6},{:.6},{:.6}", 
                step, cos_theta_values[0], cos_theta_values[1], cos_theta_values[2]).unwrap();
            
            // Progress update
            if step % (total_steps / 10) == 0 {
                println!("  Step {}: Ordered={:.4}, Anti-ordered={:.4}, Random={:.4}", 
                    step, cos_theta_values[0], cos_theta_values[1], cos_theta_values[2]);
            }
        }
    }
    
    csv_file.flush().unwrap();
    
    // Final analysis
    println!("\n📊 FINAL ANALYSIS:");
    let final_values: Vec<f64> = systems.iter().map(|(name, (graph, _))| {
        let mut obs_calc = BatchedObservables::new();
        let obs = obs_calc.measure(graph, alpha, beta);
        println!("  {}: Final <cos θ> = {:.6}", name, obs.mean_cos);
        obs.mean_cos
    }).collect();
    
    // Check convergence
    let mean_final = final_values.iter().sum::<f64>() / final_values.len() as f64;
    let max_deviation = final_values.iter()
        .map(|&val| (val - mean_final).abs())
        .fold(0.0, f64::max);
    
    println!("\nConvergence analysis:");
    println!("  Mean final value: {:.6}", mean_final);
    println!("  Max deviation: {:.6}", max_deviation);
    println!("  Expected at T={}: ~0.0 ± 0.1", temperature);
    
    // Ergodicity test
    let convergence_threshold = 0.1; // Allow for statistical fluctuations
    if max_deviation < convergence_threshold {
        println!("✅ ERGODICITY TEST PASSED");
        println!("   All initial conditions converged to the same equilibrium value");
    } else {
        println!("❌ ERGODICITY TEST FAILED");
        println!("   Initial conditions did not converge - possible ergodicity breaking");
    }
    
    // Test for expected high-temperature behavior
    if mean_final.abs() < 0.2 {
        println!("✅ HIGH TEMPERATURE BEHAVIOR CORRECT");
        println!("   Order parameter is small as expected at T={}", temperature);
    } else {
        println!("⚠️ HIGH TEMPERATURE BEHAVIOR QUESTIONABLE");
        println!("   Order parameter |M|={:.3} is large for T={}", mean_final.abs(), temperature);
    }
    
    // Calculate equilibration times
    println!("\n⏱️ EQUILIBRATION ANALYSIS:");
    for (i, (name, _)) in systems.iter().enumerate() {
        let eq_time = estimate_equilibration_time(&trajectories[i], final_values[i]);
        println!("  {}: Equilibration time ≈ {} steps", name, eq_time);
    }
    
    // Autocorrelation analysis on final portions
    println!("\n📈 AUTOCORRELATION ANALYSIS:");
    for (i, (name, _)) in systems.iter().enumerate() {
        let final_portion: Vec<f64> = trajectories[i].iter()
            .skip(trajectories[i].len() / 2) // Use second half
            .map(|(_, val)| *val)
            .collect();
        
        let autocorr_time = estimate_autocorrelation_time(&final_portion);
        println!("  {}: Autocorrelation time ≈ {:.1} measurements", name, autocorr_time);
    }
    
    println!("\n📄 Results written to 'initialization_convergence.csv'");
    println!("Plot with: python -c \"");
    println!("import pandas as pd; import matplotlib.pyplot as plt");
    println!("df = pd.read_csv('initialization_convergence.csv')");
    println!("plt.figure(figsize=(10,6))");
    println!("plt.plot(df['step'], df['ordered_cos_theta'], label='Ordered (θ=0)', alpha=0.8)");
    println!("plt.plot(df['step'], df['anti_ordered_cos_theta'], label='Anti-ordered (θ=π)', alpha=0.8)");
    println!("plt.plot(df['step'], df['random_cos_theta'], label='Random', alpha=0.8)");
    println!("plt.xlabel('Monte Carlo Steps'); plt.ylabel('<cos θ>')");
    println!("plt.title('Ergodicity Test: Convergence from Different Initial Conditions')");
    println!("plt.legend(); plt.grid(True, alpha=0.3); plt.show()\"");
}

fn create_ordered_system(n: usize, theta_value: f64) -> (FastGraph, rand_pcg::Pcg64) {
    let mut graph = FastGraph::new(n, 12345);
    let rng = rand_pcg::Pcg64::seed_from_u64(42);
    
    // Set all angles to the specified value
    for i in 0..graph.links.len() {
        graph.links[i].update_theta(theta_value);
    }
    
    (graph, rng)
}

fn create_random_system(n: usize) -> (FastGraph, rand_pcg::Pcg64) {
    // The new initialization already creates random angles
    let graph = FastGraph::new(n, 999);
    let rng = rand_pcg::Pcg64::seed_from_u64(123);
    
    (graph, rng)
}

fn estimate_equilibration_time(trajectory: &[(usize, f64)], final_value: f64) -> usize {
    let tolerance = 0.05; // 5% of final value
    let threshold = tolerance;
    
    // Find when the system gets within threshold of final value and stays there
    let mut equilibrated_start = trajectory.len();
    
    for i in (0..trajectory.len()).rev() {
        let (step, value) = trajectory[i];
        if (value - final_value).abs() > threshold {
            equilibrated_start = i + 1;
            break;
        }
    }
    
    if equilibrated_start < trajectory.len() {
        trajectory[equilibrated_start].0
    } else {
        0 // Equilibrated from the start
    }
}

fn estimate_autocorrelation_time(data: &[f64]) -> f64 {
    if data.len() < 10 {
        return 1.0;
    }
    
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    
    if variance < 1e-10 {
        return 1.0; // No fluctuations
    }
    
    let mut autocorr_sum = 0.5; // C(0) contributes 0.5
    let max_lag = (data.len() / 4).min(50); // Don't go beyond 1/4 of data or 50 lags
    
    for lag in 1..max_lag {
        let mut correlation = 0.0;
        let count = data.len() - lag;
        
        for i in 0..count {
            correlation += (data[i] - mean) * (data[i + lag] - mean);
        }
        correlation /= count as f64;
        
        let normalized_corr = correlation / variance;
        
        // Stop when correlation becomes negligible
        if normalized_corr.abs() < 0.05 {
            break;
        }
        
        autocorr_sum += normalized_corr;
    }
    
    2.0 * autocorr_sum
}