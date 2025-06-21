// Comprehensive temperature sweep to find the critical point
use scan::graph_fast::{FastGraph, BatchedObservables};
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("🌡️ COMPREHENSIVE TEMPERATURE SWEEP");
    println!("===================================");
    
    let n = 32; // Medium size for good statistics
    let alpha = 1.5; // Fixed coupling
    
    // Temperature range: 0.1 to 10.0 in steps of 0.1
    let temperatures: Vec<f64> = (1..=100).map(|i| i as f64 * 0.1).collect();
    
    let mut results = Vec::new();
    
    println!("Running temperature sweep from T=0.1 to T=10.0...");
    println!("System: N={}, α={}", n, alpha);
    println!("Each point: 10^5 equilibration + 10^5 production steps");
    
    let mut csv_file = File::create("temperature_sweep.csv").expect("Failed to create CSV file");
    writeln!(csv_file, "T,beta,mean_cos_theta,abs_mean_cos_theta,susceptibility,specific_heat,binder_cumulant,acceptance_rate").unwrap();
    
    for (i, &temperature) in temperatures.iter().enumerate() {
        let beta = 1.0 / temperature;
        
        print!("T={:.1} (β={:.3}) [{}/{}]... ", temperature, beta, i+1, temperatures.len());
        std::io::stdout().flush().unwrap();
        
        // Run simulation
        let mut rng = Pcg64::seed_from_u64(42 + i as u64);
        let mut graph = FastGraph::new(n, 12345 + i as u64);
        let mut observable_calc = BatchedObservables::new();
        
        // Optimize move sizes for this temperature
        let (delta_z, delta_theta) = optimize_move_sizes(temperature);
        
        // Equilibration
        let eq_steps = 100_000;
        for _ in 0..eq_steps {
            graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
        }
        
        // Production run with measurements
        let prod_steps = 100_000;
        let measure_interval = 100;
        let mut measurements = Vec::new();
        let mut cos_theta_values = Vec::new();
        let mut energy_values = Vec::new();
        let mut accepted_moves = 0;
        
        for step in 0..prod_steps {
            let info = graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
            if info.accept {
                accepted_moves += 1;
            }
            
            if step % measure_interval == 0 {
                let obs = observable_calc.measure(&graph, alpha, beta);
                measurements.push(obs);
                cos_theta_values.push(obs.mean_cos);
                
                // Calculate energy for specific heat
                let entropy_action = graph.entropy_action();
                let triangle_sum = graph.triangle_sum();
                let energy = beta * entropy_action + alpha * triangle_sum;
                energy_values.push(energy);
            }
        }
        
        // Calculate observables
        let mean_cos_theta = cos_theta_values.iter().sum::<f64>() / cos_theta_values.len() as f64;
        let abs_mean_cos_theta = mean_cos_theta.abs();
        
        // Susceptibility (already calculated in measurements)
        let susceptibility = measurements.last().unwrap().susceptibility;
        
        // Specific heat from energy fluctuations
        let mean_energy = energy_values.iter().sum::<f64>() / energy_values.len() as f64;
        let energy_variance = energy_values.iter()
            .map(|e| (e - mean_energy).powi(2))
            .sum::<f64>() / energy_values.len() as f64;
        let specific_heat = beta * beta * energy_variance / (n as f64);
        
        // Binder cumulant U = 1 - <M^4>/(3<M^2>^2)
        let m2_values: Vec<f64> = cos_theta_values.iter().map(|&m| m * m).collect();
        let m4_values: Vec<f64> = cos_theta_values.iter().map(|&m| m.powi(4)).collect();
        let mean_m2 = m2_values.iter().sum::<f64>() / m2_values.len() as f64;
        let mean_m4 = m4_values.iter().sum::<f64>() / m4_values.len() as f64;
        let binder_cumulant = 1.0 - mean_m4 / (3.0 * mean_m2.powi(2));
        
        let acceptance_rate = accepted_moves as f64 / prod_steps as f64 * 100.0;
        
        let result = TempResult {
            temperature,
            beta,
            mean_cos_theta,
            abs_mean_cos_theta,
            susceptibility,
            specific_heat,
            binder_cumulant,
            acceptance_rate,
        };
        
        results.push(result);
        
        // Write to CSV
        writeln!(csv_file, "{:.1},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.2}",
            temperature, beta, mean_cos_theta, abs_mean_cos_theta, 
            susceptibility, specific_heat, binder_cumulant, acceptance_rate).unwrap();
        
        println!("✓ |M|={:.3}, χ={:.1}, C={:.3}, U={:.3}, acc={:.1}%", 
            abs_mean_cos_theta, susceptibility, specific_heat, binder_cumulant, acceptance_rate);
    }
    
    csv_file.flush().unwrap();
    
    // Analysis
    println!("\n📊 ANALYSIS:");
    
    // Find susceptibility peak
    let max_chi_result = results.iter()
        .max_by(|a, b| a.susceptibility.partial_cmp(&b.susceptibility).unwrap())
        .unwrap();
    
    println!("Susceptibility peak: T={:.1}, χ={:.1}", 
        max_chi_result.temperature, max_chi_result.susceptibility);
    
    // Find specific heat peak
    let max_cv_result = results.iter()
        .max_by(|a, b| a.specific_heat.partial_cmp(&b.specific_heat).unwrap())
        .unwrap();
    
    println!("Specific heat peak: T={:.1}, C={:.3}", 
        max_cv_result.temperature, max_cv_result.specific_heat);
    
    // Find Binder cumulant crossing (should be around 2/3 ≈ 0.67 for 3D Ising)
    let binder_target = 2.0/3.0;
    let binder_crossing = results.iter()
        .min_by(|a, b| {
            let diff_a = (a.binder_cumulant - binder_target).abs();
            let diff_b = (b.binder_cumulant - binder_target).abs();
            diff_a.partial_cmp(&diff_b).unwrap()
        })
        .unwrap();
    
    println!("Binder cumulant crossing (≈2/3): T={:.1}, U={:.3}", 
        binder_crossing.temperature, binder_crossing.binder_cumulant);
    
    // Check low temperature behavior
    let low_temp_results: Vec<_> = results.iter().filter(|r| r.temperature <= 1.0).collect();
    if !low_temp_results.is_empty() {
        let avg_low_temp_order = low_temp_results.iter()
            .map(|r| r.abs_mean_cos_theta)
            .sum::<f64>() / low_temp_results.len() as f64;
        println!("Low temperature (T≤1.0) average |M|: {:.3}", avg_low_temp_order);
        
        if avg_low_temp_order < 0.5 {
            println!("⚠️ WARNING: Low temperature order parameter is small!");
        }
    }
    
    // Check high temperature behavior
    let high_temp_results: Vec<_> = results.iter().filter(|r| r.temperature >= 8.0).collect();
    if !high_temp_results.is_empty() {
        let avg_high_temp_order = high_temp_results.iter()
            .map(|r| r.abs_mean_cos_theta)
            .sum::<f64>() / high_temp_results.len() as f64;
        println!("High temperature (T≥8.0) average |M|: {:.3}", avg_high_temp_order);
        
        if avg_high_temp_order > 0.1 {
            println!("⚠️ WARNING: High temperature order parameter is large!");
        }
    }
    
    println!("\n📈 Results written to 'temperature_sweep.csv'");
    println!("Estimated critical temperature: T_c ≈ {:.1}", max_chi_result.temperature);
}

#[derive(Debug, Clone)]
struct TempResult {
    temperature: f64,
    beta: f64,
    mean_cos_theta: f64,
    abs_mean_cos_theta: f64,
    susceptibility: f64,
    specific_heat: f64,
    binder_cumulant: f64,
    acceptance_rate: f64,
}

fn optimize_move_sizes(temperature: f64) -> (f64, f64) {
    // Adaptive move sizes based on temperature
    // At high T, need larger moves; at low T, smaller moves
    
    let base_delta_z = 0.5;
    let base_delta_theta = 0.5;
    
    // Scale with temperature, but not too extreme
    let temp_factor = (temperature / 3.0).min(2.0).max(0.2);
    
    (base_delta_z * temp_factor, base_delta_theta * temp_factor)
}