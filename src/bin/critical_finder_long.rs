// Extended critical point finder with much longer equilibration
// Uses hot start and multiple passes to ensure proper thermalization

use scan::graph::Graph;
use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::time::Instant;
use rayon::prelude::*;
use std::sync::Mutex;

#[derive(Debug, Clone)]
struct MeasurementPoint {
    beta: f64,
    alpha: f64,
    chi: f64,
    chi_error: f64,
    mean_cos: f64,
    mean_cos_error: f64,
    mean_w: f64,
    binder: f64,
    acceptance: f64,
}

fn main() {
    println!("=== EXTENDED CRITICAL POINT FINDER ===");
    let n = 48;  // System size
    
    // First, do a quick hot start to get a good initial configuration
    println!("Creating hot configuration near critical point...");
    let hot_config = create_hot_configuration(n);
    
    // Phase 1: Focused sweep near expected critical region
    let start_total = Instant::now();
    let focused_points = focused_sweep(n, &hot_config);
    
    // Find maximum chi
    let best_focused = focused_points.iter()
        .max_by(|a, b| a.chi.partial_cmp(&b.chi).unwrap())
        .unwrap();
    
    println!("\nBest point from focused sweep:");
    println!("  β = {:.3}, α = {:.3}", best_focused.beta, best_focused.alpha);
    println!("  χ = {:.1} ± {:.1}", best_focused.chi, best_focused.chi_error);
    println!("  <cos θ> = {:.3} ± {:.3}", best_focused.mean_cos, best_focused.mean_cos_error);
    println!("  <w> = {:.3}", best_focused.mean_w);
    
    // Phase 2: Ultra-fine scan around best point
    println!("\nUltra-fine scan around best point:");
    let ultra_fine = ultra_fine_scan(n, best_focused.beta, best_focused.alpha, &hot_config);
    
    let best_ultra = ultra_fine.iter()
        .max_by(|a, b| a.chi.partial_cmp(&b.chi).unwrap())
        .unwrap();
    
    println!("\nFinal critical point estimate:");
    println!("  β_c = {:.4}", best_ultra.beta);
    println!("  α_c = {:.4}", best_ultra.alpha);
    println!("  χ_max = {:.1} ± {:.1}", best_ultra.chi, best_ultra.chi_error);
    println!("  <cos θ> = {:.3} ± {:.3}", best_ultra.mean_cos, best_ultra.mean_cos_error);
    println!("  <w> = {:.3}", best_ultra.mean_w);
    println!("  Binder U4 = {:.3}", best_ultra.binder);
    println!("  Acceptance = {:.1}%", best_ultra.acceptance);
    
    // Ridge analysis
    let ridge_slope = analyze_ridge(&focused_points);
    println!("\nCritical ridge: α = {:.3}β + {:.3}", ridge_slope.0, ridge_slope.1);
    
    let total_time = start_total.elapsed();
    println!("\nTotal time: {:.1} seconds", total_time.as_secs_f64());
    
    // Check if we found reasonable values
    if best_ultra.chi < 10.0 {
        println!("\n⚠️  WARNING: χ values are still too low!");
        println!("   Expected χ ≈ 30-40 at critical point");
        println!("   This suggests:");
        println!("   1. Need even longer equilibration");
        println!("   2. Wrong parameter region");
        println!("   3. Possible bug in susceptibility calculation");
    }
}

fn create_hot_configuration(n: usize) -> FastGraph {
    println!("  Thermalizing at high temperature...");
    let mut rng = Pcg64::seed_from_u64(42);
    let mut graph = Graph::complete_random_with(&mut rng, n);
    
    // Initialize with intermediate values (not fully random)
    use rand::prelude::*;
    
    // Start with z values in critical region
    for link in &mut graph.links {
        link.z = 0.8 + rng.gen_range(-0.2..0.2);  // Around exp(-z) ≈ 0.45
        link.theta = rng.gen_range(-1.0..1.0);    // Partially ordered
    }
    
    let mut fast_graph = FastGraph::from_graph(&graph);
    
    // Thermalize at high T first
    let beta_hot = 2.0;
    let alpha_hot = 1.3;
    for _ in 0..10_000 {
        fast_graph.metropolis_step(alpha_hot, beta_hot, 0.5, 0.5, &mut rng);
    }
    
    // Cool down gradually
    for i in 0..10 {
        let beta = 2.0 + 0.09 * i as f64;  // 2.0 → 2.9
        let alpha = 1.3 + 0.02 * i as f64;  // 1.3 → 1.5
        for _ in 0..5_000 {
            fast_graph.metropolis_step(alpha, beta, 0.3, 0.3, &mut rng);
        }
    }
    
    println!("  Hot configuration ready");
    fast_graph
}

fn focused_sweep(n: usize, hot_config: &FastGraph) -> Vec<MeasurementPoint> {
    println!("\nFocused sweep in critical region:");
    let start = Instant::now();
    
    // Narrow grid around expected critical point
    let beta_values: Vec<f64> = (0..=8).map(|i| 2.88 + 0.01 * i as f64).collect();
    let alpha_values: Vec<f64> = (0..=8).map(|i| 1.48 + 0.01 * i as f64).collect();
    
    let results = Mutex::new(Vec::new());
    let total_points = beta_values.len() * alpha_values.len();
    let points_done = Mutex::new(0);
    
    beta_values.par_iter().for_each(|&beta| {
        for &alpha in &alpha_values {
            let point = long_measurement(n, beta, alpha, hot_config);
            
            let mut count = points_done.lock().unwrap();
            *count += 1;
            print!("\r  Progress: {}/{} points", *count, total_points);
            use std::io::{self, Write};
            io::stdout().flush().unwrap();
            drop(count);
            
            results.lock().unwrap().push(point);
        }
    });
    
    let elapsed = start.elapsed();
    println!("\r  Focused sweep: {} points in {:.1} seconds", total_points, elapsed.as_secs_f64());
    
    let mut final_results = results.into_inner().unwrap();
    final_results.sort_by(|a, b| b.chi.partial_cmp(&a.chi).unwrap());
    final_results
}

fn ultra_fine_scan(n: usize, beta_center: f64, alpha_center: f64, hot_config: &FastGraph) -> Vec<MeasurementPoint> {
    println!("  Ultra-fine scan around (β={:.3}, α={:.3})", beta_center, alpha_center);
    let start = Instant::now();
    
    // Very fine grid
    let beta_values: Vec<f64> = (-2..=2).map(|i| beta_center + 0.005 * i as f64).collect();
    let alpha_values: Vec<f64> = (-2..=2).map(|i| alpha_center + 0.005 * i as f64).collect();
    
    let results = Mutex::new(Vec::new());
    
    beta_values.par_iter().for_each(|&beta| {
        for &alpha in &alpha_values {
            let point = very_long_measurement(n, beta, alpha, hot_config);
            results.lock().unwrap().push(point);
        }
    });
    
    let elapsed = start.elapsed();
    println!("  Ultra-fine: 25 points in {:.1} seconds", elapsed.as_secs_f64());
    
    let mut final_results = results.into_inner().unwrap();
    final_results.sort_by(|a, b| b.chi.partial_cmp(&a.chi).unwrap());
    final_results
}

fn long_measurement(n: usize, beta: f64, alpha: f64, hot_config: &FastGraph) -> MeasurementPoint {
    let mut rng = Pcg64::seed_from_u64((beta * 10000.0 + alpha * 1000.0) as u64);
    
    // Start from hot configuration
    let mut graph = hot_config.clone();
    
    // Long equilibration with adaptive steps
    let equilibration = 100_000;
    let production = 100_000;
    let mut accepts = 0;
    let mut delta_z = 0.2;
    let mut delta_theta = 0.2;
    
    // Equilibration
    for i in 0..equilibration {
        let info = graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
        if info.accept {
            accepts += 1;
        }
        
        // Adapt step sizes
        if i > 0 && i % 5000 == 0 {
            let rate = accepts as f64 / i as f64;
            if rate > 0.55 {
                delta_z *= 1.05;
                delta_theta *= 1.05;
            } else if rate < 0.45 {
                delta_z *= 0.95;
                delta_theta *= 0.95;
            }
            delta_z = delta_z.clamp(0.1, 0.4);
            delta_theta = delta_theta.clamp(0.1, 0.4);
        }
    }
    
    // Production with measurements
    let mut measurements = Vec::new();
    let measure_interval = 100;
    
    for step in 0..production {
        let info = graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
        if info.accept {
            accepts += 1;
        }
        
        if step % measure_interval == 0 {
            let m = graph.m() as f64;
            let sum_cos: f64 = graph.links.iter().map(|l| l.cos_theta).sum();
            let sum_sin_sq: f64 = graph.links.iter().map(|l| l.sin_theta * l.sin_theta).sum();
            let sum_w: f64 = graph.links.iter().map(|l| l.exp_neg_z).sum();
            let sum_w_cos: f64 = graph.links.iter()
                .map(|l| l.exp_neg_z * l.cos_theta).sum();
            
            let mean_cos = sum_cos / m;
            let mean_sin_sq = sum_sin_sq / m;
            let mean_w = sum_w / m;
            let mean_w_cos = sum_w_cos / m;
            
            // Calculate cos² for susceptibility
            let sum_cos_sq: f64 = graph.links.iter().map(|l| l.cos_theta * l.cos_theta).sum();
            let mean_cos_sq = sum_cos_sq / m;
            
            // Correct susceptibility formula: χ = Nβ(<cos²θ> - <cosθ>²)
            let chi = n as f64 * beta * (mean_cos_sq - mean_cos * mean_cos);
            
            measurements.push((mean_cos, mean_sin_sq, chi, mean_w));
        }
    }
    
    // Calculate statistics
    let n_meas = measurements.len() as f64;
    let mean_cos = measurements.iter().map(|(c, _, _, _)| c).sum::<f64>() / n_meas;
    let mean_chi = measurements.iter().map(|(_, _, x, _)| x).sum::<f64>() / n_meas;
    let mean_w = measurements.iter().map(|(_, _, _, w)| w).sum::<f64>() / n_meas;
    
    // Errors
    let cos_var = measurements.iter()
        .map(|(c, _, _, _)| (c - mean_cos).powi(2))
        .sum::<f64>() / (n_meas - 1.0);
    let chi_var = measurements.iter()
        .map(|(_, _, x, _)| (x - mean_chi).powi(2))
        .sum::<f64>() / (n_meas - 1.0);
    
    let cos_error = (cos_var / n_meas).sqrt();
    let chi_error = (chi_var / n_meas).sqrt();
    
    // Binder cumulant
    let m2 = measurements.iter()
        .map(|(c, _, _, _)| c * c)
        .sum::<f64>() / n_meas;
    let m4 = measurements.iter()
        .map(|(c, _, _, _)| c.powi(4))
        .sum::<f64>() / n_meas;
    let binder = 1.0 - m4 / (3.0 * m2 * m2);
    
    let acceptance = 100.0 * accepts as f64 / (equilibration + production) as f64;
    
    MeasurementPoint {
        beta,
        alpha,
        chi: mean_chi,
        chi_error,
        mean_cos,
        mean_cos_error: cos_error,
        mean_w,
        binder,
        acceptance,
    }
}

fn very_long_measurement(n: usize, beta: f64, alpha: f64, hot_config: &FastGraph) -> MeasurementPoint {
    // Even longer for ultra-fine scan
    let mut rng = Pcg64::seed_from_u64((beta * 10000.0 + alpha * 1000.0 + 12345.0) as u64);
    let mut graph = hot_config.clone();
    
    // Very long equilibration
    let equilibration = 200_000;
    let production = 200_000;
    let mut accepts = 0;
    let mut delta_z = 0.2;
    let mut delta_theta = 0.2;
    
    // Two-stage equilibration
    // Stage 1: Fast equilibration
    for _ in 0..50_000 {
        let info = graph.metropolis_step(alpha, beta, 0.3, 0.3, &mut rng);
        if info.accept {
            accepts += 1;
        }
    }
    
    // Stage 2: Fine equilibration with adaptive steps
    for i in 0..equilibration {
        let info = graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
        if info.accept {
            accepts += 1;
        }
        
        if i > 0 && i % 10_000 == 0 {
            let rate = accepts as f64 / (50_000 + i) as f64;
            if rate > 0.52 {
                delta_z *= 1.02;
                delta_theta *= 1.02;
            } else if rate < 0.48 {
                delta_z *= 0.98;
                delta_theta *= 0.98;
            }
            delta_z = delta_z.clamp(0.15, 0.35);
            delta_theta = delta_theta.clamp(0.15, 0.35);
        }
    }
    
    // Long production
    let mut measurements = Vec::new();
    let measure_interval = 200;  // Less frequent for decorrelation
    
    for step in 0..production {
        let info = graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
        if info.accept {
            accepts += 1;
        }
        
        if step % measure_interval == 0 {
            let m = graph.m() as f64;
            let sum_cos: f64 = graph.links.iter().map(|l| l.cos_theta).sum();
            let sum_sin_sq: f64 = graph.links.iter().map(|l| l.sin_theta * l.sin_theta).sum();
            let sum_w: f64 = graph.links.iter().map(|l| l.exp_neg_z).sum();
            let sum_w_cos: f64 = graph.links.iter()
                .map(|l| l.exp_neg_z * l.cos_theta).sum();
            
            let mean_cos = sum_cos / m;
            let mean_sin_sq = sum_sin_sq / m;
            let mean_w = sum_w / m;
            let mean_w_cos = sum_w_cos / m;
            
            // Calculate cos² for susceptibility
            let sum_cos_sq: f64 = graph.links.iter().map(|l| l.cos_theta * l.cos_theta).sum();
            let mean_cos_sq = sum_cos_sq / m;
            
            // Correct susceptibility formula: χ = Nβ(<cos²θ> - <cosθ>²)
            let chi = n as f64 * beta * (mean_cos_sq - mean_cos * mean_cos);
            
            measurements.push((mean_cos, mean_sin_sq, chi, mean_w));
        }
    }
    
    // Statistics
    let n_meas = measurements.len() as f64;
    let mean_cos = measurements.iter().map(|(c, _, _, _)| c).sum::<f64>() / n_meas;
    let mean_chi = measurements.iter().map(|(_, _, x, _)| x).sum::<f64>() / n_meas;
    let mean_w = measurements.iter().map(|(_, _, _, w)| w).sum::<f64>() / n_meas;
    
    let cos_var = measurements.iter()
        .map(|(c, _, _, _)| (c - mean_cos).powi(2))
        .sum::<f64>() / (n_meas - 1.0);
    let chi_var = measurements.iter()
        .map(|(_, _, x, _)| (x - mean_chi).powi(2))
        .sum::<f64>() / (n_meas - 1.0);
    
    let cos_error = (cos_var / n_meas).sqrt();
    let chi_error = (chi_var / n_meas).sqrt();
    
    let m2 = measurements.iter()
        .map(|(c, _, _, _)| c * c)
        .sum::<f64>() / n_meas;
    let m4 = measurements.iter()
        .map(|(c, _, _, _)| c.powi(4))
        .sum::<f64>() / n_meas;
    let binder = 1.0 - m4 / (3.0 * m2 * m2);
    
    let acceptance = 100.0 * accepts as f64 / (250_000 + production) as f64;
    
    MeasurementPoint {
        beta,
        alpha,
        chi: mean_chi,
        chi_error,
        mean_cos,
        mean_cos_error: cos_error,
        mean_w,
        binder,
        acceptance,
    }
}

fn analyze_ridge(points: &[MeasurementPoint]) -> (f64, f64) {
    // Find high-chi points (top 30%)
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| b.chi.partial_cmp(&a.chi).unwrap());
    let cutoff_idx = sorted.len() / 3;
    
    let ridge_points: Vec<(f64, f64)> = sorted[..cutoff_idx]
        .iter()
        .map(|p| (p.beta, p.alpha))
        .collect();
    
    // Linear regression
    let n = ridge_points.len() as f64;
    let sum_x: f64 = ridge_points.iter().map(|(b, _)| b).sum();
    let sum_y: f64 = ridge_points.iter().map(|(_, a)| a).sum();
    let sum_xx: f64 = ridge_points.iter().map(|(b, _)| b * b).sum();
    let sum_xy: f64 = ridge_points.iter().map(|(b, a)| b * a).sum();
    
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;
    
    (slope, intercept)
}