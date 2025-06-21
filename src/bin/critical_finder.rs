// Rapid critical point finder using optimized Monte Carlo
// Leverages 3M steps/sec speed for quick parameter scans

use scan::graph::Graph;
use scan::graph_fast::FastGraph;
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
    binder: f64,
    acceptance: f64,
}

fn main() {
    println!("=== CRITICAL POINT FINDER ===");
    let n = 48;  // System size
    
    // Phase 1: Coarse sweep
    let start_total = Instant::now();
    let coarse_points = coarse_sweep(n);
    
    // Find maximum chi
    let best_coarse = coarse_points.iter()
        .max_by(|a, b| a.chi.partial_cmp(&b.chi).unwrap())
        .unwrap();
    
    println!("\nPeak found at: β = {:.3}, α = {:.3}", best_coarse.beta, best_coarse.alpha);
    println!("  χ_max = {:.1}", best_coarse.chi);
    println!("  <cos θ> = {:.2}", best_coarse.mean_cos);
    
    // Phase 2: Fine refinement around peak
    println!("\nFine refinement:");
    let fine_points = fine_zoom(n, best_coarse.beta, best_coarse.alpha);
    
    let best_fine = fine_points.iter()
        .max_by(|a, b| a.chi.partial_cmp(&b.chi).unwrap())
        .unwrap();
    
    println!("Better peak: β = {:.3}, α = {:.3}", best_fine.beta, best_fine.alpha);
    println!("\n  χ = {:.1} ± {:.1}", best_fine.chi, best_fine.chi_error);
    println!("  <cos θ> = {:.2} ± {:.2}", best_fine.mean_cos, best_fine.mean_cos_error);
    println!("  Binder U4 = {:.2}", best_fine.binder);
    println!("  Acceptance = {:.1}%", best_fine.acceptance);
    
    // Phase 3: Ridge tracing
    let ridge_points = trace_ridge(n, &fine_points);
    let (slope, intercept) = fit_ridge(&ridge_points);
    
    println!("\nRidge detected: α = {:.3}β + {:.3}", slope, intercept);
    
    let total_time = start_total.elapsed();
    println!("\nTotal time: {:.1} seconds", total_time.as_secs_f64());
    println!("Ready for FSS at ({:.3}, {:.3})!", best_fine.beta, best_fine.alpha);
}

fn coarse_sweep(n: usize) -> Vec<MeasurementPoint> {
    println!("Coarse sweep: 121 points");
    let start = Instant::now();
    
    // Grid scan near expected critical region
    let beta_values: Vec<f64> = (0..=10).map(|i| 2.85 + 0.01 * i as f64).collect();
    let alpha_values: Vec<f64> = (0..=10).map(|i| 1.47 + 0.01 * i as f64).collect();
    
    let results = Mutex::new(Vec::new());
    let total_points = beta_values.len() * alpha_values.len();
    let points_done = Mutex::new(0);
    
    // Parallel scan over parameter grid
    beta_values.par_iter().for_each(|&beta| {
        for &alpha in &alpha_values {
            let point = quick_measurement(n, beta, alpha, 20_000, 10_000);
            
            let mut count = points_done.lock().unwrap();
            *count += 1;
            if *count % 10 == 0 {
                print!("\r  Progress: {}/{} points", *count, total_points);
                use std::io::{self, Write};
                io::stdout().flush().unwrap();
            }
            drop(count);
            
            results.lock().unwrap().push(point);
        }
    });
    
    let elapsed = start.elapsed();
    println!("\r  Coarse sweep: {} points in {:.1} seconds", total_points, elapsed.as_secs_f64());
    
    let mut final_results = results.into_inner().unwrap();
    final_results.sort_by(|a, b| b.chi.partial_cmp(&a.chi).unwrap());
    final_results
}

fn fine_zoom(n: usize, beta_center: f64, alpha_center: f64) -> Vec<MeasurementPoint> {
    println!("Fine zoom around (β={:.3}, α={:.3})", beta_center, alpha_center);
    let start = Instant::now();
    
    let beta_values: Vec<f64> = (-2..=2).map(|i| beta_center + 0.01 * i as f64).collect();
    let alpha_values: Vec<f64> = (-2..=2).map(|i| alpha_center + 0.01 * i as f64).collect();
    
    let results = Mutex::new(Vec::new());
    
    // Higher statistics for fine zoom
    beta_values.par_iter().for_each(|&beta| {
        for &alpha in &alpha_values {
            let point = quick_measurement(n, beta, alpha, 50_000, 50_000);
            results.lock().unwrap().push(point);
        }
    });
    
    let elapsed = start.elapsed();
    println!("  Fine zoom: 25 points in {:.1} seconds", elapsed.as_secs_f64());
    
    let mut final_results = results.into_inner().unwrap();
    final_results.sort_by(|a, b| b.chi.partial_cmp(&a.chi).unwrap());
    final_results
}

fn quick_measurement(n: usize, beta: f64, alpha: f64, 
                     equilibration: usize, production: usize) -> MeasurementPoint {
    let mut rng = Pcg64::seed_from_u64((beta * 1000.0 + alpha * 100.0) as u64);
    
    // Initialize with configuration closer to critical point
    let mut graph = Graph::complete_random_with(&mut rng, n);
    
    // Set theta values with some order (not fully random)
    use rand::distributions::Uniform;
    use rand::prelude::*;
    let theta_dist = Uniform::new(-0.5, 0.5);  // Small deviations from zero
    for link in &mut graph.links {
        link.theta = rng.sample(theta_dist);
        // Set z values based on expected critical region
        // For critical point, <w> ≈ 0.4-0.5, so z ≈ 0.7-0.9
        link.z = 0.8 + rng.gen_range(-0.1..0.1);
    }
    
    let mut fast_graph = FastGraph::from_graph(&graph);
    
    // Equilibration with adaptive step sizes
    let mut accepts = 0;
    let mut delta_z = 0.3;
    let mut delta_theta = 0.3;
    
    // Adjust step sizes during equilibration
    for i in 0..equilibration {
        let info = fast_graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
        if info.accept {
            accepts += 1;
        }
        
        // Adapt step sizes every 1000 steps
        if i > 0 && i % 1000 == 0 {
            let rate = accepts as f64 / i as f64;
            if rate > 0.6 {
                delta_z *= 1.1;
                delta_theta *= 1.1;
            } else if rate < 0.4 {
                delta_z *= 0.9;
                delta_theta *= 0.9;
            }
            delta_z = delta_z.clamp(0.1, 0.5);
            delta_theta = delta_theta.clamp(0.1, 0.5);
        }
    }
    
    // Production with measurements
    let mut measurements = Vec::new();
    let measure_interval = 50;
    
    for step in 0..production {
        let info = fast_graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
        if info.accept {
            accepts += 1;
        }
        
        if step % measure_interval == 0 {
            let m = fast_graph.m() as f64;
            let sum_cos: f64 = fast_graph.links.iter().map(|l| l.cos_theta).sum();
            let sum_w: f64 = fast_graph.links.iter().map(|l| l.exp_neg_z).sum();
            let sum_w_cos: f64 = fast_graph.links.iter()
                .map(|l| l.exp_neg_z * l.cos_theta).sum();
            
            let mean_cos = sum_cos / m;
            let mean_w = sum_w / m;
            let mean_w_cos = sum_w_cos / m;
            
            // Calculate cos² for susceptibility
            let sum_cos_sq: f64 = fast_graph.links.iter().map(|l| l.cos_theta * l.cos_theta).sum();
            let mean_cos_sq = sum_cos_sq / m;
            
            // Correct susceptibility formula: χ = Nβ(<cos²θ> - <cosθ>²)
            let chi = n as f64 * beta * (mean_cos_sq - mean_cos * mean_cos);
            
            measurements.push((mean_cos, chi, mean_w));
        }
    }
    
    // Calculate statistics
    let n_meas = measurements.len() as f64;
    let mean_cos = measurements.iter().map(|(c, _, _)| c).sum::<f64>() / n_meas;
    let mean_chi = measurements.iter().map(|(_, x, _)| x).sum::<f64>() / n_meas;
    
    // Errors
    let cos_var = measurements.iter()
        .map(|(c, _, _)| (c - mean_cos).powi(2))
        .sum::<f64>() / (n_meas - 1.0);
    let chi_var = measurements.iter()
        .map(|(_, x, _)| (x - mean_chi).powi(2))
        .sum::<f64>() / (n_meas - 1.0);
    
    let cos_error = (cos_var / n_meas).sqrt();
    let chi_error = (chi_var / n_meas).sqrt();
    
    // Binder cumulant U4 = 1 - <m^4>/(3<m^2>^2)
    let m2 = measurements.iter()
        .map(|(c, _, _)| c * c)
        .sum::<f64>() / n_meas;
    let m4 = measurements.iter()
        .map(|(c, _, _)| c.powi(4))
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
        binder,
        acceptance,
    }
}

fn trace_ridge(n: usize, points: &[MeasurementPoint]) -> Vec<(f64, f64)> {
    println!("\nTracing ridge:");
    
    // Find high-chi points (top 20%)
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| b.chi.partial_cmp(&a.chi).unwrap());
    let cutoff_idx = sorted.len() / 5;
    
    let ridge_points: Vec<(f64, f64)> = sorted[..cutoff_idx]
        .iter()
        .map(|p| (p.beta, p.alpha))
        .collect();
    
    // Optional: Gradient ascent to refine ridge
    // For now, just return the high-chi points
    println!("  Found {} ridge points", ridge_points.len());
    
    ridge_points
}

fn fit_ridge(points: &[(f64, f64)]) -> (f64, f64) {
    // Linear regression: alpha = slope * beta + intercept
    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|(b, _)| b).sum();
    let sum_y: f64 = points.iter().map(|(_, a)| a).sum();
    let sum_xx: f64 = points.iter().map(|(b, _)| b * b).sum();
    let sum_xy: f64 = points.iter().map(|(b, a)| b * a).sum();
    
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;
    
    (slope, intercept)
}

// Optional: More sophisticated ridge following using gradient ascent
#[allow(dead_code)]
fn gradient_ascent_ridge(n: usize, start_beta: f64, start_alpha: f64) -> Vec<(f64, f64)> {
    let mut ridge = vec![(start_beta, start_alpha)];
    let mut beta = start_beta;
    let mut alpha = start_alpha;
    let step_size = 0.005;
    let epsilon = 0.001;  // For numerical gradient
    
    for _ in 0..10 {
        // Compute gradient of chi
        let chi_0 = quick_measurement(n, beta, alpha, 5000, 5000).chi;
        let chi_beta = quick_measurement(n, beta + epsilon, alpha, 5000, 5000).chi;
        let chi_alpha = quick_measurement(n, beta, alpha + epsilon, 5000, 5000).chi;
        
        let grad_beta = (chi_beta - chi_0) / epsilon;
        let grad_alpha = (chi_alpha - chi_0) / epsilon;
        
        // Normalize gradient
        let grad_norm = (grad_beta * grad_beta + grad_alpha * grad_alpha).sqrt();
        if grad_norm < 0.1 {
            break;  // At peak
        }
        
        // Step in gradient direction
        beta += step_size * grad_beta / grad_norm;
        alpha += step_size * grad_alpha / grad_norm;
        
        ridge.push((beta, alpha));
        println!("  Ridge point: β = {:.3}, α = {:.3}, χ = {:.1}", beta, alpha, chi_0);
    }
    
    ridge
}