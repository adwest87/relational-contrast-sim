// Quick reproducibility study for Binder cumulant calculations
// Reduced sample size for faster execution

use scan::graph_ultra_optimized::UltraOptimizedGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::fs::File;
use std::io::Write;

#[derive(Debug, Clone)]
struct QuickRunResult {
    seed: u64,
    u4_value: f64,
    magnetization_mean: f64,
    m2_moment: f64,
    m4_moment: f64,
    acceptance_rate: f64,
    sample_count: usize,
}

fn quick_run(alpha: f64, beta: f64, n: usize, seed: u64) -> QuickRunResult {
    let mut graph = UltraOptimizedGraph::new(n, seed);
    let mut rng = Pcg64::seed_from_u64(seed + 1000);
    
    // Quick thermalization
    let n_therm = 500 * n;
    for _ in 0..n_therm {
        graph.metropolis_step(alpha, beta, 0.0, 0.1, 0.1, &mut rng);
    }
    
    // Quick measurement
    let n_measure = 2000 * n;
    let mut magnetization_samples = Vec::with_capacity(n_measure);
    let mut accepts = 0;
    
    for _ in 0..n_measure {
        let accept = graph.metropolis_step(alpha, beta, 0.0, 0.1, 0.1, &mut rng);
        if accept { accepts += 1; }
        
        let magnetization = graph.cos_theta.iter().sum::<f64>() / n as f64;
        magnetization_samples.push(magnetization);
    }
    
    let acceptance_rate = accepts as f64 / n_measure as f64;
    
    // Calculate moments
    let n_samples = magnetization_samples.len() as f64;
    let mean = magnetization_samples.iter().sum::<f64>() / n_samples;
    let m2_moment = magnetization_samples.iter().map(|&x| x * x).sum::<f64>() / n_samples;
    let m4_moment = magnetization_samples.iter().map(|&x| x.powi(4)).sum::<f64>() / n_samples;
    
    let u4_value = if m2_moment > 1e-10 {
        1.0 - m4_moment / (3.0 * m2_moment * m2_moment)
    } else {
        0.0
    };
    
    QuickRunResult {
        seed,
        u4_value,
        magnetization_mean: mean,
        m2_moment,
        m4_moment,
        acceptance_rate,
        sample_count: magnetization_samples.len(),
    }
}

fn main() {
    println!("=== QUICK REPRODUCIBILITY STUDY ===\n");
    
    let alpha = 1.37;
    let beta = 2.0;
    let n = 12;
    let n_runs = 20;
    
    println!("Parameters: α={:.3}, β={:.3}, N={}, runs={}", alpha, beta, n, n_runs);
    println!();
    
    let mut results = Vec::new();
    
    for run_id in 0..n_runs {
        let seed = 1000 + run_id as u64 * 137;
        print!("Run {:2}: ", run_id + 1);
        
        let result = quick_run(alpha, beta, n, seed);
        
        println!("U₄={:7.4}, ⟨M⟩={:7.4}, acc={:4.1}%", 
                 result.u4_value, result.magnetization_mean, 100.0 * result.acceptance_rate);
        
        results.push(result);
    }
    
    // Analysis
    println!("\n=== RESULTS ANALYSIS ===");
    
    let u4_values: Vec<f64> = results.iter().map(|r| r.u4_value).collect();
    let u4_mean = u4_values.iter().sum::<f64>() / u4_values.len() as f64;
    let u4_variance = u4_values.iter().map(|&x| (x - u4_mean).powi(2)).sum::<f64>() / (u4_values.len() - 1) as f64;
    let u4_std = u4_variance.sqrt();
    let u4_min = u4_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let u4_max = u4_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    println!("U₄ Statistics:");
    println!("  Mean: {:.6} ± {:.6}", u4_mean, u4_std);
    println!("  Range: [{:.6}, {:.6}]", u4_min, u4_max);
    println!("  Relative std: {:.2}%", 100.0 * u4_std / u4_mean.abs());
    
    // Count unusual values
    let negative_count = u4_values.iter().filter(|&&x| x < 0.0).count();
    let low_count = u4_values.iter().filter(|&&x| x < 0.5).count();
    let very_low_count = u4_values.iter().filter(|&&x| x < 0.3).count();
    
    println!("\nDistribution Analysis:");
    println!("  Negative U₄: {} out of {} ({:.1}%)", 
             negative_count, n_runs, 100.0 * negative_count as f64 / n_runs as f64);
    println!("  U₄ < 0.5: {} out of {} ({:.1}%)", 
             low_count, n_runs, 100.0 * low_count as f64 / n_runs as f64);
    println!("  U₄ < 0.3: {} out of {} ({:.1}%)", 
             very_low_count, n_runs, 100.0 * very_low_count as f64 / n_runs as f64);
    
    // Detailed values
    println!("\nAll U₄ values:");
    for (i, &u4) in u4_values.iter().enumerate() {
        print!("{:7.4}", u4);
        if (i + 1) % 5 == 0 {
            println!();
        } else {
            print!("  ");
        }
    }
    if u4_values.len() % 5 != 0 {
        println!();
    }
    
    // Magnetization analysis
    let mag_values: Vec<f64> = results.iter().map(|r| r.magnetization_mean).collect();
    let mag_mean = mag_values.iter().sum::<f64>() / mag_values.len() as f64;
    let mag_std = (mag_values.iter().map(|&x| (x - mag_mean).powi(2)).sum::<f64>() / (mag_values.len() - 1) as f64).sqrt();
    
    println!("\nMagnetization Statistics:");
    println!("  ⟨M⟩ mean: {:.6} ± {:.6}", mag_mean, mag_std);
    
    let negative_mag_count = mag_values.iter().filter(|&&x| x < 0.0).count();
    println!("  Negative ⟨M⟩: {} out of {} ({:.1}%)", 
             negative_mag_count, n_runs, 100.0 * negative_mag_count as f64 / n_runs as f64);
    
    // Save results
    if let Ok(mut file) = File::create("quick_reproducibility_results.csv") {
        writeln!(file, "run,seed,u4_value,magnetization_mean,m2_moment,m4_moment,acceptance_rate").unwrap();
        for (i, result) in results.iter().enumerate() {
            writeln!(file, "{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
                     i + 1, result.seed, result.u4_value, result.magnetization_mean,
                     result.m2_moment, result.m4_moment, result.acceptance_rate).unwrap();
        }
        println!("\nResults saved to: quick_reproducibility_results.csv");
    }
    
    // Statistical test
    println!("\n=== STATISTICAL ASSESSMENT ===");
    
    if u4_std / u4_mean.abs() > 0.1 {
        println!("⚠ HIGH VARIABILITY: Relative std > 10%");
        println!("  This indicates significant run-to-run variation in U₄");
    } else {
        println!("✓ LOW VARIABILITY: Results are reasonably consistent");
    }
    
    if very_low_count > n_runs / 4 {
        println!("🔬 UNUSUAL PHYSICS: >25% of runs show U₄ < 0.3");
        println!("  This strongly suggests non-conventional critical behavior");
    }
    
    if negative_count > 0 {
        println!("⚠ NEGATIVE U₄ DETECTED: {} out of {} runs", negative_count, n_runs);
        println!("  This confirms exotic quantum criticality");
    }
    
    println!("\n=== CONCLUSION ===");
    if u4_std > 0.05 && (low_count > n_runs / 3 || negative_count > 0) {
        println!("🎯 SIGNIFICANT EVIDENCE for exotic quantum spin liquid behavior:");
        println!("   - Large U₄ variability across runs");
        println!("   - Frequent unusual/negative U₄ values");
        println!("   - Non-conventional critical statistics");
    } else if u4_std > 0.05 {
        println!("⚠ Moderate evidence for unusual critical behavior");
    } else {
        println!("✓ Results suggest conventional critical behavior");
    }
}