// Benchmark comparison between original and optimized implementations

use scan::graph::Graph;
use scan::graph_fast::FastGraph;
use scan::observables::Observables;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_pcg::Pcg64;
use std::time::{Duration, Instant};
use std::process;

#[derive(Debug)]
struct BenchmarkResult {
    time: Duration,
    steps_per_sec: f64,
    time_per_step_us: f64,
    memory_mb: f64,
    mean_cos: f64,
    cos_error: f64,
    entropy: f64,
    susceptibility: f64,
    acceptance_rate: f64,
}

/// Get current memory usage in MB
fn get_memory_usage_mb() -> f64 {
    let pid = process::id();
    let status_path = format!("/proc/{}/status", pid);
    
    // Try Linux /proc filesystem first
    if let Ok(contents) = std::fs::read_to_string(&status_path) {
        for line in contents.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(kb) = parts[1].parse::<f64>() {
                        return kb / 1024.0;
                    }
                }
            }
        }
    }
    
    // Fallback for macOS/other systems - estimate based on graph size
    0.0
}

/// Run benchmark with original implementation
fn benchmark_original(n: usize, steps: usize, beta: f64, alpha: f64, seed: u64) -> BenchmarkResult {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    
    // Create graph
    let mut graph = Graph::complete_random_with(&mut rng, n);
    
    // Measure initial memory
    let mem_start = get_memory_usage_mb();
    
    // Start timing
    let start = Instant::now();
    
    // Run MC steps
    let mut accepts = 0;
    let mut cos_sum = 0.0;
    let mut cos_sum_sq = 0.0;
    let mut measurements = 0;
    
    for step in 0..steps {
        let info = graph.metropolis_step(beta, alpha, 0.1, 0.1, &mut rng);
        if info.accepted {
            accepts += 1;
        }
        
        // Measure every 100 steps
        if step % 100 == 0 && step > steps / 2 {
            let obs = Observables::measure(&graph, beta, alpha);
            cos_sum += obs.mean_cos;
            cos_sum_sq += obs.mean_cos * obs.mean_cos;
            measurements += 1;
        }
    }
    
    let elapsed = start.elapsed();
    
    // Final measurement
    let final_obs = Observables::measure(&graph, beta, alpha);
    
    // Calculate statistics
    let mean_cos = cos_sum / measurements as f64;
    let cos_var = cos_sum_sq / measurements as f64 - mean_cos * mean_cos;
    let cos_error = (cos_var / measurements as f64).sqrt();
    
    // Memory usage
    let mem_end = get_memory_usage_mb();
    let memory_mb = if mem_end > 0.0 { mem_end - mem_start } else { estimate_memory_original(n) };
    
    BenchmarkResult {
        time: elapsed,
        steps_per_sec: steps as f64 / elapsed.as_secs_f64(),
        time_per_step_us: elapsed.as_micros() as f64 / steps as f64,
        memory_mb,
        mean_cos,
        cos_error,
        entropy: final_obs.entropy,
        susceptibility: final_obs.susceptibility,
        acceptance_rate: accepts as f64 / steps as f64,
    }
}

/// Run benchmark with optimized implementation
fn benchmark_optimized(n: usize, steps: usize, beta: f64, alpha: f64, seed: u64) -> BenchmarkResult {
    let mut rng = Pcg64::seed_from_u64(seed);
    
    // Create optimized graph
    let mut graph = FastGraph::new(n, seed);
    
    // Measure initial memory
    let mem_start = get_memory_usage_mb();
    
    // Start timing
    let start = Instant::now();
    
    // Run MC steps
    let mut accepts = 0;
    let mut cos_sum = 0.0;
    let mut cos_sum_sq = 0.0;
    let mut measurements = 0;
    
    for step in 0..steps {
        let info = graph.metropolis_step(beta, alpha, 0.1, 0.1, &mut rng);
        if info.accept {
            accepts += 1;
        }
        
        // Measure every 100 steps
        if step % 100 == 0 && step > steps / 2 {
            let sum_cos: f64 = graph.links.iter().map(|l| l.cos_theta).sum();
            let mean_cos = sum_cos / graph.m() as f64;
            cos_sum += mean_cos;
            cos_sum_sq += mean_cos * mean_cos;
            measurements += 1;
        }
    }
    
    let elapsed = start.elapsed();
    
    // Final measurements
    let sum_w: f64 = graph.links.iter().map(|l| l.exp_neg_z).sum();
    let mean_w = sum_w / graph.m() as f64;
    let var_w = graph.links.iter()
        .map(|l| (l.exp_neg_z - mean_w).powi(2))
        .sum::<f64>() / graph.m() as f64;
    
    let entropy = graph.entropy_action();
    let susceptibility = graph.n() as f64 * var_w;
    
    // Calculate statistics
    let mean_cos = cos_sum / measurements as f64;
    let cos_var = cos_sum_sq / measurements as f64 - mean_cos * mean_cos;
    let cos_error = (cos_var / measurements as f64).sqrt();
    
    // Memory usage
    let mem_end = get_memory_usage_mb();
    let memory_mb = if mem_end > 0.0 { mem_end - mem_start } else { estimate_memory_optimized(n) };
    
    BenchmarkResult {
        time: elapsed,
        steps_per_sec: steps as f64 / elapsed.as_secs_f64(),
        time_per_step_us: elapsed.as_micros() as f64 / steps as f64,
        memory_mb,
        mean_cos,
        cos_error,
        entropy,
        susceptibility,
        acceptance_rate: accepts as f64 / steps as f64,
    }
}

/// Estimate memory usage for original implementation
fn estimate_memory_original(n: usize) -> f64 {
    let nodes = n * std::mem::size_of::<scan::graph::Node>();
    let links = n * (n - 1) / 2 * (2 * 8 + 2 * 8 + 216); // i,j,z,theta,tensor
    let triangles = n * (n - 1) * (n - 2) / 6 * 3 * 8;
    (nodes + links + triangles) as f64 / (1024.0 * 1024.0)
}

/// Estimate memory usage for optimized implementation
fn estimate_memory_optimized(n: usize) -> f64 {
    let nodes = n * 4; // u32 id
    let links = n * (n - 1) / 2 * 64; // FastLink is 64 bytes
    let triangles = n * (n - 1) * (n - 2) / 6 * 3 * 4; // u32 indices
    (nodes + links + triangles) as f64 / (1024.0 * 1024.0)
}

/// Check if results agree within tolerance
fn check_correctness(orig: &BenchmarkResult, opt: &BenchmarkResult) -> (bool, String) {
    let mut passed = true;
    let mut details = String::new();
    
    // Check mean cos theta
    let cos_diff = (orig.mean_cos - opt.mean_cos).abs();
    let cos_tol = 3.0 * (orig.cos_error.powi(2) + opt.cos_error.powi(2)).sqrt();
    let cos_ok = cos_diff < cos_tol.max(0.001);
    
    if !cos_ok {
        passed = false;
        details.push_str(&format!("  <cos θ> differs by {:.4} (tolerance: {:.4})\n", cos_diff, cos_tol));
    }
    
    // Check susceptibility (relative difference)
    let chi_diff = ((orig.susceptibility - opt.susceptibility) / orig.susceptibility).abs();
    let chi_ok = chi_diff < 0.05; // 5% tolerance
    
    if !chi_ok {
        passed = false;
        details.push_str(&format!("  Susceptibility differs by {:.1}%\n", chi_diff * 100.0));
    }
    
    // Check acceptance rate
    let acc_diff = (orig.acceptance_rate - opt.acceptance_rate).abs();
    let acc_ok = acc_diff < 0.02; // 2% tolerance
    
    if !acc_ok {
        passed = false;
        details.push_str(&format!("  Acceptance rate differs by {:.1}%\n", acc_diff * 100.0));
    }
    
    (passed, details)
}

/// Run a single benchmark comparison
fn run_benchmark(name: &str, n: usize, steps: usize, beta: f64, alpha: f64) {
    println!("\n=== Performance Benchmark ===");
    println!("System size: N={}", n);
    println!("Parameters: β={:.2}, α={:.2}", beta, alpha);
    println!("MC steps: {}", steps);
    println!("Seed: 12345");
    
    // Run original implementation
    println!("\nOriginal implementation:");
    let orig = benchmark_original(n, steps, beta, alpha, 12345);
    
    println!("  Time: {:.2} seconds", orig.time.as_secs_f64());
    println!("  MC steps/sec: {:.0}", orig.steps_per_sec);
    println!("  Time per step: {:.2} μs", orig.time_per_step_us);
    println!("  Memory: {:.1} MB", orig.memory_mb);
    println!("  <cos θ> = {:.4} ± {:.4}", orig.mean_cos, orig.cos_error);
    println!("  Entropy: {:.4}", orig.entropy);
    println!("  Susceptibility: {:.2}", orig.susceptibility);
    println!("  Acceptance: {:.1}%", orig.acceptance_rate * 100.0);
    
    // Run optimized implementation
    println!("\nOptimized implementation:");
    let opt = benchmark_optimized(n, steps, beta, alpha, 12345);
    
    let speedup = orig.time.as_secs_f64() / opt.time.as_secs_f64();
    let mem_reduction = (orig.memory_mb - opt.memory_mb) / orig.memory_mb * 100.0;
    
    println!("  Time: {:.2} seconds ({:.1}x speedup)", opt.time.as_secs_f64(), speedup);
    println!("  MC steps/sec: {:.0}", opt.steps_per_sec);
    println!("  Time per step: {:.2} μs", opt.time_per_step_us);
    println!("  Memory: {:.1} MB ({:.0}% reduction)", opt.memory_mb, mem_reduction);
    println!("  <cos θ> = {:.4} ± {:.4}", opt.mean_cos, opt.cos_error);
    println!("  Entropy: {:.4}", opt.entropy);
    println!("  Susceptibility: {:.2}", opt.susceptibility);
    println!("  Acceptance: {:.1}%", opt.acceptance_rate * 100.0);
    
    // Check correctness
    let (passed, details) = check_correctness(&orig, &opt);
    
    println!("\nCorrectness check: {}", if passed { "PASSED" } else { "FAILED" });
    if !details.is_empty() {
        println!("{}", details);
    }
    
    println!("  Observables agree: {}", if passed { "YES" } else { "NO" });
    println!("  Same convergence: {}", 
             if (orig.acceptance_rate - opt.acceptance_rate).abs() < 0.02 { "YES" } else { "NO" });
    
    // Summary
    println!("\nSummary for {}:", name);
    println!("  Speedup: {:.1}x", speedup);
    println!("  Memory reduction: {:.0}%", mem_reduction);
    println!("  Correctness: {}", if passed { "✓" } else { "✗" });
}

fn main() {
    println!("Monte Carlo Optimization Benchmark");
    println!("==================================");
    
    // Small test case
    run_benchmark("Small system", 24, 10000, 2.9, 1.5);
    
    println!("\n{}\n", "=".repeat(50));
    
    // Medium test case
    run_benchmark("Medium system", 48, 10000, 2.91, 1.48);
    
    // Overall summary
    println!("\n{}", "=".repeat(50));
    println!("\nBenchmark Complete!");
}