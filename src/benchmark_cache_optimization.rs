// Benchmark comparing original vs cache-optimized graph implementations

use std::time::Instant;
use rand::prelude::*;

// Import both implementations
use crate::graph::Graph as OriginalGraph;
use crate::graph_cache_optimized::{OptimizedGraph, benchmark_comparison};

/// Run comprehensive benchmark comparison
pub fn run_full_comparison() {
    println!("\n========================================");
    println!("  CACHE OPTIMIZATION BENCHMARK RESULTS");
    println!("========================================\n");
    
    // Test different graph sizes
    let sizes = vec![12, 24, 48, 96];
    let num_mc_steps = 10000;
    
    for &n in &sizes {
        compare_implementations(n, num_mc_steps);
        println!("\n----------------------------------------\n");
    }
    
    // Detailed analysis for N=24
    detailed_analysis(24);
}

/// Compare original and optimized implementations
fn compare_implementations(n: usize, num_steps: usize) {
    println!("Comparison for N = {} nodes", n);
    println!("({} links, {} triangles)\n", 
             n * (n - 1) / 2, 
             n * (n - 1) * (n - 2) / 6);
    
    let mut rng = thread_rng();
    
    // === Original Implementation ===
    println!("Original Implementation:");
    
    // Create graph
    let start = Instant::now();
    let mut orig_graph = OriginalGraph::complete_random_with(&mut rng, n);
    let orig_create_time = start.elapsed();
    
    // Benchmark entropy
    let start = Instant::now();
    let mut orig_entropy_sum = 0.0;
    for _ in 0..1000 {
        orig_entropy_sum += orig_graph.entropy_action();
    }
    let orig_entropy_time = start.elapsed();
    
    // Benchmark triangle sum
    let start = Instant::now();
    let mut orig_triangle_sum = 0.0;
    for _ in 0..100 {
        orig_triangle_sum += orig_graph.triangle_sum();
    }
    let orig_triangle_time = start.elapsed();
    
    // Benchmark MC steps
    let start = Instant::now();
    let mut orig_accepts = 0;
    for _ in 0..num_steps {
        let info = orig_graph.metropolis_step(1.0, 1.0, 0.1, 0.1, &mut rng);
        if info.accept {
            orig_accepts += 1;
        }
    }
    let orig_mc_time = start.elapsed();
    
    println!("  Creation time:      {:?}", orig_create_time);
    println!("  Entropy (1k calls): {:?} ({:.2} μs/call)", 
             orig_entropy_time, 
             orig_entropy_time.as_micros() as f64 / 1000.0);
    println!("  Triangle (100):     {:?} ({:.2} μs/call)", 
             orig_triangle_time,
             orig_triangle_time.as_micros() as f64 / 100.0);
    println!("  MC steps ({}):    {:?} ({:.2} μs/step)", 
             num_steps, orig_mc_time,
             orig_mc_time.as_micros() as f64 / num_steps as f64);
    println!("  Accept rate:        {:.1}%", 
             100.0 * orig_accepts as f64 / num_steps as f64);
    
    // === Optimized Implementation ===
    println!("\nOptimized Implementation:");
    
    // Create graph
    let start = Instant::now();
    let mut opt_graph = OptimizedGraph::new(n, &mut rng);
    let opt_create_time = start.elapsed();
    
    // Benchmark entropy
    let start = Instant::now();
    let mut opt_entropy_sum = 0.0;
    for _ in 0..1000 {
        opt_entropy_sum += opt_graph.entropy_action();
    }
    let opt_entropy_time = start.elapsed();
    
    // Benchmark triangle sum
    let start = Instant::now();
    let mut opt_triangle_sum = 0.0;
    for _ in 0..100 {
        opt_triangle_sum += opt_graph.triangle_sum();
    }
    let opt_triangle_time = start.elapsed();
    
    // Benchmark MC steps
    let start = Instant::now();
    let mut opt_accepts = 0;
    for _ in 0..num_steps {
        let info = opt_graph.metropolis_step(1.0, 1.0, 0.1, 0.1, &mut rng);
        if info.accept {
            opt_accepts += 1;
        }
    }
    let opt_mc_time = start.elapsed();
    
    println!("  Creation time:      {:?}", opt_create_time);
    println!("  Entropy (1k calls): {:?} ({:.2} μs/call)", 
             opt_entropy_time,
             opt_entropy_time.as_micros() as f64 / 1000.0);
    println!("  Triangle (100):     {:?} ({:.2} μs/call)", 
             opt_triangle_time,
             opt_triangle_time.as_micros() as f64 / 100.0);
    println!("  MC steps ({}):    {:?} ({:.2} μs/step)", 
             num_steps, opt_mc_time,
             opt_mc_time.as_micros() as f64 / num_steps as f64);
    println!("  Accept rate:        {:.1}%", 
             100.0 * opt_accepts as f64 / num_steps as f64);
    
    // === Speedup Summary ===
    println!("\nSpeedup Factors:");
    println!("  Creation:    {:.1}x", 
             orig_create_time.as_secs_f64() / opt_create_time.as_secs_f64());
    println!("  Entropy:     {:.1}x", 
             orig_entropy_time.as_secs_f64() / opt_entropy_time.as_secs_f64());
    println!("  Triangle:    {:.1}x", 
             orig_triangle_time.as_secs_f64() / opt_triangle_time.as_secs_f64());
    println!("  MC steps:    {:.1}x", 
             orig_mc_time.as_secs_f64() / opt_mc_time.as_secs_f64());
    
    // Prevent optimization
    println!("\n(Debug: {:.6} {:.6} {:.6} {:.6})", 
             orig_entropy_sum, opt_entropy_sum, 
             orig_triangle_sum, opt_triangle_sum);
}

/// Detailed analysis including cache behavior
fn detailed_analysis(n: usize) {
    println!("\n========================================");
    println!("  DETAILED ANALYSIS FOR N = {}", n);
    println!("========================================\n");
    
    let mut rng = thread_rng();
    let mut opt_graph = OptimizedGraph::new(n, &mut rng);
    
    // Memory layout analysis
    println!("Memory Layout:");
    println!("  Link struct size:        {} bytes", 
             std::mem::size_of::<crate::graph_cache_optimized::OptimizedLink>());
    println!("  Cache line size:         64 bytes");
    println!("  Links per cache line:    {}", 
             64 / std::mem::size_of::<crate::graph_cache_optimized::OptimizedLink>());
    
    let num_links = n * (n - 1) / 2;
    let num_triangles = n * (n - 1) * (n - 2) / 6;
    let link_memory = num_links * 28; // 28 bytes per optimized link
    let triangle_memory = num_triangles * 12; // 3 * 4 bytes per triangle
    
    println!("  Total link memory:       {:.1} KB", link_memory as f64 / 1024.0);
    println!("  Total triangle memory:   {:.1} KB", triangle_memory as f64 / 1024.0);
    println!("  Fits in L2 cache:        {}", 
             (link_memory + triangle_memory) < 256 * 1024);
    println!("  Fits in L3 cache:        {}", 
             (link_memory + triangle_memory) < 8 * 1024 * 1024);
    
    // Access pattern analysis
    println!("\nAccess Pattern Analysis:");
    
    // Sequential access test
    let start = Instant::now();
    let mut sum = 0.0;
    for _ in 0..1000 {
        for link in &opt_graph.links {
            sum += link.exp_neg_z as f64;
        }
    }
    let seq_time = start.elapsed();
    
    // Random access test
    let indices: Vec<usize> = (0..num_links).collect();
    let mut shuffled = indices.clone();
    shuffled.shuffle(&mut rng);
    
    let start = Instant::now();
    let mut sum2 = 0.0;
    for _ in 0..1000 {
        for &idx in &shuffled {
            sum2 += opt_graph.links[idx].exp_neg_z as f64;
        }
    }
    let rand_time = start.elapsed();
    
    println!("  Sequential access:  {:?} ({:.2} ns/access)", 
             seq_time,
             seq_time.as_nanos() as f64 / (1000.0 * num_links as f64));
    println!("  Random access:      {:?} ({:.2} ns/access)", 
             rand_time,
             rand_time.as_nanos() as f64 / (1000.0 * num_links as f64));
    println!("  Random/Sequential:  {:.1}x slower", 
             rand_time.as_secs_f64() / seq_time.as_secs_f64());
    
    // Triangle access patterns
    println!("\nTriangle Access Patterns:");
    
    // Test incremental update efficiency
    let link_idx = n / 2; // Middle link
    let start = Instant::now();
    let mut delta_sum = 0.0;
    for _ in 0..1000 {
        delta_sum += opt_graph.triangle_sum_delta(link_idx, 0.1);
    }
    let incr_time = start.elapsed();
    
    let start = Instant::now();
    let mut full_sum = 0.0;
    for _ in 0..10 {
        full_sum += opt_graph.triangle_sum();
    }
    let full_time = start.elapsed();
    
    let triangles_per_link = opt_graph.links_per_triangle[link_idx].len();
    println!("  Triangles per link:     {}", triangles_per_link);
    println!("  Incremental (1k):       {:?} ({:.2} μs/call)", 
             incr_time,
             incr_time.as_micros() as f64 / 1000.0);
    println!("  Full sum (10):          {:?} ({:.2} μs/call)", 
             full_time,
             full_time.as_micros() as f64 / 10.0);
    println!("  Speedup factor:         {:.1}x", 
             (full_time.as_secs_f64() / 10.0) / (incr_time.as_secs_f64() / 1000.0));
    
    // Prevent optimization
    println!("\n(Debug: {} {} {} {})", sum, sum2, delta_sum, full_sum);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_n24() {
        compare_implementations(24, 1000);
    }
    
    #[test]
    fn test_detailed_analysis() {
        detailed_analysis(24);
    }
}