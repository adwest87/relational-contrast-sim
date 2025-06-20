// Benchmark specifically for Apple Silicon M1 optimizations

use scan::graph::Graph;
use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_pcg::Pcg64;
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use scan::graph_m1_optimized::{M1Graph, benchmark_m1_optimizations};

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("This benchmark requires Apple Silicon (ARM64) architecture");
    eprintln!("Running fast implementation benchmark instead...\n");
    
    // Run a simple benchmark with the fast implementation
    let n = 48;
    let steps = 10_000;
    let alpha = 1.5;
    let beta = 2.9;
    let seed = 12345;
    
    let mut fast_graph = FastGraph::new(n, seed);
    let mut rng = Pcg64::seed_from_u64(seed);
    
    let start = Instant::now();
    let mut accepts = 0;
    
    for _ in 0..steps {
        let info = fast_graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
        if info.accept {
            accepts += 1;
        }
    }
    
    let elapsed = start.elapsed();
    let rate = steps as f64 / elapsed.as_secs_f64();
    
    println!("Fast implementation (non-ARM):");
    println!("  Time: {:.3} s", elapsed.as_secs_f64());
    println!("  Rate: {:.0} steps/sec", rate);
    println!("  Accept: {:.1}%", 100.0 * accepts as f64 / steps as f64);
}

#[cfg(target_arch = "aarch64")]
fn main() {
    println!("Apple Silicon M1 Optimization Benchmark");
    println!("=====================================");
    println!("Architecture: {}", std::env::consts::ARCH);
    println!("CPU cores: {}", std::thread::available_parallelism().unwrap());
    
    // Detect if running on Apple Silicon
    if let Ok(brand) = std::fs::read_to_string("/proc/cpuinfo") {
        if brand.contains("Apple") {
            println!("Detected Apple processor");
        }
    }
    
    let n = 48;
    let steps = 100_000;
    let alpha = 1.5;
    let beta = 2.9;
    let seed = 12345;
    
    println!("\nTest parameters:");
    println!("  N = {}", n);
    println!("  Steps = {}", steps);
    println!("  α = {}, β = {}", alpha, beta);
    
    // Create initial graph for fair comparison
    let mut init_rng = ChaCha20Rng::seed_from_u64(seed);
    let init_graph = Graph::complete_random_with(&mut init_rng, n);
    
    // Benchmark original implementation
    println!("\n1. Original implementation:");
    let mut orig_graph = init_graph.clone();
    let mut orig_rng = ChaCha20Rng::seed_from_u64(seed);
    
    let start = Instant::now();
    let mut orig_accepts = 0;
    for _ in 0..steps {
        let info = orig_graph.metropolis_step(beta, alpha, 0.1, 0.1, &mut orig_rng);
        if info.accepted {
            orig_accepts += 1;
        }
    }
    let orig_time = start.elapsed();
    let orig_rate = steps as f64 / orig_time.as_secs_f64();
    
    println!("   Time: {:.3} s", orig_time.as_secs_f64());
    println!("   Rate: {:.0} steps/sec", orig_rate);
    println!("   Accept: {:.1}%", 100.0 * orig_accepts as f64 / steps as f64);
    
    // Benchmark fast implementation
    println!("\n2. Fast implementation (PCG64):");
    let mut fast_graph = FastGraph::from_graph(&init_graph);
    let mut fast_rng = Pcg64::seed_from_u64(seed);
    
    let start = Instant::now();
    let mut fast_accepts = 0;
    for _ in 0..steps {
        let info = fast_graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut fast_rng);
        if info.accept {
            fast_accepts += 1;
        }
    }
    let fast_time = start.elapsed();
    let fast_rate = steps as f64 / fast_time.as_secs_f64();
    
    println!("   Time: {:.3} s", fast_time.as_secs_f64());
    println!("   Rate: {:.0} steps/sec", fast_rate);
    println!("   Accept: {:.1}%", 100.0 * fast_accepts as f64 / steps as f64);
    println!("   Speedup vs original: {:.1}x", fast_rate / orig_rate);
    
    // Benchmark M1-optimized implementation
    println!("\n3. M1-optimized implementation (NEON SIMD + parallel):");
    let mut m1_graph = M1Graph::from_graph(&init_graph);
    let mut m1_rng = Pcg64::seed_from_u64(seed);
    
    // Warmup to ensure threads are ready
    for _ in 0..1000 {
        m1_graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut m1_rng);
    }
    m1_rng = Pcg64::seed_from_u64(seed); // Reset RNG
    
    let start = Instant::now();
    let mut m1_accepts = 0;
    for _ in 0..steps {
        let info = m1_graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut m1_rng);
        if info.accept {
            m1_accepts += 1;
        }
    }
    let m1_time = start.elapsed();
    let m1_rate = steps as f64 / m1_time.as_secs_f64();
    
    println!("   Time: {:.3} s", m1_time.as_secs_f64());
    println!("   Rate: {:.0} steps/sec", m1_rate);
    println!("   Accept: {:.1}%", 100.0 * m1_accepts as f64 / steps as f64);
    println!("   Speedup vs original: {:.1}x", m1_rate / orig_rate);
    println!("   Speedup vs fast: {:.1}x", m1_rate / fast_rate);
    
    // Test SIMD operations
    println!("\n4. SIMD operation benchmarks:");
    
    // Entropy calculation
    let start = Instant::now();
    let mut entropy_sum = 0.0;
    for _ in 0..10000 {
        entropy_sum += m1_graph.entropy_action();
    }
    let simd_entropy_time = start.elapsed();
    
    let start = Instant::now();
    let mut entropy_sum2 = 0.0;
    for _ in 0..10000 {
        entropy_sum2 += fast_graph.entropy_action();
    }
    let fast_entropy_time = start.elapsed();
    
    println!("   Entropy calculation:");
    println!("     Fast: {:.2} μs/call", fast_entropy_time.as_micros() as f64 / 10000.0);
    println!("     SIMD: {:.2} μs/call", simd_entropy_time.as_micros() as f64 / 10000.0);
    println!("     Speedup: {:.1}x", 
        fast_entropy_time.as_secs_f64() / simd_entropy_time.as_secs_f64());
    
    // Triangle sum calculation
    let start = Instant::now();
    let mut tri_sum = 0.0;
    for _ in 0..100 {
        tri_sum += m1_graph.triangle_sum();
    }
    let m1_tri_time = start.elapsed();
    
    let start = Instant::now();
    let mut tri_sum2 = 0.0;
    for _ in 0..100 {
        tri_sum2 += fast_graph.triangle_sum();
    }
    let fast_tri_time = start.elapsed();
    
    println!("   Triangle sum:");
    println!("     Fast: {:.2} μs/call", fast_tri_time.as_micros() as f64 / 100.0);
    println!("     M1 parallel: {:.2} μs/call", m1_tri_time.as_micros() as f64 / 100.0);
    println!("     Speedup: {:.1}x", 
        fast_tri_time.as_secs_f64() / m1_tri_time.as_secs_f64());
    
    // Observable calculations
    println!("\n5. SIMD observable calculations:");
    let (mean_cos, mean_w, mean_w_cos) = m1_graph.calculate_observables_simd();
    println!("   <cos θ> = {:.6}", mean_cos);
    println!("   <w> = {:.6}", mean_w);
    println!("   <w cos θ> = {:.6}", mean_w_cos);
    
    // Memory usage
    println!("\n6. Memory efficiency:");
    let orig_link_size = std::mem::size_of::<scan::graph::Link>();
    let fast_link_size = std::mem::size_of::<scan::graph_fast::FastLink>();
    
    #[cfg(target_arch = "aarch64")]
    let m1_link_size = std::mem::size_of::<scan::graph_m1_optimized::M1Link>();
    #[cfg(not(target_arch = "aarch64"))]
    let m1_link_size = 128; // Size on M1
    
    println!("   Original Link: {} bytes", orig_link_size);
    println!("   Fast Link: {} bytes", fast_link_size);
    println!("   M1 Link: {} bytes (cache-aligned)", m1_link_size);
    
    let n_links = n * (n - 1) / 2;
    println!("   Total link memory (N={}):", n);
    println!("     Original: {:.1} KB", (orig_link_size * n_links) as f64 / 1024.0);
    println!("     Fast: {:.1} KB", (fast_link_size * n_links) as f64 / 1024.0);
    println!("     M1: {:.1} KB", (m1_link_size * n_links) as f64 / 1024.0);
    
    // Summary
    println!("\n7. Summary:");
    println!("   Total speedup (M1 vs original): {:.1}x", m1_rate / orig_rate);
    println!("   Performance: {:.2} million steps/sec", m1_rate / 1e6);
    
    // Detailed M1 benchmark
    println!("\n8. Detailed M1 optimization test:");
    benchmark_m1_optimizations(n, 10000);
}