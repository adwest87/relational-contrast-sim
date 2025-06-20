// Benchmark Metal GPU acceleration on Apple Silicon

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("This benchmark requires macOS with Metal support");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn main() {
    use scan::graph::Graph;
    use scan::graph_fast::FastGraph;
    use scan::graph_metal::{MetalGraph, benchmark_metal_gpu};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_pcg::Pcg64;
    use std::time::Instant;
    
    println!("Metal GPU Acceleration Benchmark");
    println!("================================");
    println!("Architecture: {}", std::env::consts::ARCH);
    println!("OS: macOS with Metal support");
    
    let n = 96;  // Larger system to show GPU benefits
    let steps = 10_000;
    let alpha = 1.5;
    let beta = 2.9;
    let seed = 12345;
    
    println!("\nTest parameters:");
    println!("  N = {} (larger system for GPU)", n);
    println!("  Steps = {}", steps);
    println!("  α = {}, β = {}", alpha, beta);
    
    // Create initial graph for fair comparison
    let mut init_rng = ChaCha20Rng::seed_from_u64(seed);
    let init_graph = Graph::complete_random_with(&mut init_rng, n);
    
    // 1. CPU Baseline (Fast implementation)
    println!("\n1. CPU Baseline (Fast implementation):");
    let mut cpu_graph = FastGraph::from_graph(&init_graph);
    let mut cpu_rng = Pcg64::seed_from_u64(seed);
    
    let start = Instant::now();
    let mut cpu_accepts = 0;
    for _ in 0..steps {
        let info = cpu_graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut cpu_rng);
        if info.accept {
            cpu_accepts += 1;
        }
    }
    let cpu_time = start.elapsed();
    let cpu_rate = steps as f64 / cpu_time.as_secs_f64();
    
    println!("   Time: {:.3} s", cpu_time.as_secs_f64());
    println!("   Rate: {:.0} steps/sec", cpu_rate);
    println!("   Accept: {:.1}%", 100.0 * cpu_accepts as f64 / steps as f64);
    
    // CPU function timings
    println!("\n   CPU function timings (100 calls):");
    
    let start = Instant::now();
    for _ in 0..100 {
        let _ = cpu_graph.entropy_action();
    }
    let cpu_entropy_time = start.elapsed();
    println!("     Entropy: {:.2} μs/call", cpu_entropy_time.as_micros() as f64 / 100.0);
    
    let start = Instant::now();
    for _ in 0..100 {
        let _ = cpu_graph.triangle_sum();
    }
    let cpu_triangle_time = start.elapsed();
    println!("     Triangle sum: {:.2} μs/call", cpu_triangle_time.as_micros() as f64 / 100.0);
    
    // 2. GPU Implementation
    println!("\n2. GPU Implementation (Metal):");
    
    match MetalGraph::from_graph(&init_graph) {
        Ok(mut gpu_graph) => {
            // Warmup
            println!("   Warming up GPU...");
            for _ in 0..100 {
                gpu_graph.metropolis_step_gpu(alpha, beta, 0.1, 0.1);
            }
            
            // Benchmark single-step Metropolis updates (old method)
            println!("\n   Single-step kernel (old method):");
            let start = Instant::now();
            let mut gpu_accepts = 0;
            for _ in 0..steps {
                let accepts = gpu_graph.metropolis_step_gpu(alpha, beta, 0.1, 0.1);
                gpu_accepts += accepts;
            }
            let gpu_time_single = start.elapsed();
            let gpu_rate_single = steps as f64 / gpu_time_single.as_secs_f64();
            
            println!("     Time: {:.3} s", gpu_time_single.as_secs_f64());
            println!("     Rate: {:.0} steps/sec", gpu_rate_single);
            println!("     Rate: {:.2} million steps/sec", gpu_rate_single / 1e6);
            
            // Benchmark batched Metropolis updates (new method)
            println!("\n   Batched kernel (new method):");
            let batch_size = 1000u32;  // Process 1000 MC steps per kernel launch
            let num_batches = (steps + batch_size as usize - 1) / batch_size as usize;
            let actual_steps = num_batches * batch_size as usize;
            
            let start = Instant::now();
            let mut gpu_accepts_batched = 0;
            for _ in 0..num_batches {
                let accepts = gpu_graph.metropolis_steps_gpu_batched(alpha, beta, 0.1, 0.1, batch_size);
                gpu_accepts_batched += accepts;
            }
            let gpu_time = start.elapsed();
            let gpu_rate = actual_steps as f64 / gpu_time.as_secs_f64();
            
            println!("     Batch size: {} steps/kernel", batch_size);
            println!("     Time: {:.3} s", gpu_time.as_secs_f64());
            println!("     Rate: {:.0} steps/sec", gpu_rate);
            println!("     Rate: {:.2} million steps/sec", gpu_rate / 1e6);
            println!("     Speedup vs single-step: {:.1}x", gpu_rate / gpu_rate_single);
            
            // Note: GPU processes all links in parallel per step
            let gpu_link_updates = (actual_steps * n * (n - 1) / 2) as f64;
            let gpu_link_rate = gpu_link_updates / gpu_time.as_secs_f64();
            println!("\n   GPU performance metrics:");
            println!("     Link updates/sec: {:.2} million", gpu_link_rate / 1e6);
            println!("     Accept rate: {:.1}%", 100.0 * gpu_accepts_batched as f64 / gpu_link_updates);
            println!("     Speedup vs CPU: {:.1}x", gpu_rate / cpu_rate);
            
            // GPU function timings
            println!("\n   GPU function timings (100 calls):");
            
            let start = Instant::now();
            for _ in 0..100 {
                let _ = gpu_graph.entropy_action_gpu();
            }
            let gpu_entropy_time = start.elapsed();
            println!("     Entropy: {:.2} μs/call", gpu_entropy_time.as_micros() as f64 / 100.0);
            println!("     Speedup: {:.1}x", cpu_entropy_time.as_secs_f64() / gpu_entropy_time.as_secs_f64());
            
            let start = Instant::now();
            for _ in 0..100 {
                let _ = gpu_graph.triangle_sum_gpu();
            }
            let gpu_triangle_time = start.elapsed();
            println!("     Triangle sum: {:.2} μs/call", gpu_triangle_time.as_micros() as f64 / 100.0);
            println!("     Speedup: {:.1}x", cpu_triangle_time.as_secs_f64() / gpu_triangle_time.as_secs_f64());
            
            // Test observables computation
            println!("\n   GPU observables computation:");
            let start = Instant::now();
            let (mean_cos, mean_w, mean_w_cos) = gpu_graph.compute_observables_gpu();
            let obs_time = start.elapsed();
            println!("     Time: {:.2} μs", obs_time.as_micros() as f64);
            println!("     <cos θ> = {:.6}", mean_cos);
            println!("     <w> = {:.6}", mean_w);
            println!("     <w cos θ> = {:.6}", mean_w_cos);
            
            // Memory bandwidth estimate
            let bytes_per_link = std::mem::size_of::<scan::graph_metal::MetalLink>();
            let total_bytes = bytes_per_link * n * (n - 1) / 2;
            let bandwidth_gb = (total_bytes as f64 * steps as f64) / (gpu_time.as_secs_f64() * 1e9);
            println!("\n   Estimated memory bandwidth: {:.1} GB/s", bandwidth_gb);
            
            // Summary
            println!("\n3. Summary:");
            println!("   System size: N = {}", n);
            println!("   Links: {}", n * (n - 1) / 2);
            println!("   Triangles: {}", n * (n - 1) * (n - 2) / 6);
            println!("   CPU rate: {:.0} steps/sec", cpu_rate);
            println!("   GPU rate: {:.0} steps/sec ({:.1}x speedup)", gpu_rate, gpu_rate / cpu_rate);
            println!("   GPU processes all {} links in parallel per step", n * (n - 1) / 2);
            
            // Run detailed benchmark
            println!("\n4. Detailed GPU benchmark:");
            if let Err(e) = benchmark_metal_gpu(48, 10000) {
                eprintln!("   GPU benchmark error: {}", e);
            }
        }
        Err(e) => {
            eprintln!("   Failed to create Metal graph: {}", e);
            eprintln!("   Make sure you're running on macOS with Metal support");
        }
    }
}