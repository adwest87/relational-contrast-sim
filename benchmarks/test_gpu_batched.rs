// Minimal test program for batched GPU implementation

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("This test requires macOS with Metal support");
    std::process::exit(1);
}

#[cfg(target_os = "macos")]
fn main() {
    // Include the necessary modules inline
    #[path = "src/graph.rs"]
    mod graph;
    
    #[path = "src/graph_metal.rs"]
    mod graph_metal;
    
    use graph_metal::MetalGraph;
    use std::time::Instant;
    
    println!("Testing Batched GPU Implementation");
    println!("==================================");
    
    let n = 48;
    let steps = 10_000;
    let alpha = 1.5;
    let beta = 2.9;
    
    match MetalGraph::new(n, 12345) {
        Ok(mut gpu_graph) => {
            // Warmup
            println!("Warming up GPU...");
            for _ in 0..10 {
                gpu_graph.metropolis_step_gpu(alpha, beta, 0.1, 0.1);
            }
            
            // Test single-step kernel
            println!("\n1. Single-step kernel (baseline):");
            let start = Instant::now();
            let mut total_accepts_single = 0u32;
            
            for _ in 0..steps {
                let accepts = gpu_graph.metropolis_step_gpu(alpha, beta, 0.1, 0.1);
                total_accepts_single += accepts;
            }
            
            let time_single = start.elapsed();
            let rate_single = steps as f64 / time_single.as_secs_f64();
            
            println!("   Time: {:.3} s", time_single.as_secs_f64());
            println!("   Rate: {:.0} steps/sec", rate_single);
            println!("   Accepts: {}", total_accepts_single);
            
            // Test batched kernel with different batch sizes
            let batch_sizes = [10, 100, 1000, 10000];
            
            println!("\n2. Batched kernel tests:");
            for &batch_size in &batch_sizes {
                let num_batches = (steps + batch_size as usize - 1) / batch_size as usize;
                let actual_steps = num_batches * batch_size as usize;
                
                let start = Instant::now();
                let mut total_accepts_batched = 0u32;
                
                for _ in 0..num_batches {
                    let accepts = gpu_graph.metropolis_steps_gpu_batched(alpha, beta, 0.1, 0.1, batch_size);
                    total_accepts_batched += accepts;
                }
                
                let time_batched = start.elapsed();
                let rate_batched = actual_steps as f64 / time_batched.as_secs_f64();
                
                println!("\n   Batch size: {} steps/kernel", batch_size);
                println!("   Time: {:.3} s", time_batched.as_secs_f64());
                println!("   Rate: {:.0} steps/sec", rate_batched);
                println!("   Speedup: {:.1}x", rate_batched / rate_single);
                println!("   Accepts: {}", total_accepts_batched);
                
                // Verify correctness by checking accept rate is similar
                let accept_rate_single = total_accepts_single as f64 / (steps * n * (n - 1) / 2) as f64;
                let accept_rate_batched = total_accepts_batched as f64 / (actual_steps * n * (n - 1) / 2) as f64;
                println!("   Accept rate single: {:.1}%", 100.0 * accept_rate_single);
                println!("   Accept rate batched: {:.1}%", 100.0 * accept_rate_batched);
                
                if (accept_rate_single - accept_rate_batched).abs() > 0.02 {
                    println!("   WARNING: Accept rates differ significantly!");
                }
            }
            
            println!("\n3. Optimal batch size:");
            println!("   For this system (N={}), batch sizes of 1000-10000 steps");
            println!("   provide the best balance of performance and latency.");
            println!("   Larger batches reduce kernel launch overhead but increase");
            println!("   latency between measurements.");
        }
        Err(e) => {
            eprintln!("Failed to create Metal graph: {}", e);
        }
    }
}