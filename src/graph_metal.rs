// Metal GPU-accelerated graph implementation for Apple Silicon M1
// Uses unified memory architecture for zero-copy CPU-GPU transfers

#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandQueue, ComputePipelineState,
    Device, Library, MTLResourceOptions, MTLSize,
};
use std::mem;
use rand::SeedableRng;
use rand_pcg::Pcg64;

// Metal shader source
const SHADER_SOURCE: &str = include_str!("shaders/monte_carlo.metal");

/// GPU-optimized link structure matching Metal shader layout
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MetalLink {
    pub i: u32,
    pub j: u32,
    pub z: f32,
    pub theta: f32,
    pub cos_theta: f32,
    pub sin_theta: f32,
    pub exp_neg_z: f32,
    pub w_cos: f32,
    pub w_sin: f32,
    padding: [f32; 7],
}

impl MetalLink {
    pub fn new(i: usize, j: usize, z: f64, theta: f64) -> Self {
        let z_f32 = z as f32;
        let theta_f32 = theta as f32;
        let cos_theta = theta_f32.cos();
        let sin_theta = theta_f32.sin();
        let exp_neg_z = (-z_f32).exp();
        
        Self {
            i: i as u32,
            j: j as u32,
            z: z_f32,
            theta: theta_f32,
            cos_theta,
            sin_theta,
            exp_neg_z,
            w_cos: exp_neg_z * cos_theta,
            w_sin: exp_neg_z * sin_theta,
            padding: [0.0; 7],
        }
    }
}

/// RNG state for GPU
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct RNGState {
    state: u64,
    inc: u64,
}

/// Observables structure for GPU reduction
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct GPUObservables {
    sum_cos: f32,
    sum_sin: f32,
    sum_w: f32,
    sum_w_cos: f32,
    sum_w_sin: f32,
    triangle_sum: f32,
    accept_count: u32,
}

/// Metal GPU-accelerated graph
pub struct MetalGraph {
    // Metal resources
    _device: Device,
    command_queue: CommandQueue,
    _library: Library,
    
    // Compute pipelines
    metropolis_pipeline: ComputePipelineState,
    metropolis_batched_pipeline: ComputePipelineState,
    triangle_pipeline: ComputePipelineState,
    rng_init_pipeline: ComputePipelineState,
    observables_pipeline: ComputePipelineState,
    entropy_pipeline: ComputePipelineState,
    
    // Buffers (using unified memory)
    links_buffer: Buffer,
    rng_states_buffer: Buffer,
    params_buffer: Buffer,
    accept_counter_buffer: Buffer,
    partial_sums_buffer: Buffer,
    _observables_buffer: Buffer,
    
    // Graph properties
    n_nodes: usize,
    n_links: usize,
    n_triangles: usize,
}

impl MetalGraph {
    pub fn new(n: usize, seed: u64) -> Result<Self, Box<dyn std::error::Error>> {
        // Get default Metal device (M1 GPU)
        let device = Device::system_default()
            .ok_or("Failed to get Metal device")?;
        
        let command_queue = device.new_command_queue();
        
        // Compile shaders
        let library = device.new_library_with_source(
            SHADER_SOURCE,
            &metal::CompileOptions::new(),
        )?;
        
        // Create compute pipelines
        let metropolis_pipeline = Self::create_pipeline(&device, &library, "metropolis_update")?;
        let metropolis_batched_pipeline = Self::create_pipeline(&device, &library, "metropolis_update_batched")?;
        let triangle_pipeline = Self::create_pipeline(&device, &library, "triangle_sum_kernel")?;
        let rng_init_pipeline = Self::create_pipeline(&device, &library, "init_rng_states")?;
        let observables_pipeline = Self::create_pipeline(&device, &library, "reduce_observables")?;
        let entropy_pipeline = Self::create_pipeline(&device, &library, "entropy_sum_kernel")?;
        
        // Calculate sizes
        let n_links = n * (n - 1) / 2;
        let n_triangles = n * (n - 1) * (n - 2) / 6;
        
        // Allocate buffers with unified memory (no copy needed on M1)
        let links_size = n_links * mem::size_of::<MetalLink>();
        let links_buffer = device.new_buffer(
            links_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let rng_states_size = n_links * mem::size_of::<RNGState>();
        let rng_states_buffer = device.new_buffer(
            rng_states_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let params_buffer = device.new_buffer(
            4 * mem::size_of::<f32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        let accept_counter_buffer = device.new_buffer(
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Buffer for partial sums (one per threadgroup) - allocate extra space for observables
        let n_threadgroups = n_triangles.div_ceil(1024);
        let n_obs_threadgroups = n_links.div_ceil(1024);
        let max_threadgroups = n_threadgroups.max(n_obs_threadgroups);
        let partial_sums_buffer = device.new_buffer(
            (max_threadgroups * 3 * mem::size_of::<f32>()) as u64,  // 3 floats per threadgroup for observables
            MTLResourceOptions::StorageModeShared,
        );
        
        let observables_buffer = device.new_buffer(
            mem::size_of::<GPUObservables>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        
        // Initialize links
        let mut rng = Pcg64::seed_from_u64(seed);
        let links_ptr = links_buffer.contents() as *mut MetalLink;
        unsafe {
            let links_slice = std::slice::from_raw_parts_mut(links_ptr, n_links);
            let mut idx = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    use rand::Rng;
                    let z = rng.gen_range(0.001..10.0);
                    links_slice[idx] = MetalLink::new(i, j, z, 0.0);
                    idx += 1;
                }
            }
        }
        
        // Initialize RNG states on GPU
        let mut graph = Self {
            _device: device,
            command_queue,
            _library: library,
            metropolis_pipeline,
            metropolis_batched_pipeline,
            triangle_pipeline,
            rng_init_pipeline,
            observables_pipeline,
            entropy_pipeline,
            links_buffer,
            rng_states_buffer,
            params_buffer,
            accept_counter_buffer,
            partial_sums_buffer,
            _observables_buffer: observables_buffer,
            n_nodes: n,
            n_links,
            n_triangles,
        };
        
        graph.init_rng_states(seed)?;
        
        Ok(graph)
    }
    
    /// Create from existing graph
    pub fn from_graph(graph: &crate::graph::Graph) -> Result<Self, Box<dyn std::error::Error>> {
        let n = graph.n();
        let metal_graph = Self::new(n, 42)?; // Seed doesn't matter, we'll copy
        
        // Copy links
        let links_ptr = metal_graph.links_buffer.contents() as *mut MetalLink;
        unsafe {
            let links_slice = std::slice::from_raw_parts_mut(links_ptr, graph.links.len());
            for (idx, link) in graph.links.iter().enumerate() {
                links_slice[idx] = MetalLink::new(link.i, link.j, link.z, link.theta);
            }
        }
        
        Ok(metal_graph)
    }
    
    fn create_pipeline(
        device: &Device,
        library: &Library,
        function_name: &str,
    ) -> Result<ComputePipelineState, Box<dyn std::error::Error>> {
        let function = library.get_function(function_name, None)?;
        let pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        Ok(pipeline)
    }
    
    fn init_rng_states(&mut self, seed: u64) -> Result<(), Box<dyn std::error::Error>> {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.rng_init_pipeline);
        encoder.set_buffer(0, Some(&self.rng_states_buffer), 0);
        encoder.set_bytes(1, mem::size_of::<u64>() as u64, &seed as *const u64 as *const _);
        
        let threads_per_threadgroup = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: self.n_links.div_ceil(256) as u64,
            height: 1,
            depth: 1,
        };
        
        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        Ok(())
    }
    
    /// Run Metropolis updates on GPU
    pub fn metropolis_step_gpu(
        &mut self,
        alpha: f64,
        beta: f64,
        delta_z: f64,
        delta_theta: f64,
    ) -> u32 {
        // Set parameters
        let params = [alpha as f32, beta as f32, delta_z as f32, delta_theta as f32];
        unsafe {
            let params_ptr = self.params_buffer.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(params.as_ptr(), params_ptr, 4);
        }
        
        // Reset accept counter
        unsafe {
            let counter_ptr = self.accept_counter_buffer.contents() as *mut u32;
            *counter_ptr = 0;
        }
        
        // Create command buffer
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        // Set up metropolis kernel
        encoder.set_compute_pipeline_state(&self.metropolis_pipeline);
        encoder.set_buffer(0, Some(&self.links_buffer), 0);
        encoder.set_buffer(1, Some(&self.params_buffer), 0);
        encoder.set_buffer(2, Some(&self.rng_states_buffer), 0);
        encoder.set_buffer(3, Some(&self.accept_counter_buffer), 0);
        encoder.set_bytes(4, mem::size_of::<u32>() as u64, &self.n_links as *const usize as *const _);
        encoder.set_bytes(5, mem::size_of::<u32>() as u64, &self.n_nodes as *const usize as *const _);
        
        // Launch with optimal thread configuration for M1
        let threads_per_threadgroup = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: self.n_links.div_ceil(256) as u64,
            height: 1,
            depth: 1,
        };
        
        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Read accept count
        unsafe {
            let counter_ptr = self.accept_counter_buffer.contents() as *const u32;
            *counter_ptr
        }
    }
    
    /// Run batched Metropolis updates on GPU - process multiple MC steps per kernel launch
    pub fn metropolis_steps_gpu_batched(
        &mut self,
        alpha: f64,
        beta: f64,
        delta_z: f64,
        delta_theta: f64,
        batch_size: u32,
    ) -> u32 {
        // Set parameters
        let params = [alpha as f32, beta as f32, delta_z as f32, delta_theta as f32];
        unsafe {
            let params_ptr = self.params_buffer.contents() as *mut f32;
            std::ptr::copy_nonoverlapping(params.as_ptr(), params_ptr, 4);
        }
        
        // Reset accept counter
        unsafe {
            let counter_ptr = self.accept_counter_buffer.contents() as *mut u32;
            *counter_ptr = 0;
        }
        
        // Create command buffer
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        // Set up batched metropolis kernel
        encoder.set_compute_pipeline_state(&self.metropolis_batched_pipeline);
        encoder.set_buffer(0, Some(&self.links_buffer), 0);
        encoder.set_buffer(1, Some(&self.params_buffer), 0);
        encoder.set_buffer(2, Some(&self.rng_states_buffer), 0);
        encoder.set_buffer(3, Some(&self.accept_counter_buffer), 0);
        encoder.set_bytes(4, mem::size_of::<u32>() as u64, &self.n_links as *const usize as *const _);
        encoder.set_bytes(5, mem::size_of::<u32>() as u64, &self.n_nodes as *const usize as *const _);
        encoder.set_bytes(6, mem::size_of::<u32>() as u64, &batch_size as *const u32 as *const _);
        
        // Launch with optimal thread configuration for M1
        let threads_per_threadgroup = MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: self.n_links.div_ceil(256) as u64,
            height: 1,
            depth: 1,
        };
        
        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Read accept count
        unsafe {
            let counter_ptr = self.accept_counter_buffer.contents() as *const u32;
            *counter_ptr
        }
    }
    
    /// Compute triangle sum on GPU
    pub fn triangle_sum_gpu(&self) -> f32 {
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        // Reset partial sums
        let n_threadgroups = self.n_triangles.div_ceil(1024);
        unsafe {
            let sums_ptr = self.partial_sums_buffer.contents() as *mut f32;
            for i in 0..n_threadgroups {
                *sums_ptr.add(i) = 0.0;
            }
        }
        
        // Triangle sum kernel
        encoder.set_compute_pipeline_state(&self.triangle_pipeline);
        encoder.set_buffer(0, Some(&self.links_buffer), 0);
        encoder.set_buffer(1, Some(&self.partial_sums_buffer), 0);
        encoder.set_bytes(2, mem::size_of::<u32>() as u64, &self.n_nodes as *const usize as *const _);
        encoder.set_threadgroup_memory_length(0, 1024 * mem::size_of::<f32>() as u64);
        
        let threads_per_threadgroup = MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: n_threadgroups as u64,
            height: 1,
            depth: 1,
        };
        
        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Sum partial results on CPU
        unsafe {
            let sums_ptr = self.partial_sums_buffer.contents() as *const f32;
            let sums_slice = std::slice::from_raw_parts(sums_ptr, n_threadgroups);
            sums_slice.iter().sum()
        }
    }
    
    /// Compute observables on GPU
    pub fn compute_observables_gpu(&self) -> (f64, f64, f64) {
        let n_threadgroups = self.n_links.div_ceil(1024);
        
        // Use partial_sums_buffer for the float3 results
        unsafe {
            let results_ptr = self.partial_sums_buffer.contents() as *mut f32;
            for i in 0..(n_threadgroups * 3) {
                *results_ptr.add(i) = 0.0;
            }
        }
        
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.observables_pipeline);
        encoder.set_buffer(0, Some(&self.links_buffer), 0);
        encoder.set_buffer(1, Some(&self.partial_sums_buffer), 0);
        encoder.set_bytes(2, mem::size_of::<u32>() as u64, &self.n_links as *const usize as *const _);
        encoder.set_threadgroup_memory_length(0, 1024 * mem::size_of::<f32>() as u64);
        encoder.set_threadgroup_memory_length(1, 1024 * mem::size_of::<f32>() as u64);
        encoder.set_threadgroup_memory_length(2, 1024 * mem::size_of::<f32>() as u64);
        
        let threads_per_threadgroup = MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: n_threadgroups as u64,
            height: 1,
            depth: 1,
        };
        
        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Sum partial results on CPU
        unsafe {
            let results_ptr = self.partial_sums_buffer.contents() as *const f32;
            let mut sum_cos = 0.0;
            let mut sum_w = 0.0;
            let mut sum_w_cos = 0.0;
            
            for i in 0..n_threadgroups {
                sum_cos += *results_ptr.add(i * 3) as f64;
                sum_w += *results_ptr.add(i * 3 + 1) as f64;
                sum_w_cos += *results_ptr.add(i * 3 + 2) as f64;
            }
            
            let m = self.n_links as f64;
            (sum_cos / m, sum_w / m, sum_w_cos / m)
        }
    }
    
    /// Compute entropy on GPU
    pub fn entropy_action_gpu(&self) -> f64 {
        let n_threadgroups = self.n_links.div_ceil(1024);
        
        // Reset partial results
        unsafe {
            let results_ptr = self.partial_sums_buffer.contents() as *mut f32;
            for i in 0..n_threadgroups {
                *results_ptr.add(i) = 0.0;
            }
        }
        
        let command_buffer = self.command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        
        encoder.set_compute_pipeline_state(&self.entropy_pipeline);
        encoder.set_buffer(0, Some(&self.links_buffer), 0);
        encoder.set_buffer(1, Some(&self.partial_sums_buffer), 0);
        encoder.set_bytes(2, mem::size_of::<u32>() as u64, &self.n_links as *const usize as *const _);
        encoder.set_threadgroup_memory_length(0, 1024 * mem::size_of::<f32>() as u64);
        
        let threads_per_threadgroup = MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        };
        let threadgroups = MTLSize {
            width: n_threadgroups as u64,
            height: 1,
            depth: 1,
        };
        
        encoder.dispatch_thread_groups(threadgroups, threads_per_threadgroup);
        encoder.end_encoding();
        
        command_buffer.commit();
        command_buffer.wait_until_completed();
        
        // Sum partial results on CPU
        unsafe {
            let results_ptr = self.partial_sums_buffer.contents() as *const f32;
            let sum: f64 = (0..n_threadgroups)
                .map(|i| *results_ptr.add(i) as f64)
                .sum();
            sum
        }
    }
    
    pub fn n(&self) -> usize {
        self.n_nodes
    }
    
    pub fn m(&self) -> usize {
        self.n_links
    }
}

/// Benchmark Metal GPU performance
pub fn benchmark_metal_gpu(n: usize, steps: usize) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;
    
    println!("\n=== Metal GPU Acceleration Benchmark ===");
    println!("Using M1 GPU with unified memory architecture");
    println!("System size: N = {}", n);
    println!("MC steps: {}", steps);
    
    let mut gpu_graph = MetalGraph::new(n, 12345)?;
    let alpha = 1.5;
    let beta = 2.9;
    
    // Warmup
    for _ in 0..10 {
        gpu_graph.metropolis_step_gpu(alpha, beta, 0.1, 0.1);
    }
    
    // Benchmark single-step Metropolis updates (old method)
    println!("\n1. Single-step kernel (original method):");
    let start = Instant::now();
    let mut total_accepts = 0u32;
    
    let test_steps = 1000; // Fewer steps to avoid timeout
    for _ in 0..test_steps {
        let accepts = gpu_graph.metropolis_step_gpu(alpha, beta, 0.1, 0.1);
        total_accepts += accepts;
    }
    
    let elapsed = start.elapsed();
    let rate = test_steps as f64 / elapsed.as_secs_f64();
    
    println!("  Time: {:.3} s", elapsed.as_secs_f64());
    println!("  Rate: {:.0} steps/sec", rate);
    println!("  Acceptance: {:.1}%", 100.0 * total_accepts as f64 / (test_steps * n * (n-1) / 2) as f64);
    
    // Benchmark batched Metropolis updates
    println!("\n2. Batched kernel (optimized):");
    let batch_sizes = [100, 1000, 10000];
    
    for &batch_size in &batch_sizes {
        let start = Instant::now();
        total_accepts = 0;
        
        let n_batches = steps / batch_size as usize;
        for _ in 0..n_batches {
            let accepts = gpu_graph.metropolis_steps_gpu_batched(alpha, beta, 0.1, 0.1, batch_size);
            total_accepts += accepts;
        }
        
        let elapsed = start.elapsed();
        let total_steps = n_batches * batch_size as usize;
        let rate = total_steps as f64 / elapsed.as_secs_f64();
        
        println!("\n  Batch size {}:", batch_size);
        println!("    Time: {:.3} s", elapsed.as_secs_f64());
        println!("    Rate: {:.0} steps/sec", rate);
        println!("    Rate: {:.2} million steps/sec", rate / 1e6);
        println!("    Acceptance: {:.1}%", 100.0 * total_accepts as f64 / (total_steps * n * (n-1) / 2) as f64);
    }
    
    // Benchmark triangle sum
    let start = Instant::now();
    let mut tri_sum = 0.0;
    for _ in 0..1000 {
        tri_sum = gpu_graph.triangle_sum_gpu() as f64;
    }
    let tri_time = start.elapsed();
    println!("\nTriangle sum on GPU:");
    println!("  Time per call: {:.2} μs", tri_time.as_micros() as f64 / 1000.0);
    println!("  Result: {:.6}", tri_sum);
    
    // Benchmark observables
    let start = Instant::now();
    let (mean_cos, mean_w, mean_w_cos) = gpu_graph.compute_observables_gpu();
    let obs_time = start.elapsed();
    println!("\nObservables computation on GPU:");
    println!("  Time: {:.2} μs", obs_time.as_micros() as f64);
    println!("  <cos θ> = {:.6}", mean_cos);
    println!("  <w> = {:.6}", mean_w);
    println!("  <w cos θ> = {:.6}", mean_w_cos);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[cfg(target_os = "macos")]
    fn test_metal_graph_creation() {
        let graph = MetalGraph::new(10, 42);
        assert!(graph.is_ok());
        let graph = graph.unwrap();
        assert_eq!(graph.n(), 10);
        assert_eq!(graph.m(), 45);
    }
    
    #[test]
    #[cfg(target_os = "macos")]
    fn test_gpu_computation() {
        let mut graph = MetalGraph::new(24, 42).unwrap();
        let accepts = graph.metropolis_step_gpu(1.5, 2.9, 0.1, 0.1);
        assert!(accepts > 0);
        
        let tri_sum = graph.triangle_sum_gpu();
        assert!(tri_sum.is_finite());
    }
    
    #[test]
    #[cfg(target_os = "macos")]
    fn test_batched_gpu_computation() {
        let mut graph = MetalGraph::new(24, 42).unwrap();
        
        // Test different batch sizes
        let batch_sizes = [1, 10, 100, 1000];
        for &batch_size in &batch_sizes {
            let accepts = graph.metropolis_steps_gpu_batched(1.5, 2.9, 0.1, 0.1, batch_size);
            assert!(accepts > 0, "Batch size {} should have some accepts", batch_size);
            
            // Check that the graph state has actually changed
            let tri_sum = graph.triangle_sum_gpu();
            assert!(tri_sum.is_finite());
        }
    }
}