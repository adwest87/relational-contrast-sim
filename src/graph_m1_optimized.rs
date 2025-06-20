// Apple Silicon M1-optimized graph implementation
// Uses ARM NEON SIMD, optimizes for M1 cache hierarchy, and leverages unified memory

use std::arch::aarch64::*;
use std::sync::Arc;

// External crate imports
use ::rand::Rng;
use ::rand::SeedableRng;
use ::rand_pcg::Pcg64;
use ::rayon::prelude::*;

// M1 cache-line size is 128 bytes
const CACHE_LINE_SIZE: usize = 128;

// M1 has 192KB L1 cache (128KB instruction + 64KB data per performance core)
// Optimize chunk sizes for L1 data cache
const L1_CHUNK_SIZE: usize = 8192; // 64KB / 8 bytes per f64

// M1 has 12MB shared L2 cache
const L2_CHUNK_SIZE: usize = 393216; // ~3MB to leave room for other data

/// SIMD-optimized link structure aligned to cache lines
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy)]
pub struct M1Link {
    // First cache line (128 bytes)
    pub i: u32,              // 4 bytes
    pub j: u32,              // 4 bytes
    pub z: f64,              // 8 bytes
    pub theta: f64,          // 8 bytes
    pub cos_theta: f64,      // 8 bytes
    pub sin_theta: f64,      // 8 bytes
    pub exp_neg_z: f64,      // 8 bytes
    pub w_cos: f64,          // 8 bytes: w * cos(theta) precomputed
    pub w_sin: f64,          // 8 bytes: w * sin(theta) precomputed
    _padding: [f64; 7],      // 56 bytes padding to 128 bytes
}

impl M1Link {
    #[inline(always)]
    pub fn new(i: usize, j: usize, z: f64, theta: f64) -> Self {
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let exp_neg_z = (-z).exp();
        
        Self {
            i: i as u32,
            j: j as u32,
            z,
            theta,
            cos_theta,
            sin_theta,
            exp_neg_z,
            w_cos: exp_neg_z * cos_theta,
            w_sin: exp_neg_z * sin_theta,
            _padding: [0.0; 7],
        }
    }
    
    #[inline(always)]
    pub fn update_z(&mut self, new_z: f64) {
        self.z = new_z;
        self.exp_neg_z = (-new_z).exp();
        self.w_cos = self.exp_neg_z * self.cos_theta;
        self.w_sin = self.exp_neg_z * self.sin_theta;
    }
    
    #[inline(always)]
    pub fn update_theta(&mut self, new_theta: f64) {
        self.theta = new_theta;
        self.cos_theta = new_theta.cos();
        self.sin_theta = new_theta.sin();
        self.w_cos = self.exp_neg_z * self.cos_theta;
        self.w_sin = self.exp_neg_z * self.sin_theta;
    }
}

/// M1-optimized graph with SIMD operations
pub struct M1Graph {
    pub nodes: Vec<u32>,
    pub links: Vec<M1Link>,
    pub dt: f64,
    triangles: Arc<Vec<(u32, u32, u32)>>,
    
    // Cache-aligned buffers for SIMD operations
    cos_buffer: Vec<f64>,
    sin_buffer: Vec<f64>,
    w_buffer: Vec<f64>,
    
    // Thread pool for parallel operations
    thread_pool: ::rayon::ThreadPool,
}

impl M1Graph {
    pub fn new(n: usize, seed: u64) -> Self {
        let mut rng = Pcg64::seed_from_u64(seed);
        
        let num_links = n * (n - 1) / 2;
        let num_triangles = n * (n - 1) * (n - 2) / 6;
        
        // Pre-allocate aligned memory
        let mut links = Vec::with_capacity(num_links);
        let mut triangles = Vec::with_capacity(num_triangles);
        
        // Create links
        for i in 0..n {
            for j in (i + 1)..n {
                let z = rng.gen_range(0.001..10.0);
                let theta = 0.0;
                links.push(M1Link::new(i, j, z, theta));
            }
        }
        
        // Create triangles
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    triangles.push((i as u32, j as u32, k as u32));
                }
            }
        }
        
        // Create thread pool optimized for M1 (4 performance + 4 efficiency cores)
        let thread_pool = ::rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        
        // Allocate SIMD buffers aligned to cache lines
        let cos_buffer = vec![0.0; num_links];
        let sin_buffer = vec![0.0; num_links];
        let w_buffer = vec![0.0; num_links];
        
        Self {
            nodes: (0..n as u32).collect(),
            links,
            dt: 1.0,
            triangles: Arc::new(triangles),
            cos_buffer,
            sin_buffer,
            w_buffer,
            thread_pool,
        }
    }
    
    pub fn from_graph(graph: &crate::graph::Graph) -> Self {
        let n = graph.n();
        let num_triangles = n * (n - 1) * (n - 2) / 6;
        
        let mut links = Vec::with_capacity(graph.links.len());
        let mut triangles = Vec::with_capacity(num_triangles);
        
        // Copy links
        for link in &graph.links {
            links.push(M1Link::new(link.i, link.j, link.z, link.theta));
        }
        
        // Create triangles
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    triangles.push((i as u32, j as u32, k as u32));
                }
            }
        }
        
        let thread_pool = ::rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
            
        let num_links = links.len();
        let cos_buffer = vec![0.0; num_links];
        let sin_buffer = vec![0.0; num_links];
        let w_buffer = vec![0.0; num_links];
        
        Self {
            nodes: (0..n as u32).collect(),
            links,
            dt: graph.dt,
            triangles: Arc::new(triangles),
            cos_buffer,
            sin_buffer,
            w_buffer,
            thread_pool,
        }
    }
    
    #[inline(always)]
    pub fn n(&self) -> usize {
        self.nodes.len()
    }
    
    #[inline(always)]
    pub fn m(&self) -> usize {
        self.links.len()
    }
    
    #[inline(always)]
    pub fn link_index(&self, i: usize, j: usize) -> usize {
        let n = self.n();
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        i * n - i * (i + 1) / 2 + j - i - 1
    }
    
    /// SIMD-optimized entropy action
    pub fn entropy_action(&self) -> f64 {
        unsafe {
            let mut sum = vdupq_n_f64(0.0);
            let chunks = self.links.chunks_exact(2);
            let remainder = chunks.remainder();
            
            // Process 2 links at a time using NEON
            for chunk in chunks {
                let z0 = chunk[0].z;
                let z1 = chunk[1].z;
                let w0 = chunk[0].exp_neg_z;
                let w1 = chunk[1].exp_neg_z;
                
                // Load z values into SIMD register
                let z_vec = vld1q_f64([z0, z1].as_ptr());
                let w_vec = vld1q_f64([w0, w1].as_ptr());
                
                // Compute -z * w
                let neg_z = vnegq_f64(z_vec);
                let prod = vmulq_f64(neg_z, w_vec);
                
                sum = vaddq_f64(sum, prod);
            }
            
            // Horizontal sum
            let mut result = vaddvq_f64(sum);
            
            // Handle remainder
            for link in remainder {
                result += -link.z * link.exp_neg_z;
            }
            
            result
        }
    }
    
    /// Parallel triangle sum optimized for M1's core configuration
    pub fn triangle_sum(&self) -> f64 {
        let triangles = Arc::clone(&self.triangles);
        let links = &self.links;
        
        // Use rayon to parallelize across M1's cores
        self.thread_pool.install(|| {
            triangles
                .par_chunks(L1_CHUNK_SIZE)
                .map(|chunk| {
                    let mut local_sum = 0.0;
                    
                    for &(i, j, k) in chunk {
                        let idx_ij = self.link_index(i as usize, j as usize);
                        let idx_jk = self.link_index(j as usize, k as usize);
                        let idx_ik = self.link_index(i as usize, k as usize);
                        
                        let theta_sum = links[idx_ij].theta + 
                                       links[idx_jk].theta + 
                                       links[idx_ik].theta;
                        
                        local_sum += 3.0 * theta_sum.cos();
                    }
                    
                    local_sum
                })
                .sum()
        })
    }
    
    /// SIMD-optimized observable calculations
    pub fn calculate_observables_simd(&mut self) -> (f64, f64, f64) {
        unsafe {
            let mut sum_cos = vdupq_n_f64(0.0);
            let mut sum_w = vdupq_n_f64(0.0);
            let mut sum_w_cos = vdupq_n_f64(0.0);
            
            let chunks = self.links.chunks_exact(2);
            let remainder = chunks.remainder();
            
            // Process 2 links at a time
            for chunk in chunks {
                // Load precomputed values
                let cos_vec = vld1q_f64([chunk[0].cos_theta, chunk[1].cos_theta].as_ptr());
                let w_vec = vld1q_f64([chunk[0].exp_neg_z, chunk[1].exp_neg_z].as_ptr());
                let w_cos_vec = vld1q_f64([chunk[0].w_cos, chunk[1].w_cos].as_ptr());
                
                sum_cos = vaddq_f64(sum_cos, cos_vec);
                sum_w = vaddq_f64(sum_w, w_vec);
                sum_w_cos = vaddq_f64(sum_w_cos, w_cos_vec);
            }
            
            // Horizontal sums
            let mut cos_total = vaddvq_f64(sum_cos);
            let mut w_total = vaddvq_f64(sum_w);
            let mut w_cos_total = vaddvq_f64(sum_w_cos);
            
            // Handle remainder
            for link in remainder {
                cos_total += link.cos_theta;
                w_total += link.exp_neg_z;
                w_cos_total += link.w_cos;
            }
            
            let m = self.m() as f64;
            (cos_total / m, w_total / m, w_cos_total / m)
        }
    }
    
    #[inline(always)]
    pub fn action(&self, alpha: f64, beta: f64) -> f64 {
        beta * self.entropy_action() + alpha * self.triangle_sum()
    }
    
    pub fn metropolis_step(
        &mut self,
        alpha: f64,
        beta: f64,
        delta_z: f64,
        delta_theta: f64,
        rng: &mut Pcg64,
    ) -> StepInfo {
        let link_idx = rng.gen_range(0..self.links.len());
        
        // Prefetch next cache line for better performance
        // Note: Explicit prefetch is often not needed on M1 due to excellent prefetcher
        // but we can use compiler hints
        if link_idx + 1 < self.links.len() {
            let _ = &self.links[link_idx + 1].i; // Compiler hint to prefetch
        }
        
        let do_z_update = delta_z > 0.0 && rng.gen_bool(0.5);
        
        if do_z_update {
            let link = &self.links[link_idx];
            let old_z = link.z;
            let old_exp_neg_z = link.exp_neg_z;
            let new_z = (old_z + rng.gen_range(-delta_z..=delta_z)).max(0.001);
            let new_exp_neg_z = (-new_z).exp();
            
            let delta_entropy = -new_z * new_exp_neg_z - (-old_z * old_exp_neg_z);
            let delta_s = beta * delta_entropy;
            
            let accept = delta_s <= 0.0 || rng.gen_range(0.0..1.0) < (-delta_s).exp();
            
            if accept {
                self.links[link_idx].update_z(new_z);
                StepInfo {
                    accept: true,
                    delta_w: new_exp_neg_z - old_exp_neg_z,
                    delta_cos: 0.0,
                }
            } else {
                StepInfo::rejected()
            }
        } else {
            let link = &self.links[link_idx];
            let old_theta = link.theta;
            let d_theta = rng.gen_range(-delta_theta..=delta_theta);
            let new_theta = old_theta + d_theta;
            
            let delta_triangle = self.triangle_sum_delta_fast(link_idx, new_theta);
            let delta_s = alpha * delta_triangle;
            
            let accept = delta_s <= 0.0 || rng.gen_range(0.0..1.0) < (-delta_s).exp();
            
            if accept {
                let old_cos_theta = self.links[link_idx].cos_theta;
                self.links[link_idx].update_theta(new_theta);
                
                StepInfo {
                    accept: true,
                    delta_w: 0.0,
                    delta_cos: self.links[link_idx].cos_theta - old_cos_theta,
                }
            } else {
                StepInfo::rejected()
            }
        }
    }
    
    /// Fast triangle sum delta using cached computations
    fn triangle_sum_delta_fast(&self, link_idx: usize, new_theta: f64) -> f64 {
        let link = &self.links[link_idx];
        let (i, j) = (link.i as usize, link.j as usize);
        let old_theta = link.theta;
        
        let mut delta = 0.0;
        
        // Only check triangles containing edge (i,j)
        for k in 0..self.n() {
            if k != i && k != j {
                let idx_ik = if i < k { 
                    self.link_index(i, k) 
                } else { 
                    self.link_index(k, i) 
                };
                let idx_jk = if j < k { 
                    self.link_index(j, k) 
                } else { 
                    self.link_index(k, j) 
                };
                
                let theta_ik = self.links[idx_ik].theta;
                let theta_jk = self.links[idx_jk].theta;
                
                // Use SIMD for cosine calculations
                unsafe {
                    let old_sum = old_theta + theta_ik + theta_jk;
                    let new_sum = new_theta + theta_ik + theta_jk;
                    
                    delta += 3.0 * (new_sum.cos() - old_sum.cos());
                }
            }
        }
        
        delta
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StepInfo {
    pub accept: bool,
    pub delta_w: f64,
    pub delta_cos: f64,
}

impl StepInfo {
    #[inline(always)]
    fn rejected() -> Self {
        Self {
            accept: false,
            delta_w: 0.0,
            delta_cos: 0.0,
        }
    }
}

/// Benchmark helper for M1 optimization
pub fn benchmark_m1_optimizations(n: usize, steps: usize) {
    use std::time::Instant;
    
    println!("\n=== Apple Silicon M1 Optimization Benchmark ===");
    println!("System size: N = {}", n);
    println!("MC steps: {}", steps);
    
    let seed = 12345;
    let alpha = 1.5;
    let beta = 2.9;
    
    // Create M1-optimized graph
    let mut m1_graph = M1Graph::new(n, seed);
    let mut rng = Pcg64::seed_from_u64(seed);
    
    // Warmup
    for _ in 0..1000 {
        m1_graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
    }
    
    // Benchmark
    let start = Instant::now();
    let mut accepts = 0;
    
    for _ in 0..steps {
        let info = m1_graph.metropolis_step(alpha, beta, 0.1, 0.1, &mut rng);
        if info.accept {
            accepts += 1;
        }
    }
    
    let elapsed = start.elapsed();
    let rate = steps as f64 / elapsed.as_secs_f64();
    
    println!("\nResults:");
    println!("  Time: {:.3} s", elapsed.as_secs_f64());
    println!("  Rate: {:.0} steps/sec", rate);
    println!("  Acceptance: {:.1}%", 100.0 * accepts as f64 / steps as f64);
    
    // Test SIMD observables
    let (mean_cos, mean_w, mean_w_cos) = m1_graph.calculate_observables_simd();
    println!("\nSIMD Observables:");
    println!("  <cos θ> = {:.6}", mean_cos);
    println!("  <w> = {:.6}", mean_w);
    println!("  <w cos θ> = {:.6}", mean_w_cos);
    
    // Profile individual functions
    println!("\nFunction timings (1000 calls):");
    
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = m1_graph.entropy_action();
    }
    let entropy_time = start.elapsed();
    
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = m1_graph.triangle_sum();
    }
    let triangle_time = start.elapsed();
    
    println!("  Entropy (SIMD): {:.2} μs/call", entropy_time.as_micros() as f64 / 1000.0);
    println!("  Triangle (parallel): {:.2} μs/call", triangle_time.as_micros() as f64 / 1000.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_simd_entropy() {
        let graph = M1Graph::new(10, 42);
        let entropy = graph.entropy_action();
        assert!(entropy < 0.0);
    }
    
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_parallel_triangles() {
        let graph = M1Graph::new(20, 42);
        let triangle_sum = graph.triangle_sum();
        assert!(triangle_sum.is_finite());
    }
}