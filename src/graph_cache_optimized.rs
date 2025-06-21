// Cache-optimized graph implementation with flat vector storage
use rand::Rng;
use std::time::Instant;

/// Cache-friendly link structure with precomputed trigonometric values
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OptimizedLink {
    pub i: u32,           // Using u32 to save space
    pub j: u32,           
    pub z: f32,           // Using f32 for better cache density
    pub theta: f32,       
    pub cos_theta: f32,   // Precomputed cos(theta)
    pub sin_theta: f32,   // Precomputed sin(theta)
    pub exp_neg_z: f32,   // Precomputed exp(-z) = w
}

impl OptimizedLink {
    #[inline(always)]
    pub fn w(&self) -> f32 {
        self.exp_neg_z
    }
    
    #[inline(always)]
    pub fn update_z(&mut self, new_z: f32) {
        self.z = new_z;
        self.exp_neg_z = (-new_z).exp();
    }
    
    #[inline(always)]
    pub fn update_theta(&mut self, new_theta: f32) {
        self.theta = new_theta;
        self.cos_theta = new_theta.cos();
        self.sin_theta = new_theta.sin();
    }
}

/// Cache-optimized complete graph with flat vector storage
pub struct OptimizedGraph {
    pub n: usize,
    pub links: Vec<OptimizedLink>,  // Flat vector with N(N-1)/2 elements
    pub triangles: Vec<(u32, u32, u32)>,  // Using u32 to save space
    pub triangle_links: Vec<[u32; 3]>,    // Precomputed link indices for each triangle
    pub links_per_triangle: Vec<Vec<u32>>, // For each link, which triangles contain it
    pub dt: f32,
}

impl OptimizedGraph {
    /// Create a new complete graph with N nodes
    pub fn new(n: usize, rng: &mut impl Rng) -> Self {
        let num_links = n * (n - 1) / 2;
        let num_triangles = n * (n - 1) * (n - 2) / 6;
        
        // Pre-allocate all vectors with exact capacity
        let mut links = Vec::with_capacity(num_links);
        let mut triangles = Vec::with_capacity(num_triangles);
        let mut triangle_links = Vec::with_capacity(num_triangles);
        let mut links_per_triangle = vec![Vec::new(); num_links];
        
        // Create links with flat indexing
        for i in 0..n {
            for j in (i + 1)..n {
                let z = rng.gen_range(0.001_f32..10.0_f32);
                let theta = 0.0_f32;
                links.push(OptimizedLink {
                    i: i as u32,
                    j: j as u32,
                    z,
                    theta,
                    cos_theta: theta.cos(),
                    sin_theta: theta.sin(),
                    exp_neg_z: (-z).exp(),
                });
            }
        }
        
        // Pre-compute triangles and their link indices
        let mut tri_idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    triangles.push((i as u32, j as u32, k as u32));
                    
                    // Pre-compute link indices for this triangle
                    let link_ij = Self::compute_link_index(n, i, j) as u32;
                    let link_jk = Self::compute_link_index(n, j, k) as u32;
                    let link_ik = Self::compute_link_index(n, i, k) as u32;
                    
                    triangle_links.push([link_ij, link_jk, link_ik]);
                    
                    // Record which triangles contain each link
                    links_per_triangle[link_ij as usize].push(tri_idx);
                    links_per_triangle[link_jk as usize].push(tri_idx);
                    links_per_triangle[link_ik as usize].push(tri_idx);
                    
                    tri_idx += 1;
                }
            }
        }
        
        Self {
            n,
            links,
            triangles,
            triangle_links,
            links_per_triangle,
            dt: 1.0,
        }
    }
    
    /// Compute link index for edge (i,j) in flat storage
    #[inline(always)]
    pub fn link_index(&self, i: usize, j: usize) -> usize {
        Self::compute_link_index(self.n, i, j)
    }
    
    /// Static version of link index computation
    #[inline(always)]
    pub fn compute_link_index(n: usize, i: usize, j: usize) -> usize {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        i * n - i * (i + 1) / 2 + j - i - 1
    }
    
    /// Reverse lookup: given index, return (i,j)
    #[inline(always)]
    pub fn index_to_nodes(&self, idx: usize) -> (usize, usize) {
        // Binary search to find i
        let n = self.n;
        let mut i = 0;
        let mut base = 0;
        
        while base + (n - i - 1) <= idx {
            base += n - i - 1;
            i += 1;
        }
        
        let j = idx - base + i + 1;
        (i, j)
    }
    
    /// Entropy action with cache-friendly access
    pub fn entropy_action(&self) -> f64 {
        let mut sum = 0.0;
        
        // Prefetch hint for sequential access
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::_mm_prefetch;
            if self.links.len() > 8 {
                _mm_prefetch(self.links.as_ptr().add(8) as *const i8, 1);
            }
        }
        
        for (i, link) in self.links.iter().enumerate() {
            // Prefetch next cache line
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::_mm_prefetch;
                if i + 8 < self.links.len() {
                    _mm_prefetch(self.links.as_ptr().add(i + 8) as *const i8, 1);
                }
            }
            
            sum += -(link.z as f64) * (link.exp_neg_z as f64);
        }
        
        sum
    }
    
    /// Triangle sum using precomputed indices and values
    pub fn triangle_sum(&self) -> f64 {
        let mut sum = 0.0;
        
        // Process triangles in chunks for better cache usage
        const CHUNK_SIZE: usize = 16;
        let chunks = self.triangle_links.chunks(CHUNK_SIZE);
        
        for chunk in chunks {
            // Prefetch links for upcoming triangles
            #[cfg(target_arch = "x86_64")]
            unsafe {
                use std::arch::x86_64::_mm_prefetch;
                for tri_links in chunk.iter().take(4) {
                    _mm_prefetch(&self.links[tri_links[0] as usize] as *const _ as *const i8, 1);
                    _mm_prefetch(&self.links[tri_links[1] as usize] as *const _ as *const i8, 1);
                    _mm_prefetch(&self.links[tri_links[2] as usize] as *const _ as *const i8, 1);
                }
            }
            
            for tri_links in chunk {
                let link_ij = &self.links[tri_links[0] as usize];
                let link_jk = &self.links[tri_links[1] as usize];
                let link_ik = &self.links[tri_links[2] as usize];
                
                let theta_sum = link_ij.theta + link_jk.theta + link_ik.theta;
                sum += (theta_sum as f64).cos();
            }
        }
        
        sum
    }
    
    /// Incremental triangle sum update for a single link change
    pub fn triangle_sum_delta(&self, link_idx: usize, new_theta: f32) -> f64 {
        let mut delta = 0.0;
        let old_theta = self.links[link_idx].theta;
        
        // Only process triangles containing this link
        for &tri_idx in &self.links_per_triangle[link_idx] {
            let tri_links = &self.triangle_links[tri_idx as usize];
            
            // Old contribution
            let mut theta_sum_old = 0.0_f32;
            for &link_idx in tri_links {
                theta_sum_old += self.links[link_idx as usize].theta;
            }
            let old_contrib = (theta_sum_old as f64).cos();
            
            // New contribution
            let theta_sum_new = theta_sum_old - old_theta + new_theta;
            let new_contrib = (theta_sum_new as f64).cos();
            
            delta += new_contrib - old_contrib;
        }
        
        delta
    }
    
    /// Optimized Metropolis step
    pub fn metropolis_step(
        &mut self,
        alpha: f64,
        beta: f64,
        delta_z: f64,
        delta_theta: f64,
        rng: &mut impl Rng,
    ) -> StepInfo {
        let link_idx = rng.gen_range(0..self.links.len());
        let link = self.links[link_idx];
        
        let do_z_update = delta_z > 0.0 && rng.gen_bool(0.5);
        
        if do_z_update {
            // Z-update
            let old_z = link.z;
            let new_z = (old_z + rng.gen_range(-delta_z as f32..=delta_z as f32)).max(0.001);
            
            let old_entropy = -(old_z as f64) * (link.exp_neg_z as f64);
            let new_w = (-new_z).exp();
            let new_entropy = -(new_z as f64) * (new_w as f64);
            
            let delta_s = beta * (new_entropy - old_entropy);
            let accept = delta_s <= 0.0 || rng.gen_range(0.0..1.0) < (-delta_s).exp();
            
            if accept {
                self.links[link_idx].update_z(new_z);
                StepInfo {
                    accept: true,
                    delta_w: (new_w - link.exp_neg_z) as f64,
                    delta_cos: 0.0,
                }
            } else {
                StepInfo::rejected()
            }
        } else {
            // Phase update
            let old_theta = link.theta;
            let new_theta = old_theta + rng.gen_range(-delta_theta as f32..=delta_theta as f32);
            
            let delta_triangle = self.triangle_sum_delta(link_idx, new_theta);
            let delta_s = alpha * delta_triangle;
            
            let accept = delta_s <= 0.0 || rng.gen_range(0.0..1.0) < (-delta_s).exp();
            
            if accept {
                self.links[link_idx].update_theta(new_theta);
                
                let delta_cos = link.exp_neg_z as f64 * 
                    (new_theta.cos() as f64 - old_theta.cos() as f64);
                
                StepInfo {
                    accept: true,
                    delta_w: 0.0,
                    delta_cos,
                }
            } else {
                StepInfo::rejected()
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct StepInfo {
    pub accept: bool,
    pub delta_w: f64,
    pub delta_cos: f64,
}

impl StepInfo {
    fn rejected() -> Self {
        Self {
            accept: false,
            delta_w: 0.0,
            delta_cos: 0.0,
        }
    }
}

/// Benchmark comparison between HashMap and flat vector implementations
pub fn benchmark_comparison(n: usize, num_steps: usize) {
    println!("\n=== Cache Optimization Benchmark (N={}) ===\n", n);
    
    let mut rng = rand::thread_rng();
    
    // Create optimized graph
    let start = Instant::now();
    let mut opt_graph = OptimizedGraph::new(n, &mut rng);
    let create_time_opt = start.elapsed();
    
    // Benchmark entropy action
    let start = Instant::now();
    let mut entropy_sum = 0.0;
    for _ in 0..1000 {
        entropy_sum += opt_graph.entropy_action();
    }
    let entropy_time = start.elapsed();
    
    // Benchmark triangle sum
    let start = Instant::now();
    let mut triangle_sum = 0.0;
    for _ in 0..100 {
        triangle_sum += opt_graph.triangle_sum();
    }
    let triangle_time = start.elapsed();
    
    // Benchmark Metropolis steps
    let start = Instant::now();
    let mut accepts = 0;
    for _ in 0..num_steps {
        let info = opt_graph.metropolis_step(1.0, 1.0, 0.1, 0.1, &mut rng);
        if info.accept {
            accepts += 1;
        }
    }
    let mc_time = start.elapsed();
    
    println!("Optimized Implementation:");
    println!("  Graph creation:    {:?}", create_time_opt);
    println!("  Entropy (1k):      {:?} ({:.2} μs/call)", 
             entropy_time, entropy_time.as_micros() as f64 / 1000.0);
    println!("  Triangle sum (100): {:?} ({:.2} μs/call)", 
             triangle_time, triangle_time.as_micros() as f64 / 100.0);
    println!("  MC steps ({}):   {:?} ({:.2} μs/step)", 
             num_steps, mc_time, mc_time.as_micros() as f64 / num_steps as f64);
    println!("  Acceptance rate:   {:.2}%", 100.0 * accepts as f64 / num_steps as f64);
    
    // Memory usage estimation
    let link_size = std::mem::size_of::<OptimizedLink>();
    let num_links = n * (n - 1) / 2;
    let num_triangles = n * (n - 1) * (n - 2) / 6;
    let memory_links = link_size * num_links;
    let memory_triangles = std::mem::size_of::<(u32, u32, u32)>() * num_triangles;
    let memory_triangle_links = std::mem::size_of::<[u32; 3]>() * num_triangles;
    let memory_total = memory_links + memory_triangles + memory_triangle_links;
    
    println!("\nMemory Usage:");
    println!("  Link size:         {} bytes", link_size);
    println!("  Links total:       {:.2} KB", memory_links as f64 / 1024.0);
    println!("  Triangle indices:  {:.2} KB", memory_triangles as f64 / 1024.0);
    println!("  Triangle links:    {:.2} KB", memory_triangle_links as f64 / 1024.0);
    println!("  Total:             {:.2} KB", memory_total as f64 / 1024.0);
    
    // Prevent optimization
    println!("\n(Sums to prevent optimization: {:.6}, {:.6})", entropy_sum, triangle_sum);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_link_indexing() {
        let n = 10;
        let mut rng = rand::thread_rng();
        let graph = OptimizedGraph::new(n, &mut rng);
        
        // Test that link_index is correct
        for (idx, link) in graph.links.iter().enumerate() {
            let computed_idx = graph.link_index(link.i as usize, link.j as usize);
            assert_eq!(idx, computed_idx);
        }
        
        // Test reverse lookup
        for idx in 0..graph.links.len() {
            let (i, j) = graph.index_to_nodes(idx);
            assert_eq!(graph.link_index(i, j), idx);
        }
    }
    
    #[test]
    fn test_triangle_consistency() {
        let n = 5;
        let mut rng = rand::thread_rng();
        let graph = OptimizedGraph::new(n, &mut rng);
        
        // Verify triangle count
        let expected_triangles = n * (n - 1) * (n - 2) / 6;
        assert_eq!(graph.triangles.len(), expected_triangles);
        
        // Verify each triangle has valid link indices
        for (tri_idx, tri_links) in graph.triangle_links.iter().enumerate() {
            let (i, j, k) = graph.triangles[tri_idx];
            
            // Check that the precomputed link indices are correct
            assert_eq!(tri_links[0], graph.link_index(i as usize, j as usize) as u32);
            assert_eq!(tri_links[1], graph.link_index(j as usize, k as usize) as u32);
            assert_eq!(tri_links[2], graph.link_index(i as usize, k as usize) as u32);
        }
    }
    
    #[test]
    fn test_metropolis_conservation() {
        let n = 6;
        let mut rng = rand::thread_rng();
        let mut graph = OptimizedGraph::new(n, &mut rng);
        
        let initial_entropy = graph.entropy_action();
        let initial_triangles = graph.triangle_sum();
        
        // Run many MC steps
        for _ in 0..1000 {
            graph.metropolis_step(1.0, 1.0, 0.1, 0.1, &mut rng);
        }
        
        // Values should change but remain finite
        let final_entropy = graph.entropy_action();
        let final_triangles = graph.triangle_sum();
        
        assert!(final_entropy.is_finite());
        assert!(final_triangles.is_finite());
        assert!((final_entropy - initial_entropy).abs() > 1e-6); // Should have changed
        assert!((final_triangles - initial_triangles).abs() > 1e-6);
    }
}

// Example usage and benchmarking
#[cfg(test)]
mod benchmarks {
    use super::*;
    
    #[test]
    fn run_benchmark() {
        benchmark_comparison(24, 10000);
    }
}