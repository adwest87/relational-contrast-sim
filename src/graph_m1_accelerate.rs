// Apple Silicon M1 optimizations using Accelerate framework
// This provides access to AMX (Apple Matrix Extension) and optimized BLAS/LAPACK

use std::os::raw::{c_double, c_int};
use std::sync::Arc;
use rayon::prelude::*;
use rand::Rng;
use rand_pcg::Pcg64;

// Link to Apple's Accelerate framework
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    // BLAS Level 1: Vector operations
    fn cblas_ddot(n: c_int, x: *const c_double, incx: c_int, 
                  y: *const c_double, incy: c_int) -> c_double;
    
    fn _cblas_daxpy(n: c_int, alpha: c_double, x: *const c_double, incx: c_int,
                   y: *mut c_double, incy: c_int);
    
    fn _cblas_dscal(n: c_int, alpha: c_double, x: *mut c_double, incx: c_int);
    
    // vDSP functions for vectorized operations
    fn _vDSP_vaddD(a: *const c_double, stride_a: c_int,
                  b: *const c_double, stride_b: c_int,
                  c: *mut c_double, stride_c: c_int,
                  n: c_int);
    
    fn vDSP_vmulD(a: *const c_double, stride_a: c_int,
                  b: *const c_double, stride_b: c_int,
                  c: *mut c_double, stride_c: c_int,
                  n: c_int);
    
    fn vDSP_vsumD(a: *const c_double, stride: c_int,
                  sum: *mut c_double, n: c_int);
    
    // Trigonometric functions (vectorized)
    fn vvcos(y: *mut c_double, x: *const c_double, n: *const c_int);
    fn vvsin(y: *mut c_double, x: *const c_double, n: *const c_int);
    fn vvexp(y: *mut c_double, x: *const c_double, n: *const c_int);
}

/// M1-optimized link with Accelerate framework
#[repr(C, align(128))]
#[derive(Debug, Clone)]
pub struct AccelerateLink {
    pub i: u32,
    pub j: u32,
    pub z: f64,
    pub theta: f64,
    pub cos_theta: f64,
    pub sin_theta: f64,
    pub exp_neg_z: f64,
    pub w_cos: f64,
    pub w_sin: f64,
    _padding: [f64; 7],
}

/// Graph optimized for Apple Accelerate framework
pub struct AccelerateGraph {
    pub links: Vec<AccelerateLink>,
    pub n: usize,
    
    // Contiguous arrays for vectorized operations
    z_array: Vec<f64>,
    theta_array: Vec<f64>,
    cos_array: Vec<f64>,
    sin_array: Vec<f64>,
    exp_array: Vec<f64>,
    
    // Triangle data
    triangles: Arc<Vec<(u32, u32, u32)>>,
    
    // GCD queue for parallel operations
    thread_pool: rayon::ThreadPool,
}

impl AccelerateGraph {
    pub fn new(n: usize, seed: u64) -> Self {
        use rand::SeedableRng;
        let mut rng = Pcg64::seed_from_u64(seed);
        
        let num_links = n * (n - 1) / 2;
        let num_triangles = n * (n - 1) * (n - 2) / 6;
        
        let mut links = Vec::with_capacity(num_links);
        let mut z_array = Vec::with_capacity(num_links);
        let mut theta_array = Vec::with_capacity(num_links);
        
        // Create links
        for i in 0..n {
            for j in (i + 1)..n {
                let z = rng.gen_range(0.001..10.0);
                let theta = 0.0;
                
                z_array.push(z);
                theta_array.push(theta);
                
                links.push(AccelerateLink {
                    i: i as u32,
                    j: j as u32,
                    z,
                    theta,
                    cos_theta: 1.0,
                    sin_theta: 0.0,
                    exp_neg_z: (-z).exp(),
                    w_cos: (-z).exp(),
                    w_sin: 0.0,
                    _padding: [0.0; 7],
                });
            }
        }
        
        // Allocate arrays for vectorized operations
        let cos_array = vec![1.0; num_links];
        let sin_array = vec![0.0; num_links];
        let mut exp_array = vec![0.0; num_links];
        
        // Use Accelerate to compute exp(-z) for all links at once
        unsafe {
            let neg_z: Vec<f64> = z_array.iter().map(|&z| -z).collect();
            let n = num_links as c_int;
            vvexp(exp_array.as_mut_ptr(), neg_z.as_ptr(), &n);
        }
        
        // Update link exp values
        for (i, link) in links.iter_mut().enumerate() {
            link.exp_neg_z = exp_array[i];
            link.w_cos = exp_array[i];
        }
        
        // Create triangles
        let mut triangles = Vec::with_capacity(num_triangles);
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    triangles.push((i as u32, j as u32, k as u32));
                }
            }
        }
        
        // Create optimized thread pool
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8) // 4 performance + 4 efficiency cores
            .build()
            .unwrap();
        
        Self {
            links,
            n,
            z_array,
            theta_array,
            cos_array,
            sin_array,
            exp_array,
            triangles: Arc::new(triangles),
            thread_pool,
        }
    }
    
    /// Vectorized entropy calculation using Accelerate
    pub fn entropy_action_accelerate(&self) -> f64 {
        unsafe {
            let n = self.links.len() as c_int;
            
            // Compute -z * exp(-z) using vectorized operations
            let mut result = vec![0.0; self.links.len()];
            
            // First compute -z
            let neg_z: Vec<f64> = self.z_array.iter().map(|&z| -z).collect();
            
            // Multiply -z * exp(-z) using vDSP
            vDSP_vmulD(
                neg_z.as_ptr(), 1,
                self.exp_array.as_ptr(), 1,
                result.as_mut_ptr(), 1,
                n
            );
            
            // Sum the result
            let mut sum = 0.0;
            vDSP_vsumD(result.as_ptr(), 1, &mut sum, n);
            
            sum
        }
    }
    
    /// Update all trigonometric values using vectorized operations
    pub fn update_trig_accelerate(&mut self) {
        unsafe {
            let n = self.theta_array.len() as c_int;
            
            // Compute cos and sin for all angles at once
            vvcos(self.cos_array.as_mut_ptr(), self.theta_array.as_ptr(), &n);
            vvsin(self.sin_array.as_mut_ptr(), self.theta_array.as_ptr(), &n);
            
            // Update links
            for (i, link) in self.links.iter_mut().enumerate() {
                link.cos_theta = self.cos_array[i];
                link.sin_theta = self.sin_array[i];
                link.w_cos = link.exp_neg_z * link.cos_theta;
                link.w_sin = link.exp_neg_z * link.sin_theta;
            }
        }
    }
    
    /// Triangle sum using parallel computation with GCD
    pub fn triangle_sum_parallel(&self) -> f64 {
        let triangles = Arc::clone(&self.triangles);
        let links = &self.links;
        let n = self.n;
        
        // Use rayon with custom thread pool
        self.thread_pool.install(|| {
            triangles
                .par_chunks(1024) // Smaller chunks for better cache usage
                .map(|chunk| {
                    let mut local_sum = 0.0;
                    
                    for &(i, j, k) in chunk {
                        let idx_ij = link_index(i as usize, j as usize, n);
                        let idx_jk = link_index(j as usize, k as usize, n);
                        let idx_ik = link_index(i as usize, k as usize, n);
                        
                        let theta_sum = links[idx_ij].theta + 
                                       links[idx_jk].theta + 
                                       links[idx_ik].theta;
                        
                        local_sum += theta_sum.cos();
                    }
                    
                    local_sum
                })
                .sum()
        })
    }
    
    /// Compute observables using BLAS operations
    pub fn compute_observables_blas(&self) -> (f64, f64, f64) {
        unsafe {
            let n = self.links.len() as c_int;
            
            // Sum of cos values using BLAS
            let ones = vec![1.0; self.links.len()];
            let cos_sum = cblas_ddot(n, self.cos_array.as_ptr(), 1, ones.as_ptr(), 1);
            
            // Sum of weights
            let w_sum = cblas_ddot(n, self.exp_array.as_ptr(), 1, ones.as_ptr(), 1);
            
            // Sum of w*cos using precomputed w_cos values
            let w_cos: Vec<f64> = self.links.iter().map(|l| l.w_cos).collect();
            let w_cos_sum = cblas_ddot(n, w_cos.as_ptr(), 1, ones.as_ptr(), 1);
            
            let m = self.links.len() as f64;
            (cos_sum / m, w_sum / m, w_cos_sum / m)
        }
    }
}

// Helper function
#[inline(always)]
fn link_index(i: usize, j: usize, n: usize) -> usize {
    let (i, j) = if i < j { (i, j) } else { (j, i) };
    i * n - i * (i + 1) / 2 + j - i - 1
}

/// Benchmark Accelerate optimizations
pub fn benchmark_accelerate(n: usize, _steps: usize) {
    use std::time::Instant;
    
    println!("\n=== Apple Accelerate Framework Benchmark ===");
    println!("Using vectorized BLAS/LAPACK and vDSP operations");
    
    let mut graph = AccelerateGraph::new(n, 12345);
    
    // Test entropy calculation
    let start = Instant::now();
    let mut _entropy_sum = 0.0;
    for _ in 0..10000 {
        _entropy_sum += graph.entropy_action_accelerate();
    }
    let entropy_time = start.elapsed();
    println!("Entropy (Accelerate): {:.2} μs/call", 
        entropy_time.as_micros() as f64 / 10000.0);
    
    // Test trigonometric updates
    let start = Instant::now();
    for _ in 0..1000 {
        graph.update_trig_accelerate();
    }
    let trig_time = start.elapsed();
    println!("Trig update (vectorized): {:.2} μs/call", 
        trig_time.as_micros() as f64 / 1000.0);
    
    // Test triangle sum
    let start = Instant::now();
    let mut _tri_sum = 0.0;
    for _ in 0..100 {
        _tri_sum += graph.triangle_sum_parallel();
    }
    let tri_time = start.elapsed();
    println!("Triangle sum (parallel): {:.2} μs/call", 
        tri_time.as_micros() as f64 / 100.0);
    
    // Test BLAS observables
    let start = Instant::now();
    let mut obs = (0.0, 0.0, 0.0);
    for _ in 0..1000 {
        obs = graph.compute_observables_blas();
    }
    let obs_time = start.elapsed();
    println!("Observables (BLAS): {:.2} μs/call", 
        obs_time.as_micros() as f64 / 1000.0);
    println!("  Results: <cos>={:.4}, <w>={:.4}, <w*cos>={:.4}", obs.0, obs.1, obs.2);
}