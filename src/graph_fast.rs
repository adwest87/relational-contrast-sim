// Optimized graph implementation with performance improvements

use rand::Rng;
use rand_pcg::Pcg64;
use rand::SeedableRng;
use std::f64::consts::TAU;

/// Cache-friendly link structure with precomputed values
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FastLink {
    pub i: u32,              // 4 bytes (supports up to 4B nodes)
    pub j: u32,              // 4 bytes
    pub z: f64,              // 8 bytes: z = -ln(w)
    pub theta: f64,          // 8 bytes
    pub cos_theta: f64,      // 8 bytes: precomputed cos(theta)
    pub sin_theta: f64,      // 8 bytes: precomputed sin(theta)
    pub exp_neg_z: f64,      // 8 bytes: precomputed exp(-z) = w
    _padding: f64,           // 8 bytes: align to 64 bytes (cache line)
}

impl FastLink {
    /// Create new link with precomputed values
    #[inline(always)]
    pub fn new(i: usize, j: usize, z: f64, theta: f64) -> Self {
        Self {
            i: i as u32,
            j: j as u32,
            z,
            theta,
            cos_theta: theta.cos(),
            sin_theta: theta.sin(),
            exp_neg_z: (-z).exp(),
            _padding: 0.0,
        }
    }
    
    /// Get weight (uses precomputed value)
    #[inline(always)]
    pub fn w(&self) -> f64 {
        self.exp_neg_z
    }
    
    /// Update z and precomputed weight
    #[inline(always)]
    pub fn update_z(&mut self, new_z: f64) {
        self.z = new_z;
        self.exp_neg_z = (-new_z).exp();
    }
    
    /// Update theta and precomputed trig values
    #[inline(always)]
    pub fn update_theta(&mut self, new_theta: f64) {
        self.theta = new_theta;
        self.cos_theta = new_theta.cos();
        self.sin_theta = new_theta.sin();
    }
}

/// Optimized graph with fast RNG and cached computations
#[derive(Clone)]
pub struct FastGraph {
    pub nodes: Vec<Node>,
    pub links: Vec<FastLink>,
    pub dt: f64,
    triangles: Vec<(u32, u32, u32)>,  // Use u32 to save memory
    
    // Autocorrelation tracking
    autocorr_window: Vec<f64>,
    autocorr_tau: f64,
    measurement_interval: usize,
    
    // Cached values
    two_pi: f64,
    inv_dt: f64,
    ln_dt: f64,
}

/// Simple node structure
#[derive(Debug, Clone, Copy)]
pub struct Node {
    pub id: u32,
}

impl FastGraph {
    /// Create new optimized graph with Pcg64 RNG
    pub fn new(n: usize, seed: u64) -> Self {
        let mut rng = Pcg64::seed_from_u64(seed);
        
        // Pre-allocate with exact sizes
        let num_links = n * (n - 1) / 2;
        let num_triangles = n * (n - 1) * (n - 2) / 6;
        
        let mut nodes = Vec::with_capacity(n);
        let mut links = Vec::with_capacity(num_links);
        let mut triangles = Vec::with_capacity(num_triangles);
        
        // Create nodes
        for i in 0..n {
            nodes.push(Node { id: i as u32 });
        }
        
        // Create links with precomputed values
        for i in 0..n {
            for j in (i + 1)..n {
                let z = rng.gen_range(0.001..10.0);
                let theta = 0.0;
                links.push(FastLink::new(i, j, z, theta));
            }
        }
        
        // Pre-compute triangles
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    triangles.push((i as u32, j as u32, k as u32));
                }
            }
        }
        
        let dt = 1.0;
        
        Self {
            nodes,
            links,
            dt,
            triangles,
            autocorr_window: Vec::with_capacity(1000),
            autocorr_tau: 10.0,  // Initial estimate
            measurement_interval: 10,
            two_pi: TAU,
            inv_dt: 1.0 / dt,
            ln_dt: dt.ln(),
        }
    }
    
    /// Create from existing graph for fair comparison
    pub fn from_graph(graph: &crate::graph::Graph) -> Self {
        let n = graph.n();
        let num_triangles = n * (n - 1) * (n - 2) / 6;
        
        let mut nodes = Vec::with_capacity(n);
        let mut links = Vec::with_capacity(graph.links.len());
        let mut triangles = Vec::with_capacity(num_triangles);
        
        // Copy nodes
        for i in 0..n {
            nodes.push(Node { id: i as u32 });
        }
        
        // Copy links with precomputed values
        for link in &graph.links {
            links.push(FastLink::new(link.i, link.j, link.z, link.theta));
        }
        
        // Pre-compute triangles
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    triangles.push((i as u32, j as u32, k as u32));
                }
            }
        }
        
        Self {
            nodes,
            links,
            dt: graph.dt,
            triangles,
            autocorr_window: Vec::with_capacity(1000),
            autocorr_tau: 10.0,
            measurement_interval: 10,
            two_pi: TAU,
            inv_dt: 1.0 / graph.dt,
            ln_dt: graph.dt.ln(),
        }
    }
    
    /// Number of nodes
    #[inline(always)]
    pub fn n(&self) -> usize {
        self.nodes.len()
    }
    
    /// Number of links
    #[inline(always)]
    pub fn m(&self) -> usize {
        self.links.len()
    }
    
    /// Fast link index calculation
    #[inline(always)]
    pub fn link_index(&self, i: usize, j: usize) -> usize {
        let n = self.n();
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        i * n - i * (i + 1) / 2 + j - i - 1
    }
    
    /// Optimized entropy action using precomputed values
    #[inline(always)]
    pub fn entropy_action(&self) -> f64 {
        let mut sum: f64 = 0.0;
        // Process in chunks for better cache usage
        for chunk in self.links.chunks(8) {
            for link in chunk {
                // Entropy = -z * exp(-z) for each link
                sum += -link.z * link.exp_neg_z;
            }
        }
        sum
    }
    
    /// Optimized triangle sum using precomputed trig values
    pub fn triangle_sum(&self) -> f64 {
        let mut sum: f64 = 0.0;
        
        // Process triangles in cache-friendly chunks
        const CHUNK_SIZE: usize = 16;
        for chunk in self.triangles.chunks(CHUNK_SIZE) {
            for &(i, j, k) in chunk {
                let idx_ij = self.link_index(i as usize, j as usize);
                let idx_jk = self.link_index(j as usize, k as usize);
                let idx_ik = self.link_index(i as usize, k as usize);
                
                let theta_sum = self.links[idx_ij].theta + 
                               self.links[idx_jk].theta + 
                               self.links[idx_ik].theta;
                
                sum += 3.0 * theta_sum.cos();
            }
        }
        
        sum
    }
    
    /// Total action
    #[inline(always)]
    pub fn action(&self, alpha: f64, beta: f64) -> f64 {
        beta * self.entropy_action() + alpha * self.triangle_sum()
    }
    
    /// Optimized Metropolis step with fast RNG
    pub fn metropolis_step(
        &mut self,
        alpha: f64,
        beta: f64,
        delta_z: f64,
        delta_theta: f64,
        rng: &mut Pcg64,
    ) -> StepInfo {
        let link_idx = rng.gen_range(0..self.links.len());
        // Pre-fetch next cache line for better performance
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::_mm_prefetch;
            if link_idx + 1 < self.links.len() {
                let next_ptr = &self.links[link_idx + 1] as *const _ as *const i8;
                _mm_prefetch(next_ptr, 1);
            }
        }
        
        let do_z_update = delta_z > 0.0 && rng.gen_bool(0.5);
        
        if do_z_update {
            // Z-update with precomputed values
            let link = &self.links[link_idx];
            let old_z = link.z;
            let old_exp_neg_z = link.exp_neg_z;
            let new_z = (old_z + rng.gen_range(-delta_z..=delta_z)).max(0.001);
            let new_exp_neg_z = (-new_z).exp();
            
            // Fast entropy change calculation
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
            // Phase update
            let link = &self.links[link_idx];
            let old_theta = link.theta;
            let d_theta = rng.gen_range(-delta_theta..=delta_theta);
            let new_theta = old_theta + d_theta;
            
            // Calculate triangle sum change for affected triangles only
            let delta_triangle = self.triangle_sum_delta(link_idx, new_theta);
            let delta_s = alpha * delta_triangle;
            
            let accept = delta_s <= 0.0 || rng.gen_range(0.0..1.0) < (-delta_s).exp();
            
            if accept {
                let old_cos_theta = self.links[link_idx].cos_theta;
                let old_exp_neg_z = self.links[link_idx].exp_neg_z;
                self.links[link_idx].update_theta(new_theta);
                
                let delta_cos = old_exp_neg_z * (new_theta.cos() - old_cos_theta);
                
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
    
    /// Calculate triangle sum change for a single link update
    fn triangle_sum_delta(&self, link_idx: usize, new_theta: f64) -> f64 {
        let link = &self.links[link_idx];
        let (i, j) = (link.i as usize, link.j as usize);
        let old_theta = link.theta;
        
        let mut delta: f64 = 0.0;
        
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
                
                // Old contribution
                let old_sum = old_theta + self.links[idx_ik].theta + self.links[idx_jk].theta;
                // New contribution
                let new_sum = new_theta + self.links[idx_ik].theta + self.links[idx_jk].theta;
                
                delta += 3.0 * (new_sum.cos() - old_sum.cos());
            }
        }
        
        delta
    }
    
    /// Update autocorrelation estimate
    pub fn update_autocorrelation(&mut self, observable: f64) {
        self.autocorr_window.push(observable);
        
        // Keep window size reasonable
        if self.autocorr_window.len() > 1000 {
            self.autocorr_window.remove(0);
        }
        
        // Estimate autocorrelation time every 100 samples
        if self.autocorr_window.len() >= 100 && self.autocorr_window.len() % 100 == 0 {
            self.autocorr_tau = self.estimate_autocorr_time();
            self.measurement_interval = (self.autocorr_tau * 15.0) as usize;
            self.measurement_interval = self.measurement_interval.clamp(10, 1000);
        }
    }
    
    /// Estimate autocorrelation time
    fn estimate_autocorr_time(&self) -> f64 {
        let data = &self.autocorr_window;
        let n = data.len();
        if n < 50 {
            return 10.0;  // Default
        }
        
        // Calculate mean
        let mean: f64 = data.iter().sum::<f64>() / n as f64;
        
        // Calculate autocorrelation function
        let mut tau_sum = 0.5;  // C(0) contributes 0.5
        let max_lag = n.min(100);
        
        for lag in 1..max_lag {
            let mut c = 0.0;
            for i in 0..n-lag {
                c += (data[i] - mean) * (data[i + lag] - mean);
            }
            c /= (n - lag) as f64;
            
            // Normalize
            let c0 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            let rho = c / c0;
            
            // Stop when autocorrelation becomes negligible
            if rho.abs() < 0.05 {
                break;
            }
            
            tau_sum += rho;
        }
        
        2.0 * tau_sum  // Integrated autocorrelation time
    }
    
    /// Get optimal measurement interval
    #[inline(always)]
    pub fn get_measurement_interval(&self) -> usize {
        self.measurement_interval
    }
    
    /// Get current autocorrelation time estimate
    #[inline(always)]
    pub fn get_autocorr_tau(&self) -> f64 {
        self.autocorr_tau
    }
}

/// Step information
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

/// Batched observable calculator for efficiency
pub struct BatchedObservables {
    rotation_counter: usize,
    cached_values: ObservableCache,
}

#[derive(Default)]
struct ObservableCache {
    mean_w: f64,
    var_w: f64,
    mean_cos: f64,
    entropy: f64,
    triangle_sum: f64,
    last_update: usize,
}

impl BatchedObservables {
    pub fn new() -> Self {
        Self {
            rotation_counter: 0,
            cached_values: ObservableCache::default(),
        }
    }
    
    /// Measure observables with rotation for expensive calculations
    pub fn measure(&mut self, graph: &FastGraph, alpha: f64, beta: f64) -> QuickObservables {
        self.rotation_counter += 1;
        
        // Always compute cheap observables
        let sum_w: f64 = graph.links.iter().map(|l| l.exp_neg_z).sum();
        let sum_cos: f64 = graph.links.iter().map(|l| l.cos_theta).sum();
        let m = graph.m() as f64;
        
        let mean_w = sum_w / m;
        let mean_cos = sum_cos / m;
        
        // Rotate expensive calculations
        match self.rotation_counter % 5 {
            0 => {
                // Update variance (expensive)
                let var_w = graph.links.iter()
                    .map(|l| (l.exp_neg_z - mean_w).powi(2))
                    .sum::<f64>() / m;
                self.cached_values.var_w = var_w;
            }
            1 => {
                // Update entropy
                self.cached_values.entropy = graph.entropy_action();
            }
            2 => {
                // Update triangle sum (most expensive)
                self.cached_values.triangle_sum = graph.triangle_sum();
            }
            _ => {}  // Use cached values
        }
        
        self.cached_values.mean_w = mean_w;
        self.cached_values.mean_cos = mean_cos;
        
        QuickObservables {
            mean_w,
            var_w: self.cached_values.var_w,
            mean_cos,
            mean_action: beta * self.cached_values.entropy + alpha * self.cached_values.triangle_sum,
            susceptibility: graph.n() as f64 * self.cached_values.var_w,
        }
    }
}

/// Lightweight observable struct
#[derive(Debug, Clone, Copy)]
pub struct QuickObservables {
    pub mean_w: f64,
    pub var_w: f64,
    pub mean_cos: f64,
    pub mean_action: f64,
    pub susceptibility: f64,
}