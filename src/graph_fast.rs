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
    _two_pi: f64,
    _inv_dt: f64,
    _ln_dt: f64,
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
                let theta = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);  // Random angle [-π, π)
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
            _two_pi: TAU,
            _inv_dt: 1.0 / dt,
            _ln_dt: dt.ln(),
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
            _two_pi: TAU,
            _inv_dt: 1.0 / graph.dt,
            _ln_dt: graph.dt.ln(),
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
    
    /// Get phase with proper antisymmetry: θ_ij if i < j, -θ_ij if i > j
    #[inline(always)]
    pub fn get_phase(&self, from_node: usize, to_node: usize) -> f64 {
        if from_node == to_node {
            return 0.0;
        }
        let link_idx = self.link_index(from_node, to_node);
        if from_node < to_node {
            self.links[link_idx].theta
        } else {
            -self.links[link_idx].theta
        }
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
    
    /// Optimized triangle sum using precomputed trig values with proper antisymmetry
    pub fn triangle_sum(&self) -> f64 {
        let mut sum: f64 = 0.0;
        
        // Process triangles in cache-friendly chunks
        const CHUNK_SIZE: usize = 16;
        for chunk in self.triangles.chunks(CHUNK_SIZE) {
            for &(i, j, k) in chunk {
                // Use proper antisymmetric phases: θ_ji = -θ_ij
                let t_ij = self.get_phase(i as usize, j as usize);
                let t_jk = self.get_phase(j as usize, k as usize);
                let t_ki = self.get_phase(k as usize, i as usize);
                
                let theta_sum = t_ij + t_jk + t_ki;
                
                sum += theta_sum.cos();
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
            
            // Check for no-op due to boundary clamping
            if (new_z - old_z).abs() < 1e-15 {
                // True no-op: always accept
                StepInfo {
                    accept: true,
                    delta_w: 0.0,
                    delta_cos: 0.0,
                }
            } else {
                let new_exp_neg_z = (-new_z).exp();
                
                // Fast entropy change calculation
                // Entropy action S = -z * w = -z * exp(-z)
                let old_entropy = -old_z * old_exp_neg_z;
                let new_entropy = -new_z * new_exp_neg_z;
                let delta_entropy = new_entropy - old_entropy;
                let delta_s = beta * delta_entropy;
                
                // Use epsilon comparison for near-zero energy changes  
                const EPSILON: f64 = 1e-6;
                
                // Log any significant energy changes that might be precision issues
                if delta_s.abs() > 1e-10 && delta_s.abs() < 1e-6 {
                    eprintln!("PRECISION WARNING (Z): |ΔS|={:.2e}", delta_s.abs());
                }
                
                let accept = if delta_s.abs() <= EPSILON { true } else { rng.gen_range(0.0..1.0) < (-delta_s).exp() };
                
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
            }
        } else {
            // Phase update
            let link = &self.links[link_idx];
            let old_theta = link.theta;
            let old_cos_theta = link.cos_theta;
            let d_theta = rng.gen_range(-delta_theta..=delta_theta);
            let new_theta = old_theta + d_theta;
            
            
            // Check for no-op due to very small theta change
            if (new_theta - old_theta).abs() < 1e-15 {
                // True no-op: always accept
                StepInfo {
                    accept: true,
                    delta_w: 0.0,
                    delta_cos: 0.0,
                }
            } else {
                // Calculate triangle sum change BEFORE applying the move
                let delta_triangle = self.triangle_sum_delta(link_idx, new_theta);
                let delta_s = alpha * delta_triangle;
                
                // Use epsilon comparison for near-zero energy changes  
                const EPSILON: f64 = 1e-6;
                
                // Log any significant energy changes that might be precision issues
                if delta_s.abs() > 1e-10 && delta_s.abs() < 1e-6 {
                    eprintln!("PRECISION WARNING (THETA): |ΔS|={:.2e}", delta_s.abs());
                }
                
                let accept = if delta_s.abs() <= EPSILON { true } else { rng.gen_range(0.0..1.0) < (-delta_s).exp() };
                
                if accept {
                    self.links[link_idx].update_theta(new_theta);
                    let delta_cos = new_theta.cos() - old_cos_theta;
                    
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
    
    /// Calculate triangle sum change for a single link update with numerical precision
    fn triangle_sum_delta(&self, link_idx: usize, new_theta: f64) -> f64 {
        let link = &self.links[link_idx];
        let (i, j) = (link.i as usize, link.j as usize);
        let old_theta = link.theta;
        let delta_theta = new_theta - old_theta;
        
        // For very small changes, use analytical derivative to avoid precision loss
        const SMALL_DELTA_THRESHOLD: f64 = 1e-8;
        
        // Collect contributions for Kahan summation
        let mut contributions = Vec::new();
        
        // Only check triangles containing edge (i,j)
        for k in 0..self.n() {
            if k != i && k != j {
                // Use proper antisymmetric phases: θ_ji = -θ_ij
                let t_ik = self.get_phase(i, k);
                let t_jk = self.get_phase(j, k);
                let t_ki = self.get_phase(k, i);  // Need t_ki, not t_ik!
                let t_ij_old = self.get_phase(i, j);
                
                let old_total = t_ij_old + t_jk + t_ki;
                
                let contribution = if delta_theta.abs() < SMALL_DELTA_THRESHOLD {
                    // Use analytical derivative: d/dx[cos(x)] = -sin(x)
                    -old_total.sin() * delta_theta
                } else {
                    // Use direct calculation for larger changes
                    let t_ij_new = if i < j { new_theta } else { -new_theta };
                    let new_total = t_ij_new + t_jk + t_ki;
                    new_total.cos() - old_total.cos()
                };
                
                contributions.push(contribution);
            }
        }
        
        // Use Kahan summation for numerical stability
        self.kahan_sum(&contributions)
    }
    
    /// Kahan summation algorithm for numerical stability
    #[inline(always)]
    fn kahan_sum(&self, values: &[f64]) -> f64 {
        let mut sum = 0.0;
        let mut c = 0.0;
        for &val in values {
            let y = val - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sum
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
    
    /// Calculate correlation length from susceptibility and finite-size scaling
    /// For a finite system: χ ~ N^(γ/ν) at criticality
    /// Away from criticality: χ ~ ξ^γ where ξ is correlation length
    /// Simple approximation: ξ ~ (χ/N)^(1/2) in mean field
    pub fn correlation_length_from_susceptibility(&self, susceptibility: f64) -> f64 {
        let n = self.n() as f64;
        // More sophisticated: account for system size effects
        // ξ = min(L/2, sqrt(χ/N)) where L ~ N^(1/d) and d is effective dimension
        let system_size = n.powf(0.5); // Effective 2D-like for complete graph
        let xi_est = (susceptibility / n).sqrt();
        xi_est.min(system_size / 2.0)
    }
    
    /// Calculate connected correlation function G(r) = <s_i s_j> - <s_i><s_j>
    /// For complete graph, all nodes are distance 1 apart
    pub fn correlation_function(&self) -> (f64, f64) {
        // For complete graph, we calculate node-based correlations
        // First, compute average phase on each node
        let n = self.n();
        let mut node_phases = vec![0.0; n];
        let mut node_degrees = vec![0; n];
        
        for link in &self.links {
            let (i, j) = (link.i as usize, link.j as usize);
            node_phases[i] += link.cos_theta;
            node_phases[j] += link.cos_theta;
            node_degrees[i] += 1;
            node_degrees[j] += 1;
        }
        
        // Average phase per node
        for i in 0..n {
            if node_degrees[i] > 0 {
                node_phases[i] /= node_degrees[i] as f64;
            }
        }
        
        let mean_phase: f64 = node_phases.iter().sum::<f64>() / n as f64;
        
        // G(0) = <s_i^2> - <s_i>^2
        let g0 = node_phases.iter()
            .map(|&s| s * s)
            .sum::<f64>() / n as f64 - mean_phase * mean_phase;
        
        // G(1) = average correlation between connected nodes minus <s>^2
        let mut g1_sum = 0.0;
        let mut g1_count = 0;
        
        for link in &self.links {
            let (i, j) = (link.i as usize, link.j as usize);
            g1_sum += node_phases[i] * node_phases[j];
            g1_count += 1;
        }
        
        let g1 = if g1_count > 0 {
            g1_sum / g1_count as f64 - mean_phase * mean_phase
        } else {
            0.0
        };
        
        (g0, g1)
    }
    
    /// Calculate correlation length from correlation function
    /// ξ = sqrt(<r²·C(r)> / <C(r)>) where C(r) is the correlation function
    pub fn calculate_correlation_length(&self) -> f64 {
        let (g0, g1) = self.correlation_function();
        
        // For complete graph, all non-self pairs are at distance 1
        // <r²·C(r)> = 0²·G(0) + 1²·G(1) = G(1)
        // <C(r)> = G(0) + G(1)
        
        let numerator = g1.abs();  // r² = 1 for all connected pairs
        let denominator = g0 + g1.abs();
        
        if denominator > 1e-10 {
            (numerator / denominator).sqrt()
        } else {
            0.0
        }
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
    // Time series accumulators for fluctuation-based observables
    energy_accumulator: TimeSeriesAccumulator,
    magnetization_accumulator: TimeSeriesAccumulator,
    // Full time series for advanced error analysis (optional)
    energy_time_series: Option<Vec<f64>>,
    magnetization_time_series: Option<Vec<f64>>,
}

#[derive(Default)]
struct ObservableCache {
    mean_w: f64,
    var_w: f64,
    mean_cos: f64,
    entropy: f64,
    triangle_sum: f64,
    _last_update: usize,
    // Additional cached values
    mean_cos_sq: f64,
    mean_cos_4th: f64,
    correlation_length: f64,
}

/// Time series accumulator for calculating fluctuations and higher moments
#[derive(Clone, Default)]
struct TimeSeriesAccumulator {
    sum: f64,
    sum_sq: f64,
    sum_4th: f64,
    count: usize,
}

impl TimeSeriesAccumulator {
    fn push(&mut self, value: f64) {
        self.sum += value;
        self.sum_sq += value * value;
        self.sum_4th += value.powi(4);
        self.count += 1;
    }
    
    fn mean(&self) -> f64 {
        if self.count > 0 {
            self.sum / self.count as f64
        } else {
            0.0
        }
    }
    
    fn variance(&self) -> f64 {
        if self.count > 1 {
            let n = self.count as f64;
            (self.sum_sq - self.sum * self.sum / n) / (n - 1.0)
        } else {
            0.0
        }
    }
    
    fn moment2(&self) -> f64 {
        if self.count > 0 {
            self.sum_sq / self.count as f64
        } else {
            0.0
        }
    }
    
    fn moment4(&self) -> f64 {
        if self.count > 0 {
            self.sum_4th / self.count as f64
        } else {
            0.0
        }
    }
    
    fn reset(&mut self) {
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.sum_4th = 0.0;
        self.count = 0;
    }
}

impl Default for BatchedObservables {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchedObservables {
    pub fn new() -> Self {
        Self {
            rotation_counter: 0,
            cached_values: ObservableCache::default(),
            energy_accumulator: TimeSeriesAccumulator::default(),
            magnetization_accumulator: TimeSeriesAccumulator::default(),
            energy_time_series: None,
            magnetization_time_series: None,
        }
    }
    
    /// Enable full time series collection for advanced error analysis
    pub fn enable_time_series(&mut self) {
        self.energy_time_series = Some(Vec::new());
        self.magnetization_time_series = Some(Vec::new());
    }
    
    /// Measure observables with rotation for expensive calculations
    pub fn measure(&mut self, graph: &FastGraph, alpha: f64, beta: f64) -> QuickObservables {
        self.rotation_counter += 1;
        
        // Always compute cheap observables
        let sum_w: f64 = graph.links.iter().map(|l| l.exp_neg_z).sum();
        let sum_cos: f64 = graph.links.iter().map(|l| l.cos_theta).sum();
        let m = graph.m() as f64;
        let n = graph.n() as f64;
        
        let mean_w = sum_w / m;
        let mean_cos = sum_cos / m;
        
        // Calculate energy for time series
        let current_energy = graph.action(alpha, beta);
        self.energy_accumulator.push(current_energy);
        
        // Calculate magnetization: m = (1/N)∑cos(θ) where sum is over all links
        // For a complete graph with N nodes and N(N-1)/2 links
        let magnetization = sum_cos / n;  // Normalized by number of nodes
        self.magnetization_accumulator.push(magnetization);
        
        // Store in full time series if enabled
        if let Some(ref mut series) = self.energy_time_series {
            series.push(current_energy);
        }
        if let Some(ref mut series) = self.magnetization_time_series {
            series.push(magnetization);
        }
        
        // Rotate expensive calculations
        match self.rotation_counter % 6 {
            0 => {
                // Update variance and higher moments for cos
                let (var_w, cos_sq, cos_4th) = graph.links.iter()
                    .fold((0.0, 0.0, 0.0), |(vw, c2, c4), l| {
                        let w = l.exp_neg_z;
                        let c = l.cos_theta;
                        (vw + (w - mean_w).powi(2), c2 + c*c, c4 + c.powi(4))
                    });
                self.cached_values.var_w = var_w / m;
                self.cached_values.mean_cos_sq = cos_sq / m;
                self.cached_values.mean_cos_4th = cos_4th / m;
            }
            1 => {
                // Update entropy
                self.cached_values.entropy = graph.entropy_action();
            }
            2 => {
                // Update triangle sum (most expensive)
                self.cached_values.triangle_sum = graph.triangle_sum();
            }
            3 => {
                // Update correlation length from correlation function
                self.cached_values.correlation_length = graph.calculate_correlation_length();
            }
            _ => {}  // Use cached values
        }
        
        self.cached_values.mean_w = mean_w;
        self.cached_values.mean_cos = mean_cos;
        
        // Calculate specific heat from energy fluctuations
        let specific_heat = if self.energy_accumulator.count > 10 {
            // C = (1/N) * (<E²> - <E>²)
            let energy_var = self.energy_accumulator.variance();
            energy_var / n
        } else {
            0.0
        };
        
        // Calculate susceptibility from magnetization fluctuations
        let susceptibility = if self.magnetization_accumulator.count > 10 {
            // χ = N * β * (<m²> - <m>²)
            let mag_var = self.magnetization_accumulator.variance();
            n * beta * mag_var
        } else {
            0.0  // Not enough data for variance calculation
        };
        
        // Calculate Binder cumulant from magnetization moments
        let binder_cumulant = if self.magnetization_accumulator.count > 10 {
            // U4 = 1 - <m⁴>/(3<m²>²)
            let m2 = self.magnetization_accumulator.moment2();
            let m4 = self.magnetization_accumulator.moment4();
            if m2 > 1e-10 {
                1.0 - m4 / (3.0 * m2 * m2)
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        QuickObservables {
            mean_w,
            var_w: self.cached_values.var_w,
            mean_cos,
            mean_action: beta * self.cached_values.entropy + alpha * self.cached_values.triangle_sum,
            susceptibility,
            specific_heat,
            binder_cumulant,
            correlation_length: self.cached_values.correlation_length,
            // Error estimates initialized to 0 (need separate jackknife analysis)
            specific_heat_err: 0.0,
            susceptibility_err: 0.0,
            binder_cumulant_err: 0.0,
        }
    }
    
    /// Reset time series accumulators (useful when changing parameters)
    pub fn reset_accumulators(&mut self) {
        self.energy_accumulator.reset();
        self.magnetization_accumulator.reset();
        if let Some(ref mut series) = self.energy_time_series {
            series.clear();
        }
        if let Some(ref mut series) = self.magnetization_time_series {
            series.clear();
        }
    }
    
    /// Get number of accumulated samples
    pub fn sample_count(&self) -> usize {
        self.energy_accumulator.count
    }
    
    /// Get energy time series (if enabled)
    pub fn energy_time_series(&self) -> Option<&[f64]> {
        self.energy_time_series.as_deref()
    }
    
    /// Get magnetization time series (if enabled)
    pub fn magnetization_time_series(&self) -> Option<&[f64]> {
        self.magnetization_time_series.as_deref()
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
    pub specific_heat: f64,
    pub binder_cumulant: f64,
    pub correlation_length: f64,
    // Error estimates (filled in by jackknife analysis)
    pub specific_heat_err: f64,
    pub susceptibility_err: f64,
    pub binder_cumulant_err: f64,
}

/// Jackknife error estimation
pub struct JackknifeEstimator {
    samples: Vec<f64>,
}

impl JackknifeEstimator {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }
    
    pub fn add_sample(&mut self, value: f64) {
        self.samples.push(value);
    }
    
    pub fn estimate_error<F>(&self, estimator: F) -> (f64, f64)
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = self.samples.len();
        if n < 2 {
            return (0.0, 0.0);
        }
        
        // Full sample estimate
        let full_estimate = estimator(&self.samples);
        
        // Jackknife estimates
        let mut jackknife_estimates = Vec::with_capacity(n);
        
        for i in 0..n {
            // Create subsample excluding i-th element
            let mut subsample = Vec::with_capacity(n - 1);
            for (j, &val) in self.samples.iter().enumerate() {
                if i != j {
                    subsample.push(val);
                }
            }
            
            let jack_estimate = estimator(&subsample);
            jackknife_estimates.push(jack_estimate);
        }
        
        // Jackknife mean
        let jack_mean: f64 = jackknife_estimates.iter().sum::<f64>() / n as f64;
        
        // Jackknife variance
        let jack_var: f64 = jackknife_estimates.iter()
            .map(|&x| (x - jack_mean).powi(2))
            .sum::<f64>() * (n - 1) as f64 / n as f64;
        
        let error = jack_var.sqrt();
        
        (full_estimate, error)
    }
    
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}