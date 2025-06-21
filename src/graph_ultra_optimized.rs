// Ultra-optimized graph implementation combining all performance improvements
// Includes incremental updates for both triangle sum and spectral term

use rand::Rng;
use rand_pcg::Pcg64;
use rand::SeedableRng;
use std::f64::consts::TAU;
use nalgebra::{DMatrix, SymmetricEigen};

/// Ultra-optimized graph with all performance enhancements
pub struct UltraOptimizedGraph {
    // Core structure - optimized memory layout
    nodes: usize,  // Just store count, not vector
    
    // Link data in structure-of-arrays format for better cache usage
    pub link_i: Vec<u32>,
    pub link_j: Vec<u32>,
    pub z_values: Vec<f64>,
    pub theta_values: Vec<f64>,
    
    // Pre-computed values
    pub cos_theta: Vec<f64>,
    pub sin_theta: Vec<f64>,
    pub exp_neg_z: Vec<f64>,
    
    // Triangle optimization
    pub triangle_sum_cache: f64,
    triangles_per_edge: Vec<Vec<usize>>,  // For each edge, which triangles contain it
    
    // Spectral term optimization
    spectral_cache: Option<SpectralCache>,
    use_spectral: bool,
    
    // Performance tracking
    mc_steps: usize,
    cache_hits: usize,
}

/// Cache for spectral term computation
struct SpectralCache {
    eigenvalues: Vec<f64>,
    eigenvectors: DMatrix<f64>,
    last_full_compute: usize,
    recompute_interval: usize,
}

impl SpectralCache {
    /// Update eigenvalues using first-order perturbation theory
    /// When link (i,j) weight changes by Δw:
    /// λ'_k ≈ λ_k + Δw * v_k[i] * v_k[j]
    fn update_eigenvalues(&mut self, i: usize, j: usize, delta_w: f64) {
        let n = self.eigenvectors.nrows();
        for k in 0..n {
            let v_ki = self.eigenvectors[(k, i)];
            let v_kj = self.eigenvectors[(k, j)];
            // First-order perturbation
            self.eigenvalues[k] += delta_w * v_ki * v_kj;
        }
    }
    
    /// Check if we need full recomputation (perturbation theory can accumulate errors)
    fn needs_recompute(&self, current_step: usize) -> bool {
        current_step - self.last_full_compute >= self.recompute_interval
    }
}

impl UltraOptimizedGraph {
    /// Create new ultra-optimized graph
    pub fn new(n: usize, seed: u64) -> Self {
        let mut rng = Pcg64::seed_from_u64(seed);
        Self::new_with_rng(&mut rng, n)
    }
    
    /// Create UltraOptimizedGraph from Reference Graph (ensures identical initial state)
    pub fn from_graph(reference: &crate::graph::Graph) -> Self {
        let n = reference.n();
        let num_links = n * (n - 1) / 2;
        
        // Allocate all arrays
        let mut link_i = Vec::with_capacity(num_links);
        let mut link_j = Vec::with_capacity(num_links);
        let mut z_values = Vec::with_capacity(num_links);
        let mut theta_values = Vec::with_capacity(num_links);
        let mut cos_theta = Vec::with_capacity(num_links);
        let mut sin_theta = Vec::with_capacity(num_links);
        let mut exp_neg_z = Vec::with_capacity(num_links);
        
        // Copy data from reference graph (ensure identical ordering)
        for link in &reference.links {
            // Reference links are already ordered with i < j
            assert!(link.i < link.j, "Reference link ordering violated");
            
            link_i.push(link.i as u32);
            link_j.push(link.j as u32);
            z_values.push(link.z);
            theta_values.push(link.theta);
            cos_theta.push(link.theta.cos());
            sin_theta.push(link.theta.sin());
            exp_neg_z.push((-link.z).exp());
        }
        
        // Pre-compute triangle membership for each edge
        let mut triangles_per_edge = vec![Vec::new(); num_links];
        for link_idx in 0..num_links {
            let i = link_i[link_idx] as usize;
            let j = link_j[link_idx] as usize;
            
            // Find all k such that (i,j,k) forms a triangle
            for k in 0..n {
                if k != i && k != j {
                    triangles_per_edge[link_idx].push(k);
                }
            }
        }
        
        let mut graph = Self {
            nodes: n,
            link_i,
            link_j,
            z_values,
            theta_values,
            cos_theta,
            sin_theta,
            exp_neg_z,
            triangle_sum_cache: 0.0,
            triangles_per_edge,
            spectral_cache: None,
            use_spectral: false,
            mc_steps: 0,
            cache_hits: 0,
        };
        
        // Initialize triangle sum cache
        graph.triangle_sum_cache = graph.compute_full_triangle_sum();
        
        graph
    }
    
    /// Create new ultra-optimized graph with existing RNG
    pub fn new_with_rng(rng: &mut impl Rng, n: usize) -> Self {
        let num_links = n * (n - 1) / 2;
        
        // Allocate all arrays
        let mut link_i = Vec::with_capacity(num_links);
        let mut link_j = Vec::with_capacity(num_links);
        let mut z_values = Vec::with_capacity(num_links);
        let mut theta_values = Vec::with_capacity(num_links);
        let mut cos_theta = Vec::with_capacity(num_links);
        let mut sin_theta = Vec::with_capacity(num_links);
        let mut exp_neg_z = Vec::with_capacity(num_links);
        
        // Initialize links
        for i in 0..n {
            for j in (i + 1)..n {
                let z = rng.gen_range(0.001..10.0);
                let theta = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
                
                link_i.push(i as u32);
                link_j.push(j as u32);
                z_values.push(z);
                theta_values.push(theta);
                cos_theta.push(theta.cos());
                sin_theta.push(theta.sin());
                exp_neg_z.push((-z as f64).exp());
            }
        }
        
        // Pre-compute triangle membership for each edge
        let mut triangles_per_edge = vec![Vec::new(); num_links];
        for link_idx in 0..num_links {
            let i = link_i[link_idx] as usize;
            let j = link_j[link_idx] as usize;
            
            // Find all k such that (i,j,k) forms a triangle
            for k in 0..n {
                if k != i && k != j {
                    triangles_per_edge[link_idx].push(k);
                }
            }
        }
        
        let mut graph = Self {
            nodes: n,
            link_i,
            link_j,
            z_values,
            theta_values,
            cos_theta,
            sin_theta,
            exp_neg_z,
            triangle_sum_cache: 0.0,
            triangles_per_edge,
            spectral_cache: None,
            use_spectral: false,
            mc_steps: 0,
            cache_hits: 0,
        };
        
        // Initialize triangle sum cache
        graph.triangle_sum_cache = graph.compute_full_triangle_sum();
        
        graph
    }
    
    /// Enable spectral term with caching
    pub fn enable_spectral(&mut self, _n_cut: usize, _gamma: f64) {
        self.use_spectral = true;
        
        // Compute initial eigendecomposition
        let laplacian = self.build_laplacian();
        let eigen = SymmetricEigen::new(laplacian);
        
        self.spectral_cache = Some(SpectralCache {
            eigenvalues: eigen.eigenvalues.as_slice().to_vec(),
            eigenvectors: eigen.eigenvectors,
            last_full_compute: 0,
            recompute_interval: 100, // Recompute every 100 steps to avoid error accumulation
        });
    }
    
    /// Build Laplacian matrix
    fn build_laplacian(&self) -> DMatrix<f64> {
        let n = self.nodes;
        let mut laplacian = DMatrix::zeros(n, n);
        
        for idx in 0..self.link_i.len() {
            let i = self.link_i[idx] as usize;
            let j = self.link_j[idx] as usize;
            let w = self.exp_neg_z[idx];
            
            laplacian[(i, j)] = -w;
            laplacian[(j, i)] = -w;
            laplacian[(i, i)] += w;
            laplacian[(j, j)] += w;
        }
        
        laplacian
    }
    
    /// Fast link index calculation
    #[inline(always)]
    fn link_index(&self, i: usize, j: usize) -> usize {
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        let n = self.nodes;
        i * n - (i * (i + 1) >> 1) + j - i - 1
    }
    
    /// Get phase with proper antisymmetry: θ_ij if i < j, -θ_ij if i > j
    #[inline(always)]
    pub fn get_phase(&self, from_node: usize, to_node: usize) -> f64 {
        let link_idx = self.link_index(from_node, to_node);
        if from_node < to_node {
            self.theta_values[link_idx]
        } else {
            -self.theta_values[link_idx]
        }
    }
    
    /// Compute full triangle sum (only called once at initialization)
    pub fn compute_full_triangle_sum(&self) -> f64 {
        let n = self.nodes;
        let mut sum = 0.0;
        
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    // Use proper antisymmetric phases
                    let t_ij = self.get_phase(i, j);
                    let t_jk = self.get_phase(j, k);
                    let t_ki = self.get_phase(k, i);
                    
                    let phase_sum = t_ij + t_jk + t_ki;
                    sum += phase_sum.cos();
                }
            }
        }
        
        sum
    }
    
    /// Incremental triangle sum update - O(N) instead of O(N³)
    #[inline]
    pub fn triangle_sum_delta(&self, link_idx: usize, new_theta: f64) -> f64 {
        let old_theta = self.theta_values[link_idx];
        let delta_theta = new_theta - old_theta;
        
        // For very small changes, use derivative
        if delta_theta.abs() < 1e-8 {
            return self.triangle_sum_derivative(link_idx, delta_theta);
        }
        
        let i = self.link_i[link_idx] as usize;
        let j = self.link_j[link_idx] as usize;
        let mut delta = 0.0;
        
        // Calculate triangles containing edge (i,j) with proper antisymmetry
        // Using the same approach as the original implementation
        for &k in &self.triangles_per_edge[link_idx] {
            // Calculate triangle phases with proper antisymmetry
            let t_ij_old = self.get_phase(i, j);
            let t_jk = self.get_phase(j, k);
            let t_ki = self.get_phase(k, i);
            
            let old_sum = t_ij_old + t_jk + t_ki;
            
            // Calculate new triangle with updated theta for link (i,j)
            let t_ij_new = if i < j { new_theta } else { -new_theta };
            let new_sum = t_ij_new + t_jk + t_ki;
            
            delta += new_sum.cos() - old_sum.cos();
        }
        
        delta
    }
    
    /// Analytical derivative for very small theta changes
    #[inline]
    fn triangle_sum_derivative(&self, link_idx: usize, delta_theta: f64) -> f64 {
        let i = self.link_i[link_idx] as usize;
        let j = self.link_j[link_idx] as usize;
        let theta_ij = self.theta_values[link_idx];
        let mut derivative = 0.0;
        
        for &k in &self.triangles_per_edge[link_idx] {
            let idx_ik = self.link_index(i, k);
            let idx_jk = self.link_index(j, k);
            let sum = theta_ij + self.theta_values[idx_ik] + self.theta_values[idx_jk];
            derivative -= sum.sin(); // d/dx cos(x) = -sin(x)
        }
        
        derivative * delta_theta
    }
    
    /// Spectral action with caching and perturbation updates
    fn spectral_action_delta(&mut self, link_idx: usize, new_z: f64) -> f64 {
        if !self.use_spectral || self.spectral_cache.is_none() {
            return 0.0;
        }
        
        let old_w = self.exp_neg_z[link_idx];
        let new_w = (-new_z).exp();
        let delta_w = new_w - old_w;
        
        // Check if we need full recomputation
        let needs_recompute = self.spectral_cache.as_ref().unwrap().needs_recompute(self.mc_steps);
        
        if needs_recompute {
            // Full recomputation
            let laplacian = self.build_laplacian();
            let eigen = SymmetricEigen::new(laplacian);
            let cache = self.spectral_cache.as_mut().unwrap();
            cache.eigenvalues = eigen.eigenvalues.as_slice().to_vec();
            cache.eigenvectors = eigen.eigenvectors;
            cache.last_full_compute = self.mc_steps;
        } else {
            let cache = self.spectral_cache.as_mut().unwrap();
            // Perturbative update
            let i = self.link_i[link_idx] as usize;
            let j = self.link_j[link_idx] as usize;
            cache.update_eigenvalues(i, j, delta_w);
            self.cache_hits += 1;
        }
        
        // Compute spectral action (simplified for demo)
        let cache = self.spectral_cache.as_ref().unwrap();
        let mean_eigenvalue = cache.eigenvalues.iter().sum::<f64>() / cache.eigenvalues.len() as f64;
        cache.eigenvalues.iter()
            .map(|&lambda| (lambda - mean_eigenvalue).powi(2))
            .sum::<f64>()
    }
    
    /// Ultra-optimized Metropolis step
    pub fn metropolis_step(
        &mut self,
        alpha: f64,
        beta: f64,
        gamma: f64,
        delta_z: f64,
        delta_theta: f64,
        rng: &mut Pcg64,
    ) -> bool {
        self.mc_steps += 1;
        let link_idx = rng.gen_range(0..self.link_i.len());
        
        let do_z_update = delta_z > 0.0 && rng.gen_bool(0.5);
        
        let (delta_s, update_fn): (f64, Box<dyn FnOnce(&mut Self)>) = if do_z_update {
            // Z-update
            let old_z = self.z_values[link_idx];
            let new_z = (old_z + rng.gen_range(-delta_z..=delta_z)).max(0.001);
            
            // Entropy change
            let old_entropy = -old_z * self.exp_neg_z[link_idx];
            let new_entropy = -new_z * (-new_z).exp();
            let delta_entropy = new_entropy - old_entropy;
            
            // Spectral change (if enabled)
            let delta_spectral = if self.use_spectral {
                let old_spectral = self.spectral_action_delta(link_idx, old_z);
                let new_spectral = self.spectral_action_delta(link_idx, new_z);
                new_spectral - old_spectral
            } else {
                0.0
            };
            
            let delta_s = beta * delta_entropy + gamma * delta_spectral;
            
            let update = Box::new(move |graph: &mut Self| {
                graph.z_values[link_idx] = new_z;
                graph.exp_neg_z[link_idx] = (-new_z).exp();
            });
            
            (delta_s, update)
        } else {
            // Phase update
            let old_theta = self.theta_values[link_idx];
            let d_theta = rng.gen_range(-delta_theta..=delta_theta);
            let new_theta = old_theta + d_theta;
            
            // Triangle sum change
            let delta_triangle = self.triangle_sum_delta(link_idx, new_theta);
            let delta_s = alpha * delta_triangle;
            
            let update = Box::new(move |graph: &mut Self| {
                graph.theta_values[link_idx] = new_theta;
                graph.cos_theta[link_idx] = new_theta.cos();
                graph.sin_theta[link_idx] = new_theta.sin();
                graph.triangle_sum_cache += delta_triangle;
            });
            
            (delta_s, update)
        };
        
        // Metropolis criterion
        let accept = delta_s <= 0.0 || rng.gen_range(0.0..1.0) < (-delta_s).exp();
        
        if accept {
            update_fn(self);
        }
        
        accept
    }
    
    /// Get current action value
    pub fn action(&self, alpha: f64, beta: f64, gamma: f64) -> f64 {
        let entropy: f64 = self.z_values.iter()
            .zip(&self.exp_neg_z)
            .map(|(&z, &w)| -z * w)
            .sum();
        
        let spectral = if self.use_spectral && self.spectral_cache.is_some() {
            let cache = self.spectral_cache.as_ref().unwrap();
            let mean = cache.eigenvalues.iter().sum::<f64>() / cache.eigenvalues.len() as f64;
            cache.eigenvalues.iter()
                .map(|&lambda| (lambda - mean).powi(2))
                .sum::<f64>()
        } else {
            0.0
        };
        
        beta * entropy + alpha * self.triangle_sum_cache + gamma * spectral
    }
    
    /// Get performance statistics
    pub fn performance_stats(&self) -> String {
        format!(
            "MC steps: {}, Spectral cache hits: {} ({:.1}%)",
            self.mc_steps,
            self.cache_hits,
            100.0 * self.cache_hits as f64 / self.mc_steps.max(1) as f64
        )
    }
    
    /// Getters for analysis
    pub fn n(&self) -> usize { self.nodes }
    pub fn m(&self) -> usize { self.link_i.len() }
    pub fn triangle_sum(&self) -> f64 { self.triangle_sum_cache }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_incremental_triangle_update() {
        let mut graph = UltraOptimizedGraph::new(10, 42);
        let original_sum = graph.triangle_sum();
        
        // Manually compute a small change
        let link_idx = 0;
        let old_theta = graph.theta_values[link_idx];
        let new_theta = old_theta + 0.1;
        
        let delta = graph.triangle_sum_delta(link_idx, new_theta);
        graph.theta_values[link_idx] = new_theta;
        graph.triangle_sum_cache += delta;
        
        // Compare with full recomputation
        let full_sum = graph.compute_full_triangle_sum();
        let incremental_sum = graph.triangle_sum();
        
        assert!((full_sum - incremental_sum).abs() < 1e-10,
                "Incremental update diverged: {} vs {}", full_sum, incremental_sum);
    }
}