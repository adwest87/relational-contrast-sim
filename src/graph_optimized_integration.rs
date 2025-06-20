// Integration layer for using OptimizedGraph with existing codebase

use crate::graph_cache_optimized::{OptimizedGraph, OptimizedLink, StepInfo};
use crate::observables::Observables;
use rand::Rng;

/// Adapter to make OptimizedGraph compatible with existing interfaces
pub struct GraphAdapter {
    graph: OptimizedGraph,
    // Cache for f64 conversions if needed
    action_cache: Option<(f64, f64, f64)>, // (alpha, beta, action)
}

impl GraphAdapter {
    pub fn new(n: usize, rng: &mut impl Rng) -> Self {
        Self {
            graph: OptimizedGraph::new(n, rng),
            action_cache: None,
        }
    }
    
    /// Number of nodes
    pub fn n(&self) -> usize {
        self.graph.n
    }
    
    /// Number of links
    pub fn m(&self) -> usize {
        self.graph.links.len()
    }
    
    /// Get link by node indices
    pub fn get_link(&self, i: usize, j: usize) -> LinkView {
        let idx = self.graph.link_index(i, j);
        LinkView::from(&self.graph.links[idx])
    }
    
    /// Sum of all weights
    pub fn sum_weights(&self) -> f64 {
        self.graph.links.iter()
            .map(|l| l.exp_neg_z as f64)
            .sum()
    }
    
    /// Sum of cos(theta) over all links
    pub fn links_cos_sum(&self) -> f64 {
        self.graph.links.iter()
            .map(|l| l.cos_theta as f64)
            .sum()
    }
    
    /// Entropy action with f64 precision
    pub fn entropy_action(&self) -> f64 {
        self.graph.entropy_action()
    }
    
    /// Triangle sum with f64 precision
    pub fn triangle_sum(&self) -> f64 {
        self.graph.triangle_sum()
    }
    
    /// Total action
    pub fn action(&mut self, alpha: f64, beta: f64) -> f64 {
        // Check cache
        if let Some((cached_alpha, cached_beta, cached_action)) = self.action_cache {
            if (cached_alpha - alpha).abs() < 1e-10 && (cached_beta - beta).abs() < 1e-10 {
                return cached_action;
            }
        }
        
        let action = beta * self.graph.entropy_action() + alpha * self.graph.triangle_sum();
        self.action_cache = Some((alpha, beta, action));
        action
    }
    
    /// Run Metropolis step
    pub fn metropolis_step(
        &mut self,
        alpha: f64,
        beta: f64,
        delta_z: f64,
        delta_theta: f64,
        rng: &mut impl Rng,
    ) -> StepInfo {
        self.action_cache = None; // Invalidate cache
        self.graph.metropolis_step(alpha, beta, delta_z, delta_theta, rng)
    }
    
    /// Measure observables efficiently
    pub fn measure_observables(&self, alpha: f64, beta: f64) -> Observables {
        let n = self.n() as f64;
        let m = self.m() as f64;
        
        // Basic sums
        let sum_w = self.sum_weights();
        let sum_cos = self.links_cos_sum();
        
        // Action components
        let entropy = self.entropy_action();
        let triangles = self.triangle_sum();
        let action = beta * entropy + alpha * triangles;
        
        // Weight statistics
        let weights: Vec<f64> = self.graph.links.iter()
            .map(|l| l.exp_neg_z as f64)
            .collect();
        
        let mean_w = sum_w / m;
        let var_w = weights.iter()
            .map(|&w| (w - mean_w).powi(2))
            .sum::<f64>() / m;
        
        // Phase statistics
        let mean_cos = sum_cos / m;
        let sum_sin: f64 = self.graph.links.iter()
            .map(|l| l.sin_theta as f64)
            .sum();
        let mean_sin = sum_sin / m;
        
        // Effective dimension (simplified)
        let d_eff = if var_w > 1e-10 {
            mean_w * mean_w / var_w
        } else {
            m
        };
        
        Observables {
            mean_action: action,
            mean_w,
            var_w,
            sum_w,
            mean_cos,
            mean_sin,
            mean_phase: mean_sin.atan2(mean_cos),
            susceptibility: n * var_w,
            binder: 1.0 - var_w / (3.0 * mean_w * mean_w),
            d_eff,
            entropy,
            triangle_sum: triangles,
        }
    }
}

/// Read-only view of a link with f64 values
pub struct LinkView {
    pub i: usize,
    pub j: usize,
    pub z: f64,
    pub theta: f64,
    pub w: f64,
}

impl From<&OptimizedLink> for LinkView {
    fn from(link: &OptimizedLink) -> Self {
        Self {
            i: link.i as usize,
            j: link.j as usize,
            z: link.z as f64,
            theta: link.theta as f64,
            w: link.exp_neg_z as f64,
        }
    }
}

/// Drop-in replacement for existing simulation code
pub fn run_optimized_simulation(
    n: usize,
    alpha: f64,
    beta: f64,
    steps: usize,
    equilibration: usize,
    delta_z: f64,
    delta_theta: f64,
    rng: &mut impl Rng,
) -> Vec<Observables> {
    let mut graph = GraphAdapter::new(n, rng);
    let mut observations = Vec::new();
    
    // Equilibration
    for _ in 0..equilibration {
        graph.metropolis_step(alpha, beta, delta_z, delta_theta, rng);
    }
    
    // Measurement
    let measure_interval = 10; // Measure every 10 steps
    for step in 0..steps {
        graph.metropolis_step(alpha, beta, delta_z, delta_theta, rng);
        
        if step % measure_interval == 0 {
            observations.push(graph.measure_observables(alpha, beta));
        }
    }
    
    observations
}

/// Benchmark helper comparing original vs optimized for real workload
pub fn benchmark_real_workload(n: usize, steps: usize) {
    use std::time::Instant;
    use crate::graph::Graph as OriginalGraph;
    
    let mut rng = rand::thread_rng();
    let alpha = 1.5;
    let beta = 2.9;
    let delta_z = 0.1;
    let delta_theta = 0.1;
    
    println!("\nReal Workload Benchmark (N={}, {} steps)", n, steps);
    println!("Parameters: α={}, β={}, Δz={}, Δθ={}", alpha, beta, delta_z, delta_theta);
    
    // Original implementation
    let start = Instant::now();
    let mut orig_graph = OriginalGraph::complete_random_with(&mut rng, n);
    let mut orig_accepts = 0;
    let mut orig_obs = Vec::new();
    
    for step in 0..steps {
        let info = orig_graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
        if info.accept {
            orig_accepts += 1;
        }
        if step % 100 == 0 {
            orig_obs.push(orig_graph.measure_observables(alpha, beta));
        }
    }
    let orig_time = start.elapsed();
    
    // Optimized implementation
    let start = Instant::now();
    let mut opt_graph = GraphAdapter::new(n, &mut rng);
    let mut opt_accepts = 0;
    let mut opt_obs = Vec::new();
    
    for step in 0..steps {
        let info = opt_graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
        if info.accept {
            opt_accepts += 1;
        }
        if step % 100 == 0 {
            opt_obs.push(opt_graph.measure_observables(alpha, beta));
        }
    }
    let opt_time = start.elapsed();
    
    println!("\nResults:");
    println!("  Original:   {:?} ({:.2} μs/step)", 
             orig_time, orig_time.as_micros() as f64 / steps as f64);
    println!("  Optimized:  {:?} ({:.2} μs/step)", 
             opt_time, opt_time.as_micros() as f64 / steps as f64);
    println!("  Speedup:    {:.1}x", orig_time.as_secs_f64() / opt_time.as_secs_f64());
    println!("\nAcceptance rates:");
    println!("  Original:   {:.1}%", 100.0 * orig_accepts as f64 / steps as f64);
    println!("  Optimized:  {:.1}%", 100.0 * opt_accepts as f64 / steps as f64);
    
    // Verify consistency
    if !orig_obs.is_empty() && !opt_obs.is_empty() {
        let orig_mean_w: f64 = orig_obs.iter().map(|o| o.mean_w).sum::<f64>() / orig_obs.len() as f64;
        let opt_mean_w: f64 = opt_obs.iter().map(|o| o.mean_w).sum::<f64>() / opt_obs.len() as f64;
        println!("\nConsistency check:");
        println!("  Original <w>:  {:.6}", orig_mean_w);
        println!("  Optimized <w>: {:.6}", opt_mean_w);
        println!("  Difference:    {:.2}%", 100.0 * (opt_mean_w - orig_mean_w).abs() / orig_mean_w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_adapter_consistency() {
        let mut rng = rand::thread_rng();
        let n = 10;
        
        let mut adapter = GraphAdapter::new(n, &mut rng);
        
        // Test basic properties
        assert_eq!(adapter.n(), n);
        assert_eq!(adapter.m(), n * (n - 1) / 2);
        
        // Test action caching
        let action1 = adapter.action(1.0, 1.0);
        let action2 = adapter.action(1.0, 1.0); // Should use cache
        assert_eq!(action1, action2);
        
        // Test observable measurement
        let obs = adapter.measure_observables(1.0, 1.0);
        assert!(obs.mean_w > 0.0);
        assert!(obs.mean_action.is_finite());
    }
    
    #[test]
    fn test_workload_benchmark() {
        benchmark_real_workload(12, 1000);
    }
}