// Integration of importance sampling with existing Monte Carlo simulations

use crate::graph::Graph;
use crate::importance_sampling::{AdaptiveRidgeSampler, ImportanceMetropolis};
use crate::observables::Observables;
use rand::Rng;
use std::collections::HashMap;

/// Run importance-sampled Monte Carlo simulation focusing on the critical ridge
pub struct ImportanceSampledMC {
    graph: Graph,
    sampler: AdaptiveRidgeSampler,
    _current_beta: f64,
    _current_alpha: f64,
    measurements: Vec<WeightedMeasurement>,
}

#[derive(Clone)]
pub struct WeightedMeasurement {
    pub beta: f64,
    pub alpha: f64,
    pub weight: f64,
    pub observables: Observables,
}

impl ImportanceSampledMC {
    /// Create new importance-sampled MC simulation
    pub fn new(n: usize, rng: &mut impl Rng) -> Self {
        let graph = Graph::complete_random_with(rng, n);
        let sampler = AdaptiveRidgeSampler::new();
        
        // Start at estimated critical point
        let current_beta = 2.90;
        let current_alpha = 0.06 * current_beta + 1.31;
        
        Self {
            graph,
            sampler,
            _current_beta: current_beta,
            _current_alpha: current_alpha,
            measurements: Vec::new(),
        }
    }
    
    /// Run importance-sampled parameter sweep
    pub fn run_ridge_scan(
        &mut self,
        n_points: usize,
        mc_steps_per_point: usize,
        equilibration_steps: usize,
        delta_z: f64,
        delta_theta: f64,
        rng: &mut impl Rng,
    ) -> Vec<WeightedMeasurement> {
        println!("Starting importance-sampled ridge scan...");
        println!("N = {}, {} points, {} MC steps each", self.graph.n(), n_points, mc_steps_per_point);
        
        for point in 0..n_points {
            // Sample new (β, α) point with importance weight
            let (beta, alpha, weight) = self.sampler.sample_point(rng);
            
            // Run MC at this point
            let obs = self.run_mc_at_point(
                beta, alpha, 
                mc_steps_per_point, 
                equilibration_steps,
                delta_z, delta_theta, 
                rng
            );
            
            // Record measurement with importance weight
            self.measurements.push(WeightedMeasurement {
                beta,
                alpha,
                weight,
                observables: obs.clone(),
            });
            
            // Adapt sampler based on susceptibility
            self.sampler.record_measurement(beta, alpha, obs.susceptibility);
            
            // Progress report
            if (point + 1) % 10 == 0 {
                let (slope, intercept, _width) = self.sampler.get_ridge_params();
                println!("Point {}/{}: β={:.3}, α={:.3}, χ={:.2}, ridge={:.3}β+{:.3}", 
                         point + 1, n_points, beta, alpha, obs.susceptibility, slope, intercept);
            }
        }
        
        self.measurements.clone()
    }
    
    /// Run Monte Carlo at a specific (β, α) point
    fn run_mc_at_point(
        &mut self,
        beta: f64,
        alpha: f64,
        mc_steps: usize,
        equilibration: usize,
        delta_z: f64,
        delta_theta: f64,
        rng: &mut impl Rng,
    ) -> Observables {
        // Equilibrate at new parameters
        for _ in 0..equilibration {
            self.graph.metropolis_step(alpha, beta, delta_z, delta_theta, rng);
        }
        
        // Measure observables
        let sum_obs = Observables::default();
        let measure_interval = 10;
        let _n_measurements = mc_steps / measure_interval;
        
        for step in 0..mc_steps {
            self.graph.metropolis_step(alpha, beta, delta_z, delta_theta, rng);
            
            if step % measure_interval == 0 {
                // let obs = Observables::measure(&self.graph, beta, alpha);
                // sum_obs = sum_obs + obs;
                // TODO: Implement Add trait for Observables
            }
        }
        
        // Return average
        // sum_obs / (n_measurements as f64)
        // TODO: Implement Div trait for Observables
        sum_obs
    }
    
    /// Compute importance-weighted averages
    pub fn compute_weighted_averages(&self) -> HashMap<String, f64> {
        let mut results = HashMap::new();
        
        // Normalize weights
        let total_weight: f64 = self.measurements.iter().map(|m| m.weight).sum();
        
        // Compute weighted averages
        let mut weighted_susceptibility = 0.0;
        let mut weighted_binder = 0.0;
        let weighted_action = 0.0;
        let mut weighted_entropy = 0.0;
        
        for m in &self.measurements {
            let w = m.weight / total_weight;
            weighted_susceptibility += w * m.observables.susceptibility;
            weighted_binder += w * m.observables.binder_cumulant;
            // weighted_action += w * m.observables.action; // TODO: Add action field
            weighted_entropy += w * m.observables.entropy;
        }
        
        results.insert("susceptibility".to_string(), weighted_susceptibility);
        results.insert("binder".to_string(), weighted_binder);
        results.insert("action".to_string(), weighted_action);
        results.insert("entropy".to_string(), weighted_entropy);
        
        results
    }
    
    /// Find the peak susceptibility location using importance weighting
    pub fn find_weighted_peak(&self) -> (f64, f64, f64) {
        let mut best_beta = 0.0;
        let mut best_alpha = 0.0;
        let mut best_chi = 0.0;
        
        for m in &self.measurements {
            if m.observables.susceptibility > best_chi {
                best_chi = m.observables.susceptibility;
                best_beta = m.beta;
                best_alpha = m.alpha;
            }
        }
        
        (best_beta, best_alpha, best_chi)
    }
}

/// Parallel tempering with importance sampling for enhanced sampling
pub struct ImportanceTempering {
    replicas: Vec<Graph>,
    temperatures: Vec<f64>,
    parameters: Vec<(f64, f64)>,  // (β, α) for each replica
    importance_sampler: ImportanceMetropolis,
}

impl ImportanceTempering {
    /// Create parallel tempering system with importance-biased parameter updates
    pub fn new(n: usize, n_replicas: usize, rng: &mut impl Rng) -> Self {
        let mut replicas = Vec::new();
        let mut temperatures = Vec::new();
        let mut parameters = Vec::new();
        
        // Set up temperature ladder
        let t_min = 0.8;
        let t_max = 1.2;
        
        for i in 0..n_replicas {
            replicas.push(Graph::complete_random_with(rng, n));
            
            let t = t_min * f64::powf(t_max / t_min, i as f64 / (n_replicas - 1) as f64);
            temperatures.push(t);
            
            // Start near the ridge
            let beta = 2.85 + 0.1 * i as f64 / n_replicas as f64;
            let alpha = 0.06 * beta + 1.31;
            parameters.push((beta, alpha));
        }
        
        let importance_sampler = ImportanceMetropolis::new(2.90, 1.48);
        
        Self {
            replicas,
            temperatures,
            parameters,
            importance_sampler,
        }
    }
    
    /// Run one parallel tempering step with importance-biased parameter updates
    pub fn step(&mut self, delta_z: f64, delta_theta: f64, rng: &mut impl Rng) {
        let n_replicas = self.replicas.len();
        
        // MC moves for each replica
        for i in 0..n_replicas {
            let (beta, alpha) = self.parameters[i];
            let t = self.temperatures[i];
            
            // Scale parameters by temperature
            let beta_eff = beta / t;
            let alpha_eff = alpha / t;
            
            self.replicas[i].metropolis_step(alpha_eff, beta_eff, delta_z, delta_theta, rng);
        }
        
        // Attempt replica exchange
        if rng.gen_bool(0.1) {  // 10% chance
            let i = rng.gen_range(0..n_replicas-1);
            let j = i + 1;
            
            let (beta_i, alpha_i) = self.parameters[i];
            let (beta_j, alpha_j) = self.parameters[j];
            
            let action_i = self.replicas[i].action(alpha_i, beta_i);
            let action_j = self.replicas[j].action(alpha_j, beta_j);
            
            let delta = (1.0/self.temperatures[i] - 1.0/self.temperatures[j]) * 
                        (action_j - action_i);
            
            if delta <= 0.0 || rng.gen_range(0.0..1.0) < (-delta).exp() {
                self.replicas.swap(i, j);
            }
        }
        
        // Attempt importance-biased parameter update
        if rng.gen_bool(0.05) {  // 5% chance
            let i = rng.gen_range(0..n_replicas);
            let (old_beta, old_alpha) = self.parameters[i];
            
            // Propose new parameters using importance sampling
            let (new_beta, new_alpha, weight_ratio) = self.importance_sampler.propose_move(rng);
            
            // Calculate action change
            let old_action = self.replicas[i].action(old_alpha, old_beta);
            let new_action = self.replicas[i].action(new_alpha, new_beta);
            let delta_action = new_action - old_action;
            
            // Accept/reject with importance correction
            if self.importance_sampler.accept_move(
                new_beta, new_alpha, delta_action, weight_ratio, rng
            ) {
                self.parameters[i] = (new_beta, new_alpha);
            }
        }
    }
}

/// Analyze importance sampling efficiency
pub fn analyze_sampling_efficiency(measurements: &[WeightedMeasurement]) {
    println!("\n=== Importance Sampling Efficiency Analysis ===");
    
    // Effective sample size
    let sum_w: f64 = measurements.iter().map(|m| m.weight).sum();
    let sum_w2: f64 = measurements.iter().map(|m| m.weight * m.weight).sum();
    let n_eff = sum_w * sum_w / sum_w2;
    let efficiency = n_eff / measurements.len() as f64;
    
    println!("Total samples: {}", measurements.len());
    println!("Effective samples: {:.1}", n_eff);
    println!("Sampling efficiency: {:.1}%", efficiency * 100.0);
    
    // Weight distribution
    let weights: Vec<f64> = measurements.iter().map(|m| m.weight).collect();
    let min_w = weights.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_w = weights.iter().fold(0.0f64, |a, &b| a.max(b));
    let mean_w = sum_w / measurements.len() as f64;
    
    println!("\nWeight statistics:");
    println!("  Min:  {:.3}", min_w);
    println!("  Mean: {:.3}", mean_w);
    println!("  Max:  {:.3}", max_w);
    println!("  Ratio max/min: {:.1}", max_w / min_w);
    
    // Coverage analysis
    let mut beta_min = f64::INFINITY;
    let mut beta_max = 0.0f64;
    let mut alpha_min = f64::INFINITY;
    let mut alpha_max = 0.0f64;
    
    for m in measurements {
        beta_min = beta_min.min(m.beta);
        beta_max = beta_max.max(m.beta);
        alpha_min = alpha_min.min(m.alpha);
        alpha_max = alpha_max.max(m.alpha);
    }
    
    println!("\nParameter coverage:");
    println!("  β range: [{:.3}, {:.3}]", beta_min, beta_max);
    println!("  α range: [{:.3}, {:.3}]", alpha_min, alpha_max);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_importance_mc() {
        let mut rng = rand::thread_rng();
        let mut mc = ImportanceSampledMC::new(24, &mut rng);
        
        // Run short scan
        let measurements = mc.run_ridge_scan(
            10,      // n_points
            1000,    // mc_steps
            100,     // equilibration
            0.1,     // delta_z
            0.1,     // delta_theta
            &mut rng
        );
        
        assert_eq!(measurements.len(), 10);
        
        // Check that weights are reasonable
        for m in &measurements {
            assert!(m.weight > 0.0);
            assert!(m.weight.is_finite());
        }
        
        // Analyze efficiency
        analyze_sampling_efficiency(&measurements);
    }
}