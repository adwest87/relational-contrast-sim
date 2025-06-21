// Importance sampling for efficient exploration of the critical region
use rand::Rng;
use rand_distr::{Normal, Distribution};
use std::f64::consts::PI;

/// Ridge-biased importance sampling for the critical region
/// The ridge follows α = 0.06β + 1.31
pub struct RidgeImportanceSampler {
    ridge_slope: f64,
    ridge_intercept: f64,
    ridge_width: f64,  // Width of the Gaussian around the ridge
    beta_range: (f64, f64),
    alpha_range: (f64, f64),
}

impl Default for RidgeImportanceSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl RidgeImportanceSampler {
    /// Create a new ridge-biased sampler
    pub fn new() -> Self {
        Self {
            ridge_slope: 0.06,
            ridge_intercept: 1.31,
            ridge_width: 0.02,  // Tune based on system size
            beta_range: (2.8, 3.0),
            alpha_range: (1.45, 1.55),
        }
    }
    
    /// Create sampler with custom parameters
    pub fn with_params(slope: f64, intercept: f64, width: f64) -> Self {
        Self {
            ridge_slope: slope,
            ridge_intercept: intercept,
            ridge_width: width,
            beta_range: (2.8, 3.0),
            alpha_range: (1.45, 1.55),
        }
    }
    
    /// Set the sampling range for beta
    pub fn set_beta_range(&mut self, min: f64, max: f64) {
        self.beta_range = (min, max);
    }
    
    /// Set the sampling range for alpha
    pub fn set_alpha_range(&mut self, min: f64, max: f64) {
        self.alpha_range = (min, max);
    }
    
    /// Sample a point (β, α) from the ridge-biased distribution
    pub fn sample_point(&self, rng: &mut impl Rng) -> (f64, f64, f64) {
        // Sample β uniformly in the range
        let beta = rng.gen_range(self.beta_range.0..self.beta_range.1);
        
        // Calculate the ridge location for this β
        let alpha_ridge = self.ridge_slope * beta + self.ridge_intercept;
        
        // Sample α from a Gaussian centered on the ridge
        let normal = Normal::new(alpha_ridge, self.ridge_width).unwrap();
        let mut alpha = normal.sample(rng);
        
        // Ensure α is within bounds
        alpha = alpha.clamp(self.alpha_range.0, self.alpha_range.1);
        
        // Calculate the importance weight
        let weight = self.importance_weight(beta, alpha);
        
        (beta, alpha, weight)
    }
    
    /// Calculate the importance weight for a given (β, α) point
    pub fn importance_weight(&self, beta: f64, alpha: f64) -> f64 {
        // Ridge location for this β
        let alpha_ridge = self.ridge_slope * beta + self.ridge_intercept;
        
        // Distance from ridge
        let distance = f64::abs(alpha - alpha_ridge);
        
        // Proposal probability: Gaussian around ridge
        let proposal_prob = (-0.5 * (distance / self.ridge_width).powi(2)).exp() 
            / (self.ridge_width * (2.0 * PI).sqrt());
        
        // Target probability: uniform over the region
        let target_prob = 1.0 / ((self.beta_range.1 - self.beta_range.0) * 
                                 (self.alpha_range.1 - self.alpha_range.0));
        
        // Importance weight
        target_prob / proposal_prob
    }
    
    /// Generate a batch of importance samples
    pub fn sample_batch(&self, n_samples: usize, rng: &mut impl Rng) -> Vec<(f64, f64, f64)> {
        (0..n_samples)
            .map(|_| self.sample_point(rng))
            .collect()
    }
}

/// Adaptive importance sampler that learns the optimal ridge location
pub struct AdaptiveRidgeSampler {
    base_sampler: RidgeImportanceSampler,
    history: Vec<(f64, f64, f64)>,  // (β, α, susceptibility)
    adaptation_rate: f64,
    min_samples: usize,
}

impl Default for AdaptiveRidgeSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveRidgeSampler {
    pub fn new() -> Self {
        Self {
            base_sampler: RidgeImportanceSampler::new(),
            history: Vec::new(),
            adaptation_rate: 0.1,
            min_samples: 100,
        }
    }
    
    /// Record a measurement (β, α, susceptibility)
    pub fn record_measurement(&mut self, beta: f64, alpha: f64, susceptibility: f64) {
        self.history.push((beta, alpha, susceptibility));
        
        // Adapt ridge parameters after enough samples
        if self.history.len() >= self.min_samples && self.history.len() % 50 == 0 {
            self.adapt_ridge_parameters();
        }
    }
    
    /// Adapt ridge parameters based on observed susceptibility peaks
    fn adapt_ridge_parameters(&mut self) {
        if self.history.len() < self.min_samples {
            return;
        }
        
        // Weight points by their susceptibility
        let total_chi: f64 = self.history.iter().map(|(_, _, chi)| chi).sum();
        
        // Weighted linear regression
        let mut sum_w = 0.0;
        let mut sum_wx = 0.0;
        let mut sum_wy = 0.0;
        let mut sum_wxx = 0.0;
        let mut sum_wxy = 0.0;
        
        for &(beta, alpha, chi) in &self.history {
            let w = chi / total_chi;  // Weight by susceptibility
            sum_w += w;
            sum_wx += w * beta;
            sum_wy += w * alpha;
            sum_wxx += w * beta * beta;
            sum_wxy += w * beta * alpha;
        }
        
        // Calculate new ridge parameters
        let det = sum_w * sum_wxx - sum_wx * sum_wx;
        if f64::abs(det) > 1e-10 {
            let new_slope = (sum_w * sum_wxy - sum_wx * sum_wy) / det;
            let new_intercept = (sum_wy - new_slope * sum_wx) / sum_w;
            
            // Smooth adaptation
            self.base_sampler.ridge_slope = (1.0 - self.adaptation_rate) * self.base_sampler.ridge_slope 
                + self.adaptation_rate * new_slope;
            self.base_sampler.ridge_intercept = (1.0 - self.adaptation_rate) * self.base_sampler.ridge_intercept 
                + self.adaptation_rate * new_intercept;
        }
        
        // Adapt width based on scatter around ridge
        let mut sum_sq_dist = 0.0;
        for &(beta, alpha, chi) in &self.history {
            let alpha_ridge = self.base_sampler.ridge_slope * beta + self.base_sampler.ridge_intercept;
            let dist = f64::abs(alpha - alpha_ridge);
            sum_sq_dist += chi * dist * dist / total_chi;
        }
        let new_width = (sum_sq_dist / sum_w).sqrt();
        
        // Update width with smoothing
        self.base_sampler.ridge_width = (1.0 - self.adaptation_rate) * self.base_sampler.ridge_width 
            + self.adaptation_rate * new_width;
    }
    
    /// Sample using current adapted parameters
    pub fn sample_point(&self, rng: &mut impl Rng) -> (f64, f64, f64) {
        self.base_sampler.sample_point(rng)
    }
    
    /// Get current ridge parameters
    pub fn get_ridge_params(&self) -> (f64, f64, f64) {
        (self.base_sampler.ridge_slope, 
         self.base_sampler.ridge_intercept, 
         self.base_sampler.ridge_width)
    }
}

/// Metropolis-Hastings with importance sampling proposal
pub struct ImportanceMetropolis {
    sampler: RidgeImportanceSampler,
    current_beta: f64,
    current_alpha: f64,
    step_size: f64,
}

impl ImportanceMetropolis {
    pub fn new(beta: f64, alpha: f64) -> Self {
        Self {
            sampler: RidgeImportanceSampler::new(),
            current_beta: beta,
            current_alpha: alpha,
            step_size: 0.01,
        }
    }
    
    /// Propose a new (β, α) using ridge-biased jumps
    pub fn propose_move(&self, rng: &mut impl Rng) -> (f64, f64, f64) {
        // Mix local moves with ridge-biased jumps
        if rng.gen_bool(0.3) {
            // 30% ridge-biased jump
            self.sampler.sample_point(rng)
        } else {
            // 70% local move
            let d_beta = rng.gen_range(-self.step_size..self.step_size);
            let d_alpha = rng.gen_range(-self.step_size..self.step_size);
            
            let new_beta = (self.current_beta + d_beta)
                .clamp(self.sampler.beta_range.0, self.sampler.beta_range.1);
            let new_alpha = (self.current_alpha + d_alpha)
                .clamp(self.sampler.alpha_range.0, self.sampler.alpha_range.1);
            
            let weight = self.sampler.importance_weight(new_beta, new_alpha);
            (new_beta, new_alpha, weight)
        }
    }
    
    /// Accept or reject the proposed move
    pub fn accept_move(&mut self, new_beta: f64, new_alpha: f64, 
                      delta_action: f64, weight_ratio: f64, rng: &mut impl Rng) -> bool {
        // Metropolis criterion with importance weight correction
        let accept_prob = (weight_ratio * (-delta_action).exp()).min(1.0);
        
        if rng.gen_range(0.0..1.0) < accept_prob {
            self.current_beta = new_beta;
            self.current_alpha = new_alpha;
            true
        } else {
            false
        }
    }
}

/// Example usage in a simulation
pub fn example_critical_scan() {
    use rand::thread_rng;
    
    let mut rng = thread_rng();
    let mut sampler = AdaptiveRidgeSampler::new();
    
    println!("Starting importance-sampled critical region scan...");
    
    // Generate importance-weighted samples
    for i in 0..1000 {
        let (beta, alpha, _weight) = sampler.sample_point(&mut rng);
        
        // Run simulation at (beta, alpha)
        // let susceptibility = run_simulation(beta, alpha);
        let susceptibility = 100.0; // Placeholder
        
        // Record for adaptation
        sampler.record_measurement(beta, alpha, susceptibility);
        
        // Use importance weight in averaging
        // weighted_average += susceptibility * weight;
        
        if i % 100 == 0 {
            let (slope, intercept, width) = sampler.get_ridge_params();
            println!("Step {}: Ridge = {:.3}β + {:.3}, width = {:.3}", 
                     i, slope, intercept, width);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ridge_sampling() {
        let mut rng = rand::thread_rng();
        let sampler = RidgeImportanceSampler::new();
        
        // Sample many points
        let samples = sampler.sample_batch(1000, &mut rng);
        
        // Check that samples cluster around ridge
        let mut sum_dist = 0.0;
        for (beta, alpha, _) in &samples {
            let alpha_ridge = 0.06 * beta + 1.31;
            sum_dist += f64::abs(alpha - alpha_ridge);
        }
        let mean_dist = sum_dist / samples.len() as f64;
        
        // Mean distance should be comparable to ridge width
        assert!(mean_dist < 2.0 * sampler.ridge_width);
    }
    
    #[test]
    fn test_adaptive_sampler() {
        let mut rng = rand::thread_rng();
        let mut sampler = AdaptiveRidgeSampler::new();
        
        // Simulate measurements with peak at slightly different ridge
        for _ in 0..200 {
            let beta = rng.gen_range(2.85..2.95);
            let alpha_true = 0.065 * beta + 1.305;  // Slightly different ridge
            let alpha = alpha_true + rng.gen_range(-0.01..0.01);
            
            // Susceptibility peaks at true ridge
            let dist = ((alpha - alpha_true) as f64).abs();
            let chi = 100.0 * (-dist * dist / 0.001).exp();
            
            sampler.record_measurement(beta, alpha, chi);
        }
        
        // Check that sampler adapted toward true ridge
        let (slope, intercept, _) = sampler.get_ridge_params();
        assert!(f64::abs(slope - 0.065) < 0.01);
        assert!(f64::abs(intercept - 1.305) < 0.01);
    }
}