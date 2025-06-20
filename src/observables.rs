// observables.rs - Extended measurements for phase transition analysis

use crate::graph::Graph;

#[derive(Debug, Clone, Default)]
pub struct Observables {
    // Basic observables
    pub mean_w: f64,
    pub mean_cos: f64,
    pub entropy: f64,
    pub triangle_sum: f64,
    
    // Phase transition indicators
    pub susceptibility: f64,
    pub specific_heat: f64,
    pub binder_cumulant: f64,
    
    // Order parameters
    pub magnetization: f64,      // For gauge symmetry breaking
    pub link_variance: f64,      // For geometric phase
    
    // Correlation functions
    pub correlation_length: f64,
    pub spectral_gap: f64,
}

impl Observables {
    /// Measure all observables from current graph state
    pub fn measure(graph: &Graph, beta: f64, alpha: f64) -> Self {
        let n_links = graph.m() as f64;
        let n_tri = graph.n_tri() as f64;
        
        // Basic quantities
        let mean_w = graph.sum_weights() / n_links;
        let mean_cos = graph.links_cos_sum() / n_links;
        let entropy = graph.entropy_action();
        let triangle_sum = graph.triangle_sum();
        
        // Calculate variance for susceptibility
        let w_squared: f64 = graph.links.iter().map(|l| l.w().powi(2)).sum::<f64>() / n_links;
        let link_variance = w_squared - mean_w.powi(2);
        
        // Magnetic susceptibility (gauge sector)
        let cos_squared: f64 = graph.links.iter()
            .map(|l| l.theta.cos().powi(2))
            .sum::<f64>() / n_links;
        let susceptibility = n_links * (cos_squared - mean_cos.powi(2));
        
        // Specific heat (from action fluctuations - placeholder)
        let action = graph.action(alpha, beta);
        let specific_heat = 0.0; // Would need time series for proper calculation
        
        // Binder cumulant (needs 4th moment - placeholder)
        let binder_cumulant = 0.0; // Requires moment calculations
        
        // Magnetization-like order parameter
        let magnetization = mean_cos.abs();
        
        // Placeholder for correlation length and spectral gap
        // These require eigenvalue calculations
        let correlation_length = 0.0;
        let spectral_gap = 0.0;
        
        Observables {
            mean_w,
            mean_cos,
            entropy,
            triangle_sum,
            susceptibility,
            specific_heat,
            binder_cumulant,
            magnetization,
            link_variance,
            correlation_length,
            spectral_gap,
        }
    }
}

/// Time series accumulator for calculating fluctuations
pub struct TimeSeriesAccumulator {
    samples: Vec<f64>,
    sum: f64,
    sum_sq: f64,
    sum_4th: f64,
}

impl TimeSeriesAccumulator {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
            sum: 0.0,
            sum_sq: 0.0,
            sum_4th: 0.0,
        }
    }
    
    pub fn push(&mut self, value: f64) {
        self.samples.push(value);
        self.sum += value;
        self.sum_sq += value * value;
        self.sum_4th += value.powi(4);
    }
    
    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() { 0.0 } else { self.sum / self.samples.len() as f64 }
    }
    
    pub fn variance(&self) -> f64 {
        if self.samples.len() < 2 { return 0.0; }
        let n = self.samples.len() as f64;
        (self.sum_sq / n) - (self.sum / n).powi(2)
    }
    
    pub fn moment4(&self) -> f64 {
        if self.samples.is_empty() { 0.0 } else { self.sum_4th / self.samples.len() as f64 }
    }
    
    pub fn binder_cumulant(&self) -> f64 {
        let m2 = self.variance() + self.mean().powi(2);
        let m4 = self.moment4();
        if m2 > 0.0 {
            1.0 - m4 / (3.0 * m2 * m2)
        } else {
            0.0
        }
    }
    
    pub fn autocorrelation_time(&self) -> f64 {
        // Simplified integrated autocorrelation time
        if self.samples.len() < 100 { return 1.0; }
        
        let mean = self.mean();
        let var = self.variance();
        if var == 0.0 { return 1.0; }
        
        let mut sum = 0.5; // τ = 0 contribution
        for t in 1..50.min(self.samples.len() / 4) {
            let mut c_t = 0.0;
            for i in 0..self.samples.len() - t {
                c_t += (self.samples[i] - mean) * (self.samples[i + t] - mean);
            }
            c_t /= (self.samples.len() - t) as f64 * var;
            
            if c_t < 0.1 { break; } // Cutoff when correlation becomes small
            sum += c_t;
        }
        
        2.0 * sum
    }
}