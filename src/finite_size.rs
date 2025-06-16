// finite_size_scaling.rs - Module for extracting critical exponents

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct FSAData {
    pub lattice_size: usize,
    pub beta: f64,
    pub alpha: f64,
    pub susceptibility: f64,
    pub specific_heat: f64,
    pub binder_cumulant: f64,
    pub correlation_length: f64,
}

/// Extract critical exponents from finite-size scaling
pub struct FiniteSizeAnalysis {
    data: Vec<FSAData>,
}

impl FiniteSizeAnalysis {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
    
    pub fn add_data(&mut self, data: FSAData) {
        self.data.push(data);
    }
    
    /// Extract γ/ν from susceptibility scaling: χ ~ L^(γ/ν)
    pub fn extract_gamma_over_nu(&self, beta_c: f64, alpha_c: f64) -> f64 {
        let mut sizes_and_chi: Vec<(f64, f64)> = Vec::new();
        
        // Collect data near critical point
        for d in &self.data {
            if (d.beta - beta_c).abs() < 0.1 && (d.alpha - alpha_c).abs() < 0.1 {
                let l = d.lattice_size as f64;
                sizes_and_chi.push((l.ln(), d.susceptibility.ln()));
            }
        }
        
        // Linear regression in log-log space
        linear_regression(&sizes_and_chi).0
    }
    
    /// Extract α/ν from specific heat scaling: C ~ L^(α/ν)
    pub fn extract_alpha_over_nu(&self, beta_c: f64, alpha_c: f64) -> f64 {
        let mut sizes_and_c: Vec<(f64, f64)> = Vec::new();
        
        for d in &self.data {
            if (d.beta - beta_c).abs() < 0.1 && (d.alpha - alpha_c).abs() < 0.1 {
                let l = d.lattice_size as f64;
                sizes_and_c.push((l.ln(), d.specific_heat.ln()));
            }
        }
        
        linear_regression(&sizes_and_c).0
    }
    
    /// Find crossing point of Binder cumulants for different sizes
    pub fn find_critical_point(&self) -> (f64, f64) {
        // Group by size
        let mut by_size: HashMap<usize, Vec<&FSAData>> = HashMap::new();
        for d in &self.data {
            by_size.entry(d.lattice_size).or_insert(Vec::new()).push(d);
        }
        
        // Find where Binder cumulants cross
        // This is simplified - in practice you'd interpolate
        let sizes: Vec<usize> = by_size.keys().copied().collect();
        if sizes.len() < 2 {
            return (3.0, 1.5); // Default estimate
        }
        
        // Find point where different sizes have similar Binder cumulant
        let mut best_beta = 3.0;
        let mut best_alpha = 1.5;
        let mut min_variance = f64::INFINITY;
        
        for d in &self.data {
            let mut binders = Vec::new();
            for &size in &sizes {
                if let Some(size_data) = by_size.get(&size) {
                    for sd in size_data {
                        if (sd.beta - d.beta).abs() < 0.05 && (sd.alpha - d.alpha).abs() < 0.05 {
                            binders.push(sd.binder_cumulant);
                        }
                    }
                }
            }
            
            if binders.len() >= 2 {
                let mean = binders.iter().sum::<f64>() / binders.len() as f64;
                let variance = binders.iter().map(|&b| (b - mean).powi(2)).sum::<f64>() / binders.len() as f64;
                
                if variance < min_variance {
                    min_variance = variance;
                    best_beta = d.beta;
                    best_alpha = d.alpha;
                }
            }
        }
        
        (best_beta, best_alpha)
    }
}

/// Simple linear regression returning (slope, intercept)
fn linear_regression(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
    let sum_xx: f64 = points.iter().map(|(x, _)| x * x).sum();
    let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
    
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;
    
    (slope, intercept)
}

/// Calculate Binder cumulant from 4th and 2nd moments
pub fn binder_cumulant(moment2: f64, moment4: f64) -> f64 {
    1.0 - moment4 / (3.0 * moment2 * moment2)
}