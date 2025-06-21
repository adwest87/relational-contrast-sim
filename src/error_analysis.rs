// error_analysis.rs - Comprehensive error analysis for Monte Carlo observables

use std::f64::consts::PI;

/// Comprehensive error analysis for Monte Carlo data
pub struct ErrorAnalysis {
    /// Raw time series data
    data: Vec<f64>,
    /// Integrated autocorrelation time
    tau_int: f64,
    /// Effective sample size
    n_eff: f64,
    /// Statistical error (from autocorrelation)
    stat_error: f64,
    /// Jackknife error estimate
    jack_error: f64,
}

impl ErrorAnalysis {
    /// Create error analysis from time series data
    pub fn new(data: Vec<f64>) -> Self {
        let n = data.len() as f64;
        
        // Calculate integrated autocorrelation time
        let tau_int = Self::integrated_autocorr_time(&data);
        
        // Effective sample size
        let n_eff = n / (2.0 * tau_int);
        
        // Statistical error accounting for autocorrelation
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1.0);
        let stat_error = (variance / n_eff).sqrt();
        
        // Jackknife error
        let jack_error = Self::jackknife_error(&data, |x| {
            x.iter().sum::<f64>() / x.len() as f64
        });
        
        Self {
            data,
            tau_int,
            n_eff,
            stat_error,
            jack_error,
        }
    }
    
    /// Calculate integrated autocorrelation time using automatic windowing
    fn integrated_autocorr_time(data: &[f64]) -> f64 {
        let n = data.len();
        if n < 10 {
            return 0.5; // Minimal correlation for very short series
        }
        
        // Calculate mean
        let mean = data.iter().sum::<f64>() / n as f64;
        
        // Calculate normalized autocorrelation function
        let c0 = data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / n as f64;
        
        if c0 == 0.0 {
            return 0.5; // No variance, no correlation
        }
        
        // Automatic windowing (Sokal 1989)
        let mut tau_sum = 0.5; // C(0) contributes 0.5
        let mut window = 0;
        
        for t in 1..n.min(n/4) {
            // Calculate C(t)
            let mut ct = 0.0;
            for i in 0..n-t {
                ct += (data[i] - mean) * (data[i+t] - mean);
            }
            ct /= (n - t) as f64;
            
            let rho_t = ct / c0;
            tau_sum += rho_t;
            
            // Automatic windowing condition
            if t >= (6.0 * tau_sum) as usize {
                window = t;
                break;
            }
            
            // Stop if correlation becomes negligible
            if rho_t.abs() < 0.05 && t > 10 {
                window = t;
                break;
            }
        }
        
        // Add tail correction if window was not reached
        if window == 0 {
            window = n.min(n/4);
        }
        
        tau_sum.max(0.5) // Ensure positive and at least 0.5
    }
    
    /// Calculate jackknife error for a given estimator function
    fn jackknife_error<F>(data: &[f64], estimator: F) -> f64
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = data.len();
        if n < 2 {
            return 0.0;
        }
        
        // Full sample estimate
        let _full_estimate = estimator(data);
        
        // Jackknife samples
        let mut jack_estimates = Vec::with_capacity(n);
        let mut subsample = Vec::with_capacity(n - 1);
        
        for i in 0..n {
            subsample.clear();
            for (j, &val) in data.iter().enumerate() {
                if i != j {
                    subsample.push(val);
                }
            }
            jack_estimates.push(estimator(&subsample));
        }
        
        // Jackknife variance
        let jack_mean = jack_estimates.iter().sum::<f64>() / n as f64;
        let jack_var = jack_estimates.iter()
            .map(|&x| (x - jack_mean).powi(2))
            .sum::<f64>() * (n - 1) as f64 / n as f64;
        
        jack_var.sqrt()
    }
    
    /// Get all error estimates
    pub fn errors(&self) -> ErrorEstimates {
        ErrorEstimates {
            tau_int: self.tau_int,
            n_eff: self.n_eff,
            stat_error: self.stat_error,
            jack_error: self.jack_error,
            relative_error: self.stat_error / self.mean().abs(),
        }
    }
    
    /// Mean of the data
    pub fn mean(&self) -> f64 {
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }
    
    /// Variance of the data
    pub fn variance(&self) -> f64 {
        let mean = self.mean();
        self.data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (self.data.len() - 1) as f64
    }
}

/// Container for different error estimates
#[derive(Debug, Clone, Copy)]
pub struct ErrorEstimates {
    pub tau_int: f64,
    pub n_eff: f64,
    pub stat_error: f64,
    pub jack_error: f64,
    pub relative_error: f64,
}

/// Chi-squared test for goodness of fit
pub struct ChiSquaredTest {
    chi2: f64,
    dof: usize,
    p_value: f64,
}

impl ChiSquaredTest {
    /// Perform chi-squared test comparing data to expected values
    pub fn new(observed: &[f64], expected: &[f64], errors: &[f64]) -> Self {
        assert_eq!(observed.len(), expected.len());
        assert_eq!(observed.len(), errors.len());
        
        let chi2: f64 = observed.iter()
            .zip(expected.iter())
            .zip(errors.iter())
            .map(|((&obs, &exp), &err)| {
                if err > 0.0 {
                    ((obs - exp) / err).powi(2)
                } else {
                    0.0
                }
            })
            .sum();
        
        let dof = observed.len().saturating_sub(1);
        let p_value = Self::chi2_p_value(chi2, dof);
        
        Self { chi2, dof, p_value }
    }
    
    /// Approximate p-value using Wilson-Hilferty transformation
    fn chi2_p_value(chi2: f64, dof: usize) -> f64 {
        if dof == 0 {
            return 1.0;
        }
        
        let k = dof as f64;
        
        // Wilson-Hilferty transformation for large dof
        if dof > 30 {
            let z = ((chi2 / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) 
                / (2.0 / (9.0 * k)).sqrt();
            return 0.5 * (1.0 + erf(-z / std::f64::consts::SQRT_2));
        }
        
        // For small dof, use incomplete gamma function approximation
        // This is a simple approximation; for production use a proper implementation
        let x = chi2 / 2.0;
        let a = k / 2.0;
        
        // Regularized incomplete gamma function P(a,x) approximation
        if x < a + 1.0 {
            // Series representation
            let mut sum = 1.0 / a;
            let mut term = 1.0 / a;
            for n in 1..100 {
                term *= x / (a + n as f64);
                sum += term;
                if term < 1e-10 * sum {
                    break;
                }
            }
            1.0 - sum * (-x + a * x.ln() - ln_gamma(a)).exp()
        } else {
            // Continued fraction (simplified)
            0.5 * (1.0 + erf(-(chi2 - k).abs() / (2.0 * k).sqrt()))
        }
    }
    
    pub fn chi2(&self) -> f64 { self.chi2 }
    pub fn dof(&self) -> usize { self.dof }
    pub fn p_value(&self) -> f64 { self.p_value }
    pub fn chi2_per_dof(&self) -> f64 { 
        if self.dof > 0 { self.chi2 / self.dof as f64 } else { 0.0 }
    }
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    
    sign * y
}

/// Log gamma function approximation
fn ln_gamma(x: f64) -> f64 {
    // Stirling's approximation
    if x > 12.0 {
        let inv_x = 1.0 / x;
        let inv_x2 = inv_x * inv_x;
        x * x.ln() - x + 0.5 * (2.0 * PI * x).ln() 
            + inv_x / 12.0 - inv_x2 * inv_x / 360.0
    } else {
        // For small x, use recursion and table
        let mut z = x;
        let mut result = 0.0;
        while z < 12.0 {
            result -= z.ln();
            z += 1.0;
        }
        result + ln_gamma(z)
    }
}

/// Systematic error estimation from finite-size effects
pub struct FiniteSizeError {
    /// System size
    n: usize,
    /// Critical exponent ν (correlation length exponent)
    nu: f64,
    /// Scaling dimension of the observable
    scaling_dim: f64,
}

impl FiniteSizeError {
    pub fn new(n: usize, nu: f64, scaling_dim: f64) -> Self {
        Self { n, nu, scaling_dim }
    }
    
    /// Estimate relative systematic error from finite-size effects
    /// Based on scaling: O(N^{-dimension/ν})
    pub fn relative_error(&self) -> f64 {
        let n = self.n as f64;
        n.powf(-self.scaling_dim / self.nu)
    }
    
    /// Estimate absolute systematic error given the observable value
    pub fn absolute_error(&self, observable_value: f64) -> f64 {
        observable_value.abs() * self.relative_error()
    }
}

/// Complete error budget for an observable
#[derive(Debug, Clone)]
pub struct ErrorBudget {
    pub name: String,
    pub value: f64,
    pub stat_error: f64,
    pub jack_error: f64,
    pub systematic_error: f64,
    pub tau_int: f64,
    pub n_eff: f64,
    pub total_error: f64,
}

impl ErrorBudget {
    pub fn new(
        name: String,
        value: f64,
        error_analysis: &ErrorAnalysis,
        finite_size_error: Option<f64>,
    ) -> Self {
        let errors = error_analysis.errors();
        let systematic_error = finite_size_error.unwrap_or(0.0);
        
        // Total error: add statistical and systematic in quadrature
        let total_error = (errors.stat_error.powi(2) + systematic_error.powi(2)).sqrt();
        
        Self {
            name,
            value,
            stat_error: errors.stat_error,
            jack_error: errors.jack_error,
            systematic_error,
            tau_int: errors.tau_int,
            n_eff: errors.n_eff,
            total_error,
        }
    }
    
    /// Format as a row for the error budget table
    pub fn format_row(&self) -> String {
        format!(
            "{:<20} {:>12.6} ± {:>10.6} {:>10.6} {:>10.6} {:>8.2} {:>8.0} {:>10.6}",
            self.name,
            self.value,
            self.stat_error,
            self.jack_error,
            self.systematic_error,
            self.tau_int,
            self.n_eff,
            self.total_error
        )
    }
}

/// Print error budget table
pub fn print_error_budget_table(budgets: &[ErrorBudget]) {
    println!("\n{:=<110}", "");
    println!("ERROR BUDGET TABLE");
    println!("{:=<110}", "");
    println!(
        "{:<20} {:>12} {:>10} {:>10} {:>10} {:>8} {:>8} {:>10}",
        "Observable", "Value", "Stat Err", "Jack Err", "Sys Err", "τ_int", "N_eff", "Total Err"
    );
    println!("{:-<110}", "");
    
    for budget in budgets {
        println!("{}", budget.format_row());
    }
    
    println!("{:=<110}", "");
    println!("τ_int: Integrated autocorrelation time");
    println!("N_eff: Effective sample size = N_samples / (2τ)");
    println!("Total error: √(stat² + sys²)");
}