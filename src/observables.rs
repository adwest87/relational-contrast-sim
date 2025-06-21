// Minimal observables for compatibility with existing code
// Most functionality has been integrated into graph implementations

#[derive(Debug, Clone, Default)]
pub struct Observables {
    pub mean_w: f64,
    pub mean_cos: f64,
    pub entropy: f64,
    pub triangle_sum: f64,
    pub susceptibility: f64,
    pub specific_heat: f64,
    pub binder_cumulant: f64,
    pub magnetization: f64,
    pub link_variance: f64,
    pub correlation_length: f64,
    pub spectral_gap: f64,
}