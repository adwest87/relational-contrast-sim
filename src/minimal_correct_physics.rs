// Minimal correct implementation of Relational Contrast Framework physics
// No optimization, no Monte Carlo, just the correct action calculation

/// Minimal 3-node complete graph for testing correct physics
pub struct MinimalGraph {
    // Weights w_ij = exp(-z_ij)
    z_12: f64,
    z_13: f64,
    z_23: f64,
    
    // U(1) phases (antisymmetric: θ_ji = -θ_ij)
    theta_12: f64,
    theta_13: f64,
    theta_23: f64,
}

impl MinimalGraph {
    /// Create a new 3-node graph with specified z-values and phases
    pub fn new(z_12: f64, z_13: f64, z_23: f64, 
               theta_12: f64, theta_13: f64, theta_23: f64) -> Self {
        Self {
            z_12,
            z_13,
            z_23,
            theta_12,
            theta_13,
            theta_23,
        }
    }
    
    /// Create with all weights = 0.5 and all phases = 0
    pub fn symmetric_half() -> Self {
        // w = 0.5 means z = -ln(0.5) = ln(2) ≈ 0.693147
        let z = -0.5_f64.ln();
        Self::new(z, z, z, 0.0, 0.0, 0.0)
    }
    
    /// Get weight from z-value: w = exp(-z)
    fn w(z: f64) -> f64 {
        (-z).exp()
    }
    
    /// CORRECT entropy term: S_entropy = Σ w_ij * ln(w_ij)
    /// For w < 1, ln(w) < 0, so this is negative
    pub fn entropy_action(&self) -> f64 {
        let w_12 = Self::w(self.z_12);
        let w_13 = Self::w(self.z_13);
        let w_23 = Self::w(self.z_23);
        
        let s_12 = w_12 * w_12.ln();
        let s_13 = w_13 * w_13.ln();
        let s_23 = w_23 * w_23.ln();
        
        s_12 + s_13 + s_23
    }
    
    /// Triangle term: cos(θ_12 + θ_23 + θ_31)
    /// Note: θ_31 = -θ_13 due to antisymmetry
    pub fn triangle_sum(&self) -> f64 {
        let theta_sum = self.theta_12 + self.theta_23 + (-self.theta_13);
        theta_sum.cos()
    }
    
    /// Total action: S = β * S_entropy + α * S_triangle
    pub fn action(&self, alpha: f64, beta: f64) -> f64 {
        beta * self.entropy_action() + alpha * self.triangle_sum()
    }
    
    /// Print detailed calculation for verification
    pub fn print_calculation(&self, alpha: f64, beta: f64) {
        println!("=== Minimal Graph Physics Calculation ===\n");
        
        // Weights
        let w_12 = Self::w(self.z_12);
        let w_13 = Self::w(self.z_13);
        let w_23 = Self::w(self.z_23);
        
        println!("Weights:");
        println!("  z_12 = {:.6}, w_12 = exp(-z_12) = {:.6}", self.z_12, w_12);
        println!("  z_13 = {:.6}, w_13 = exp(-z_13) = {:.6}", self.z_13, w_13);
        println!("  z_23 = {:.6}, w_23 = exp(-z_23) = {:.6}", self.z_23, w_23);
        
        // Entropy contributions
        println!("\nEntropy contributions (w * ln(w)):");
        println!("  w_12 * ln(w_12) = {:.6} * {:.6} = {:.6}", 
                 w_12, w_12.ln(), w_12 * w_12.ln());
        println!("  w_13 * ln(w_13) = {:.6} * {:.6} = {:.6}", 
                 w_13, w_13.ln(), w_13 * w_13.ln());
        println!("  w_23 * ln(w_23) = {:.6} * {:.6} = {:.6}", 
                 w_23, w_23.ln(), w_23 * w_23.ln());
        
        let entropy = self.entropy_action();
        println!("  Total entropy = {:.6}", entropy);
        
        // Triangle term
        println!("\nTriangle term:");
        println!("  θ_12 = {:.6}", self.theta_12);
        println!("  θ_23 = {:.6}", self.theta_23);
        println!("  θ_31 = -θ_13 = {:.6}", -self.theta_13);
        let theta_sum = self.theta_12 + self.theta_23 + (-self.theta_13);
        println!("  Sum = {:.6}", theta_sum);
        println!("  cos(sum) = {:.6}", theta_sum.cos());
        
        // Total action
        println!("\nAction calculation:");
        println!("  S = β * entropy + α * triangle");
        println!("  S = {:.2} * {:.6} + {:.2} * {:.6}", 
                 beta, entropy, alpha, self.triangle_sum());
        println!("  S = {:.6} + {:.6}", 
                 beta * entropy, alpha * self.triangle_sum());
        println!("  S = {:.6}", self.action(alpha, beta));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_symmetric_half_calculation() {
        // Test case: all weights = 0.5, all phases = 0
        let graph = MinimalGraph::symmetric_half();
        
        // Hand calculation:
        // z = -ln(0.5) = ln(2) ≈ 0.693147
        // w = 0.5
        // w * ln(w) = 0.5 * ln(0.5) = 0.5 * (-0.693147) = -0.346574
        // Total entropy = 3 * (-0.346574) = -1.039721
        // Triangle = cos(0 + 0 + 0) = 1
        
        let entropy = graph.entropy_action();
        assert!((entropy - (-1.039721)).abs() < 1e-5);
        
        let triangle = graph.triangle_sum();
        assert!((triangle - 1.0).abs() < 1e-10);
        
        // With α = 1, β = 1:
        // S = 1 * (-1.039721) + 1 * 1 = -0.039721
        let action = graph.action(1.0, 1.0);
        assert!((action - (-0.039721)).abs() < 1e-5);
    }
    
    #[test]
    fn test_entropy_negativity() {
        // For any w < 1, entropy should be negative
        let graph = MinimalGraph::new(0.1, 0.5, 1.0, 0.0, 0.0, 0.0);
        assert!(graph.entropy_action() < 0.0);
    }
}

/// Example usage showing the calculation
pub fn demonstrate_calculation() {
    let graph = MinimalGraph::symmetric_half();
    graph.print_calculation(1.0, 1.0);
    
    println!("\n=== Verification ===");
    println!("Expected entropy: 3 * 0.5 * ln(0.5) = -1.039721...");
    println!("Expected triangle: cos(0) = 1");
    println!("Expected action (α=β=1): -1.039721 + 1 = -0.039721");
}