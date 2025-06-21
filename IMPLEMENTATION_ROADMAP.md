# Implementation Roadmap - From Current State to Full Model

**Date**: 2025-06-21  
**Purpose**: Practical path forward with honest assessment

## Current State Summary

### What We Have (Correct)
- ✓ Entropy term: w*ln(w) implemented as -z*exp(-z)
- ✓ Triangle term: Simplified to U(1) phases
- ✓ Monte Carlo machinery: Working Metropolis algorithm
- ✓ Dougal rescaling: Implemented correctly

### What We Have (Wrong)
- ✗ Magnetic observables (magnetization, susceptibility)
- ✗ Wrong physics interpretation (spin system)
- ✗ Missing spectral calculations

### What We're Missing
- ✗ Spectral regularization term: γ Σ (λ_n - λ̄)²
- ✗ Full gauge structure: SU(3) × SU(2) × U(1)
- ✗ Minimal-weight quantum axiom implementation
- ✗ Fermions and AIB projector

## Immediate Priority: Fix Physics Interpretation

### Step 1: Add Spectral Gap Calculation (1-2 days)
```rust
use nalgebra::{DMatrix, SymmetricEigen};

impl Graph {
    pub fn weighted_laplacian(&self) -> DMatrix<f64> {
        let n = self.n();
        let mut lap = DMatrix::zeros(n, n);
        
        for link in &self.links {
            let w = link.w();
            lap[(link.i, link.i)] += w;
            lap[(link.j, link.j)] += w;
            lap[(link.i, link.j)] -= w;
            lap[(link.j, link.i)] -= w;
        }
        lap
    }
    
    pub fn spectral_gap(&self) -> Result<f64, &'static str> {
        if self.n() < 2 {
            return Err("Need at least 2 nodes");
        }
        
        let lap = self.weighted_laplacian();
        let eigen = SymmetricEigen::new(lap);
        let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        Ok(eigenvalues[1] - eigenvalues[0])
    }
}
```

### Step 2: Remove Magnetic Observables (1 day)
- Delete magnetization calculations
- Remove susceptibility from all analysis tools
- Update bin/* tools to use spectral observables

### Step 3: Add Basic Spectral Term (2-3 days)
```rust
impl Graph {
    pub fn spectral_regularization(&self, n_cut: usize, lambda_bar: f64) -> f64 {
        let eigenvalues = self.compute_smallest_eigenvalues(n_cut + 1);
        eigenvalues.iter()
            .take(n_cut)
            .map(|&lambda| (lambda - lambda_bar).powi(2))
            .sum()
    }
    
    pub fn action(&self, alpha: f64, beta: f64, gamma: f64) -> f64 {
        let triangle = self.triangle_sum();
        let entropy = self.entropy_action();
        let spectral = self.spectral_regularization(10, 1.0); // TODO: tune
        
        alpha * triangle + beta * entropy + gamma * spectral
    }
}
```

## Medium Term: Computational Optimizations

### Challenge: Spectral Calculations at Each MC Step

For N = 1000:
- Full eigendecomposition: ~1 second
- MC steps needed: ~10^6
- Total time: ~11 days!

### Solution: Approximate Methods

1. **Chebyshev Polynomial Approximation**
```rust
// Approximate tr(f(L)) without computing eigenvalues
pub fn chebyshev_spectral_sum(&self, n_terms: usize) -> f64 {
    // Use Chebyshev polynomials to approximate spectral function
    // O(n_terms * N) instead of O(N³)
}
```

2. **Lanczos Method for Extremal Eigenvalues**
```rust
// Only compute smallest k eigenvalues
pub fn lanczos_eigenvalues(&self, k: usize) -> Vec<f64> {
    // O(k * N * iterations) ~ O(N) for fixed k
}
```

3. **Delayed Updates**
```rust
pub struct SpectralCache {
    eigenvalues: Vec<f64>,
    last_update: usize,
    update_interval: usize, // e.g., 100 steps
}
```

## Long Term: Full Physics Implementation

### Phase 1: Extended Gauge Structure (1 month)

1. **Data Structure**
```rust
pub struct GaugeLink {
    w: f64,           // weight ∈ (0,1]
    su3: Matrix3<Complex<f64>>,   // SU(3) color
    su2: Matrix2<Complex<f64>>,   // SU(2) weak
    u1_phase: f64,    // U(1) hypercharge
}
```

2. **Memory Impact**
- Current: ~16 bytes/link
- Full gauge: ~170 bytes/link
- At N=1000: 85 MB → 850 MB

### Phase 2: Minimal-Weight Quantum (2 weeks)

**Interpretation**: The axiom sets the overall scale, not a dynamic constraint.

```rust
impl Graph {
    pub fn calibrate_action_scale(&mut self) -> f64 {
        let w_min = self.links.iter()
            .map(|l| l.w())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        
        // Set α so that minimal triangle has action 3h
        let h = 1.0; // Planck's constant in natural units
        let minimal_triangle_weight = w_min.powi(3);
        let alpha = 3.0 * h / minimal_triangle_weight;
        
        alpha
    }
}
```

### Phase 3: Fermions and AIB (2 months)
- Complex implementation requiring Grassmann algebra
- RHMC for det(D†D)
- Beyond current scope

## Realistic Timeline

### Week 1-2: Core Fixes
- [x] Understand paper physics
- [ ] Add spectral gap calculation
- [ ] Remove magnetic observables
- [ ] Update analysis tools

### Week 3-4: Spectral Term
- [ ] Implement efficient eigenvalue methods
- [ ] Add spectral regularization to action
- [ ] Test computational scaling

### Month 2: Validation
- [ ] Search for 4D emergence (d_eff ≈ 4)
- [ ] Study phase diagram
- [ ] Document correct physics

### Month 3+: Extensions
- [ ] Full gauge structure (optional)
- [ ] Fermions (very optional)
- [ ] Connect to continuum limit

## Key Insights and Warnings

1. **Computational Reality**: Full spectral calculations at every MC step are prohibitive. Need approximations.

2. **Memory Scaling**: Full gauge structure increases memory by ~10×. May need distributed implementation.

3. **Conceptual Clarity**: Focus on emergent geometry, not phase transitions.

4. **Minimal Viable Product**: U(1) model with spectral gap is sufficient to demonstrate key physics.

## Success Metrics

1. **Correct observables**: Spectral gap, not magnetization
2. **4D emergence**: Find parameters where d_eff ≈ 4
3. **Computational efficiency**: MC step < 0.1 seconds for N=1000
4. **Clean code**: Remove all magnetic concepts

## Final Recommendation

**Start small**: Fix the observables first. The U(1) model with spectral gap calculations is sufficient to demonstrate the core physics of emergent spacetime. Full gauge structure can wait.

**Be realistic**: The computational challenges are severe. Focus on what's achievable with current resources.

**Document everything**: This is novel physics. Clear documentation is essential.

---

*"Perfect is the enemy of good. A working U(1) model with correct physics beats a non-working full gauge implementation."*