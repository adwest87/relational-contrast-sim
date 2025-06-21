# Physics from the Relational Contrast Framework Paper

**Source**: papers/relational_contrast_framework/relational_contrast_framework.tex  
**Date Extracted**: 2025-06-21

## Core Axioms (Section 2)

### Axiom 1: Relational Weight
- Fundamental data: (V, {w_ij}, {G_ij})
- V = countable set of nodes
- w_ij = w_ji ∈ (0,1] (symmetric positive weights)
- G_ij = G_ji^{-1} ∈ SU(3)×SU(2)×U(1)_Y (gauge group elements)

### Axiom 2: Dougal Invariance
- Simultaneous rescaling w_ij → λw_ij and Δt → λΔt leaves physics invariant
- All couplings (α,β,γ) are dimensionless

### Axiom 3: Minimal-Weight Quantum
- w_min = min_{i<j} w_ij > 0
- A link with minimal weight contributes exactly one quantum of action: S_link(w_min) = h

## The Action (Section 3, Equation 117-122)

```
S[R] = α Σ_△ Re tr(R_ij R_jk R_ki) + β Σ_{i<j} w_ij ln w_ij + γ Σ_{n≤n_cut} (λ_n - λ̄)²
```

Where:
- R_ij = w_ij G_ij
- First term: seeds curvature and gauge holonomy (triangle term)
- Second term: **log-convex entropy ensuring well-defined Monte Carlo measure**
- Third term: spectral self-consistency (matter back-reacts on geometry)

**CRITICAL**: The entropy term is `w_ij ln w_ij` which is **POSITIVE** (since ln w < 0 for w < 1).

## Numerical Implementation (Appendix B)

From the Python reference code (lines 439):
```python
grad = attr['w'] * np.log(attr['w'])  # d/dw w ln w term
```

This confirms the entropy term is w*ln(w).

## Key Physics Points

1. **NOT a spin system** - No magnetization mentioned anywhere
2. **Gauge fields**: G_ij are SU(3)×SU(2)×U(1) matrices, not just U(1) phases
3. **Spectral properties central**: The third term in action involves eigenvalues
4. **Goal**: Emergent 4D spacetime and Standard Model gauge fields
5. **Observables** (Section 7.4):
   - Spectral dimension
   - Wilson triangles  
   - Laplacian gap
   - NO MAGNETIZATION!

## Hierarchy of Bosonic Bits (Section 4)

- 3^d contrast bits per link in dimension d
- After projections: 3:8:20 hierarchy
- 20 bits → 2 graviton polarizations in continuum

## What Our Code Should Implement

1. **Action with three terms**: Triangle + Entropy + Spectral
2. **Entropy**: w*ln(w) NOT -z*exp(-z) (though mathematically equivalent)
3. **Observables**: Spectral gap, NOT magnetization
4. **Gauge fields**: Full SU(3)×SU(2)×U(1), not just U(1) phases
5. **Focus**: Emergent geometry, not phase transitions

## Current Implementation Status

Our graph.rs has:
- ✓ Correct entropy (though written as -z*exp(-z) which is equivalent)
- ✓ Triangle term
- ✗ Missing spectral term in action
- ✗ Wrong observables (magnetization instead of spectral)
- ✗ Simplified to U(1) only (acceptable for testing)
- ✗ Wrong interpretation (magnetic system vs emergent geometry)

## Conclusion

The paper is crystal clear:
1. This is about emergent spacetime from relational networks
2. The entropy term w*ln(w) drives disorder (positive for Monte Carlo)
3. No magnetic physics whatsoever
4. Spectral properties determine spacetime dimensionality
5. Our implementation has the right math but wrong physics interpretation