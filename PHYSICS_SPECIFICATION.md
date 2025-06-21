# PHYSICS SPECIFICATION - Relational Contrast Framework

**Version**: 2.0  
**Date**: 2025-06-21  
**Status**: CORRECTED FROM PAPER - FUNDAMENTAL REVISION

## Model Definition - From Relational Contrast Framework Paper

### 1. The Action (NOT Hamiltonian)

The system is defined by the following action:

```
S = β ∑_{⟨ij⟩} w_ij ln(w_ij) + α ∑_{⟨ijk⟩} cos(holonomy around triangle)
```

Where:
- `w_ij ∈ (0,1]` are the fundamental relational weights on edges
- For numerical stability, we use `z_ij = -ln(w_ij)`, so `w_ij = exp(-z_ij)`
- The entropy term is `w ln(w)` which is **negative** for w < 1
- In the simplified U(1) model without full gauge fields: holonomy = θ_ij + θ_jk + θ_ki
- `α` controls geometric frustration (triangle term)
- `β` controls entropic effects (weight term)

**CRITICAL**: This is NOT a Hamiltonian! The entropy term has opposite sign from typical statistical mechanics.

### 2. Degrees of Freedom

#### Primary Variables: Relational Weights w_ij
- **Definition**: Relational connection strengths between entities
- **Domain**: w_ij ∈ (0, 1] (weights cannot exceed unity)
- **Numerical representation**: z_ij = -ln(w_ij) ∈ [0, ∞)
- **Physical meaning**: Strength of relational connection in emergent spacetime
- **Dougal invariance**: Physics invariant under w_ij → λw_ij (global rescaling)

#### Secondary Variables: Gauge Fields/Phases
In the full model:
- **Gauge fields**: G_ij ∈ SU(3) × SU(2) × U(1) (Standard Model gauge group)
- **Simplified model**: θ_ij ∈ [0, 2π) (U(1) phases for testing)
- **Physical meaning**: Gauge connections that will become Standard Model fields

### 3. Constraints

#### Weight Symmetry
```
w_ji = w_ij  (undirected graph)
z_ji = z_ij
```

#### Phase Antisymmetry (for U(1) simplified model)
```
θ_ji = -θ_ij
```

#### Dougal Invariance
The physics must be invariant under:
```
w_ij → λ w_ij  for all edges
dt → λ dt
```
This is a fundamental symmetry of the relational framework.

### 4. Observables

#### 4.1 Order Parameters

**Weight Order Parameter**:
```
W = (1/N_edges) ∑_{⟨ij⟩} w_ij
```

**Triangle Frustration**:
```
F = (1/N_triangles) ∑_{⟨ijk⟩} cos(θ_ij + θ_jk + θ_ki)
```

#### 4.2 Emergent Geometry Observables

**Spectral Gap** (key for emergent dimensionality):
```
Δλ = λ_2 - λ_1
```
Where λ_i are eigenvalues of the weighted Laplacian L_ij = δ_ij ∑_k w_ik - w_ij

**Effective Dimension**:
```
d_eff = -2 ln(N) / ln(Δλ)
```

#### 4.3 NO Traditional Magnetization!

**CRITICAL**: There is NO magnetization in this model! Previous implementations incorrectly imported spin system concepts. The phases θ_ij are gauge fields, not spins.

### 5. Physical Interpretation - Emergent Spacetime and Gauge Fields

#### What This Model Actually Represents

1. **NOT a spin system or spin liquid** - Previous interpretations were completely wrong
2. **Relational weights w_ij** represent connection strengths in pre-geometric phase
3. **Entropy term** drives weights toward disorder (small w_ij)
4. **Triangle term** creates geometric frustration, organizing into emergent geometry
5. **Competition** between entropy and geometry leads to critical behavior
6. **Emergent spacetime** arises from spectral properties of weight matrix
7. **Gauge fields** (in full model) give rise to Standard Model interactions

#### Expected Phases

1. **Pre-geometric phase** (large β): Disordered weights, no emergent geometry
2. **Geometric phase** (small β): Organized weights forming emergent spacetime
3. **Critical ridge**: Where 4D spacetime and gauge fields emerge

#### This is NOT About:
- Magnetic ordering
- Spin liquids
- Conventional phase transitions
- Condensed matter physics

### 6. Key Differences from Previous Implementation

1. **Entropy sign**: Should be `w ln(w)` (negative for w<1), NOT `-z*w`
2. **No magnetization**: Phases are gauge fields, not spins
3. **Spectral properties**: Central to physics, not optional
4. **Dougal invariance**: Must be maintained
5. **Interpretation**: Emergent geometry, not magnetism

## Implementation Requirements

### Correct Action Implementation

```rust
// CORRECT entropy term
pub fn entropy_action(&self) -> f64 {
    self.links.iter().map(|l| {
        let w = l.w();  // exp(-z)
        w * w.ln()      // w ln(w), negative for w < 1
    }).sum()
}

// CORRECT total action
pub fn action(&self, alpha: f64, beta: f64) -> f64 {
    beta * self.entropy_action() + alpha * self.triangle_sum()
}
```

### Initialization
1. Initialize z_ij uniformly in [0, 5] (corresponding to w_ij ∈ [exp(-5), 1])
2. Initialize θ_ij uniformly in [0, 2π) for U(1) model
3. Enforce constraints

### Critical: What to Measure
1. **Spectral gap** and eigenvalue distribution
2. **Weight statistics** (mean, variance, distribution)
3. **Triangle frustration**
4. **Effective dimension** from spectral properties
5. **NOT magnetization or susceptibility**

## Current Implementation Assessment

### graph.rs - INCORRECT but Closest
- Uses z-variables ✓
- Has Dougal rescaling ✓
- **WRONG**: entropy_action computes `-z*w` instead of `w*ln(w)`
- **WRONG**: Includes magnetization concepts
- **FIX NEEDED**: Change one line in entropy_action

### Other Implementations - Fundamentally Wrong
- All include magnetization/susceptibility
- None understand the relational physics
- Optimizations can be salvaged after physics correction

## Action Plan

1. **Fix graph.rs entropy term** - One line change
2. **Remove all magnetization code** - Not physical
3. **Add spectral gap calculations** - Essential observable  
4. **Update analysis tools** - Focus on geometry emergence
5. **Archive other implementations** - Keep optimizations for later

## Validation

The model should show:
1. **Dougal invariance**: Rescaling w_ij doesn't change physics
2. **Entropy dominance** at large β: weights → 0
3. **Geometric organization** at small β: non-trivial spectral gap
4. **NO magnetic behavior**: This isn't a magnet!

---

**CRITICAL**: We have been solving the wrong physics. This is about emergent spacetime from relational networks, NOT magnetic systems. All "spin liquid" terminology and magnetic observables must be removed.