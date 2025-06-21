# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 📖 CRITICAL: READ THE PHYSICS FIRST 📖

**BEFORE analyzing any code or making changes, ALWAYS read the physics papers first:**

```bash
# Essential reading to understand the physics model
papers/relational_contrast_framework/relational_contrast_framework.tex
papers/relational_contrast_framework/relational_contrast_framework.pdf
```

**This is NOT a condensed matter or spin system - it's a quantum gravity model for emergent spacetime!**

Without reading the papers, you will misinterpret the physics and implement the wrong observables.

## 🚨 CRITICAL DEVELOPMENT PRINCIPLE 🚨

**ALWAYS MODIFY EXISTING CODE RATHER THAN CREATING NEW FILES**

When implementing features or fixes:
1. **First** search for existing implementations that can be extended
2. **Modify** existing files to add functionality rather than creating duplicates
3. **Refactor** existing code instead of writing parallel implementations
4. **Only** create new files when absolutely necessary for architectural reasons

This prevents:
- Code duplication and bloat
- Multiple implementations of the same functionality
- Confusion about which version to use
- Maintenance overhead from redundant files

## ✅ PHYSICS MODEL: Relational Contrast Framework ✅

**What This Code Actually Models** (updated from rc_update.tex):
- **Emergent 4D spacetime** from weighted complete graphs with bounded relational weights w_ij ∈ (0,1]
- **Background-independent quantum gravity** where geometry emerges from relationship patterns
- **Spectral dimension d_s ≈ 4** achieved through natural scaling laws, not fine-tuning
- **Complete connectivity requirement** for unique geometric emergence
- **Discrete-to-continuum bridge** via explicit coarse-graining procedures

**This is NOT**:
- ❌ A magnetic system
- ❌ A spin model 
- ❌ Condensed matter physics
- ❌ Traditional statistical mechanics

**Core Philosophy** (3 fundamental principles):
1. **No background**: Spacetime emerges from more primitive relations
2. **Relational weights**: Only fundamental quantities are w_ij ∈ (0,1] relationship strengths  
3. **Bounded interactions**: Unit strength maximum provides natural UV cutoff

**The Action** (updated formulation):
```
S[w] = α S_entropy + β S_triangle + γ S_spectral

Where:
S_entropy = -Σ w_ij ln(w_ij)           (relational entropy, spreads weights)
S_triangle = Σ [-ln(w_ij w_jk w_ki) + ln(w_ij + w_jk + w_ki)]  (discrete curvature)
S_spectral = Σ (λ_n - λ_target(n))²   (enforces d-dimensional spectrum)
```

**Key Insight**: "Relational contrast" = varying relationship strengths encode both connectivity and emergent distance

## Build and Development Commands

### Basic Operations
```bash
# Build the project (always use --release for performance)
cargo build --release

# Run tests
cargo test

# Run clippy linter
cargo clippy

# Format code
cargo fmt

# Check for compilation errors without building
cargo check
```

### Key Binaries
```bash
# Spectral analysis (correct physics for emergent spacetime)
cargo run --release --bin test_spectral_properties
cargo run --release --bin test_complete_graph_spectrum
cargo run --release --bin test_spectral_gap_methods
cargo run --release --bin test_erdos_renyi_alternative

# Physics validation and verification
cargo run --release --bin comprehensive_physics_validation
cargo run --release --bin verify_entropy_implementations

# Current analysis tools (WARNING: Some still use wrong observables!)
cargo run --release --bin wide_scan              # FIXME: Remove magnetic observables
cargo run --release --bin critical_finder        # FIXME: Remove magnetic observables

# Performance benchmarking
cargo run --release --bin benchmark_m1
cargo run --release --bin benchmark_metal
cargo run --release --bin benchmark_ultra_optimized
```

### Performance Testing
```bash
# Benchmark M1 optimizations
cargo run --release --bin benchmark_m1

# Test Metal GPU acceleration (macOS only)
cargo run --release --bin benchmark_metal
```

## Architecture Overview

### Relational Contrast Framework Physics
The system models **emergent 4D spacetime and Standard Model gauge fields**:

**Core Variables**:
- **Relational weights**: w_ij ∈ (0,1] encode spacetime distances
- **Gauge matrices**: G_ij ∈ SU(3)×SU(2)×U(1) on graph edges
- **Holonomies**: θ_ij phases around triangles (simplified U(1) model)

**Key Observables** (NOT magnetic quantities):
- **Spectral dimension**: d_s = -2 lim_{t→0} d ln Z(t)/d ln t where Z(t) = Tr[e^{-tL}]
- **Spectral gap**: Δλ = λ₂ - λ₁ of weighted graph Laplacian
- **Effective distance**: d_ij² = -ξ² ln(w_ij) (weight-distance relation)
- **Local metric tensor**: g_ab(i) = Σ w_ij w_ik ê_j^a ê_k^b (emergent from weights)
- **Heat kernel trace**: Z(t) = Σ e^{-λ_n t} (characterizes spectral dimension)

**Physical Goals** (updated from rc_update.tex):
- **Natural 4D emergence**: d_s ≈ 4 through scaling law w ~ N^{-3/2} (not fine-tuning)
- **Unique geometry**: Complete graphs provide necessary constraints for unambiguous emergence
- **Einstein-Hilbert recovery**: Discrete action → ∫ √g R in continuum limit  
- **Explicit coarse-graining**: Rigorous discrete → continuous bridge procedure
- **Spectral regularization**: Target spectrum λ_target(n) = c·n^{2/d} enforces dimensionality

### Core Module Structure

1. **Graph Implementations**:
   - `graph.rs` - **CANONICAL** implementation with correct physics
   - `graph_fast.rs` - Optimized version (may have wrong observables)
   - `graph_ultra_optimized.rs` - Ultra-optimized (may have wrong observables)
   - `graph_m1_optimized.rs` - Apple Silicon NEON SIMD 
   - `graph_metal.rs` - GPU acceleration via Metal (macOS)
   
   **STATUS**: Use `graph.rs` as reference. Others need validation for correct physics.

2. **Monte Carlo Integration**:
   - `fast_mc_integration.rs` - High-performance MC runner
   - `importance_mc_integration.rs` - Importance sampling on critical ridge
   - Both use adaptive measurement intervals based on autocorrelation

3. **Utility Modules** (`src/utils/`):
   - `config.rs` - Configuration structures
   - `output.rs` - Output formatting
   - `ridge.rs` - Ridge-related utilities
   - `rng.rs` - Random number generation

4. **Key Algorithms**:
   - **Metropolis updates**: Separate z-updates (weights) and θ-updates (phases)
   - **Triangle sums**: O(N) incremental updates instead of O(N³)
   - **Spectral calculations**: Eigenvalues of weighted Laplacian for spacetime
   - **Entropy term**: w*ln(w) = -z*exp(-z) where w = exp(-z)

### Current Implementation Status

**✅ SPECTRAL METHODS IMPLEMENTED (2025-06-21)**:
- `graph.spectral_gap()` - Computes λ₂ - λ₁ (fundamental spacetime observable)
- `graph.effective_dimension()` - Computes d_eff = -2ln(N)/ln(Δλ)
- `graph.laplacian_eigenvalues()` - Full eigenvalue spectrum
- `graph.spectral_action()` - Regularization term γ Σ (λ_n - λ̄)²

**❌ STILL USING WRONG OBSERVABLES**:
Many analysis tools still compute magnetic quantities that are meaningless for this physics:
- Magnetization (should be removed)
- Susceptibility (should be removed)  
- Specific heat (should be removed)
- Binder cumulants (should be removed)

**🎯 TARGET OBSERVABLES FOR SPACETIME**:
- Spectral gap Δλ
- Effective dimension d_eff
- Spectral density ρ(λ)
- Clustering coefficients
- Distance correlations
- Geometric scaling exponents

### Performance Considerations

- Always compile with `--release` (150x faster than debug)
- Spectral calculations are O(N³) - most expensive part
- Metal GPU gives 100x speedup for large systems
- Cache optimization critical for CPU performance
- Target: <0.1s per MC step for N=1000

### Data Flow
```
src/bin/* → results/data/*.csv → analysis/scripts/* → analysis/figures/*
                     ↓
          Spectral gap data → 4D emergence analysis
```

### Project Organization
```
src/
├── graph.rs (canonical implementation)
├── mc runners (fast_mc_integration.rs, etc.)  
├── utils/ (config, output, ridge, rng)
└── bin/ (analysis tools - many need physics updates)

papers/
└── relational_contrast_framework/ (ESSENTIAL READING)

results/
├── data/ (all CSV files - gitignored)
└── figures/ (all generated plots - gitignored)

docs/
├── scripts/ (maintenance scripts)
├── patches/ (patch files)
└── guides/ (documentation)
```

## Critical Physics Understanding

### ✅ ENTROPY IMPLEMENTATION IS CORRECT
**Previously thought wrong, now verified correct**:
- Implementation uses w*ln(w) = -z*exp(-z) where w = exp(-z)
- This is mathematically identical: exp(-z) * ln(exp(-z)) = exp(-z) * (-z) = -z*exp(-z)
- The entropy term provides the Monte Carlo measure, not thermodynamics
- No "entropy error" exists - this was a false alarm

### ✅ SPECTRAL GAP IS THE KEY OBSERVABLE  
**For emergent spacetime physics**:
- Spectral gap Δλ = λ₂ - λ₁ determines effective dimension
- Target: d_eff = -2ln(N)/ln(Δλ) ≈ 4 for realistic spacetime
- Complete graphs with uniform weights: λ₂ = Nw (degenerate spectrum)
- Need weight variation or non-complete topologies for rich spectral structure

### ❌ REMOVE ALL MAGNETIC PHYSICS REFERENCES
**The following concepts are NOT relevant to this model**:
- Magnetization (no magnetic moments exist)
- Susceptibility (no magnetic response)
- Spin correlations (no spins exist)
- Phase transitions in magnetic sense
- Ising model analogies
- "Quantum spin liquid" claims (wrong physics interpretation)

## Common Pitfalls

- Debug builds are unusably slow - always use `--release`
- Spectral calculations are expensive - consider approximations for large N
- Finite-size effects significant for small systems (N < 50)
- Complete graphs may be too symmetric for realistic spacetime
- Many analysis tools still compute wrong (magnetic) observables
- **CRITICAL**: Don't interpret this as condensed matter physics

### Key Files to Understand

**Essential Physics**:
- `papers/relational_contrast_framework/relational_contrast_framework.tex` - Original comprehensive physics
- `papers/rc_update/rc_update.tex` - **UPDATED RIGOROUS FRAMEWORK** with scaling laws and geometry emergence
- `RIGOROUS_FOUNDATIONS.md` - Mathematical foundations with proven vs conjectural claims
- `src/graph.rs` - Canonical implementation with correct observables
- `src/bin/test_spectral_gap_methods.rs` - Test correct physics observables

**Analysis Tools** (need updates):
- `src/bin/comprehensive_physics_validation.rs` - Physics verification  
- `src/bin/verify_entropy_implementations.rs` - Proves entropy is correct
- `src/fast_mc_integration.rs` - Production MC runner
- `CLAUDE.md` - This file with development principles

**Spectral Analysis**:
- `src/bin/test_complete_graph_spectrum.rs` - Mathematical verification
- `COMPLETE_GRAPH_SPECTRAL_ANALYSIS.md` - Detailed mathematical analysis
- `SPECTRAL_ANALYSIS_CONCLUSIONS.md` - Key insights on graph topology

## Recent Major Progress (2025-06-21)

### ✅ PHYSICS FRAMEWORK MATURED - RC_UPDATE.TEX
**NEW RIGOROUS FRAMEWORK**: Complete mathematical treatment in `papers/rc_update/rc_update.tex`
**KEY BREAKTHROUGHS**:
- **Scaling law not fine-tuning**: w ~ N^{-3/2} is natural scaling for 4D, not arbitrary requirement
- **Complete graphs proven necessary**: Provide unique constraints for geometry emergence 
- **Explicit coarse-graining**: Rigorous discrete → continuous procedure with metric tensor emergence
- **Action principle derived**: Physical entropy + geometric + spectral terms → Einstein-Hilbert limit

### ✅ PHYSICS CRISIS RESOLVED  
**ROOT PROBLEM**: Previous work interpreted this as a magnetic/spin system
**SOLUTION**: Read the actual physics papers - this is emergent spacetime!
**RESULT**: Now implementing correct observables (spectral dimension, not magnetization)

### ✅ SPECTRAL GAP METHODS ADDED TO graph.rs
**New methods implemented**:
- `spectral_gap()` - Returns λ₂ - λ₁ 
- `effective_dimension()` - Returns d_eff = -2ln(N)/ln(Δλ)
- Integration with existing `laplacian_eigenvalues()`

**Verification results**:
- Uniform complete graph: Perfect theoretical agreement (gap = Nw)
- 4D emergence: N=20, w=N^(-3/2) gives d_eff = 4.000 exactly
- Mathematical consistency: All tests pass

### ✅ ENTROPY "ERROR" WAS FALSE ALARM  
**Previous concern**: Different entropy formulas between implementations
**Reality**: w*ln(w) = -z*exp(-z) mathematically (proven by verification tools)
**Status**: All entropy implementations are correct

## Next Steps for Complete Implementation

### Immediate Priority (Week 1-2) - **UPDATED from rc_update.tex**
1. **Implement heat kernel spectral dimension**:
   - Add d_s = -2 lim_{t→0} d ln Z(t)/d ln t calculation to graph.rs  
   - Replace effective dimension d_eff with proper spectral dimension d_s
   - Test heat kernel trace Z(t) = Σ e^{-λ_n t} convergence

2. **Update action formulation**:
   - Implement corrected entropy S_entropy = -Σ w_ij ln(w_ij) (negative sign)
   - Add triangle curvature S_triangle = Σ [-ln(w_ij w_jk w_ki) + ln(w_ij + w_jk + w_ki)]
   - Implement spectral targeting S_spectral = Σ (λ_n - c·n^{2/4})² for 4D spectrum

3. **Remove magnetic observables** from analysis tools:
   - Delete magnetization calculations from bin/* tools  
   - Replace susceptibility with spectral dimension measurements
   - Update output CSV headers to remove magnetic quantities

### Medium Term (Month 2-3) - **UPDATED from rc_update.tex**
1. **Implement coarse-graining procedures**:
   - Add local metric tensor calculation g_ab(i) = Σ w_ij w_ik ê_j^a ê_k^b
   - Implement self-consistent embedding Σ_j w_ij (x_j - x_i) = 0
   - Test discrete → continuous field φ(x) = Σ_i K_σ(x - x_i) φ_i / Σ_i K_σ(x - x_i)

2. **Validate scaling law theory**:
   - Verify w ~ N^{-3/2} gives natural d_s ≈ 4 (not fine-tuning)
   - Test latent 4D point embedding → weight generation → dimension recovery
   - Compare complete vs sparse graphs for geometric emergence quality

3. **Extended gauge structure** (optional):
   - SU(3)×SU(2)×U(1) matrices on edges
   - Full Standard Model implementation  
   - Memory scaling challenges

## Success Metrics - **UPDATED from rc_update.tex**

1. **Correct observables**: All tools measure spectral dimension d_s, not magnetic quantities
2. **Natural 4D emergence**: Verify d_s ≈ 4 through scaling law w ~ N^{-3/2} without fine-tuning  
3. **Geometric consistency**: Local metric tensor emergence from weights with unique geometry
4. **Coarse-graining validation**: Successful discrete → continuous bridge with Einstein-Hilbert recovery
5. **Performance**: Heat kernel calculations <0.1s for N=1000 with spectral dimension measurement
6. **Complete graph necessity**: Demonstrate superior dimensional emergence vs sparse graphs

## Development Best Practices

### Avoid Code Duplication
1. **Search before creating**: Use grep/find to locate existing implementations
2. **Extend don't duplicate**: Add features to existing files
3. **Refactor don't rewrite**: Improve existing code rather than starting fresh
4. **One implementation per feature**: Don't create multiple versions

### When Adding Features
1. Check if similar functionality exists in:
   - `src/bin/` - Look for related analysis tools
   - `src/` - Check core implementations  
   - `analysis/scripts/` - Review analysis scripts
2. Modify the closest existing implementation
3. Only create new files for genuinely new architectural components

### File Naming
- Use descriptive names that indicate purpose
- Avoid versioned names (e.g., `analysis_v2.rs`, `improved_scan.rs`)
- If improving code, update the original file instead of creating variants

This approach keeps the codebase maintainable and prevents proliferation of redundant implementations.

---

**Remember**: This is quantum gravity research, not condensed matter. Read the physics papers first!