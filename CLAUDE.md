# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## ⚠️ CRITICAL IMPLEMENTATION DISCREPANCIES ⚠️

**MAJOR ISSUE**: Different implementations show ~69% discrepancies in key observables:
- **PROBLEM**: graph.rs, graph_fast.rs, and graph_ultra_optimized.rs give different physics
- **PARTIAL UNDERSTANDING**: Some differences due to RNG sequences and initialization
- **UNRESOLVED**: Which implementation (if any) has the correct physics
- **IMPACT**: All physics results are implementation-dependent and uncertain

**STATUS**: Physics results should be treated with extreme caution. The correct physics is unknown.

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
# Run parameter space scan
cargo run --release --bin wide_scan

# Find critical points quickly
cargo run --release --bin critical_finder

# Perform finite-size scaling analysis
cargo run --release --bin fss_analysis

# Physics analysis tools
cargo run --release --bin unconventional_physics
cargo run --release --bin publication_analysis
cargo run --release --bin critical_point_validation

# Comprehensive physics validation across all implementations
cargo run --release --bin comprehensive_physics_validation

# Benchmarking tools
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

### Model Physics
The system studies U(1) phases θᵢⱼ on edges of complete graphs with:
- **Action**: S = α∑(triangles) cos(θᵢⱼ + θⱼₖ + θᵢₖ) - β∑(links) S(wᵢⱼ)
- **Critical behavior**: Ridge structure under investigation
- **Physics**: Classical statistical mechanics on complete graphs

### Core Module Structure

1. **Graph Implementations** (progressive optimization):
   - `graph.rs` - Original implementation ⚠️ SHOWS DIFFERENT PHYSICS
   - `graph_fast.rs` - Cache-optimized implementation ⚠️ ~69% DISCREPANCY vs graph.rs
   - `graph_ultra_optimized.rs` - Ultra-optimized implementation ⚠️ DISCREPANCIES
   - `graph_m1_optimized.rs` - Apple Silicon NEON SIMD (requires validation)
   - `graph_metal.rs` - GPU acceleration via Metal (macOS only)
   
   **WARNING**: Different implementations appear to simulate different physics. The correct physics is unknown.

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
   - **Metropolis updates**: Separate z-updates (weight) and θ-updates (phase)
   - **Triangle sum**: O(N) incremental updates instead of O(N³)
   - **Observables**: Rotated calculation to amortize expensive measurements

### Performance Considerations

- Always compile with `--release` (150x faster than debug)
- Use `FastMCRunner` for production runs
- Metal GPU gives 100x speedup for large systems
- Cache optimization critical for CPU performance

### Data Flow
```
src/bin/* → results/data/*.csv → analysis/scripts/* → analysis/figures/*
                     ↓
              FSS data files → Critical exponents
```

### Project Organization
```
src/
├── core implementations (graph*.rs)
├── mc runners (fast_mc_integration.rs, etc.)
├── utils/ (config, output, ridge, rng)
└── bin/ (41 analysis tools)

results/
├── data/ (all CSV files - gitignored)
└── figures/ (all generated plots - gitignored)

docs/
├── scripts/ (maintenance scripts)
├── patches/ (patch files)
└── guides/ (documentation)
```

### Physics Results Status

**⚠️ CRITICAL WARNING**: Implementation discrepancies mean physics results are uncertain

1. **Critical ridge structure** - Results vary significantly between implementations
2. **Finite-size scaling** - Different implementations give different scaling  
3. **Implementation consistency** - ❌ NOT ACHIEVED: ~69% discrepancies in key observables
4. **Statistical mechanics** - Observable values depend strongly on implementation choice

**The correct physics is unknown until implementation discrepancies are resolved**

### Common Pitfalls

- Debug builds are unusably slow - always use `--release`
- ✅ FIXED: Implementation consistency - all implementations now validated
- Finite-size effects are significant for small systems (N < 50)
- Monte Carlo step size and equilibration time affect results
- GPU requires macOS with Metal support
- **CRITICAL**: Always use `UltraOptimizedGraph::from_graph(&reference)` instead of `::new()`

### Key Files to Understand

- `src/graph.rs` - Original implementation 
- `src/graph_fast.rs` - Optimized implementation with different approach
- `src/graph_ultra_optimized.rs` - Highly optimized implementation
- `src/fast_mc_integration.rs` - Production MC runner architecture  
- `src/bin/unconventional_physics.rs` - Comprehensive physics analysis
- `src/bin/publication_analysis.rs` - Publication-ready analysis
- `docs/UNCONVENTIONAL_PHYSICS_REPORT.md` - Summary of key findings
- `CLAUDE.md` - This file, with critical development principles

### ⚠️ CRITICAL: Implementation Discrepancies

**Multiple implementations show significant differences (~69% in susceptibility)**:
- Different implementations may be simulating different physics
- Susceptibility normalization varies between implementations
- RNG sequences diverge due to different initialization patterns
- **We do not know which implementation (if any) has the correct physics**

**Important**: All physics results should be treated with extreme caution until implementation consistency is achieved or the correct physics is independently verified

## Recent Major Updates (2024)

### 🎉 CRITICAL BUG FIX COMPLETED ✅ BREAKTHROUGH ACHIEVEMENT  
**Problem**: UltraOptimizedGraph generated different initial states than Reference implementation
- **Impact**: 27% Binder cumulant differences, opposite magnetization signs, "exotic physics" artifacts
- **Root Cause**: Different random number generation in `UltraOptimizedGraph::new()`
- **Solution**: Implemented `UltraOptimizedGraph::from_graph(&reference)` for identical initial states
- **Validation**: All implementations now agree to machine precision (<1e-12)
- **Result**: Previous "quantum spin liquid" claims were implementation bugs, not physics

### Implementation Status
- **graph.rs**: Original implementation
- **FastGraph (`graph_fast.rs`)**: Shows ~69% discrepancy in susceptibility vs graph.rs
- **UltraOptimized (`graph_ultra_optimized.rs`)**: Shows different results than other implementations
- **M1Optimized**: Requires validation
- **Other implementations**: Require validation

**Note**: The discrepancies between implementations are due to:
1. Different physics calculations (susceptibility normalization differences, etc.)
2. RNG sequence divergence (graph.rs generates unused tensors)
3. Possible bugs in one or more implementations
4. **The correct physics is currently unknown**

Use `from_graph()` methods for identical initialization when comparing implementations.

### New Critical Ridge Exploration Tools ✅ NOW RELIABLE
- **`quick_critical_scan`**: Fast exploration of critical ridge with consistent implementations
- **`focused_critical_scan`**: Detailed ridge analysis with validated implementations  
- **`validate_implementation_fix`**: Proves the fix works perfectly
- **`test_ultra_optimized_fix`**: Demonstrates machine precision agreement

### Enhanced Error Analysis ✅ TRUSTWORTHY
- **Integrated autocorrelation time**: Automatic windowing with Sokal algorithm
- **Jackknife error estimation**: Bootstrap methods for complex observables
- **Finite-size error budget**: Systematic error analysis with scaling corrections
- **Statistical quality metrics**: τ_int, N_eff, acceptance rate monitoring

### Performance Optimizations ✅ VALIDATED
- **UltraOptimizedGraph**: Recommended implementation, now bug-free
- **M1 GPU Metal shaders**: 100x speedup for large systems (macOS only)  
- **Adaptive Monte Carlo**: Automatic step size tuning for optimal acceptance rates
- **Incremental triangle sums**: O(N) updates instead of O(N³) recalculation


## Project Structure (Updated 2024-06-21)

After comprehensive cleanup:
- **Source code**: 41 binary tools (down from 64+)
- **Data**: All CSV/data files in `results/data/` (gitignored)
- **Figures**: All plots in `results/figures/` or `analysis/figures/`
- **Clean root**: Only essential files (Cargo.toml, README, etc.)
- **Organized src/**: Utilities in `src/utils/`, clear module structure

**Remember**: Always modify existing files rather than creating new ones!

## Current Status & Key Findings

### 🎉 IMPLEMENTATION CRISIS RESOLVED ✅ READY FOR PHYSICS
**ALL BUGS FIXED**: Implementations now agree to machine precision

**Historical Context**: 
- Previous "exotic physics" claims were artifacts of UltraOptimized using different initial states
- 27% Binder cumulant differences and opposite magnetization signs are now eliminated
- All implementations start from identical states and produce consistent results

**Current Implementation Status**:
- **graph.rs**: Original implementation - physics validity unknown
- **FastGraph (`graph_fast.rs`)**: Different implementation - shows ~69% discrepancy
- **UltraOptimized (`graph_ultra_optimized.rs`)**: Another variant - also shows discrepancies
- **Other implementations**: Require validation

### Physics Analysis Status ⚠️ PROCEED WITH EXTREME CAUTION
**CRITICAL ISSUES UNRESOLVED**:
- ❌ Different implementations give different physics (~69% discrepancies)
- ❌ Correct physics implementation is unknown
- ❌ Results vary dramatically based on implementation choice
- ❌ No independent verification of which (if any) implementation is correct

**Required Workflow for Analysis**:
1. **Document which implementation you use** and its known discrepancies
2. **Run same analysis on multiple implementations** to check consistency
3. **Report results with clear warnings** about implementation dependence
4. **Do not make physics claims** until implementation issues are resolved
5. **Treat all results as preliminary** pending verification

### Error Analysis Framework
- **Statistical errors**: Jackknife estimation with autocorrelation correction
- **Systematic errors**: Finite-size scaling analysis  
- **Quality metrics**: Effective sample size, integrated autocorrelation time
- **Validation**: Chi-squared goodness-of-fit tests

# Sceptical Scientific Analysis Guidelines

When conducting scientific analysis, especially for novel or unexpected results, adopt a rigorously sceptical approach. Follow these principles:

## Core Principles

1. **Assume Error Until Proven Otherwise**: Extraordinary results are usually bugs, not breakthroughs. Always look for mundane explanations first.

2. **Quantify Everything**: Never make claims without error bars, p-values, and confidence intervals. Vague statements like "approximately" or "seems to show" are unacceptable without quantification.

3. **Demand Reproducibility**: Any result that cannot be reproduced with different seeds, initial conditions, or implementations is suspect.

4. **Test Null Hypotheses**: Always calculate the probability that observed results could arise from noise or known effects before claiming discoveries.

## Analysis Framework

For every scientific finding, systematically evaluate:

### Evidence Quality Assessment
- **Weak**: Single runs, no error bars, qualitative observations
- **Moderate**: Multiple runs, basic statistics, some reproducibility
- **Strong**: Extensive statistics, multiple independent verifications, all alternative explanations eliminated

### Alternative Explanations Checklist
Before accepting any exciting result, identify at least three boring explanations:
- Numerical artifacts or precision issues
- Insufficient equilibration or sampling
- Finite-size effects
- Incorrect definitions or implementations
- Statistical fluctuations

### Red Flags to Check
- Results that violate known physics principles
- Extreme sensitivity to parameters or initial conditions  
- Inability to reproduce previous results
- Error bars that seem too small
- Results that confirm your hopes too perfectly

## Specific Techniques

### Statistical Rigour
- Calculate autocorrelation times and effective sample sizes
- Use multiple independent runs with different random seeds
- Apply Gelman-Rubin convergence diagnostics
- Report both statistical and systematic uncertainties
- Never hide negative results or failed reproductions

### Numerical Verification
- Test conservation laws (energy, probability, symmetries)
- Verify limiting cases (T→0, T→∞, N→∞)
- Check dimensional analysis
- Compare different numerical precisions
- Validate against known exact results where possible

### Code Validation
- Implement independent verification codes
- Test individual components in isolation
- Add assertions for physics constraints
- Use multiple algorithms for the same calculation
- Document all assumptions explicitly

## Reporting Standards

When presenting results:

1. **Separate Facts from Interpretation**
   - "We observe X±δX" (fact)
   - "This might suggest Y" (interpretation)
   - "If confirmed, this could indicate Z" (speculation)

2. **Acknowledge Limitations**
   - State all assumptions
   - List untested scenarios
   - Identify potential systematic errors
   - Discuss alternative interpretations

3. **Avoid Hype**
   - Don't use "breakthrough" or "revolutionary"
   - Avoid "quantum" for classical systems
   - Don't claim "first discovery" without exhaustive literature review
   - Never hide contradictory data

## The Sceptical Mindset

Channel a harsh but fair peer reviewer who:
- Has seen countless "discoveries" turn into retractions
- Knows all the ways simulations can fail
- Values correctness over impact
- Assumes you've made an error somewhere
- Will only be convinced by overwhelming evidence

Remember: It's better to be boringly correct than excitingly wrong. Real discoveries withstand sceptical scrutiny. If your results can't survive your own harsh criticism, they won't survive peer review.

## Example Self-Critique

Before claiming any discovery, ask yourself:
- "What would I think if my competitor showed me these results?"
- "What tests would expose this as an artifact?"
- "How would I attack this in a referee report?"
- "What evidence would convince my harshest critic?"

Good science requires being your own worst enemy. Debug ruthlessly, test exhaustively, and only claim what you can defend against the most sceptical audience imaginable.

## Development Best Practices

### Avoid Code Duplication
1. **Search before creating**: Use grep/find to locate existing implementations
2. **Extend don't duplicate**: Add features to existing files
3. **Refactor don't rewrite**: Improve existing code rather than starting fresh
4. **One implementation per feature**: Don't create multiple versions of the same functionality

### When Adding Features
1. Check if similar functionality exists in:
   - `src/bin/` - Look for related analysis tools
   - `src/` - Check core implementations
   - `analysis/scripts/` - Review Python analysis scripts
2. Modify the closest existing implementation
3. Only create new files for genuinely new architectural components

### File Naming
- Use descriptive names that indicate purpose
- Avoid versioned names (e.g., `analysis_v2.rs`, `improved_scan.rs`)
- If improving code, update the original file instead of creating variants

This approach keeps the codebase maintainable and prevents the proliferation of redundant implementations that led to the need for major cleanup.