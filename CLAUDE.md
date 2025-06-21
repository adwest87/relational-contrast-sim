# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

# Quick validation of physics
cargo run --release --bin quick_validation

# Investigate unconventional physics
cargo run --release --bin unconventional_physics
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
- **Critical behavior**: Ridge at α ≈ 0.06β + 1.31
- **Unconventional physics**: Quantum spin liquid-like behavior

### Core Module Structure

1. **Graph Implementations** (progressive optimization):
   - `graph.rs` - Basic implementation with z-variables
   - `graph_fast.rs` - Cache-optimized with batched observables
   - `graph_optimized.rs` - Triangle sum optimizations
   - `graph_m1_optimized.rs` - Apple Silicon NEON SIMD
   - `graph_metal.rs` - GPU acceleration via Metal

2. **Monte Carlo Integration**:
   - `fast_mc_integration.rs` - High-performance MC runner
   - `importance_mc_integration.rs` - Importance sampling on critical ridge
   - Both use adaptive measurement intervals based on autocorrelation

3. **Key Algorithms**:
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
src/bin/* → CSV files → Python analysis scripts → Figures/Results
                ↓
         FSS data files → Critical exponents
```

### Important Physics Results

1. **Not a conventional phase transition** - susceptibility saturates
2. **Extensive ground state degeneracy** - entropy remains finite as T→0  
3. **Novel universality class** - not Ising, XY, or Heisenberg
4. **Possible experimental realizations** in artificial spin ice, photonics

### Common Pitfalls

- Debug builds are unusably slow - always use `--release`
- The model shows spin liquid behavior, not conventional critical phenomena
- Finite-size scaling behaves unusually due to saturation
- GPU requires macOS with Metal support

### Key Files to Understand

- `src/graph.rs` - Core data structures and metropolis algorithm
- `src/fast_mc_integration.rs` - Production MC runner architecture  
- `src/bin/unconventional_physics.rs` - Comprehensive physics analysis
- `UNCONVENTIONAL_PHYSICS_REPORT.md` - Summary of key findings