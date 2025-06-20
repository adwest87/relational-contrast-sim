# Relational Contrast Simulation

High-performance Monte Carlo simulation for studying phase transitions in relational contrast models on complete graphs.

## Overview

This project implements an optimized Monte Carlo simulation for investigating the critical behavior of a relational contrast model. The simulation has been heavily optimized for performance, achieving up to 157x speedup through various techniques including cache optimization, SIMD vectorization, and GPU acceleration.

## Features

- **High Performance**: 3M+ Monte Carlo steps/second on Apple Silicon
- **GPU Acceleration**: Metal compute shaders for 100x speedup
- **Finite-Size Scaling**: Automated FSS analysis for critical exponents
- **Critical Point Finding**: Rapid location of phase transitions
- **Comprehensive Analysis**: Full suite of analysis tools and visualizations

## Quick Start

```bash
# Build the project
cargo build --release

# Run a parameter scan
cargo run --release --bin wide_scan

# Find critical points
cargo run --release --bin critical_finder

# Run FSS analysis
cargo run --release --bin fss_analysis
```

## Project Structure

See `PROJECT_STRUCTURE.md` for detailed directory organization.

Key directories:
- `/src/` - Core simulation code
- `/analysis/` - Analysis scripts and results
- `/data/` - Simulation output data
- `/docs/` - Documentation and guides

## Performance

Optimization achievements:
- Basic implementation: ~20k steps/sec
- Cache optimized: ~600k steps/sec (30x)
- Triangle optimization: ~1.5M steps/sec (75x) 
- SIMD + parallel: ~3M steps/sec (150x)
- GPU batched: ~200M steps/sec (10,000x)

## Physics

The model studies phase transitions in a relational contrast system where:
- Links have angular variables θ and weights w = exp(-z)
- Action: S = α∑ᵢⱼ zᵢⱼ + β∑ₜᵣᵢ wₜᵣᵢ cos(θₜᵣᵢ)
- Critical behavior shows emergent geometric phase

Critical point: (β ≈ 2.93, α ≈ 1.50) with exponents γ/ν ≈ 1.75

## Documentation

- `/docs/guides/` - Implementation and optimization guides
- `/docs/results/` - Key findings and results
- `/analysis/` - Analysis scripts and notebooks

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:
```
[Citation information to be added]
```