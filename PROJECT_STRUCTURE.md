# Relational Contrast Simulation - Project Structure

## Directory Organization

### Core Directories

#### `/src/`
Source code for the Monte Carlo simulation
- `lib.rs` - Main library exports
- `graph.rs` - Core graph structure
- `graph_fast.rs` - Optimized graph implementation
- `graph_m1_optimized.rs` - Apple Silicon M1 optimizations
- `graph_metal.rs` - Metal GPU acceleration
- `observables.rs` - Physics measurements
- `bin/` - Executable programs
  - `wide_scan.rs` - Parameter space scanning
  - `fss_analysis.rs` - Finite-size scaling
  - `critical_finder.rs` - Critical point finder
  - `benchmark_metal.rs` - GPU benchmarking

#### `/analysis/`
Analysis scripts and results
- `scripts/` - Python analysis scripts organized by purpose
  - `fss/` - Finite-size scaling analysis
  - `ridge/` - Critical ridge analysis
  - `visualization/` - Figure generation
- `data/` - Analysis input/output data
- `figures/` - Generated figures organized by type
- `results/` - Analysis results and summaries

#### `/notebooks/`
Jupyter notebooks for interactive analysis
- `mc_exploration.ipynb` - Monte Carlo exploration
- `phase_scan.ipynb` - Phase diagram scanning
- `analysis.ipynb` - General analysis

#### `/data/`
Simulation output data
- CSV files with simulation results
- Organized by scan type (fss, ridge, etc.)

#### `/fss_data/`
Finite-size scaling data
- Results for different system sizes (N=24, 48, 96)

#### `/papers/`
LaTeX papers and related materials
- Each paper in its own subdirectory
- Includes figures used in papers

#### `/docs/`
Documentation
- `guides/` - Implementation and optimization guides
- `results/` - Result summaries and findings
- `papers/` - Paper drafts (if different from /papers/)

#### `/benchmarks/`
Performance benchmarking files
- Patch files showing optimizations
- Benchmark scripts and results

#### `/examples/`
Example usage code
- `m1_example.rs` - M1 optimization example
- `metal_gpu_example.rs` - GPU acceleration example

#### `/tests/`
Unit and integration tests
- Test files for core functionality

#### `/archive/`
Archived/deprecated files
- `old_figures/` - Previous figure versions
- `old_scripts/` - Superseded scripts
- `old_data/` - Old data files

### Key Files

- `Cargo.toml` - Rust project configuration
- `README.md` - Main project documentation
- `LICENSE` - Project license
- `.gitignore` - Git ignore rules

### Workflow

1. **Development**: Code in `/src/`
2. **Simulation**: Run binaries, output to `/data/` or `/fss_data/`
3. **Analysis**: Use scripts in `/analysis/scripts/` to process data
4. **Visualization**: Generate figures saved to `/analysis/figures/`
5. **Documentation**: Update `/docs/` with findings
6. **Publication**: Prepare papers in `/papers/`

### Data Flow

```
src/bin/* → data/*.csv → analysis/scripts/* → analysis/figures/*
                ↓                                      ↓
           fss_data/*                          analysis/results/*
                                                      ↓
                                                  papers/*
```

### Optimization History

The project has undergone several optimization phases:
1. Basic implementation
2. Cache optimization (30x speedup)
3. Triangle sum optimization (O(N³) → O(N))
4. Apple Silicon M1 optimization (NEON SIMD)
5. Metal GPU acceleration (100x with batching)

See `/docs/guides/` for detailed optimization guides.