# Performance Optimization Plan

## Executive Summary

The current implementation has several major performance bottlenecks, especially with the spectral term requiring O(N³) eigenvalue computations. This document outlines concrete optimizations that could provide 10-1000x speedup.

## Critical Optimizations (Highest Impact)

### 1. Spectral Term Optimization (1000x potential speedup)

**Current Problem**: Full eigendecomposition costs O(N³) per MC step.

**Solution A - Perturbation Theory (Recommended)**:
```rust
// When updating link (i,j) with weight change Δw:
// λ'_k ≈ λ_k + Δw * v_k[i] * v_k[j]
// where v_k is the k-th eigenvector

struct SpectralCache {
    eigenvalues: Vec<f64>,
    eigenvectors: DMatrix<f64>,
    last_update: usize,
}

impl SpectralCache {
    fn update_single_link(&mut self, i: usize, j: usize, delta_w: f64) {
        for k in 0..self.eigenvalues.len() {
            let v_ki = self.eigenvectors[(k, i)];
            let v_kj = self.eigenvectors[(k, j)];
            self.eigenvalues[k] += delta_w * v_ki * v_kj;
        }
    }
}
```

**Solution B - Approximate Methods**:
```rust
// Use Lanczos iteration for just the first few eigenvalues
fn spectral_action_approx(&self, n_cut: usize) -> f64 {
    let eigenvalues = lanczos_eigenvalues(&self.laplacian(), n_cut + 1);
    // Only compute what we need
}
```

**Solution C - Disable for Large Systems**:
```rust
// For N > 20, use a mean-field approximation
if self.n() > 20 {
    return self.spectral_action_mean_field();
}
```

### 2. Incremental Triangle Sum Updates (100x speedup)

**Current Problem**: Recalculating all triangles costs O(N²) per update.

**Solution - Port from graph_optimized.rs**:
```rust
impl FastGraph {
    fn triangle_sum_delta_fast(&self, link_idx: usize, new_theta: f64) -> f64 {
        let link = &self.links[link_idx];
        let (i, j) = (link.i as usize, link.j as usize);
        
        // Only triangles containing edge (i,j) are affected
        let mut delta = 0.0;
        
        // Use pre-computed triangle membership
        for &k in &self.triangles_containing_edge[link_idx] {
            let old_sum = self.get_triangle_phase_sum(i, j, k);
            let new_sum = old_sum - link.theta + new_theta;
            delta += new_sum.cos() - old_sum.cos();
        }
        
        delta
    }
}
```

### 3. Eliminate Redundant Calculations (10x speedup)

**Problem**: Action calculated twice per MC step.

**Solution**:
```rust
pub fn metropolis_step_optimized(&mut self, alpha: f64, beta: f64, 
                                 delta_z: f64, delta_theta: f64, 
                                 rng: &mut Pcg64) -> bool {
    let link_idx = rng.gen_range(0..self.links.len());
    
    // Calculate energy change directly
    let delta_s = if rng.gen_bool(0.5) {
        // Z update
        let new_z = ...;
        beta * self.entropy_change(link_idx, new_z)
    } else {
        // Phase update  
        let new_theta = ...;
        alpha * self.triangle_sum_delta_fast(link_idx, new_theta)
    };
    
    // Single Metropolis criterion
    if delta_s <= 0.0 || rng.gen_range(0.0..1.0) < (-delta_s).exp() {
        // Apply update
        true
    } else {
        false
    }
}
```

### 4. Memory Layout Optimization (2x speedup)

**Problem**: Poor cache locality.

**Solution A - Structure of Arrays**:
```rust
struct FastGraphSOA {
    // Hot data in contiguous arrays
    link_i: Vec<u32>,
    link_j: Vec<u32>, 
    z_values: Vec<f64>,
    theta_values: Vec<f64>,
    cos_theta: Vec<f64>,  // Pre-computed
    
    // Cold data separate
    metadata: GraphMetadata,
}
```

**Solution B - Remove Unused Fields**:
```rust
// Remove tensor field from Link struct
// Store separately if needed for analysis
```

### 5. Parallel Measurement Strategy (4x speedup)

**Problem**: Sequential observable calculations.

**Solution**:
```rust
use rayon::prelude::*;

impl BatchedObservables {
    fn measure_parallel(&self, graph: &FastGraph) -> QuickObservables {
        let (sum_w, sum_cos, var_w) = graph.links
            .par_iter()
            .map(|link| (link.w(), link.cos_theta, link.w()))
            .reduce(
                || (0.0, 0.0, 0.0),
                |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2)
            );
            
        // Parallel triangle sum if needed
        let triangle_sum = if self.needs_triangle_sum() {
            graph.triangles.par_iter()
                .map(|&(i, j, k)| {
                    let phase_sum = graph.get_triangle_phase_sum(i, j, k);
                    phase_sum.cos()
                })
                .sum()
        } else {
            self.cached_triangle_sum
        };
        
        // ... rest of observables
    }
}
```

## Implementation Priority

1. **Week 1**: Implement incremental triangle sum updates
   - Port logic from graph_optimized.rs
   - Add pre-computed triangle membership lists
   - Benchmark improvement

2. **Week 2**: Optimize spectral term
   - Implement perturbation theory updates
   - Add caching layer
   - Provide mean-field fallback for large N

3. **Week 3**: Memory optimizations
   - Remove unused tensor field
   - Implement SOA layout for hot paths
   - Profile cache misses

4. **Week 4**: Parallelize measurements
   - Use rayon for reductions
   - Batch similar calculations
   - Consider SIMD for variance

## Benchmarking Plan

Create benchmarks for each optimization:
```rust
#[bench]
fn bench_triangle_sum_original(b: &mut Bencher) {
    let graph = FastGraph::new(20, 42);
    b.iter(|| graph.triangle_sum());
}

#[bench]
fn bench_triangle_sum_incremental(b: &mut Bencher) {
    let graph = FastGraph::new(20, 42);
    b.iter(|| graph.triangle_sum_delta_fast(0, 1.0));
}
```

## Expected Performance Gains

For a system with N=50:
- Current: ~1000 MC steps/second
- With triangle optimization: ~100,000 MC steps/second  
- With spectral caching: ~1,000,000 MC steps/second
- With all optimizations: ~2,000,000 MC steps/second

## Additional Recommendations

1. **Use Profile-Guided Optimization**:
   ```bash
   RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data" cargo build --release
   ./target/release/wide_scan
   RUSTFLAGS="-Cprofile-use=/tmp/pgo-data" cargo build --release
   ```

2. **Enable Link-Time Optimization**:
   ```toml
   [profile.release]
   lto = true
   codegen-units = 1
   ```

3. **Consider Alternative Algorithms**:
   - Cluster algorithms for critical region
   - Parallel tempering for better sampling
   - Multigrid methods for large systems

4. **GPU Acceleration**:
   - Already started with Metal implementation
   - Consider CUDA for broader compatibility
   - Especially valuable for eigenvalue computations

## Conclusion

The most impactful optimization is fixing the O(N³) spectral term calculation. Combined with incremental triangle updates and better memory layout, we can achieve 100-1000x speedup, making simulations of N=100+ systems practical.