# Triangle Sum Optimization: From O(N³) to O(N) per MC Step

## Overview

The relational contrast simulation involves a complete graph with N nodes where the action includes a triangle sum term:

```
S = β × (entropy term) + α × Σ_triangles 3 cos(θ_ij + θ_jk + θ_ki)
```

The naive implementation recalculates the entire triangle sum (O(N³) triangles) at each MC step. This optimization reduces it to O(N) by only updating triangles affected by the changed link.

## Key Insight

In a complete graph:
- Total number of triangles: C(N,3) = N(N-1)(N-2)/6 = O(N³)
- Each edge (i,j) is part of exactly (N-2) triangles (one for each third vertex k)
- When edge (i,j) changes, only these (N-2) triangles need updating

## Implementation Strategy

### 1. Pre-computation (Graph Construction)

```rust
// Build edge-to-triangle index
triangles_by_edge: HashMap<(usize, usize), Vec<usize>>

// For each triangle (i,j,k), add its index to all three edges
for (tri_idx, (i,j,k)) in triangles.enumerate() {
    triangles_by_edge[(i,j)].push(tri_idx);
    triangles_by_edge[(j,k)].push(tri_idx);
    triangles_by_edge[(i,k)].push(tri_idx);
}
```

### 2. Cache Triangle Sum

```rust
triangle_sum_cache: f64  // Maintained incrementally
```

### 3. Incremental Update

When link (i,j) phase changes from θ_old to θ_new:

```rust
fn compute_triangle_sum_delta(i, j, old_theta, new_theta) -> f64 {
    let mut delta = 0.0;
    
    // Only iterate over (N-2) triangles containing edge (i,j)
    for tri_idx in triangles_by_edge[(i,j)] {
        let (a,b,c) = triangles[tri_idx];
        
        // Old contribution
        old_contrib = 3 * cos(θ_ab + θ_bc + θ_ca)
        
        // New contribution (with updated θ for edge (i,j))
        new_contrib = 3 * cos(θ'_ab + θ'_bc + θ'_ca)
        
        delta += new_contrib - old_contrib;
    }
    
    return delta;
}
```

### 4. Update Metropolis Step

```rust
// Instead of:
let s_before = self.action(alpha, beta);  // O(N³)
let s_after = self.action(alpha, beta);   // O(N³)
let delta_s = s_after - s_before;

// Use:
let delta_triangle = compute_triangle_sum_delta(i, j, old_theta, new_theta);  // O(N)
let delta_s = alpha * delta_triangle;
```

## Performance Analysis

### Before Optimization
- Per MC step: O(N³) for triangle sum calculation
- For N=96: ~148,000 triangle evaluations per step
- Dominates runtime for large N

### After Optimization
- Graph construction: O(N³) one-time cost
- Per MC step: O(N) for incremental update
- For N=96: ~94 triangle evaluations per step (1500x speedup)
- Memory overhead: O(N²) for edge-to-triangle map

## Verification

To ensure correctness:
1. Periodically verify: `triangle_sum_cache == compute_full_triangle_sum()`
2. Check conservation after accept/reject decisions
3. Monitor for numerical drift over long runs

## Integration Steps

1. Add new fields to Graph struct:
   - `triangle_sum_cache: f64`
   - `triangles_by_edge: HashMap<(usize,usize), Vec<usize>>`

2. Modify Graph::new() to build the index and initialize cache

3. Replace `triangle_sum()` to return cached value

4. Modify `metropolis_step()` to use incremental updates

5. Update cache when proposals are accepted

## Expected Speedup

For typical simulation parameters:
- N=24: ~3x speedup
- N=48: ~25x speedup  
- N=96: ~100x speedup
- N=192: ~400x speedup

The speedup increases quadratically with system size, making larger systems feasible.

## Testing

Run benchmarks comparing:
1. Time per MC sweep (N × steps)
2. Total time for equilibration + measurement
3. Verify identical physics (same averages, fluctuations)

## Code Examples

See:
- `src/graph_optimized.rs` - Full optimized implementation
- `src/incremental_triangle_update.patch` - Patch for existing code