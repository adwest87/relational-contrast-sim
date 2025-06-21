# Observable Calculations Implementation

## Summary

I have successfully implemented all the missing observables in `graph_fast.rs` with the correct theoretical normalizations:

### 1. Specific Heat ✓
**Formula**: `C = (1/N) * (<E²> - <E>²)`
- Correctly normalized by 1/N (not β²/N)
- Uses time series accumulation of total action E
- Calculates variance over accumulated samples

### 2. Binder Cumulant ✓
**Formula**: `U₄ = 1 - <m⁴>/(3<m²>²)`
- Uses proper magnetization: `m = (1/N)∑cos(θ)`
- Sum is over all links, normalized by number of nodes
- Accumulates 2nd and 4th moments separately

### 3. Susceptibility ✓
**Formula**: `χ = N * (<m²> - <m>²)`
- Fixed to use node-based magnetization variance
- N is the number of nodes (not links)
- Calculated from the same magnetization time series as Binder cumulant

### 4. Correlation Length ✓
**Formula**: `ξ = sqrt(<r²·C(r)> / <C(r)>)`
- Properly implemented using correlation function C(r)
- For complete graphs: all non-self pairs are at distance r=1
- Node-based correlation calculation (not link-based)

### 5. Jackknife Error Estimation ✓
- Added `JackknifeEstimator` class
- Can estimate errors for any observable function
- Proper jackknife variance formula: `σ² = (n-1)/n * Σ(θᵢ - θ̄)²`

## Key Corrections Made

1. **Magnetization Definition**: Changed from `|<cos θ>|` to `(1/N)∑cos(θ)` where the sum is over all links but normalized by nodes

2. **Specific Heat**: Removed incorrect β² factor, now just `(1/N) * Var(E)`

3. **Susceptibility**: Now uses magnetization variance with proper node normalization

4. **Correlation Function**: Implemented node-based correlations instead of link-based

## Implementation Details

### Time Series Accumulation
```rust
struct TimeSeriesAccumulator {
    sum: f64,
    sum_sq: f64,
    sum_4th: f64,
    count: usize,
}
```
- Efficiently accumulates moments online
- Allows calculation of variance and 4th moment
- Reset capability for parameter changes

### Batched Measurements
- Expensive calculations (entropy, triangle sum) are rotated
- Time series always accumulated for each measurement
- Correlation length updated periodically

### Error Estimation
```rust
pub fn estimate_error<F>(&self, estimator: F) -> (f64, f64)
where F: Fn(&[f64]) -> f64
```
- Generic over any estimator function
- Returns (estimate, error) tuple
- Handles edge cases (n < 2)

## Usage Example

```rust
let mut observables = BatchedObservables::new();

// Equilibrate
for _ in 0..equilibration_steps {
    graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
}

// Reset accumulators after equilibration
observables.reset_accumulators();

// Measure
for _ in 0..measurement_steps {
    graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut rng);
    let obs = observables.measure(&graph, alpha, beta);
    
    // obs now contains:
    // - specific_heat with C = (1/N) * Var(E)
    // - susceptibility with χ = N * Var(m)
    // - binder_cumulant with U₄ = 1 - <m⁴>/(3<m²>²)
    // - correlation_length with ξ = sqrt(<r²C(r)>/<C(r)>)
}
```

## Notes

- For complete graphs, the correlation length calculation is simplified since all nodes are equidistant
- Error estimates require separate jackknife analysis on collected samples
- The implementation is optimized for performance with batched calculations