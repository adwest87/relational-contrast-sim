# Canonical Implementation Plan

**Date**: 2025-06-21  
**Status**: DECISIVE ACTION REQUIRED

## Decision: graph.rs as Canonical Implementation

### Why graph.rs?
1. **Closest to correct physics** - Only needs one-line entropy fix
2. **Has z-variable formulation** - Correct numerical approach  
3. **Includes Dougal rescaling** - Fundamental symmetry preserved
4. **Original implementation** - Least contaminated by magnetic concepts
5. **Simplest to fix** - Minimal changes needed

### Required Fix for graph.rs

```rust
// CHANGE THIS (line ~161):
pub fn entropy_action(&self) -> f64 {
    self.links.iter().map(|l| -l.z * l.w()).sum()
}

// TO THIS:
pub fn entropy_action(&self) -> f64 {
    self.links.iter().map(|l| {
        let w = l.w();
        w * w.ln()  // w ln(w), negative for w < 1
    }).sum()
}
```

## Implementation Cleanup Plan

### 1. Archive Non-Canonical Implementations

```bash
# Create archive directory
mkdir -p archived_implementations

# Move all non-canonical implementations
mv src/graph_fast.rs archived_implementations/
mv src/graph_ultra_optimized.rs archived_implementations/
mv src/graph_m1_optimized.rs archived_implementations/
mv src/graph_m1_accelerate.rs archived_implementations/
mv src/graph_metal.rs archived_implementations/

# Add README to archive
echo "These implementations solve the WRONG PHYSICS MODEL. 
Archived on 2025-06-21 after discovering fundamental physics errors.
DO NOT USE without complete physics revision." > archived_implementations/README.md
```

### 2. Clean Up lib.rs

Remove modules for archived implementations:
```rust
// Remove these lines from lib.rs
pub mod graph_fast;
pub mod graph_ultra_optimized;
#[cfg(target_arch = "aarch64")]
pub mod graph_m1_optimized;
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub mod graph_m1_accelerate;
#[cfg(target_os = "macos")]
pub mod graph_metal;
```

### 3. Remove Magnetic Observables

Delete or archive:
- All magnetization calculations
- All susceptibility code  
- Binder cumulant calculations
- Spin-related analysis tools

### 4. Delete All Previous Results

```bash
# Remove all previous data - it's physically meaningless
rm -rf results/data/*
rm -rf results/figures/*
rm -rf analysis/figures/*

# Add warning file
echo "All previous results deleted on 2025-06-21.
They were based on incorrect physics model.
DO NOT attempt to recover or use old data." > results/RESULTS_DELETED.txt
```

## Verification Tests

### Test 1: Entropy Sign Verification
```rust
#[test]
fn test_entropy_sign() {
    let graph = Graph::complete_random(3);
    let entropy = graph.entropy_action();
    // For w < 1, w ln(w) < 0
    assert!(entropy < 0.0, "Entropy should be negative for w < 1");
}
```

### Test 2: Dougal Invariance
```rust
#[test]
fn test_dougal_invariance() {
    let mut graph1 = Graph::complete_random(10);
    let mut graph2 = graph1.clone();
    
    let action1 = graph1.action(1.0, 1.0);
    graph2.rescale(2.0);  // w → 2w
    let action2 = graph2.action(1.0, 1.0);
    
    // Physics should be invariant
    assert!((action1 - action2).abs() < 1e-10);
}
```

### Test 3: Weight Bounds
```rust
#[test]
fn test_weight_bounds() {
    let graph = Graph::complete_random(10);
    for link in &graph.links {
        assert!(link.w() > 0.0 && link.w() <= 1.0);
    }
}
```

## New Observables to Implement

### 1. Spectral Gap Calculator
```rust
pub fn spectral_gap(&self) -> f64 {
    let lap = self.weighted_laplacian();
    let eigenvalues = lap.symmetric_eigenvalues();
    eigenvalues[1] - eigenvalues[0]
}
```

### 2. Effective Dimension
```rust
pub fn effective_dimension(&self) -> f64 {
    let gap = self.spectral_gap();
    -2.0 * (self.n() as f64).ln() / gap.ln()
}
```

### 3. Weight Statistics
```rust
pub fn weight_stats(&self) -> (f64, f64) {
    let weights: Vec<f64> = self.links.iter().map(|l| l.w()).collect();
    let mean = weights.iter().sum::<f64>() / weights.len() as f64;
    let var = weights.iter().map(|&w| (w - mean).powi(2)).sum::<f64>() / weights.len() as f64;
    (mean, var)
}
```

## Timeline

### Day 1 (Immediate)
1. ✓ Create PHYSICS_SPECIFICATION.md
2. ✓ Create IMPLEMENTATION_ANALYSIS.md  
3. ✓ Create this plan
4. Fix graph.rs entropy term
5. Archive other implementations

### Day 2
1. Remove magnetic observables
2. Add spectral calculations
3. Create verification tests
4. Delete all old results

### Day 3
1. Run verification tests
2. Validate basic physics
3. Document correct usage
4. Update README

### Day 4+
1. Begin legitimate exploration
2. Study emergent geometry
3. Look for 4D spacetime emergence
4. Investigate gauge field dynamics

## Success Criteria

The canonical implementation will be considered correct when:

1. **Entropy term** returns w*ln(w) < 0 for w < 1
2. **Dougal invariance** is verified numerically
3. **No magnetic terms** remain in code
4. **Spectral gap** is computed correctly
5. **All tests pass** with correct physics

## Final Notes

This disaster occurred because:
- We had multiple implementations without clear physics specification
- Each implementation drifted toward different physics
- No one questioned the magnetic interpretation
- Optimizations obscured the fundamental issues

**LESSON**: One physics model, one implementation, verified against theory.

---

*"When you have two implementations, you have two physics models. When you have five implementations, you have a physics crisis."*