# Critical Physics Bug Report

## Issue: Missing Antisymmetry in Optimized Graph Implementations

### Problem
The optimized graph implementations (`FastGraph` and `UltraOptimizedGraph`) do not properly handle the antisymmetric nature of phases, leading to **incorrect physics**.

### Root Cause
In the original `Graph` implementation, phases are antisymmetric: `θ_ji = -θ_ij`. This is enforced by the `get_phase()` method:

```rust
pub fn get_phase(&self, from_node: usize, to_node: usize) -> f64 {
    if from_node < to_node {
        self.theta
    } else {
        -self.theta  // Antisymmetric!
    }
}
```

However, the optimized implementations directly use `link.theta` without considering direction, violating this crucial antisymmetry.

### Impact
This fundamentally changes the physics of the model:
- Triangle sums are calculated incorrectly
- Action values differ between implementations
- Equilibrium properties are wrong
- The model loses important symmetries

### Evidence
Debug output shows action discrepancies:
- Original: Action = -4.136547328748, Triangle sum = 0.262710276865
- FastGraph: Action = 7.196319192550, Triangle sum = 7.817954624397
- UltraOptimized: Action = -15.057118715286, Triangle sum = -7.353861553509

Even after copying identical link states, triangle sums remain different due to the antisymmetry issue.

### Fix Required
All optimized implementations must properly handle antisymmetric phases in triangle calculations.

#### For FastGraph:
```rust
// Instead of:
let theta_sum = self.links[idx_ij].theta + 
               self.links[idx_jk].theta + 
               self.links[idx_ik].theta;

// Use:
let t_ij = self.get_phase(i, j);  // Need to implement get_phase
let t_jk = self.get_phase(j, k);
let t_ki = self.get_phase(k, i);
let theta_sum = t_ij + t_jk + t_ki;
```

#### For UltraOptimizedGraph:
```rust
// Need to implement proper phase lookup with antisymmetry
fn get_phase(&self, i: usize, j: usize) -> f64 {
    let link_idx = self.link_index(i, j);
    if i < j {
        self.theta_values[link_idx]
    } else {
        -self.theta_values[link_idx]
    }
}
```

### Priority
**CRITICAL** - This bug invalidates all physics results from optimized implementations.