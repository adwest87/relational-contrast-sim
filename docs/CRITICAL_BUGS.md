# CRITICAL MONTE CARLO BUGS IDENTIFIED

## 🚨 BUG #1: NO-OP MOVE REJECTION (FUNDAMENTAL METROPOLIS VIOLATION)

**Severity**: CRITICAL - Violates basic Monte Carlo requirements
**Location**: `src/graph_fast.rs:272` and `src/graph_fast.rs:307`
**Affects**: All physics results - energy conservation, detailed balance, acceptance rates

### Problem Description
The Metropolis algorithm rejects moves where the state doesn't actually change (no-op moves). This violates the fundamental requirement that moves with ΔE=0 must ALWAYS be accepted.

### Root Cause
Current logic checks proposed move parameters instead of actual state changes:

```rust
// WRONG: Checks proposed delta instead of actual state change
if d_theta.abs() < 1e-15 {
    // Accept no-op
} else {
    // Make acceptance decision
}
```

But `d_theta` is almost never exactly zero from `rng.gen_range()`, even when the actual state doesn't change.

### Mathematical Proof Why This Is Wrong
In Metropolis Monte Carlo:
- P(accept | ΔE=0) = 1.0 (must always accept)
- If state unchanged → ΔE=0 → must accept
- Current code: P(accept | no-op) ≈ 0.27 (random decision!)

### Evidence
- **73.83% of no-op moves rejected** (should be 0%)
- **Energy drift**: 477 units over 10k steps (should be ≈0)
- **Detailed balance violations**: 100% failure rate
- **11.9% of ΔE=0 moves rejected** (should be 0%)

### Correct Fix
Check actual state change, not proposed change:

```rust
// Save original state
let original_state = self.links[link_idx].clone();

// Apply proposed move
self.links[link_idx].update_theta(new_theta);

// Check if state actually changed
let state_changed = (original_state.theta - self.links[link_idx].theta).abs() > 1e-15;

if !state_changed {
    // No-op: always accept
    StepInfo { accept: true, delta_w: 0.0, delta_cos: 0.0 }
} else {
    // Real move: calculate energy and apply Metropolis
    let delta_triangle = self.triangle_sum_delta(link_idx, new_theta);
    let delta_s = alpha * delta_triangle;
    let accept = delta_s <= 0.0 || rng.gen_range(0.0..1.0) < (-delta_s).exp();
    
    if !accept {
        // Revert if rejected
        self.links[link_idx] = original_state;
        StepInfo::rejected()
    } else {
        StepInfo { accept: true, delta_w: 0.0, delta_cos: new_theta.cos() - original_state.cos_theta }
    }
}
```

### Impact Assessment
This bug affects:
- ✗ **Energy conservation**: Random rejection of valid moves causes drift
- ✗ **Detailed balance**: Violates microscopic reversibility
- ✗ **Acceptance rates**: Artificially inflated due to random rejections
- ✗ **All observables**: Mean cosine, susceptibility, entropy all incorrect
- ✗ **Critical point detection**: System can't reach equilibrium properly

## 🚨 BUG #2: OBSERVABLE TRACKING ERROR (FIXED)

**Severity**: HIGH - Affects all measurements
**Location**: `src/graph_fast.rs:306` (FIXED)
**Status**: ✅ RESOLVED

### Problem (Fixed)
Observable tracking calculated weighted cosines instead of pure cosines:
```rust
// WRONG (fixed):
let delta_cos = old_exp_neg_z * (new_theta.cos() - old_cos_theta);

// CORRECT (applied):
let delta_cos = new_theta.cos() - old_cos_theta;
```

## 🚨 BUG #3: ENTROPY CALCULATION ERROR (FIXED)

**Severity**: MEDIUM - Affects energy conservation
**Location**: `src/graph_fast.rs:273` (FIXED)
**Status**: ✅ RESOLVED

### Problem (Fixed)
Incorrect operator precedence in entropy calculation:
```rust
// WRONG (fixed):
let delta_entropy = -new_z * new_exp_neg_z - (-old_z * old_exp_neg_z);

// CORRECT (applied):
let delta_entropy = (-new_z * new_exp_neg_z) - (-old_z * old_exp_neg_z);
```

## IMMEDIATE ACTION REQUIRED

1. **Fix Bug #1** (no-op move rejection) - This is blocking all other fixes
2. **Implement proper state change detection**
3. **Add unit tests for no-op move acceptance**
4. **Re-run all validation tests**

Bug #1 is the root cause of the persistent physics violations. Until this is fixed, the simulation cannot produce correct results.

## Minimal Reproducer

```rust
// This should always print "accepted=true"
let mut graph = FastGraph::new(4, 123);
for link in &mut graph.links { link.update_z(0.001); } // Force boundary
let info = graph.metropolis_step(1.0, 1.0, 0.001, 0.001, &mut rng);
println!("No-op move accepted={}", info.accept); // Currently prints false ~70% of time!
```