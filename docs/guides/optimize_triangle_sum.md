# Optimizing Triangle Sum Calculation: Implementation Guide

## Current Implementation (O(N³) per MC step)

The current code in `src/graph.rs` recalculates the entire triangle sum at each Monte Carlo step:

```rust
pub fn triangle_sum(&self) -> f64 {
    self.triangles.iter().map(|&(i, j, k)| {
        let t_ij = self.links[self.link_index(i, j)].theta;
        let t_jk = self.links[self.link_index(j, k)].theta;
        let t_ki = self.links[self.link_index(k, i)].theta;
        3.0 * (t_ij + t_jk + t_ki).cos()
    }).sum()
}
```

This iterates over all N(N-1)(N-2)/6 triangles, making it O(N³).

## Optimized Implementation (O(N) per MC step)

### Step 1: Add New Fields to Graph Struct

```rust
use std::collections::HashMap;

pub struct Graph {
    nodes: Vec<Node>,
    links: Vec<Link>,
    triangles: Vec<(usize, usize, usize)>,
    
    // NEW: Cache and index for O(N) updates
    triangle_sum_cache: f64,
    triangles_by_edge: HashMap<(usize, usize), Vec<usize>>,
}
```

### Step 2: Build Triangle Index During Construction

In `Graph::new()`, after creating the triangles vector:

```rust
// Build edge-to-triangle index
let mut triangles_by_edge = HashMap::new();
for (tri_idx, &(i, j, k)) in triangles.iter().enumerate() {
    // Each triangle has 3 edges
    let edges = vec![(i, j), (j, k), (i, k)];
    for (a, b) in edges {
        let canonical = if a < b { (a, b) } else { (b, a) };
        triangles_by_edge
            .entry(canonical)
            .or_insert_with(Vec::new)
            .push(tri_idx);
    }
}

// Initialize cache
let mut graph = Self {
    nodes,
    links,
    triangles,
    triangle_sum_cache: 0.0,
    triangles_by_edge,
};

// Compute initial triangle sum
graph.triangle_sum_cache = graph.compute_full_triangle_sum();
```

### Step 3: Add Helper Methods

```rust
impl Graph {
    /// Get cached triangle sum - O(1)
    pub fn triangle_sum(&self) -> f64 {
        self.triangle_sum_cache
    }
    
    /// Full calculation - O(N³), only for initialization
    fn compute_full_triangle_sum(&self) -> f64 {
        self.triangles.iter().map(|&(i, j, k)| {
            let t_ij = self.links[self.link_index(i, j)].theta;
            let t_jk = self.links[self.link_index(j, k)].theta;
            let t_ki = self.links[self.link_index(k, i)].theta;
            3.0 * (t_ij + t_jk + t_ki).cos()
        }).sum()
    }
    
    /// Incremental update - O(N)
    fn triangle_sum_delta_for_phase_change(
        &self, 
        link_idx: usize,
        new_theta: f64
    ) -> f64 {
        let link = &self.links[link_idx];
        let (i, j) = (link.i, link.j);
        let old_theta = link.theta;
        let canonical = if i < j { (i, j) } else { (j, i) };
        
        let mut delta = 0.0;
        
        if let Some(tri_indices) = self.triangles_by_edge.get(&canonical) {
            for &tri_idx in tri_indices {
                let (a, b, c) = self.triangles[tri_idx];
                
                // Old contribution
                let t_ab = self.links[self.link_index(a, b)].theta;
                let t_bc = self.links[self.link_index(b, c)].theta;
                let t_ca = self.links[self.link_index(c, a)].theta;
                let old_contrib = 3.0 * (t_ab + t_bc + t_ca).cos();
                
                // New contribution (update the changed edge)
                let t_ab_new = if (a == i && b == j) || (a == j && b == i) {
                    new_theta
                } else {
                    t_ab
                };
                let t_bc_new = if (b == i && c == j) || (b == j && c == i) {
                    new_theta
                } else {
                    t_bc
                };
                let t_ca_new = if (c == i && a == j) || (c == j && a == i) {
                    new_theta
                } else {
                    t_ca
                };
                let new_contrib = 3.0 * (t_ab_new + t_bc_new + t_ca_new).cos();
                
                delta += new_contrib - old_contrib;
            }
        }
        
        delta
    }
}
```

### Step 4: Modify Metropolis Step

Replace the current metropolis_step implementation:

```rust
pub fn metropolis_step(
    &mut self,
    alpha: f64,
    beta: f64,
    delta_z: f64,
    delta_theta: f64,
    rng: &mut impl Rng,
) -> StepInfo {
    let link_index = rng.gen_range(0..self.links.len());
    let link = &self.links[link_index];
    
    let phase_only = delta_z == 0.0;
    let do_z_update = !phase_only && rng.gen_bool(0.5);
    
    if do_z_update {
        // Z-update: only affects entropy, not triangles
        let old_z = link.z;
        let old_w = (-old_z).exp();
        let new_z = (old_z + rng.gen_range(-delta_z..=delta_z)).max(0.001);
        let new_w = (-new_z).exp();
        
        let delta_entropy = new_w * new_z - old_w * old_z;
        let delta_s = beta * delta_entropy;
        
        let accept = delta_s <= 0.0 || rng.gen_range(0.0..1.0) < (-delta_s).exp();
        
        if accept {
            self.links[link_index].z = new_z;
            StepInfo {
                accept: true,
                delta_w: new_w - old_w,
                delta_cos: 0.0,
            }
        } else {
            StepInfo { accept: false, delta_w: 0.0, delta_cos: 0.0 }
        }
    } else {
        // Phase update: use incremental triangle calculation
        let old_theta = link.theta;
        let new_theta = old_theta + rng.gen_range(-delta_theta..=delta_theta);
        
        // O(N) calculation instead of O(N³)
        let delta_triangle = self.triangle_sum_delta_for_phase_change(
            link_index, 
            new_theta
        );
        let delta_s = alpha * delta_triangle;
        
        let accept = delta_s <= 0.0 || rng.gen_range(0.0..1.0) < (-delta_s).exp();
        
        if accept {
            self.links[link_index].theta = new_theta;
            self.triangle_sum_cache += delta_triangle;  // Update cache!
            
            let w = (-link.z).exp();
            let delta_cos = w * (new_theta.cos() - old_theta.cos());
            
            StepInfo {
                accept: true,
                delta_w: 0.0,
                delta_cos,
            }
        } else {
            StepInfo { accept: false, delta_w: 0.0, delta_cos: 0.0 }
        }
    }
}
```

## Performance Impact

- **Memory overhead**: O(N²) for the HashMap (acceptable)
- **Speedup factor**: ~N²/2
  - N=24: ~12x speedup
  - N=48: ~48x speedup
  - N=96: ~192x speedup
  
## Testing

To verify correctness:

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_incremental_update() {
        let mut graph = Graph::new(10, &mut rand::thread_rng());
        
        // Verify cache matches full calculation
        assert!((graph.triangle_sum_cache - graph.compute_full_triangle_sum()).abs() < 1e-10);
        
        // Do some MC steps
        for _ in 0..1000 {
            graph.metropolis_step(1.0, 1.0, 0.0, 0.1, &mut rand::thread_rng());
        }
        
        // Verify cache still matches
        assert!((graph.triangle_sum_cache - graph.compute_full_triangle_sum()).abs() < 1e-10);
    }
}
```

## Integration Checklist

1. [ ] Add HashMap import
2. [ ] Add new fields to Graph struct
3. [ ] Update Graph::new() to build index and initialize cache
4. [ ] Replace triangle_sum() to return cached value
5. [ ] Add compute_full_triangle_sum() helper
6. [ ] Add triangle_sum_delta_for_phase_change() helper
7. [ ] Modify metropolis_step() to use incremental updates
8. [ ] Add tests to verify correctness
9. [ ] Benchmark performance improvement