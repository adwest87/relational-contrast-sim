use rand::Rng;
use std::collections::HashMap;

/// Optimized graph structure with incremental triangle sum updates
/// 
/// When a link (i,j) is modified, only triangles containing that edge need updating.
/// For a complete graph, link (i,j) is part of exactly (N-2) triangles.
/// This reduces triangle sum update from O(N³) to O(N) per MC step.
pub struct OptimizedGraph {
    // Original fields
    nodes: Vec<Node>,
    links: Vec<Link>,
    triangles: Vec<(usize, usize, usize)>,
    
    // New fields for optimization
    triangle_sum_cache: f64,
    triangles_by_edge: HashMap<(usize, usize), Vec<usize>>, // edge -> triangle indices
}

impl OptimizedGraph {
    /// Create new optimized graph with triangle indexing
    pub fn new(n: usize, rng: &mut impl Rng) -> Self {
        // Initialize nodes and links as before
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            nodes.push(Node {
                id: i,
            });
        }

        let mut links = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let z: f64 = -(rng.gen_range(0.0..1.0)).ln(); // exponential distribution
                links.push(Link {
                    i,
                    j,
                    z,
                    theta: 0.0,
                    tensor: random_tensor(rng),
                });
            }
        }

        // Pre-compute triangles
        let mut triangles = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    triangles.push((i, j, k));
                }
            }
        }

        // Build edge-to-triangle index
        let mut triangles_by_edge = HashMap::new();
        for (tri_idx, &(i, j, k)) in triangles.iter().enumerate() {
            // Each triangle has 3 edges
            let edges = vec![(i, j), (j, k), (i, k)];
            for edge in edges {
                let canonical_edge = if edge.0 < edge.1 { edge } else { (edge.1, edge.0) };
                triangles_by_edge
                    .entry(canonical_edge)
                    .or_insert_with(Vec::new)
                    .push(tri_idx);
            }
        }

        let mut graph = Self {
            nodes,
            links,
            triangles,
            triangle_sum_cache: 0.0,
            triangles_by_edge,
        };

        // Initialize triangle sum cache
        graph.triangle_sum_cache = graph.compute_full_triangle_sum();
        graph
    }

    /// Compute triangle sum from scratch - O(N³)
    fn compute_full_triangle_sum(&self) -> f64 {
        self.triangles.iter().map(|&(i, j, k)| {
            let t_ij = self.links[self.link_index(i, j)].theta;
            let t_jk = self.links[self.link_index(j, k)].theta;
            let t_ki = self.links[self.link_index(k, i)].theta;
            (t_ij + t_jk + t_ki).cos()
        }).sum()
    }

    /// Get cached triangle sum - O(1)
    pub fn triangle_sum(&self) -> f64 {
        self.triangle_sum_cache
    }

    /// Compute triangle contribution for a single triangle
    fn triangle_contribution(&self, i: usize, j: usize, k: usize) -> f64 {
        let t_ij = self.links[self.link_index(i, j)].theta;
        let t_jk = self.links[self.link_index(j, k)].theta;
        let t_ki = self.links[self.link_index(k, i)].theta;
        (t_ij + t_jk + t_ki).cos()
    }

    /// Compute change in triangle sum when link (i,j) phase changes by delta_theta - O(N)
    fn triangle_sum_delta_phase(&self, i: usize, j: usize, delta_theta: f64) -> f64 {
        let canonical_edge = if i < j { (i, j) } else { (j, i) };
        
        if let Some(triangle_indices) = self.triangles_by_edge.get(&canonical_edge) {
            let mut delta = 0.0;
            
            for &tri_idx in triangle_indices {
                let (ti, tj, tk) = self.triangles[tri_idx];
                
                // Old contribution
                let old_contrib = self.triangle_contribution(ti, tj, tk);
                
                // New contribution (temporarily update phase)
                let link_idx = self.link_index(i, j);
                let old_theta = self.links[link_idx].theta;
                
                // Calculate new triangle sum with updated phase
                let t_ij = if (ti == i && tj == j) || (ti == j && tj == i) {
                    old_theta + delta_theta
                } else if (tj == i && tk == j) || (tj == j && tk == i) {
                    old_theta + delta_theta
                } else {
                    self.links[self.link_index(ti, tj)].theta
                };
                
                let t_jk = if (tj == i && tk == j) || (tj == j && tk == i) {
                    old_theta + delta_theta
                } else if (ti == i && tk == j) || (ti == j && tk == i) {
                    old_theta + delta_theta
                } else {
                    self.links[self.link_index(tj, tk)].theta
                };
                
                let t_ki = if (tk == i && ti == j) || (tk == j && ti == i) {
                    old_theta + delta_theta
                } else if (ti == i && tj == j) || (ti == j && tj == i) {
                    old_theta + delta_theta
                } else {
                    self.links[self.link_index(tk, ti)].theta
                };
                
                let new_contrib = (t_ij + t_jk + t_ki).cos();
                delta += new_contrib - old_contrib;
            }
            
            delta
        } else {
            0.0
        }
    }

    /// Optimized Metropolis step with incremental updates
    pub fn metropolis_step_optimized(
        &mut self,
        alpha: f64,
        beta: f64,
        delta_z: f64,
        delta_theta: f64,
        rng: &mut impl Rng,
    ) -> StepInfo {
        // Choose random link
        let link_index = rng.gen_range(0..self.links.len());
        let link = &self.links[link_index];
        let (i, j) = (link.i, link.j);
        
        // Decide update type
        let phase_only = delta_z == 0.0;
        let do_z_update = !phase_only && rng.gen_bool(0.5);
        
        if do_z_update {
            // Z-update: only affects entropy term, not triangles
            let old_z = link.z;
            let new_z = (old_z + rng.gen_range(-delta_z..=delta_z)).max(0.001);
            let old_w = (-old_z).exp();
            let new_w = (-new_z).exp();
            
            // Entropy change
            let delta_entropy = new_w * new_z - old_w * old_z;
            let delta_s = beta * delta_entropy;
            
            // Accept/reject
            let accept = if delta_s <= 0.0 {
                true
            } else {
                rng.gen_range(0.0..1.0) < (-delta_s).exp()
            };
            
            if accept {
                self.links[link_index].z = new_z;
                StepInfo {
                    accept: true,
                    delta_w: new_w - old_w,
                    delta_cos: 0.0, // No change to triangles
                }
            } else {
                StepInfo {
                    accept: false,
                    delta_w: 0.0,
                    delta_cos: 0.0,
                }
            }
        } else {
            // Phase update: use incremental triangle sum
            let link = &self.links[link_index];
            let (i, j) = (link.i, link.j);
            let d_theta = rng.gen_range(-delta_theta..=delta_theta);
            
            // Calculate triangle sum change - O(N) instead of O(N³)
            let delta_triangle = self.triangle_sum_delta_phase(i, j, d_theta);
            let delta_s = alpha * delta_triangle;
            
            // Accept/reject
            let accept = if delta_s <= 0.0 {
                true
            } else {
                rng.gen_range(0.0..1.0) < (-delta_s).exp()
            };
            
            if accept {
                self.links[link_index].theta += d_theta;
                self.triangle_sum_cache += delta_triangle;
                
                // For cosine observable: compute change in cos(theta) for single link
                let link = &self.links[link_index];
                let old_cos = link.theta.cos();
                let new_cos = (link.theta + d_theta).cos();
                let w = link.w();
                
                StepInfo {
                    accept: true,
                    delta_w: 0.0,
                    delta_cos: w * (new_cos - old_cos),
                }
            } else {
                StepInfo {
                    accept: false,
                    delta_w: 0.0,
                    delta_cos: 0.0,
                }
            }
        }
    }

    /// Number of nodes in the graph
    pub fn n(&self) -> usize {
        self.nodes.len()
    }
    
    /// Link index calculation (same as original)
    pub fn link_index(&self, i: usize, j: usize) -> usize {
        let n = self.n();
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        i * (n - 1) - i * (i + 1) / 2 + (j - i - 1)
    }

    /// Action calculation using cached triangle sum
    pub fn action(&self, alpha: f64, beta: f64) -> f64 {
        beta * self.entropy_action() + alpha * self.triangle_sum_cache
    }

    /// Entropy action (same as original)
    pub fn entropy_action(&self) -> f64 {
        self.links.iter().map(|link| {
            let w = (-link.z).exp();
            w * link.z
        }).sum::<f64>()
    }
}

#[derive(Debug)]
pub struct StepInfo {
    pub accept: bool,
    pub delta_w: f64,
    pub delta_cos: f64,
}

// Node structure matching the actual implementation
#[derive(Debug, Clone)]
struct Node {
    id: usize,
}

// Link structure matching the actual implementation
#[derive(Debug, Clone)]
struct Link {
    i: usize,
    j: usize,
    z: f64,
    theta: f64,
    tensor: [[[f64; 3]; 3]; 3],
}

impl Link {
    /// Get the weight w = exp(-z)
    pub fn w(&self) -> f64 {
        (-self.z).exp()
    }
}

// Generate random 3D tensor matching actual implementation
fn random_tensor<R: Rng>(rng: &mut R) -> [[[f64; 3]; 3]; 3] {
    let mut t = [[[0.0; 3]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                t[a][b][c] = rng.gen_range(-1.0..1.0);
            }
        }
    }
    t
}