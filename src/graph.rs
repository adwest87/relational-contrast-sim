// src/graph.rs - Enhanced implementation with missing features

use crate::projector::{aib_project, frobenius_norm};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;
use std::collections::HashMap;

/// A vertex in the network
#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
}

/// An undirected edge with weight, U(1) phase, and contrast tensor
#[derive(Debug, Clone)]
pub struct Link {
    pub i: usize,
    pub j: usize,
    pub w: f64,
    pub theta: f64,
    pub tensor: [[[f64; 3]; 3]; 3],
}

/// Information returned by metropolis_step for O(1) bookkeeping
#[derive(Debug, Clone, Copy)]
pub struct StepInfo {
    pub accepted: bool,
    pub delta_w: f64,   // change in Σ w
    pub delta_cos: f64, // change in Σ cos θ
}

/// Complete undirected graph with contrast structure
#[derive(Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
    pub dt: f64,
    triangles: Vec<(usize, usize, usize)>,
    // New: adjacency structure for efficient queries
    adjacency: HashMap<(usize, usize), usize>, // (i,j) -> link index
}

impl Graph {
    /// Build a complete graph with random weights and phases
    pub fn complete_random_with(rng: &mut impl Rng, n: usize) -> Self {
        let nodes = (0..n).map(|id| Node { id }).collect::<Vec<_>>();
        
        let mut links = Vec::with_capacity(n * (n - 1) / 2);
        let mut adjacency = HashMap::new();
        
        let mut link_idx = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                adjacency.insert((i, j), link_idx);
                adjacency.insert((j, i), link_idx);
                
                links.push(Link {
                    i,
                    j,
                    w: rng.gen_range(0.000_001..=1.0),
                    theta: 0.0,
                    tensor: random_tensor(rng),
                });
                link_idx += 1;
            }
        }

        // Pre-compute all triangles
        let mut triangles = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    triangles.push((i, j, k));
                }
            }
        }

        Self {
            nodes,
            links,
            dt: 1.0,
            triangles,
            adjacency,
        }
    }

    pub fn complete_random(n: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self::complete_random_with(&mut rng, n)
    }

    // Graph properties
    #[inline(always)]
    pub fn n(&self) -> usize { self.nodes.len() }
    
    #[inline(always)]
    pub fn m(&self) -> usize { self.links.len() }
    
    pub fn n_tri(&self) -> usize { self.triangles.len() }

    // Observable calculations
    pub fn sum_weights(&self) -> f64 {
        self.links.iter().map(|l| l.w).sum()
    }

    pub fn links_cos_sum(&self) -> f64 {
        self.links.iter().map(|l| l.theta.cos()).sum()
    }

    pub fn entropy_action(&self) -> f64 {
        self.links.iter().map(|l| l.w * l.w.ln()).sum()
    }

    pub fn invariant_action(&self) -> f64 {
        let s = self.entropy_action();
        let sum_w = self.sum_weights();
        (s - self.dt.ln() * sum_w) / self.dt
    }

    pub fn rescale(&mut self, lambda: f64) {
        for link in &mut self.links {
            link.w *= lambda;
        }
        self.dt *= lambda;
    }

    // Triangle sum for gauge action
    pub fn triangle_sum(&self) -> f64 {
        self.triangles.iter().map(|&(i, j, k)| {
            let t_ij = self.links[self.link_index(i, j)].theta;
            let t_jk = self.links[self.link_index(j, k)].theta;
            let t_ki = self.links[self.link_index(k, i)].theta;
            3.0 * (t_ij + t_jk + t_ki).cos()
        }).sum()
    }

    pub fn triangle_action(&self, alpha: f64) -> f64 {
        alpha * self.triangle_sum()
    }

    pub fn triangle_sum_norm(&self) -> f64 {
        self.triangle_sum() / self.n_tri() as f64
    }

    pub fn action(&self, alpha: f64, beta: f64) -> f64 {
        beta * self.entropy_action() + alpha * self.triangle_sum()
    }

    // Fast link lookup using adjacency map
    pub fn link_index(&self, i: usize, j: usize) -> usize {
        *self.adjacency.get(&(i.min(j), i.max(j))).unwrap()
    }

    // Metropolis implementation
    pub fn metropolis_step(
        &mut self,
        beta: f64,
        alpha: f64,
        delta_w: f64,
        delta_theta: f64,
        rng: &mut impl Rng,
    ) -> StepInfo {
        let s_before = self.action(alpha, beta);
        let proposal = self.propose_update(delta_w, delta_theta, rng);
        let s_after = self.action(alpha, beta);
        let delta_s = s_after - s_before;
        
        let accept = if delta_s <= 0.0 {
            true
        } else {
            rng.gen::<f64>() < (-delta_s).exp()
        };

        match proposal {
            Proposal::Weight { idx, old_w, new_w } => {
                if accept {
                    StepInfo {
                        accepted: true,
                        delta_w: new_w - old_w,
                        delta_cos: 0.0,
                    }
                } else {
                    self.links[idx].w = old_w;
                    StepInfo { accepted: false, delta_w: 0.0, delta_cos: 0.0 }
                }
            }
            Proposal::Phase { idx, old_th, new_th } => {
                if accept {
                    StepInfo {
                        accepted: true,
                        delta_w: 0.0,
                        delta_cos: new_th.cos() - old_th.cos(),
                    }
                } else {
                    self.links[idx].theta = old_th;
                    StepInfo { accepted: false, delta_w: 0.0, delta_cos: 0.0 }
                }
            }
        }
    }

    fn propose_update(
        &mut self,
        delta_w: f64,
        delta_theta: f64,
        rng: &mut impl Rng,
    ) -> Proposal {
        let link_index = rng.gen_range(0..self.links.len());
        
        let phase_only = delta_w == 0.0;
        let do_weight = !phase_only && rng.gen_bool(0.5);

        if do_weight {
            let eps: f64 = Uniform::new_inclusive(-delta_w, delta_w).sample(rng);
            let old_w = self.links[link_index].w;
            let new_w = old_w * eps.exp();
            self.links[link_index].w = new_w;
            Proposal::Weight { idx: link_index, old_w, new_w }
        } else {
            let dtheta: f64 = Uniform::new_inclusive(-delta_theta, delta_theta).sample(rng);
            let old_th = self.links[link_index].theta;
            let new_th = old_th + dtheta;
            self.links[link_index].theta = new_th;
            Proposal::Phase { idx: link_index, old_th, new_th }
        }
    }

    // Tensor projection
    pub fn project_all(&mut self) -> (f64, f64) {
        let mut norm_before = 0.0;
        let mut norm_after = 0.0;

        for link in &mut self.links {
            norm_before += frobenius_norm(&link.tensor);
            link.tensor = aib_project(link.tensor);
            norm_after += frobenius_norm(&link.tensor);
        }
        (norm_before, norm_after)
    }

    // New: Spectral analysis methods
    pub fn laplacian_matrix(&self) -> Vec<Vec<f64>> {
        let n = self.n();
        let mut lap = vec![vec![0.0; n]; n];
        
        for link in &self.links {
            lap[link.i][link.j] -= link.w;
            lap[link.j][link.i] -= link.w;
            lap[link.i][link.i] += link.w;
            lap[link.j][link.j] += link.w;
        }
        lap
    }

    // New: Find minimal weight
    pub fn min_weight(&self) -> f64 {
        self.links.iter().map(|l| l.w).fold(f64::INFINITY, f64::min)
    }

    // Iterator over triangles
    pub fn triangles(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.triangles.iter().copied()
    }
}

#[derive(Debug)]
enum Proposal {
    Weight { idx: usize, old_w: f64, new_w: f64 },
    Phase { idx: usize, old_th: f64, new_th: f64 },
}

fn random_tensor(rng: &mut impl Rng) -> [[[f64; 3]; 3]; 3] {
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