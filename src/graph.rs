// Modified graph.rs with z-variable implementation

use crate::projector::{aib_project, frobenius_norm};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;

/// A vertex in the network.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
}

/// An undirected edge with z = -ln(w) and a U(1) phase.
#[derive(Debug, Clone)]
pub struct Link {
    pub i: usize,
    pub j: usize,
    pub z: f64,  // z = -ln(w), so w = exp(-z)
    pub theta: f64,
    pub tensor: [[[f64; 3]; 3]; 3],
}

impl Link {
    /// Get the weight w = exp(-z)
    pub fn w(&self) -> f64 {
        (-self.z).exp()
    }
    
    /// Set weight by updating z
    pub fn set_w(&mut self, w: f64) {
        self.z = -w.ln();
    }
}

/// A simple undirected complete graph.
#[derive(Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
    pub dt: f64,
    triangles: Vec<(usize, usize, usize)>,
}

/// Helper: generate a random third‑rank tensor with entries in (‑1, 1).
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

/// A proposed update during Metropolis.
#[derive(Debug)]
enum Proposal {
    ZUpdate { idx: usize, old_z: f64, new_z: f64 },
    Phase  { idx: usize, old_th: f64, new_th: f64 },
}

/// Returned by `metropolis_step`, allows O(1) book‑keeping in the driver.
#[derive(Debug, Clone, Copy)]
pub struct StepInfo {
    pub accepted: bool,
    pub delta_w:  f64,   // change in Σ w
    pub delta_cos: f64,  // change in Σ cos θ
}

impl Graph {
    /// Build a complete graph on `n` nodes with random weights and phases,
    /// using a caller‑supplied RNG (preferred for reproducibility).
    pub fn complete_random_with(rng: &mut impl Rng, n: usize) -> Self {
        let nodes = (0..n).map(|id| Node { id }).collect::<Vec<_>>();

        let mut links = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                // Generate z uniformly in a reasonable range
                let z = rng.gen_range(0.001..10.0);  // This gives w ∈ [exp(-10), exp(-0.001)]
                links.push(Link {
                    i,
                    j,
                    z,
                    theta: 0.0,
                    tensor: random_tensor(rng),
                });
            }
        }

        // Pre‑compute all unordered triangles (i < j < k).
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
        }
    }

    /// Convenience wrapper that uses `thread_rng`.
    pub fn complete_random(n: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self::complete_random_with(&mut rng, n)
    }

    /// Number of vertices.
    #[inline(always)]
    pub fn n(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    #[inline(always)]
    pub fn m(&self) -> usize {
        self.links.len()
    }

    /// Σ w over all links.
    pub fn sum_weights(&self) -> f64 {
        self.links.iter().map(|l| l.w()).sum()
    }

    /// Σ cos θ over all links.
    pub fn links_cos_sum(&self) -> f64 {
        self.links.iter().map(|l| l.theta.cos()).sum()
    }

    /// Entropy term `S = Σ w ln w` using z-variables
    /// S = Σ exp(-z) * (-z) = -Σ z * exp(-z)
    pub fn entropy_action(&self) -> f64 {
        self.links.iter().map(|l| -l.z * l.w()).sum()
    }

    /// Invariant combination `I = (S - ln(dt) Σ w) / dt`.
    pub fn invariant_action(&self) -> f64 {
        let s = self.entropy_action();
        let sum_w = self.sum_weights();
        (s - self.dt.ln() * sum_w) / self.dt
    }

    /// Rescale all link weights and `dt` by λ (Dougal transformation).
    pub fn rescale(&mut self, lambda: f64) {
        for link in &mut self.links {
            link.z -= lambda.ln();  // w → λw means z → z - ln(λ)
        }
        self.dt *= lambda;
    }

    /// S = β Σ w ln w  +  α Σ triangle
    pub fn action(&self, alpha: f64, beta: f64) -> f64 {
        beta * self.entropy_action() + alpha * self.triangle_sum()
    }

    /// ∑_{triangles} 3 cos(θ_ij+θ_jk+θ_ki)  (no coupling prefactor)
    pub fn triangle_sum(&self) -> f64 {
        self.triangles.iter().map(|&(i, j, k)| {
            let t_ij = self.links[self.link_index(i, j)].theta;
            let t_jk = self.links[self.link_index(j, k)].theta;
            let t_ki = self.links[self.link_index(k, i)].theta;
            3.0 * (t_ij + t_jk + t_ki).cos()
        }).sum()
    }

    /// S_Δ(α) = α × triangle_sum
    pub fn triangle_action(&self, alpha: f64) -> f64 {
        alpha * self.triangle_sum()
    }

    /// number of (unordered) triangles
    pub fn n_tri(&self) -> usize {
        self.n() * (self.n() - 1) * (self.n() - 2) / 6
    }

    /// ΣΔ divided by number of triangles
    pub fn triangle_sum_norm(&self) -> f64 {
        self.triangle_sum() / self.n_tri() as f64
    }

    /// Index in `self.links` for the unordered pair (i, j).
    /// Works only for a complete graph.
    pub fn link_index(&self, i: usize, j: usize) -> usize {
        let n = self.n();
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        i * (n - 1) - i * (i + 1) / 2 + (j - i - 1)
    }

    /// Create either a z-update or phase perturbation.
    fn propose_update(
        &mut self,
        delta_z: f64,
        delta_theta: f64,
        rng: &mut impl Rng,
    ) -> Proposal {
        let link_index = rng.gen_range(0..self.links.len());

        let phase_only = delta_z == 0.0;
        let do_z_update = !phase_only && rng.gen_bool(0.5);

        if do_z_update {
            // Additive update in z-space
            let dz: f64 = Uniform::new_inclusive(-delta_z, delta_z).sample(rng);
            let old_z = self.links[link_index].z;
            let new_z = (old_z + dz).max(0.001);  // Keep z positive (w < 1)
            self.links[link_index].z = new_z;
            Proposal::ZUpdate { idx: link_index, old_z, new_z }
        } else {
            let dtheta: f64 = Uniform::new_inclusive(-delta_theta, delta_theta).sample(rng);
            let old_th = self.links[link_index].theta;
            let new_th = old_th + dtheta;
            self.links[link_index].theta = new_th;
            Proposal::Phase { idx: link_index, old_th, new_th }
        }
    }

    /// Perform one Metropolis step with z-variable updates
    pub fn metropolis_step(
        &mut self,
        beta: f64,
        alpha: f64,
        delta_z: f64,
        delta_theta: f64,
        rng: &mut impl Rng,
    ) -> StepInfo {
        let s_before = self.action(alpha, beta);
        let proposal = self.propose_update(delta_z, delta_theta, rng);
        let s_after  = self.action(alpha, beta);
        let delta_s  = s_after - s_before;
        let accept = if delta_s <= 0.0 {
            true
        } else {
            rng.gen::<f64>() < (-delta_s).exp()
        };

        if accept {
            match proposal {
                Proposal::ZUpdate { idx, old_z, new_z } => {
                    let old_w = (-old_z).exp();
                    let new_w = (-new_z).exp();
                    StepInfo {
                        accepted: true,
                        delta_w:  new_w - old_w,
                        delta_cos: 0.0,
                    }
                },
                Proposal::Phase { old_th, new_th, .. } => StepInfo {
                    accepted: true,
                    delta_w:  0.0,
                    delta_cos: new_th.cos() - old_th.cos(),
                },
            }
        } else {
            // Revert.
            match proposal {
                Proposal::ZUpdate { idx, old_z, .. } => self.links[idx].z = old_z,
                Proposal::Phase  { idx, old_th, .. } => self.links[idx].theta = old_th,
            }
            StepInfo { accepted: false, delta_w: 0.0, delta_cos: 0.0 }
        }
    }

    /// Project every link tensor with the AIB projector.
    /// Returns the Frobenius norm before and after projection.
    pub fn project_all(&mut self) -> (f64, f64) {
        let mut norm_before = 0.0;
        let mut norm_after  = 0.0;

        for link in &mut self.links {
            norm_before += frobenius_norm(&link.tensor);
            link.tensor  = aib_project(link.tensor);
            norm_after  += frobenius_norm(&link.tensor);
        }
        (norm_before, norm_after)
    }

    /// Iterate over all unordered triangles (i < j < k).
    pub fn triangles(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.triangles.iter().copied()
    }
    
    /// Get minimum weight in the graph
    pub fn min_weight(&self) -> f64 {
        self.links.iter()
            .map(|l| l.w())
            .fold(f64::INFINITY, f64::min)
    }
}