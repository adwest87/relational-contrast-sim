//! Minimal graph data structure for the Relational‑Contrast model
//! • Node  : identified by an integer id
//! • Link  : unordered pair (i, j) with weight w and U(1) phase θ
//! • Graph : complete, undirected, with pre‑computed triangle list
//! ------------------------------------------------------------------

use crate::projector::{aib_project, frobenius_norm};
use rand::distributions::{Distribution, Uniform};
use rand::Rng;          // `Rng` brings `.gen()` into scope

/// A vertex in the network.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
}

/// An undirected edge with a weight and a U(1) phase.
#[derive(Debug, Clone)]
pub struct Link {
    pub i: usize,
    pub j: usize,
    pub w: f64,
    pub theta: f64,
    pub tensor: [[[f64; 3]; 3]; 3],
}

/// A simple undirected complete graph.
#[derive(Debug)]
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
    Weight { idx: usize, old_w: f64, new_w: f64 },
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
    // ---------------------------------------------------------------------
    // Constructors
    // ---------------------------------------------------------------------

    /// Build a complete graph on `n` nodes with random weights and phases,
    /// using a caller‑supplied RNG (preferred for reproducibility).
    pub fn complete_random_with(rng: &mut impl Rng, n: usize) -> Self {
        let nodes = (0..n).map(|id| Node { id }).collect::<Vec<_>>();

        let mut links = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                links.push(Link {
                    i,
                    j,
                    w: rng.gen_range(0.000_001..=1.0),
                    theta: 0.0,                       // unit holonomy by default
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

    // ---------------------------------------------------------------------
    // Cheap accessors
    // ---------------------------------------------------------------------

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
        self.links.iter().map(|l| l.w).sum()
    }

    /// Σ cos θ over all links.
    pub fn links_cos_sum(&self) -> f64 {
        self.links.iter().map(|l| l.theta.cos()).sum()
    }

    // ---------------------------------------------------------------------
    // Dougal‑invariant observables
    // ---------------------------------------------------------------------

    /// Entropy term `S = Σ w ln w`.
    pub fn entropy_action(&self) -> f64 {
        self.links.iter().map(|l| l.w * l.w.ln()).sum()
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
            link.w *= lambda;
        }
        self.dt *= lambda;
    }

    // ---------------------------------------------------------------------
    // Action and geometry
    // ---------------------------------------------------------------------

    /// S = β Σ w ln w  +  α Σ triangle
    pub fn action(&self, alpha: f64, beta: f64) -> f64 {
        beta * self.entropy_action()          // β Σ w ln w
      + alpha * self.triangle_sum()           // α Σ triangle
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

    /// Index in `self.links` for the unordered pair (i, j).
    /// Works only for a complete graph.
    fn link_index(&self, i: usize, j: usize) -> usize {
        let n = self.n();
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        i * (n - 1) - i * (i + 1) / 2 + (j - i - 1)
    }

    // ---------------------------------------------------------------------
    // Metropolis machinery
    // ---------------------------------------------------------------------

    /// Create either a weight or phase perturbation.
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

    /// Perform one Metropolis step.  The caller supplies the RNG to keep
    /// random streams reproducible.  `StepInfo` tells the driver how Σ w
    /// and Σ cos θ changed, enabling O(1) updates.
    pub fn metropolis_step(
        &mut self,
        beta: f64,
        alpha: f64,
        delta_w: f64,
        delta_theta: f64,
        rng: &mut impl Rng,      // `Rng` so `.gen()` is available
    ) -> StepInfo {
        let s_before = self.action(alpha, beta);
        let proposal = self.propose_update(delta_w, delta_theta, rng);
        let s_after  = self.action(alpha, beta);
        let delta_s  = s_after - s_before;
        let accept = if delta_s <= 0.0 {
            true
        } else {
            rng.gen::<f64>() < (-delta_s).exp()
        };

        if accept {
            match proposal {
                Proposal::Weight { old_w, new_w, .. } => StepInfo {
                    accepted: true,
                    delta_w:  new_w - old_w,
                    delta_cos: 0.0,
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
                Proposal::Weight { idx, old_w, .. } => self.links[idx].w = old_w,
                Proposal::Phase  { idx, old_th, .. } => self.links[idx].theta = old_th,
            }
            StepInfo { accepted: false, delta_w: 0.0, delta_cos: 0.0 }
        }
    }

    // ---------------------------------------------------------------------
    // Tensor projection (unchanged)
    // ---------------------------------------------------------------------

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

    // ---------------------------------------------------------------------
    // Utility iterations
    // ---------------------------------------------------------------------

    /// Iterate over all unordered triangles (i < j < k).
    pub fn triangles(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {
        self.triangles.iter().copied()
    }
}
