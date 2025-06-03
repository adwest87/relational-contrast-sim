//! Minimal graph data structure for Relational Contrast
//! • Node: identified by an integer id
//! • Link: unordered pair (i, j) with weight w
use crate::projector::{aib_project, frobenius_norm};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};

/// One vertex in the network
#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
}

#[derive(Debug, Clone)]
pub struct Link {
    pub i: usize,
    pub j: usize,
    pub w: f64,                // weight
    pub theta: f64,            // NEW: U(1) phase angle
    pub tensor: [[[f64; 3]; 3]; 3],
}



/// A simple undirected graph
#[derive(Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
    pub dt: f64,
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

#[derive(Debug)]
pub enum Proposal {
    Weight { idx: usize, old: f64 },
    Phase  { idx: usize, old: f64 },
}


impl Graph {
    pub fn propose_update(&mut self, delta_w: f64, delta_theta: f64) -> Proposal {
        let mut rng = rand::thread_rng();
        let link_index = rng.gen_range(0..self.links.len());

        if delta_w == 0.0 {
            // --- weights frozen: do phase update only ---
            let mut rng = rand::thread_rng();
            let link_index = rng.gen_range(0..self.links.len());
            let dtheta: f64 = Uniform::new_inclusive(-delta_theta, delta_theta).sample(&mut rng);
            let old = self.links[link_index].theta;
            self.links[link_index].theta = old + dtheta;
            Proposal::Phase { idx: link_index, old }
        } else if rng.gen_bool(0.5) {
            // --- normal weight update ---
            let mut rng = rand::thread_rng();
            let link_index = rng.gen_range(0..self.links.len());
            let eps: f64 = Uniform::new_inclusive(-delta_w, delta_w).sample(&mut rng);
            let old = self.links[link_index].w;
            self.links[link_index].w = old * eps.exp();
            Proposal::Weight { idx: link_index, old }
        } else {
            // --- phase update ---
            let mut rng = rand::thread_rng();
            let link_index = rng.gen_range(0..self.links.len());
            let dtheta: f64 = Uniform::new_inclusive(-delta_theta, delta_theta).sample(&mut rng);
            let old = self.links[link_index].theta;
            self.links[link_index].theta = old + dtheta;
            Proposal::Phase { idx: link_index, old }
        }
    }

    /// Construct a complete graph on `n` nodes with random
    /// weights w ∈ (0, 1].
    pub fn complete_random(n: usize) -> Self {
        let mut rng = rand::thread_rng();
        let nodes = (0..n).map(|id| Node { id }).collect();

        let mut links = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                links.push(Link {
                    i,
                    j,
                    w: rng.gen_range(0.000_001..=1.0),
                    theta: rng.gen_range(-std::f64::consts::PI..=std::f64::consts::PI),
                    tensor: random_tensor(&mut rng),
                });
            }
        }

        let dt = 1.0;               // default time increment

        Self { nodes, links, dt }
    }

    /// Project every link tensor with the AIB projector.
    /// Returns the total Frobenius norm *before* and *after* so you
    /// can see how much content was removed.
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

    /// Convenience: number of nodes
    pub fn n(&self) -> usize {
        self.nodes.len()
    }

    /// Convenience: number of links
    pub fn m(&self) -> usize {
        self.links.len()
    }
    
    /// Entropy term  S = Σ w ln w  (Dougal-invariant)
    pub fn entropy_action(&self) -> f64 {
        self.links
            .iter()
            .map(|link| link.w * link.w.ln())
            .sum()
    }

    /// Multiply every link weight by the same factor λ
    pub fn rescale_weights(&mut self, lambda: f64) {
        for link in &mut self.links {
            link.w *= lambda;
        }
    }
    /// Sum of all link weights  Σ w
    pub fn sum_weights(&self) -> f64 {
        self.links.iter().map(|l| l.w).sum()
    }

    /// The Dougal-invariant combination
    /// I = (S - ln(dt) Σ w) / dt
    pub fn invariant_action(&self) -> f64 {
        let s = self.entropy_action();
        let sum_w = self.sum_weights();
        (s - self.dt.ln() * sum_w) / self.dt
    }

    /// Rescale all weights *and* dt by λ  (Dougal transformation)
    pub fn rescale(&mut self, lambda: f64) {
        for link in &mut self.links {
            link.w *= lambda;
        }
        self.dt *= lambda;
    }
    /// Current action. For now this is just the Dougal-invariant
    /// entropy combination. Later you'll add holonomy, projector, etc.
    pub fn action(&self) -> f64 {
        let i_term = self.invariant_action();
        let triangle_term = self.triangle_action(1.0); // α = 1 for now
        i_term + triangle_term
    }

    /// Propose: pick one link at random and multiply its weight by e^{ε},
    /// where ε ~ U[-δ, +δ]. Returns (link_index, old_w, new_w).
    pub fn propose_weight_update(&mut self, delta: f64) -> (usize, f64, f64) {
        let mut rng = rand::thread_rng();
        let link_index = rng.gen_range(0..self.links.len());

        // perturbation
        let eps: f64 = Uniform::new_inclusive(-delta, delta).sample(&mut rng);
        let old_w = self.links[link_index].w;
        let new_w = old_w * eps.exp();

        self.links[link_index].w = new_w;
        (link_index, old_w, new_w)
    }
    /// Perform one Metropolis step at inverse temperature β.
    /// Picks a random link, perturbs its weight, and accepts/rejects.
    pub fn metropolis_step(&mut self, beta: f64, delta_w: f64, delta_theta: f64) -> bool {
        let s_before = self.action();

        let proposal = self.propose_update(delta_w, delta_theta);

        let s_after = self.action();
        let delta_s = s_after - s_before;

        let accept = delta_s <= 0.0 || {
            let mut rng = rand::thread_rng();
            rng.gen_range(0.0..1.0) < (-beta * delta_s).exp()
        };

        if !accept {
            match proposal {
                Proposal::Weight { idx, old } => self.links[idx].w = old,
                Proposal::Phase  { idx, old } => self.links[idx].theta = old,
            }
        }
        accept
    }

    /// Iterate over all unordered triangles (i < j < k).
    pub fn triangles(&self) -> impl Iterator<Item = (usize, usize, usize)> + '_ {   
        let n = self.n();
        (0..n).flat_map(move |i| {
            ((i + 1)..n).flat_map(move |j| {
                ((j + 1)..n).map(move |k| (i, j, k))
            })
        })
    }
    /// Triangle term  S_Δ  with coefficient α
    pub fn triangle_action(&self, alpha: f64) -> f64 {
        let mut sum = 0.0;
        for (i, j, k) in self.triangles() {
            let t_ij = self.links[self.link_index(i, j)].theta;
            let t_jk = self.links[self.link_index(j, k)].theta;
            let t_ki = self.links[self.link_index(k, i)].theta;

            let loop_theta = t_ij + t_jk + t_ki;
            let trace = 3.0 * loop_theta.cos();

            sum += trace;
        }
        alpha * sum
    }

    /// Return the index in self.links for the unordered pair (i,j)
    fn link_index(&self, i: usize, j: usize) -> usize {
        // Works only for complete graph with ordering i<j.
        // index = i*(n-1) - i*(i+1)/2 + (j-i-1)
        let n = self.n();
        let (i, j) = if i < j { (i, j) } else { (j, i) };
        i * (n - 1) - (i * (i + 1)) / 2 + (j - i - 1)
    }
}    
