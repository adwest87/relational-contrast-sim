//! Minimal graph data structure for Relational Contrast
//! • Node: identified by an integer id
//! • Link: unordered pair (i, j) with weight w
use crate::projector::{aib_project, frobenius_norm};
use rand::Rng;

/// One vertex in the network
#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
}

#[derive(Debug, Clone)]
pub struct Link {
    pub i: usize,
    pub j: usize,
    pub w: f64,                              // weight
    pub tensor: [[[f64; 3]; 3]; 3],          // raw 3×3×3 data
    pub holonomy: [[f64; 3]; 3],             // 3×3 gauge matrix (placeholder)
}


/// A simple undirected graph
#[derive(Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
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

fn identity_matrix() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0],
     [0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0]]
}


impl Graph {
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
                    tensor: random_tensor(&mut rng),
                    holonomy: identity_matrix(),
                });
            }
        }

        Self { nodes, links }
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
}

