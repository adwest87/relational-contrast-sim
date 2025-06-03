//! Minimal graph data structure for Relational Contrast
//! • Node: identified by an integer id
//! • Link: unordered pair (i, j) with weight w

use rand::Rng;

/// One vertex in the network
#[derive(Debug, Clone)]
pub struct Node {
    pub id: usize,
}

/// One undirected link (i, j) with weight w
#[derive(Debug, Clone)]
pub struct Link {
    pub i: usize,
    pub j: usize,
    pub w: f64,
}

/// A simple undirected graph
#[derive(Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
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
                    w: rng.gen_range(0.000_001..=1.0), // avoid exact zero
                });
            }
        }

        Self { nodes, links }
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

