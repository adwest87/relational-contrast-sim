/// Record time-series observables and compute graph properties
use crate::graph::{Graph, Link};
use rand::Rng;

#[derive(Default)]
pub struct Recorder {
    pub cos_theta: Vec<f64>,
}

impl Recorder {
    /// Push a new measurement: average cos θ over all links
    pub fn push(&mut self, links: &[Link]) {
        let avg = links.iter()
            .map(|l| l.theta.cos())
            .sum::<f64>() / links.len() as f64;
        self.cos_theta.push(avg);
    }
}

/// Compute spectral dimension using random walk return probability
pub fn compute_spectral_dimension(graph: &Graph, max_steps: usize, n_walks: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let n = graph.n();
    
    // Build adjacency information
    let mut adj: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
    for link in &graph.links {
        let w = link.w();
        adj[link.i].push((link.j, w));
        adj[link.j].push((link.i, w));
    }
    
    // Normalize transition probabilities
    for neighbors in &mut adj {
        let total: f64 = neighbors.iter().map(|(_, w)| w).sum();
        if total > 0.0 {
            for (_, w) in neighbors {
                *w /= total;
            }
        }
    }
    
    // Count returns for different walk lengths
    let mut returns = vec![0usize; max_steps + 1];
    
    for _ in 0..n_walks {
        let start = rng.gen_range(0..n);
        let mut current = start;
        
        for step in 1..=max_steps {
            // Take a step
            if adj[current].is_empty() {
                break;
            }
            
            let r: f64 = rng.gen();
            let mut cumsum = 0.0;
            
            for &(next, prob) in &adj[current] {
                cumsum += prob;
                if r < cumsum {
                    current = next;
                    break;
                }
            }
            
            if current == start {
                returns[step] += 1;
            }
        }
    }
    
    // Estimate spectral dimension from P(t) ~ t^(-ds/2)
    // Use log-log regression in the range [10, max_steps/2]
    let mut points = Vec::new();
    for t in 10..(max_steps/2) {
        if returns[t] > 0 {
            let p_t = returns[t] as f64 / n_walks as f64;
            points.push(((t as f64).ln(), p_t.ln()));
        }
    }
    
    if points.len() < 5 {
        return 0.0; // Not enough data
    }
    
    // Linear regression on log-log data
    let n_points = points.len() as f64;
    let sum_x: f64 = points.iter().map(|(x, _)| *x).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| *y).sum();
    let sum_xx: f64 = points.iter().map(|(x, _)| *x * *x).sum();
    let sum_xy: f64 = points.iter().map(|(x, y)| *x * *y).sum();
    
    let slope = (n_points * sum_xy - sum_x * sum_y) / (n_points * sum_xx - sum_x * sum_x);
    
    // ds = -2 * slope
    -2.0 * slope
}