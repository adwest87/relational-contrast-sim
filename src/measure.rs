/// Record time-series observables (simple example)
#[derive(Default)]
pub struct Recorder {
    pub cos_theta: Vec<f64>,
}

impl Recorder {
    /// Push a new measurement: average cos θ over all links
    pub fn push(&mut self, links: &[crate::graph::Link]) {
        let avg = links.iter()
            .map(|l| l.theta.cos())
            .sum::<f64>() / links.len() as f64;
        self.cos_theta.push(avg);
    }
}
