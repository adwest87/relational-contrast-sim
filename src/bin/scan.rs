//! High-resolution β-scan with replicas and error bars
//! ---------------------------------------------------
//!  – 16-node complete graph (120 links)
//!  – 50k MC steps, first 20k discarded (equilibration)
//!  – 5 replicas per β
//!  – β range 0.5 … 5.0 in steps of 0.1
//!  – Records <w>, Var(w), <cosθ>, Var(cosθ) and
//!    susceptibility  χ = N_links * Var(cosθ)
//!
//!  Run with:  cargo run --bin scan
//! ---------------------------------------------------
use rc_sim::graph::Graph;
use std::fs::File;
use std::io::Write;
use rand::rngs::StdRng;
use rand::{SeedableRng, Rng};

// ---------- Welford online mean & variance ----------
#[derive(Default, Clone)]
struct OnlineStats {
    n: u64,
    mean: f64,
    m2: f64,
}
impl OnlineStats {
    fn push(&mut self, x: f64) {
        self.n += 1;
        let delta  = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2   += delta * delta2;
    }
    fn mean(&self) -> f64 { self.mean }
    fn var(&self) -> f64 { if self.n > 1 { self.m2 / (self.n - 1) as f64 } else { 0.0 } }
}

fn main() {
    // ---- scan parameters ----------------------------------------
    let n_nodes       = 16;                               // bigger lattice
    let links_per_g   = n_nodes * (n_nodes - 1) / 2;      // 120
    let n_steps       = 50_000;
    let equil_steps   = 20_000;
    let beta_vals: Vec<f64> = (5..=50).map(|i| 0.5 + 0.1 * i as f64).collect(); // 0.5 .. 5.0
    let n_rep         = 5;                                // replicas
    let alpha         = 1.0;
    // -------------------------------------------------------------

    let mut csv = File::create("scan_results.csv")
        .expect("cannot create scan_results.csv");
    writeln!(
        csv,
        "beta,alpha,mean_w,std_w,mean_cos,std_cos,susceptibility"
    ).unwrap();

    println!("# β-scan  N_nodes={n_nodes}  steps={n_steps}  replicas={n_rep}");

    for &beta in &beta_vals {
        // accumulators over replicas
        let mut stats_w   = OnlineStats::default();
        let mut stats_cos = OnlineStats::default();

        for rep in 0..n_rep {
            // deterministic seed so runs are reproducible
            let seed = [rep as u8; 32];
            let mut rng: StdRng = SeedableRng::from_seed(seed);
            let mut g = Graph::complete_random(n_nodes);

            // local tuners
            let mut tuner_w  = Tuner::new(0.10, 200, 0.30, 0.05);
            let mut tuner_th = Tuner::new(0.20, 200, 0.30, 0.05);

            for step in 1..=n_steps {
                let accepted = g.metropolis_step(
                    beta,
                    tuner_w.delta,
                    tuner_th.delta,
                );
                tuner_w.update(accepted);
                tuner_th.update(accepted);

                // collect measurements only after equilibration
                if step > equil_steps {
                    let avg_w   = g.sum_weights() / g.m() as f64;
                    let avg_cos = g.links.iter()
                        .map(|l| l.theta.cos())
                        .sum::<f64>() / g.m() as f64;

                    stats_w.push(avg_w);
                    stats_cos.push(avg_cos);
                }
            }
            println!(
                "β={beta:>4.1}  rep {rep} done  ⟨w⟩={:.3}  ⟨cosθ⟩={:.3}",
                stats_w.mean(), stats_cos.mean()
            );
        }

        let chi = links_per_g as f64 * stats_cos.var();

        writeln!(
            csv,
            "{:.2},{:.2},{:.6},{:.6},{:.6},{:.6},{:.6}",
            beta,
            alpha,
            stats_w.mean(), stats_w.var().sqrt(),
            stats_cos.mean(), stats_cos.var().sqrt(),
            chi
        ).unwrap();
    }
    println!("Scan complete → scan_results.csv");
}

// ---------- same little Tuner struct as before ----------
struct Tuner { delta: f64, acc: usize, win: usize, tgt: f64, band: f64 }
impl Tuner {
    fn new(delta: f64, win: usize, tgt: f64, band: f64) -> Self {
        Self { delta, acc: 0, win, tgt, band }
    }
    fn update(&mut self, accepted: bool) {
        if accepted { self.acc += 1; }
        if self.acc + 1 == self.win {
            let r = self.acc as f64 / self.win as f64;
            if r > self.tgt + self.band { self.delta *= 1.1; }
            else if r < self.tgt - self.band { self.delta *= 0.9; }
            self.acc = 0;
        }
    }
}
