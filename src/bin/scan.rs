//! Scan over a range of β (inverse temperature) and/or α (triangle coeff).
//! Writes scan_results.csv with one row per run.
//
//  Usage: cargo run --bin scan
//
use rc_sim::graph::Graph;
use std::fs::File;
use std::io::Write;

fn main() {
    // ------------------------------------------------------------
    // Scan parameters
    // ------------------------------------------------------------
    let n_nodes        = 8;
    let n_steps        = 10_000;
    let equil_steps    = 4_000;                    // discard first 4k as warm-up
    let report_every   = 1_000;                    // silent run

    // Range of β to test.  Adjust as you like:
    let beta_vals: Vec<f64> = (0..=20).map(|i| 0.5 + i as f64 * 0.25).collect();
    // Single α value here.  Later you can scan α similarly.
    let alpha = 1.0;

    // ------------------------------------------------------------
    let mut csv = File::create("scan_results.csv")
        .expect("cannot create scan_results.csv");
    writeln!(
        csv,
        "beta,alpha,avg_w,avg_cos_theta,S_entropy,S_triangle,action"
    ).unwrap();

    println!("# β-scan   nodes = {n_nodes}, steps = {n_steps}, α = {alpha}");

    // ------------------------------------------------------------
    for beta in beta_vals {
        let mut g = Graph::complete_random(n_nodes);

        // --- auto-tuning proposal widths ---
        let mut tuner_w  = Tuner::new(0.1, 200, 0.3, 0.05);
        let mut tuner_th = Tuner::new(0.2, 200, 0.3, 0.05);

        // --- Metropolis loop ---
        for step in 1..=n_steps {
            let accepted =
                g.metropolis_step(beta, tuner_w.delta, tuner_th.delta);
            tuner_w.update(accepted);
            tuner_th.update(accepted);

            if step % report_every == 0 {
                // silent; you can print progress if desired
            }
        }

        // ------- measure at end of run -------
        // simple averages
        let avg_w   = g.sum_weights() / g.m() as f64;
        let avg_cos = g.links.iter().map(|l| l.theta.cos()).sum::<f64>()
                        / g.m() as f64;
        let s_entropy  = g.entropy_action();
        let s_triangle = g.triangle_action(alpha);
        let act        = g.action();

        println!("β={:>5.2}  ⟨w⟩={:>6.3}  ⟨cosθ⟩={:>6.3}", beta, avg_w, avg_cos);

        writeln!(
            csv, "{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
            beta, alpha, avg_w, avg_cos, s_entropy, s_triangle, act
        ).unwrap();
    }

    println!("Results written to scan_results.csv");
}

// ----------------------------------------------------------------
// Tiny tuner helper (same as in mc_demo.rs; duplicated for brevity)
// ----------------------------------------------------------------
struct Tuner { delta: f64, acc: usize, window: usize, target: f64, band: f64 }
impl Tuner {
    fn new(delta: f64, window: usize, target: f64, band: f64) -> Self {
        Self { delta, acc: 0, window, target, band }
    }
    fn update(&mut self, accepted: bool) {
        if accepted { self.acc += 1; }
        if self.acc + 1 == self.window {
            let rate = self.acc as f64 / self.window as f64;
            if rate > self.target + self.band { self.delta *= 1.1; }
            else if rate < self.target - self.band { self.delta *= 0.9; }
            self.acc = 0;
        }
    }
}

