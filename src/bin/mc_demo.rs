//! MC demo: single‑temperature Monte‑Carlo run on a small graph.
//!
//! This version
//!   • uses the new `StepInfo` return to update ⟨w⟩ and ⟨cos θ⟩ in O(1)
//!   • injects a caller‑owned RNG for full reproducibility
//!   • employs the corrected Tuner that counts attempts
//!   • writes CSV with the `csv` crate
//!   • shows a live progress bar with `indicatif`

use scan::graph::{Graph, StepInfo};
use scan::measure::Recorder;

use rand::{SeedableRng, RngCore};
use rand_chacha::ChaCha20Rng;
use indicatif::{ProgressBar, ProgressStyle};
use csv::WriterBuilder;
use std::fs::File;

/// Proportional tuner for proposal widths (counts attempts, not acceptances).
struct Tuner {
    delta:    f64,
    attempts: usize,
    accepted: usize,
    window:   usize,
    target:   f64,
    band:     f64,
}
impl Tuner {
    fn new(delta: f64, window: usize, target: f64, band: f64) -> Self {
        Self { delta, attempts: 0, accepted: 0, window, target, band }
    }
    fn update(&mut self, accepted: bool) {
        self.attempts += 1;
        if accepted   { self.accepted += 1; }
        if self.attempts == self.window {
            let r = self.accepted as f64 / self.window as f64;
            if r > self.target + self.band { self.delta *= 1.1; }
            else if r < self.target - self.band { self.delta *= 0.9; }
            self.attempts = 0;
            self.accepted = 0;
        }
    }
}

fn main() {
    // -------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------
    let n_nodes          = 8;
    let n_steps: usize   = 100_000;
    let equil_steps      = 20_000;
    let report_every     = 1_000;

    let beta   = 1.0;
    let alpha  = 1.0;

    let mut tuner_w  = Tuner::new(0.10, 200, 0.30, 0.05);
    let mut tuner_th = Tuner::new(0.20, 200, 0.30, 0.05);

    // RNG seeded from entropy; record seed if desired.
    let mut rng = ChaCha20Rng::from_entropy();

    // Build graph with explicit RNG for reproducibility.
    let mut g = Graph::complete_random_with(&mut rng, n_nodes);

    // Cached running sums for O(1) averages.
    let mut sum_w   = g.sum_weights();
    let mut sum_cos = g.links_cos_sum();

    // Recorder for time series after equilibration.
    let mut recorder = Recorder::default();

    // Progress bar.
    let bar = ProgressBar::new(n_steps as u64);
    bar.set_style(ProgressStyle::with_template(
        " {bar:40.green/white} {pos}/{len} [{elapsed_precise}] acc={percent:>3}%"
    ).unwrap());

    // CSV writers.
    let mut obs_csv = WriterBuilder::new()
        .from_path("mc_observables.csv")
        .expect("cannot create mc_observables.csv");

    obs_csv.write_record(&[
        "step","acc_rate","delta_w","delta_theta",
        "avg_w","avg_cos_theta",
        "S_entropy","S_triangle","action"
    ]).unwrap();

    let mut theta_csv = WriterBuilder::new()
        .from_path("theta_final.csv")
        .expect("cannot create theta_final.csv");
    theta_csv.write_record(&["link_i","link_j","theta"]).unwrap();

    // -------------------------------------------------------------------
    // Main loop
    // -------------------------------------------------------------------
    let mut accepted_total = 0usize;

    println!(
        "# step  acc%   δw     δθ     ⟨w⟩     ⟨cosθ⟩   Sₑ        SΔ        A"
    );

    for step in 1..=n_steps {
        let StepInfo { accepted, delta_w, delta_cos } = g.metropolis_step(
            beta, alpha, tuner_w.delta, tuner_th.delta, &mut rng
        );

        if accepted {
            accepted_total += 1;
            sum_w   += delta_w;
            sum_cos += delta_cos;
        }
        tuner_w.update(accepted);
        tuner_th.update(accepted);

        if step >= equil_steps {
            recorder.push(&g.links);
        }

        if step % report_every == 0 {
            let avg_w   = sum_w   / g.m() as f64;
            let avg_cos = sum_cos / g.m() as f64;
            let s_entropy = g.entropy_action();
            let s_tri     = g.triangle_action(alpha);
            let total_a   = g.action(alpha);
            let acc_rate  = accepted_total as f64 / step as f64;

            println!(
                "{:>6} {:>5.2}% {:>6.3} {:>6.3} {:>7.3} {:>8.3} {:>10.2} {:>10.2} {:>10.2}",
                step,
                100.0 * acc_rate,
                tuner_w.delta,
                tuner_th.delta,
                avg_w,
                avg_cos,
                s_entropy,
                s_tri,
                total_a,
            );

            obs_csv.write_record(&[
                step.to_string(),
                acc_rate.to_string(),
                tuner_w.delta.to_string(),
                tuner_th.delta.to_string(),
                avg_w.to_string(),
                avg_cos.to_string(),
                s_entropy.to_string(),
                s_tri.to_string(),
                total_a.to_string(),
            ]).unwrap();
        }

        bar.inc(1);
    }
    bar.finish();

    // -------------------------------------------------------------------
    // Save final phases
    // -------------------------------------------------------------------
    for link in &g.links {
        theta_csv.write_record(&[
            link.i.to_string(),
            link.j.to_string(),
            link.theta.to_string(),
        ]).unwrap();
    }
    theta_csv.flush().unwrap();
    println!("Saved final U(1) phases to theta_final.csv");

    // Time series of cos θ after equilibration.
    let mut cos_series = File::create("timeseries_cos.csv")
        .expect("cannot create timeseries_cos.csv");
    for v in &recorder.cos_theta {
        use std::io::Write;
        writeln!(cos_series, "{v}").unwrap();
    }
    println!("Saved time series to timeseries_cos.csv");
}
