//! Parameter scan for the relational‑connection model
//! (see `Config` below for all run parameters).

use scan::graph::{Graph, StepInfo};          // `StepInfo` is new, see comment below.
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use csv::WriterBuilder;
use std::sync::Mutex;

/// Run‑time configuration (single source of truth).
#[derive(Debug, Clone)]
struct Config {
    n_nodes:       usize,
    n_steps:       usize,
    equil_steps:   usize,
    sample_every:  usize,
    beta_vals:     Vec<f64>,
    alpha_vals:    Vec<f64>,
    n_rep:         usize,
    tune_win:      usize,
    tune_tgt:      f64,
    tune_band:     f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            n_nodes:      48,
            n_steps:      200_000,
            equil_steps:  80_000,
            sample_every: 10,                // evaluate every 10th step
            beta_vals:    (120..=150).map(|i| 0.02 * i as f64).collect(),
            alpha_vals:   (0..=20).map(|i| 0.1  * i as f64).collect(),
            n_rep:        5,
            tune_win:     200,
            tune_tgt:     0.30,
            tune_band:    0.05,
        }
    }
}

/// Welford online stats.
#[derive(Default, Clone)]
struct OnlineStats {
    n:    u64,
    mean: f64,
    m2:   f64,
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
    fn var(&self)  -> f64 { if self.n > 1 { self.m2 / (self.n - 1) as f64 } else { 0.0 } }
    fn std(&self)  -> f64 { self.var().sqrt() }
}

/// Adaptive step‑size tuner that counts *attempts*.
struct Tuner { delta: f64, attempts: usize, accepted: usize, win: usize, tgt: f64, band: f64 }
impl Tuner {
    fn new(delta: f64, win: usize, tgt: f64, band: f64) -> Self {
        Self { delta, attempts: 0, accepted: 0, win, tgt, band }
    }
    fn update(&mut self, accepted: bool) {
        self.attempts += 1;
        if accepted { self.accepted += 1; }
        if self.attempts == self.win {
            let r = self.accepted as f64 / self.win as f64;
            if r > self.tgt + self.band { self.delta *= 1.1; }
            else if r < self.tgt - self.band { self.delta *= 0.9; }
            self.attempts = 0;
            self.accepted = 0;
        }
    }
}

/// Row to be written to CSV.
#[derive(Debug)]
struct Row {
    beta: f64,
    alpha: f64,
    mean_w: f64,
    std_w: f64,
    mean_cos: f64,
    std_cos: f64,
    chi: f64,
}

fn main() {
    // ------------------------------------------------------------
    let cfg = Config::default();
    println!("Configuration:\n{cfg:#?}");

    let links_per = cfg.n_nodes * (cfg.n_nodes - 1) / 2;

    // A progress bar that tracks β, α pairs (outer loop).
    let total_tasks = cfg.beta_vals.len() * cfg.alpha_vals.len();
    let bar = ProgressBar::new(total_tasks as u64);
    bar.set_style(ProgressStyle::with_template(
        " {bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}]",
    ).unwrap());

    let results: Mutex<Vec<Row>> = Mutex::new(Vec::new());

    // Parallel scan over (β, α). Replica loop stays serial for determinism.
    cfg.beta_vals.par_iter().enumerate().for_each(|(b_idx, &beta)| {
        // Separate master RNG per β to avoid contention; seed from OS.
        let mut master = ChaCha20Rng::from_entropy();

        for (a_idx, &alpha) in cfg.alpha_vals.iter().enumerate() {
            let mut stats_w   = OnlineStats::default();
            let mut stats_cos = OnlineStats::default();

            for rep in 0..cfg.n_rep {
                // Derive a unique u64 seed from (β index, α index, replica).
                let seed = ((b_idx as u64) << 40) | ((a_idx as u64) << 20) | rep as u64;
                let mut rng = ChaCha20Rng::seed_from_u64(seed ^ master.next_u64());

                // Build random graph with an explicit RNG.
                let mut g = Graph::complete_random_with(&mut rng, cfg.n_nodes);

                // Tuners for weight and angle moves.
                let mut tun_w  = Tuner::new(0.10, cfg.tune_win, cfg.tune_tgt, cfg.tune_band);
                let mut tun_th = Tuner::new(0.20, cfg.tune_win, cfg.tune_tgt, cfg.tune_band);

                // Cached running sums so we can update in O(1) per step.
                let mut sum_w   = g.sum_weights();
                let mut sum_cos = g.links_cos_sum();

                for step in 1..=cfg.n_steps {
                    // Metropolis move returns per‑link deltas.
                    let StepInfo { accepted, delta_w, delta_cos } =
                        g.metropolis_step(beta, alpha, tun_w.delta, tun_th.delta, &mut rng);

                    if accepted {
                        sum_w   += delta_w;
                        sum_cos += delta_cos;
                    }

                    tun_w.update(accepted);
                    tun_th.update(accepted);

                    // Record after equilibration, sampling every `sample_every`.
                    if step > cfg.equil_steps && step % cfg.sample_every == 0 {
                        let avg_w   = sum_w   / g.m() as f64;
                        let avg_cos = sum_cos / g.m() as f64;
                        stats_w.push(avg_w);
                        stats_cos.push(avg_cos);
                    }
                }
            }

            let chi = links_per as f64 * stats_cos.var();

            results.lock().unwrap().push(Row {
                beta, alpha,
                mean_w:   stats_w.mean(), std_w:   stats_w.std(),
                mean_cos: stats_cos.mean(), std_cos: stats_cos.std(),
                chi,
            });

            bar.inc(1);
        }
    });
    bar.finish();

    // ------------------------------------------------------------
    // Sort for deterministic CSV order.
    let mut rows = results.into_inner().unwrap();
    rows.sort_by(|a, b| a.beta.partial_cmp(&b.beta).unwrap()
        .then(a.alpha.partial_cmp(&b.alpha).unwrap()));

    // Write CSV.
    let mut wtr = WriterBuilder::new().from_path("scan_results.csv")
        .expect("cannot create scan_results.csv");
    wtr.write_record([
        "beta","alpha","mean_w","std_w","mean_cos","std_cos","susceptibility"
    ]).unwrap();
    for r in &rows {
        wtr.write_record(&[
            r.beta.to_string(),  r.alpha.to_string(),
            r.mean_w.to_string(), r.std_w.to_string(),
            r.mean_cos.to_string(), r.std_cos.to_string(),
            r.chi.to_string(),
        ]).unwrap();
    }
    wtr.flush().unwrap();
    println!("Scan complete → scan_results.csv");
}

/* ----------------------------------------------------------------
   Changes required in `rc_sim::graph`

   1. Add `links_cos_sum(&self) -> f64` that returns Σ cos θ over all links.
   2. Change `metropolis_step` to

      fn metropolis_step(
          &mut self,
          beta: f64,
          alpha: f64,
          delta_w: f64,
          delta_th: f64,
          rng: &mut impl rand::RngCore,
      ) -> StepInfo

      where

      pub struct StepInfo {
          pub accepted: bool,
          pub delta_w:  f64,   // change in Σ w  (0 if rejected)
          pub delta_cos: f64,  // change in Σ cosθ (0 if rejected)
      }

   This lets the driver maintain O(1) running sums.
   ---------------------------------------------------------------- */
