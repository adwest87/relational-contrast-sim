//! High‑resolution β–α scan with replicas and error bars.
//!
//! Parameters are kept in one `Config` struct so comments never
//! drift out of sync with the executable settings.
//
//  Compile & run:  `cargo run --bin beta_alpha_scan`

use scan::graph::{Graph, StepInfo};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand::RngCore;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use csv::WriterBuilder;
use std::sync::Mutex;

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------
#[derive(Clone, Debug)]
struct WideConfig {
    n_nodes:      usize,
    n_steps:      usize,
    equil_steps:  usize,
    sample_every: usize,
    beta_vals:    Vec<f64>,
    alpha_vals:   Vec<f64>,
    n_rep:        usize,
    tune_win:     usize,
    tune_tgt:     f64,
    tune_band:    f64,
}
impl Default for WideConfig {
    fn default() -> Self {
        Self {
            n_nodes:      128,
            n_steps:      60_000,
            equil_steps:  10_000,
            sample_every: 10,
            beta_vals:    (0..=12)           // 0.00 … 3.00, step 0.25
                              .map(|i| 0.25 * i as f64)
                              .collect(),
            alpha_vals:   (0..=12)           // 0.0 … 6.0, step 0.5
                              .map(|i| 0.50 * i as f64)
                              .collect(),
            n_rep:        2,
            tune_win:     200,
            tune_tgt:     0.30,
            tune_band:    0.05,
        }
    }
}

// -----------------------------------------------------------------------------
// Online mean / variance (Welford)
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Adaptive tuner that counts attempts
// -----------------------------------------------------------------------------
struct Tuner { delta: f64, attempts: usize, accepted: usize, win: usize, tgt: f64, band: f64 }
impl Tuner {
    fn new(delta: f64, win: usize, tgt: f64, band: f64) -> Self {
        Self { delta, attempts: 0, accepted: 0, win, tgt, band }
    }
    fn update(&mut self, acc: bool) {
        self.attempts += 1;
        if acc { self.accepted += 1; }
        if self.attempts == self.win {
            let r = self.accepted as f64 / self.win as f64;
            if r > self.tgt + self.band { self.delta *= 1.1; }
            else if r < self.tgt - self.band { self.delta *= 0.9; }
            self.attempts = 0;
            self.accepted = 0;
        }
    }
}

// -----------------------------------------------------------------------------
// CSV row
// -----------------------------------------------------------------------------
#[derive(Debug)]
struct Row {
    beta:      f64,
    alpha:     f64,
    mean_w:    f64,
    std_w:     f64,
    mean_cos:  f64,
    std_cos:   f64,
    chi:       f64,
    c_spec:    f64,
    s_bar:     f64,
    delta_bar: f64,
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
fn main() {
    let cfg = WideConfig::default();
    println!("Running scan with configuration:\n{cfg:#?}");

    let links_per = cfg.n_nodes * (cfg.n_nodes - 1) / 2;
    let n_tri = cfg.n_nodes * (cfg.n_nodes - 1) * (cfg.n_nodes - 2) / 6;


    // Progress bar counts (β, α) pairs.
    let bar = ProgressBar::new((cfg.beta_vals.len() * cfg.alpha_vals.len()) as u64);
    bar.set_style(ProgressStyle::with_template(
        " {bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}]"
    ).unwrap());

    let rows = Mutex::new(Vec::new());

    // Parallel outer loop over β.
    cfg.beta_vals.par_iter().enumerate().for_each(|(b_idx, &beta)| {
        let mut master = ChaCha20Rng::from_entropy();               // separate master per β

        for (a_idx, &alpha) in cfg.alpha_vals.iter().enumerate() {
            let mut stats_w   = OnlineStats::default();
            let mut stats_cos = OnlineStats::default();
            let mut stats_se   = OnlineStats::default();   // entropy
            let mut stats_tri  = OnlineStats::default();   // triangle sum
            let mut stats_stot = OnlineStats::default();   // total action

            for rep in 0..cfg.n_rep {
                // Unique deterministic seed from indices.
                let seed = ((b_idx as u64) << 40) | ((a_idx as u64) << 20) | rep as u64;
                let mut rng = ChaCha20Rng::seed_from_u64(seed ^ master.next_u64());

                // Graph and tuners.
                let mut g = Graph::complete_random_with(&mut rng, cfg.n_nodes);
                let mut tuner_w  = Tuner::new(0.10, cfg.tune_win, cfg.tune_tgt, cfg.tune_band);
                let mut tuner_th = Tuner::new(0.20, cfg.tune_win, cfg.tune_tgt, cfg.tune_band);

                // Running sums for O(1) averages.
                let mut sum_w   = g.sum_weights();
                let mut sum_cos = g.links_cos_sum();

                for step in 1..=cfg.n_steps {
                    let StepInfo { accepted, delta_w, delta_cos } =
                        g.metropolis_step(
                            beta,
                            alpha,
                            tuner_w.delta,
                            tuner_th.delta,
                            &mut rng,
                        );

                    if accepted {
                        sum_w   += delta_w;
                        sum_cos += delta_cos;
                    }
                    tuner_w.update(accepted);
                    tuner_th.update(accepted);

                    if step > cfg.equil_steps && step % cfg.sample_every == 0 {
                        let avg_w   = sum_w   / g.m() as f64;
                        let avg_cos = sum_cos / g.m() as f64;
                        let s_entropy = g.entropy_action();
                        let tri_sum   = g.triangle_sum();
                        let s_total   = beta * s_entropy + alpha * tri_sum;
                        stats_se  .push(s_entropy);
                        stats_tri .push(tri_sum);
                        stats_stot.push(s_total);
                        stats_w.push(avg_w);
                        stats_cos.push(avg_cos);
                    }
                } // end step loop
            }     // end replica loop

            let chi = links_per as f64 * stats_cos.var();
            let mut stats_stot = OnlineStats::default();   // total action
            let mut stats_se   = OnlineStats::default();   // entropy
            let mut stats_tri  = OnlineStats::default();   // triangle sum
            let c_spec   = stats_stot.var()  / links_per as f64;   // C = Var(S_tot)/m
            let s_bar    = stats_se  .mean() / links_per as f64;   // S̄  = ⟨S_e⟩/m
            let delta_bar= stats_tri .mean() / n_tri     as f64;   // Δ̄  = ⟨ΣΔ⟩/n_tri


            rows.lock().unwrap().push(Row {
                beta,
                alpha,
                mean_w:   stats_w.mean(),
                std_w:    stats_w.std(),
                mean_cos: stats_cos.mean(),
                std_cos:  stats_cos.std(),
                chi,
                c_spec,
                s_bar,
                delta_bar,
            });

            bar.inc(1);
        } // end α loop
    });   // end β loop
    bar.finish();

    // Sort for deterministic CSV order.
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by(|a, b| a.beta.partial_cmp(&b.beta).unwrap()
        .then(a.alpha.partial_cmp(&b.alpha).unwrap()));

    // ---------------------------------------------------------------------
    // Write CSV
    // ---------------------------------------------------------------------
    let mut wtr = WriterBuilder::new()
        .from_path("scan_coarse.csv")   // new name
        .expect("cannot create scan_wide.csv");

    wtr.write_record(&[
    "beta","alpha",
    //"mean_w","std_w",
    "mean_cos","std_cos","susceptibility","C","S_bar","Delta_bar"
    ]).unwrap();

    for r in &rows {
        wtr.write_record(&[
            r.beta.to_string(),
            r.alpha.to_string(),
            //r.mean_w.to_string(),
            //r.std_w.to_string(),
            r.mean_cos.to_string(),
            r.std_cos.to_string(),
            r.chi.to_string(),
            r.c_spec.to_string(),
            r.s_bar.to_string(),
            r.delta_bar.to_string(),
        ]).unwrap();
    }
    wtr.flush().unwrap();

    println!("Scan complete → scan_coarse.csv");
}
