//! Targeted (“narrow”) β–α scan.
//!
//! 1.  Read a CSV called   pairs.csv   that contains the
//!     two columns  beta,alpha  (no header needed – exactly what you
//!     exported from the Jupyter ridge‑finder).
//! 2.  For each row call  run_single()  – the *unchanged* MC engine that
//!     used to sit in your β–α double loop.
//! 3.  Append observables to   narrow_scan_results.csv
//!
//! Compile & run:   cargo run --bin narrow_scan
//! Or pass a different CSV with  --pairs other.csv

use std::{fs::File, io::BufReader, path::PathBuf};

use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use scan::graph::{Graph, StepInfo};

/// All MC / lattice parameters stay exactly as before.
#[derive(Clone, Debug)]
struct MC {
    n_nodes:      usize,
    n_steps:      usize,
    equil_steps:  usize,
    sample_every: usize,
    n_rep:        usize,
    tune_win:     usize,
    tune_tgt:     f64,
    tune_band:    f64,
}
impl Default for MC {
    fn default() -> Self {
        Self {
            n_nodes:      12,
            n_steps:      50_000,
            equil_steps:  10_000,
            sample_every: 10,
            n_rep:        5,
            tune_win:     200,
            tune_tgt:     0.30,
            tune_band:    0.05,
        }
    }
}

/// Small CLI helper –‐ let you specify a different list on the fly.
#[derive(Parser)]
struct Cli {
    /// CSV with 2 columns: beta, alpha
    #[arg(long, default_value = "pairs.csv")]
    pairs: PathBuf,
}

#[derive(Debug)]
struct Row {
    beta:      f64,
    alpha:     f64,
    mean_w:    f64,
    std_w:     f64,
    mean_cos:  f64,
    std_cos:   f64,
    chi:       f64,
}

// ────────────────────────────────────────────────────────────────────────────
// Engine that *used to be inside the nested loops*.
// Pulling it into its own fn lets Python decide what (β,α) to call.
// ────────────────────────────────────────────────────────────────────────────
fn run_single(beta: f64, alpha: f64, mc: &MC, seed_modifier: u64) -> Row {
    // ---- bookkeeping helpers identical to your original file ----
    let links_per = mc.n_nodes * (mc.n_nodes - 1) / 2;

    #[derive(Default, Clone)]
    struct OnlineStats { n: u64, mean: f64, m2: f64 }
    impl OnlineStats {
        fn push(&mut self, x: f64) {
            self.n += 1;
            let d  = x - self.mean;
            self.mean += d / self.n as f64;
            let d2 = x - self.mean;
            self.m2 += d * d2;
        }
        fn mean(&self) -> f64 { self.mean }
        fn var (&self) -> f64 { if self.n > 1 { self.m2 / (self.n-1) as f64 } else { 0.0 } }
        fn std (&self) -> f64 { self.var().sqrt() }
    }

    struct Tuner { delta: f64, attempts: usize, accepted: usize, win: usize, tgt: f64, band: f64 }
    impl Tuner {
        fn new(delta: f64, win: usize, tgt: f64, band: f64) -> Self {
            Self { delta, attempts: 0, accepted: 0, win, tgt, band }
        }
        fn update(&mut self, acc: bool) {
            self.attempts += 1; if acc { self.accepted += 1 }
            if self.attempts == self.win {
                let r = self.accepted as f64 / self.win as f64;
                if r > self.tgt + self.band { self.delta *= 1.1 }
                else if r < self.tgt - self.band { self.delta *= 0.9 }
                self.attempts = 0; self.accepted = 0;
            }
        }
    }

    let mut master = ChaCha20Rng::seed_from_u64(seed_modifier);
    let mut stats_w   = OnlineStats::default();
    let mut stats_cos = OnlineStats::default();

    for rep in 0..mc.n_rep {
        let mut rng = ChaCha20Rng::seed_from_u64(master.next_u64() ^ rep as u64);

        let mut g = Graph::complete_random_with(&mut rng, mc.n_nodes);
        let mut tuner_w  = Tuner::new(0.10, mc.tune_win, mc.tune_tgt, mc.tune_band);
        let mut tuner_th = Tuner::new(0.20, mc.tune_win, mc.tune_tgt, mc.tune_band);

        let mut sum_w   = g.sum_weights();
        let mut sum_cos = g.links_cos_sum();

        for step in 1..=mc.n_steps {
            let StepInfo { accepted, delta_w, delta_cos } =
                g.metropolis_step(beta, alpha, tuner_w.delta, tuner_th.delta, &mut rng);

            if accepted { sum_w += delta_w; sum_cos += delta_cos; }
            tuner_w.update(accepted); tuner_th.update(accepted);

            if step > mc.equil_steps && step % mc.sample_every == 0 {
                stats_w  .push(sum_w   / g.m() as f64);
                stats_cos.push(sum_cos / g.m() as f64);
            }
        }
    }

    let chi = links_per as f64 * stats_cos.var();
    Row {
        beta, alpha,
        mean_w:   stats_w.mean(), std_w:    stats_w.std(),
        mean_cos: stats_cos.mean(), std_cos: stats_cos.std(),
        chi,
    }
}

// ────────────────────────────────────────────────────────────────────────────
// MAIN
// ────────────────────────────────────────────────────────────────────────────
fn main() {
    let args = Cli::parse();
    let mc   = MC::default();

    // ------------------------------------------------------------
    // 1. Read the target (β, α) pairs
    // ------------------------------------------------------------
    let file   = File::open(&args.pairs).expect("cannot open pairs.csv");
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(BufReader::new(file));

    let pairs: Vec<(f64,f64)> = rdr.records()
        .map(|r| {
            let rec = r.expect("bad CSV");
            let β: f64 = rec[0].parse().unwrap();
            let α: f64 = rec[1].parse().unwrap();
            (β, α)
        })
        .collect();

    println!("Narrow scan – {} (β,α) points", pairs.len());

    // Progress bar
    let bar = ProgressBar::new(pairs.len() as u64);
    bar.set_style(ProgressStyle::with_template(
        " {bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}]"
    ).unwrap());

    // ------------------------------------------------------------
    // 2. Run in parallel over the list
    // ------------------------------------------------------------
    let rows: Vec<Row> = pairs
        .par_iter()
        .enumerate()
        .map(|(idx, &(β, α))| {
            let row = run_single(β, α, &mc, idx as u64);
            bar.inc(1);
            row
        })
        .collect();

    bar.finish();

    // ------------------------------------------------------------
    // 3. Write CSV
    // ------------------------------------------------------------
    let mut wtr = WriterBuilder::new()
        .from_path("narrow_scan_results.csv")
        .expect("cannot create result file");

    wtr.write_record([
        "beta","alpha","mean_w","std_w","mean_cos","std_cos","susceptibility"
    ]).unwrap();

    let mut rows = rows;
    rows.sort_by(|a,b| a.beta.partial_cmp(&b.beta).unwrap()
                      .then(a.alpha.partial_cmp(&b.alpha).unwrap()));

    for r in &rows {
        wtr.write_record([
            r.beta.to_string(),
            r.alpha.to_string(),
            r.mean_w.to_string(),
            r.std_w.to_string(),
            r.mean_cos.to_string(),
            r.std_cos.to_string(),
            r.chi.to_string(),
        ]).unwrap();
    }
    wtr.flush().unwrap();
    println!("Done → narrow_scan_results.csv");
}
