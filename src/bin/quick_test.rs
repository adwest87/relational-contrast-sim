// src/bin/quick_test.rs
//! Quick sanity check - should run in under a minute

use scan::graph::{Graph, StepInfo};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand::RngCore;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use csv::WriterBuilder;
use std::sync::Mutex;

#[derive(Clone, Debug)]
struct QuickConfig {
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

impl Default for QuickConfig {
    fn default() -> Self {
        Self {
            // REDUCED PARAMETERS FOR QUICK TEST
            n_nodes:      12,      // Small graph (66 links)
            n_steps:      5_000,   // Much shorter runs
            equil_steps:  1_000,   // Quick equilibration
            sample_every: 10,      // Same sampling rate
            
            // Coarse grid: 3x3 = 9 points only
            beta_vals:    vec![1.0, 2.0, 3.0],
            alpha_vals:   vec![0.0, 1.0, 2.0],
            
            n_rep:        2,       // Just 2 replicas
            tune_win:     100,     // Faster tuning
            tune_tgt:     0.30,
            tune_band:    0.05,
        }
    }
}

// Copy the same OnlineStats and Tuner from wide_scan.rs
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

#[derive(Debug)]
struct Row {
    beta:      f64,
    alpha:     f64,
    mean_cos:  f64,
    std_cos:   f64,
    chi:       f64,
    c_spec:    f64,
    s_bar:     f64,
    delta_bar: f64,
}

fn main() {
    let cfg = QuickConfig::default();
    println!("Quick test with configuration:\n{cfg:#?}");
    println!("\nThis should take < 1 minute on a modern CPU\n");

    let links_per = cfg.n_nodes * (cfg.n_nodes - 1) / 2;
    let n_tri = cfg.n_nodes * (cfg.n_nodes - 1) * (cfg.n_nodes - 2) / 6;

    let bar = ProgressBar::new((cfg.beta_vals.len() * cfg.alpha_vals.len()) as u64);
    bar.set_style(ProgressStyle::with_template(
        " {bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}] eta: {eta}"
    ).unwrap());

    let rows = Mutex::new(Vec::new());

    // Simple serial loop for testing (faster for small problems)
    let start_time = std::time::Instant::now();
    
    for (b_idx, &beta) in cfg.beta_vals.iter().enumerate() {
        let mut master = ChaCha20Rng::seed_from_u64(b_idx as u64);

        for (a_idx, &alpha) in cfg.alpha_vals.iter().enumerate() {
            let mut stats_cos = OnlineStats::default();
            let mut stats_se   = OnlineStats::default();
            let mut stats_tri  = OnlineStats::default();
            let mut stats_stot = OnlineStats::default();

            for rep in 0..cfg.n_rep {
                let seed = ((b_idx as u64) << 40) | ((a_idx as u64) << 20) | rep as u64;
                let mut rng = ChaCha20Rng::seed_from_u64(seed ^ master.next_u64());

                let mut g = Graph::complete_random_with(&mut rng, cfg.n_nodes);
                let mut tuner_w  = Tuner::new(0.10, cfg.tune_win, cfg.tune_tgt, cfg.tune_band);
                let mut tuner_th = Tuner::new(0.20, cfg.tune_win, cfg.tune_tgt, cfg.tune_band);

                let mut sum_w   = g.sum_weights();
                let mut sum_cos = g.links_cos_sum();
                
                // Quick equilibration
                for _ in 1..=cfg.equil_steps {
                    let StepInfo { accepted, delta_w, delta_cos } =
                        g.metropolis_step(beta, alpha, tuner_w.delta, tuner_th.delta, &mut rng);

                    if accepted {
                        sum_w   += delta_w;
                        sum_cos += delta_cos;
                    }
                    tuner_w.update(accepted);
                    tuner_th.update(accepted);
                }

                // Production
                for step in 1..=(cfg.n_steps - cfg.equil_steps) {
                    let StepInfo { accepted, delta_w, delta_cos } =
                        g.metropolis_step(beta, alpha, tuner_w.delta, tuner_th.delta, &mut rng);

                    if accepted {
                        sum_w   += delta_w;
                        sum_cos += delta_cos;
                    }
                    tuner_w.update(accepted);
                    tuner_th.update(accepted);

                    if step % cfg.sample_every == 0 {
                        let avg_cos = sum_cos / g.m() as f64;
                        let s_entropy = g.entropy_action();
                        let tri_sum   = g.triangle_sum_norm();
                        let s_total   = beta * s_entropy + alpha * tri_sum * n_tri as f64;
                        
                        stats_cos.push(avg_cos);
                        stats_se.push(s_entropy);
                        stats_tri.push(tri_sum);
                        stats_stot.push(s_total);
                    }
                }
            }

            let chi = links_per as f64 * stats_cos.var();
            let c_spec   = stats_stot.var() / links_per as f64;
            let s_bar    = stats_se.mean() / links_per as f64;
            let delta_bar= stats_tri.mean() / n_tri as f64;

            rows.lock().unwrap().push(Row {
                beta,
                alpha,
                mean_cos: stats_cos.mean(),
                std_cos:  stats_cos.std(),
                chi,
                c_spec,
                s_bar,
                delta_bar,
            });

            bar.inc(1);
            
            // Print intermediate results
            println!("β={:.1} α={:.1}: C={:.3}, χ={:.3}, ⟨cos⟩={:.3}", 
                     beta, alpha, c_spec, chi, stats_cos.mean());
        }
    }
    bar.finish();

    let elapsed = start_time.elapsed();
    println!("\nCompleted in {:.1} seconds", elapsed.as_secs_f64());

    // Sort and save
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by(|a, b| a.beta.partial_cmp(&b.beta).unwrap()
        .then(a.alpha.partial_cmp(&b.alpha).unwrap()));

    let mut wtr = WriterBuilder::new()
        .from_path("quick_test_results.csv")
        .expect("cannot create file");

    wtr.write_record(&[
        "beta","alpha","mean_cos","std_cos","susceptibility","C","S_bar","Delta_bar"
    ]).unwrap();

    for r in &rows {
        wtr.write_record(&[
            r.beta.to_string(),
            r.alpha.to_string(),
            r.mean_cos.to_string(),
            r.std_cos.to_string(),
            r.chi.to_string(),
            r.c_spec.to_string(),
            r.s_bar.to_string(),
            r.delta_bar.to_string(),
        ]).unwrap();
    }
    wtr.flush().unwrap();

    println!("\nResults saved → quick_test_results.csv");
    
    // Quick analysis
    let max_c = rows.iter().map(|r| r.c_spec).fold(0.0, f64::max);
    let max_chi = rows.iter().map(|r| r.chi).fold(0.0, f64::max);
    
    println!("\nMax specific heat C = {:.3}", max_c);
    println!("Max susceptibility χ = {:.3}", max_chi);
    
    if max_c > 1.0 || max_chi > 10.0 {
        println!("\n✓ Sanity check PASSED - seeing non-trivial phase structure");
    } else {
        println!("\n⚠ Warning: values seem low - check your Graph implementation");
    }
}