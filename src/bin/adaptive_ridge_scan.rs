// src/bin/adaptive_ridge_scan.rs
//! Adaptive scanning that concentrates samples near the critical ridge
//! with error bar estimation and multiple observable tracking

use std::{fs::File, io::BufReader, path::PathBuf};
use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use scan::graph::{Graph, StepInfo};
use std::sync::Mutex;
use std::collections::HashMap;

#[derive(Clone, Debug)]
struct AdaptiveConfig {
    n_nodes:      usize,
    n_steps:      usize,
    equil_steps:  usize,
    sample_every: usize,
    n_rep:        usize,
    tune_win:     usize,
    tune_tgt:     f64,
    tune_band:    f64,
    // New: adaptive parameters
    min_samples:  usize,  // minimum replicas per point
    max_samples:  usize,  // maximum replicas per point
    error_threshold: f64, // target relative error
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            n_nodes:      48,
            n_steps:      100_000,
            equil_steps:  20_000,
            sample_every: 10,
            n_rep:        5,      // initial replicas
            tune_win:     200,
            tune_tgt:     0.30,
            tune_band:    0.05,
            min_samples:  5,
            max_samples:  20,
            error_threshold: 0.05, // 5% relative error target
        }
    }
}

#[derive(Parser)]
struct Cli {
    /// Initial ridge points CSV
    #[arg(long, default_value = "ridge_points.csv")]
    ridge: PathBuf,
    
    /// Output file
    #[arg(long, default_value = "adaptive_scan_results.csv")]
    output: PathBuf,
}

#[derive(Debug, Clone)]
struct Measurement {
    beta:         f64,
    alpha:        f64,
    mean_w:       f64,
    std_w:        f64,
    mean_cos:     f64,
    std_cos:      f64,
    chi:          f64,
    c_spec:       f64,
    s_bar:        f64,
    delta_bar:    f64,
    // New: error estimates
    err_chi:      f64,
    err_c:        f64,
    n_samples:    usize,
    // New: additional observables
    spectral_dim: f64,
    min_weight:   f64,
    wilson_3:     f64,  // Wilson triangle average
}

// Helper structs remain the same
#[derive(Default, Clone)]
struct OnlineStats { n: u64, mean: f64, m2: f64 }

impl OnlineStats {
    fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }
    fn mean(&self) -> f64 { self.mean }
    fn var(&self) -> f64 { if self.n > 1 { self.m2 / (self.n - 1) as f64 } else { 0.0 } }
    fn std(&self) -> f64 { self.var().sqrt() }
    fn stderr(&self) -> f64 { if self.n > 0 { self.std() / (self.n as f64).sqrt() } else { 0.0 } }
}

struct Tuner { 
    delta: f64, attempts: usize, accepted: usize, 
    win: usize, tgt: f64, band: f64 
}

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

// New: compute spectral dimension via random walk
fn compute_spectral_dimension(g: &Graph, n_walks: usize, t_max: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let mut return_probs = vec![0.0; t_max];
    
    for _ in 0..n_walks {
        let start = rng.gen_range(0..g.n());
        let mut current = start;
        
        for t in 1..=t_max {
            // Random walk step
            let neighbors: Vec<_> = g.links.iter()
                .filter(|l| l.i == current || l.j == current)
                .collect();
            
            if !neighbors.is_empty() {
                let link = neighbors[rng.gen_range(0..neighbors.len())];
                current = if link.i == current { link.j } else { link.i };
            }
            
            if current == start {
                return_probs[t-1] += 1.0 / n_walks as f64;
            }
        }
    }
    
    // Fit log P(t) ~ -ds/2 * log(t)
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;
    let mut count = 0.0;
    
    for t in 10..t_max {  // Skip early times
        if return_probs[t] > 0.0 {
            let x = (t as f64).ln();
            let y = return_probs[t].ln();
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
            count += 1.0;
        }
    }
    
    if count > 0.0 {
        let slope = (count * sum_xy - sum_x * sum_y) / (count * sum_xx - sum_x * sum_x);
        -2.0 * slope
    } else {
        0.0
    }
}

// Enhanced measurement function
fn run_adaptive(
    beta: f64, 
    alpha: f64, 
    cfg: &AdaptiveConfig, 
    seed: u64
) -> Measurement {
    let links_per = cfg.n_nodes * (cfg.n_nodes - 1) / 2;
    let n_tri = cfg.n_nodes * (cfg.n_nodes - 1) * (cfg.n_nodes - 2) / 6;
    
    let mut master = ChaCha20Rng::seed_from_u64(seed);
    
    // Adaptive sampling: start with min_samples
    let mut all_chi = Vec::new();
    let mut all_c = Vec::new();
    let mut stats_w = OnlineStats::default();
    let mut stats_cos = OnlineStats::default();
    let mut stats_se = OnlineStats::default();
    let mut stats_tri = OnlineStats::default();
    let mut stats_spec = OnlineStats::default();
    let mut min_weights = Vec::new();
    let mut wilson_3_vals = Vec::new();
    
    let mut total_samples = 0;
    
    // Run initial samples
    for _ in 0..cfg.min_samples {
        let (chi, c, w, cos, se, tri, spec, minw, w3) = 
            run_single_replica(beta, alpha, cfg, &mut master);
        
        all_chi.push(chi);
        all_c.push(c);
        stats_w.push(w);
        stats_cos.push(cos);
        stats_se.push(se);
        stats_tri.push(tri);
        stats_spec.push(spec);
        min_weights.push(minw);
        wilson_3_vals.push(w3);
        total_samples += 1;
    }
    
    // Adaptive refinement based on error
    while total_samples < cfg.max_samples {
        // Compute current errors
        let chi_mean = all_chi.iter().sum::<f64>() / all_chi.len() as f64;
        let chi_std = (all_chi.iter().map(|x| (x - chi_mean).powi(2)).sum::<f64>() 
                       / (all_chi.len() - 1) as f64).sqrt();
        let chi_err = chi_std / (all_chi.len() as f64).sqrt();
        let chi_rel_err = chi_err / chi_mean.abs().max(1e-10);
        
        let c_mean = all_c.iter().sum::<f64>() / all_c.len() as f64;
        let c_std = (all_c.iter().map(|x| (x - c_mean).powi(2)).sum::<f64>() 
                     / (all_c.len() - 1) as f64).sqrt();
        let c_err = c_std / (all_c.len() as f64).sqrt();
        let c_rel_err = c_err / c_mean.abs().max(1e-10);
        
        // Check if we've reached target accuracy
        if chi_rel_err < cfg.error_threshold && c_rel_err < cfg.error_threshold {
            break;
        }
        
        // Add more samples
        let (chi, c, w, cos, se, tri, spec, minw, w3) = 
            run_single_replica(beta, alpha, cfg, &mut master);
        
        all_chi.push(chi);
        all_c.push(c);
        stats_w.push(w);
        stats_cos.push(cos);
        stats_se.push(se);
        stats_tri.push(tri);
        stats_spec.push(spec);
        min_weights.push(minw);
        wilson_3_vals.push(w3);
        total_samples += 1;
    }
    
    // Final statistics
    let chi = links_per as f64 * stats_cos.var();
    let c_spec = (all_c.iter().sum::<f64>() / all_c.len() as f64) / links_per as f64;
    let s_bar = stats_se.mean() / links_per as f64;
    let delta_bar = stats_tri.mean() / n_tri as f64;
    
    // Error estimates
    let chi_mean = all_chi.iter().sum::<f64>() / all_chi.len() as f64;
    let chi_std = (all_chi.iter().map(|x| (x - chi_mean).powi(2)).sum::<f64>() 
                   / (all_chi.len() - 1) as f64).sqrt();
    let err_chi = chi_std / (all_chi.len() as f64).sqrt();
    
    let c_mean = all_c.iter().sum::<f64>() / all_c.len() as f64;
    let c_std = (all_c.iter().map(|x| (x - c_mean).powi(2)).sum::<f64>() 
                 / (all_c.len() - 1) as f64).sqrt();
    let err_c = c_std / (all_c.len() as f64).sqrt();
    
    Measurement {
        beta,
        alpha,
        mean_w: stats_w.mean(),
        std_w: stats_w.std(),
        mean_cos: stats_cos.mean(),
        std_cos: stats_cos.std(),
        chi,
        c_spec,
        s_bar,
        delta_bar,
        err_chi,
        err_c,
        n_samples: total_samples,
        spectral_dim: stats_spec.mean(),
        min_weight: min_weights.iter().sum::<f64>() / min_weights.len() as f64,
        wilson_3: wilson_3_vals.iter().sum::<f64>() / wilson_3_vals.len() as f64,
    }
}

// Single replica runner
fn run_single_replica(
    beta: f64,
    alpha: f64,
    cfg: &AdaptiveConfig,
    master_rng: &mut ChaCha20Rng
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f64) {
    let mut rng = ChaCha20Rng::seed_from_u64(master_rng.next_u64());
    let mut g = Graph::complete_random_with(&mut rng, cfg.n_nodes);
    
    let mut tuner_w = Tuner::new(0.10, cfg.tune_win, cfg.tune_tgt, cfg.tune_band);
    let mut tuner_th = Tuner::new(0.20, cfg.tune_win, cfg.tune_tgt, cfg.tune_band);
    
    let mut sum_w = g.sum_weights();
    let mut sum_cos = g.links_cos_sum();
    
    let mut cos_samples = Vec::new();
    let mut se_samples = Vec::new();
    let mut tri_samples = Vec::new();
    let mut action_samples = Vec::new();
    
    // Thermalization
    for _ in 1..=cfg.equil_steps {
        let StepInfo { accepted, delta_w, delta_cos } =
            g.metropolis_step(beta, alpha, tuner_w.delta, tuner_th.delta, &mut rng);
        
        if accepted {
            sum_w += delta_w;
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
            sum_w += delta_w;
            sum_cos += delta_cos;
        }
        tuner_w.update(accepted);
        tuner_th.update(accepted);
        
        if step % cfg.sample_every == 0 {
            let avg_cos = sum_cos / g.m() as f64;
            let s_entropy = g.entropy_action();
            let tri_sum = g.triangle_sum_norm();
            let s_total = beta * s_entropy + alpha * tri_sum * g.n_tri() as f64;
            
            cos_samples.push(avg_cos);
            se_samples.push(s_entropy);
            tri_samples.push(tri_sum);
            action_samples.push(s_total);
        }
    }
    
    // Compute observables
    let avg_w = sum_w / g.m() as f64;
    let avg_cos = cos_samples.iter().sum::<f64>() / cos_samples.len() as f64;
    let avg_se = se_samples.iter().sum::<f64>() / se_samples.len() as f64;
    let avg_tri = tri_samples.iter().sum::<f64>() / tri_samples.len() as f64;
    
    // Variance for susceptibility and specific heat
    let cos_var = cos_samples.iter()
        .map(|&x| (x - avg_cos).powi(2))
        .sum::<f64>() / (cos_samples.len() - 1) as f64;
    let chi = g.m() as f64 * cos_var;
    
    let action_mean = action_samples.iter().sum::<f64>() / action_samples.len() as f64;
    let action_var = action_samples.iter()
        .map(|&x| (x - action_mean).powi(2))
        .sum::<f64>() / (action_samples.len() - 1) as f64;
    let c = action_var;
    
    // Additional observables
    let spec_dim = compute_spectral_dimension(&g, 100, 50);
    let min_w = g.min_weight();
    
    // Wilson triangle average
    let wilson_3 = g.triangles()
        .take(100)  // Sample some triangles
        .map(|(i, j, k)| {
            let theta_sum = g.links[g.link_index(i, j)].theta +
                           g.links[g.link_index(j, k)].theta +
                           g.links[g.link_index(k, i)].theta;
            theta_sum.cos()
        })
        .sum::<f64>() / 100.0;
    
    (chi, c, avg_w, avg_cos, avg_se, avg_tri, spec_dim, min_w, wilson_3)
}

fn main() {
    let args = Cli::parse();
    let cfg = AdaptiveConfig::default();
    
    // Read initial ridge points
    let file = File::open(&args.ridge).expect("cannot open ridge file");
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(BufReader::new(file));
    
    let points: Vec<(f64, f64)> = rdr.records()
        .map(|r| {
            let rec = r.expect("bad CSV");
            let beta: f64 = rec[0].parse().unwrap();
            let alpha: f64 = rec[1].parse().unwrap();
            (beta, alpha)
        })
        .collect();
    
    println!("Adaptive scan - {} (β,α) points", points.len());
    
    let bar = ProgressBar::new(points.len() as u64);
    bar.set_style(ProgressStyle::with_template(
        " {bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}] eta: {eta}"
    ).unwrap());
    
    // Run adaptive measurements
    let measurements: Vec<Measurement> = points
        .par_iter()
        .enumerate()
        .map(|(idx, &(beta, alpha))| {
            let m = run_adaptive(beta, alpha, &cfg, idx as u64);
            bar.inc(1);
            m
        })
        .collect();
    
    bar.finish();
    
    // Write results
    let mut wtr = WriterBuilder::new()
        .from_path(&args.output)
        .expect("cannot create output file");
    
    wtr.write_record(&[
        "beta", "alpha", "mean_w", "std_w", "mean_cos", "std_cos",
        "susceptibility", "C", "S_bar", "Delta_bar",
        "err_chi", "err_c", "n_samples",
        "spectral_dim", "min_weight", "wilson_3"
    ]).unwrap();
    
    for m in &measurements {
        wtr.write_record(&[
            m.beta.to_string(),
            m.alpha.to_string(),
            m.mean_w.to_string(),
            m.std_w.to_string(),
            m.mean_cos.to_string(),
            m.std_cos.to_string(),
            m.chi.to_string(),
            m.c_spec.to_string(),
            m.s_bar.to_string(),
            m.delta_bar.to_string(),
            m.err_chi.to_string(),
            m.err_c.to_string(),
            m.n_samples.to_string(),
            m.spectral_dim.to_string(),
            m.min_weight.to_string(),
            m.wilson_3.to_string(),
        ]).unwrap();
    }
    
    wtr.flush().unwrap();
    println!("Done → {}", args.output.display());
    
    // Summary statistics
    let total_samples: usize = measurements.iter().map(|m| m.n_samples).sum();
    let avg_samples = total_samples as f64 / measurements.len() as f64;
    println!("\nTotal samples: {}", total_samples);
    println!("Average samples per point: {:.1}", avg_samples);
    
    // Points with highest specific heat
    let mut sorted = measurements.clone();
    sorted.sort_by(|a, b| b.c_spec.partial_cmp(&a.c_spec).unwrap());
    
    println!("\nTop 5 critical points by specific heat:");
    for m in sorted.iter().take(5) {
        println!("  β={:.3} α={:.3} C={:.3}±{:.3} ds={:.2}",
                 m.beta, m.alpha, m.c_spec, m.err_c, m.spectral_dim);
    }
}