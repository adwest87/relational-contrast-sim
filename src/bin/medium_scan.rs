// src/bin/medium_scan_improved.rs
//! Medium-resolution scan with improved adaptive step sizes
//! Targets 30% acceptance rate across all phases

use scan::graph::{Graph, StepInfo};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand::RngCore;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use csv::WriterBuilder;
use std::sync::Mutex;

#[derive(Clone, Debug)]
struct MediumConfig {
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

impl Default for MediumConfig {
    fn default() -> Self {
        Self {
            n_nodes:      24,
            n_steps:      20_000,
            equil_steps:  5_000,
            sample_every: 10,
            
            // Standard 7x7 grid
            beta_vals:    vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            alpha_vals:   vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            
            n_rep:        3,
            tune_win:     200,
            tune_tgt:     0.30,    // Target 30% acceptance
            tune_band:    0.05,    // ±5% tolerance
        }
    }
}

impl MediumConfig {
    fn focused() -> Self {
        Self {
            n_nodes:      24,
            n_steps:      25_000,
            equil_steps:  5_000,
            sample_every: 10,
            
            // Focus on critical region
            beta_vals:    (0..=8).map(|i| 0.4 + 0.2 * i as f64).collect(),
            alpha_vals:   (0..=10).map(|i| 0.5 + 0.2 * i as f64).collect(),
            
            n_rep:        3,
            tune_win:     200,
            tune_tgt:     0.30,
            tune_band:    0.05,
        }
    }
}

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

// Improved adaptive tuner with bounds
struct AdaptiveTuner {
    delta:    f64,
    attempts: usize,
    accepted: usize,
    window:   usize,
    target:   f64,
    band:     f64,
    min_delta: f64,
    max_delta: f64,
    adaptation_rate: f64,
}

impl AdaptiveTuner {
    fn new(initial_delta: f64, window: usize, target: f64, band: f64) -> Self {
        Self {
            delta: initial_delta,
            attempts: 0,
            accepted: 0,
            window,
            target,
            band,
            min_delta: initial_delta * 0.1,   // Don't go below 10% of initial
            max_delta: initial_delta * 10.0,  // Don't go above 10x initial
            adaptation_rate: 1.2,             // How fast to adapt
        }
    }
    
    fn update(&mut self, accepted: bool) {
        self.attempts += 1;
        if accepted { self.accepted += 1; }
        
        // Check more frequently during equilibration
        let check_window = if self.attempts < 1000 { 
            self.window / 2 
        } else { 
            self.window 
        };
        
        if self.attempts % check_window == 0 {
            let rate = self.accepted as f64 / check_window as f64;
            
            // Adaptive scaling based on how far we are from target
            let error = (rate - self.target).abs();
            let scale = if error > 0.2 {
                self.adaptation_rate.powf(2.0)  // Aggressive adaptation if far from target
            } else if error > 0.1 {
                self.adaptation_rate.powf(1.5)
            } else {
                self.adaptation_rate
            };
            
            if rate > self.target + self.band {
                self.delta *= scale;
            } else if rate < self.target - self.band {
                self.delta /= scale;
            }
            
            // Enforce bounds
            self.delta = self.delta.clamp(self.min_delta, self.max_delta);
            
            // Reset counters for next window
            self.accepted = 0;
            self.attempts = 0;
        }
    }
    
    fn acceptance_rate(&self) -> f64 {
        if self.attempts > 0 {
            self.accepted as f64 / self.attempts as f64
        } else {
            0.0
        }
    }
}

// Get phase-appropriate initial step sizes
fn get_initial_steps(beta: f64, alpha: f64) -> (f64, f64) {
    // Based on the analysis of your acceptance rates:
    // - Very high acceptance at α=0 (need much larger steps)
    // - Moderate acceptance in transition region
    // - Lower acceptance in ordered phase
    
    let delta_w = if alpha < 0.1 {
        0.50  // Very large for α≈0 where acceptance was 80-88%
    } else if alpha < 0.5 {
        0.35  // Large for small α
    } else if alpha < 1.5 {
        0.20  // Moderate near transition
    } else if beta > 2.0 && alpha > 2.0 {
        0.08  // Smaller in strongly ordered phase
    } else {
        0.15  // Default
    };
    
    let delta_theta = if alpha < 0.1 {
        1.0   // Very weak gauge coupling, can take huge steps
    } else if alpha < 0.5 {
        0.60  // Large steps for weak coupling
    } else if alpha < 1.5 {
        0.35  // Moderate near transition
    } else {
        0.20  // Conservative in strong coupling
    };
    
    (delta_w, delta_theta)
}

#[derive(Debug)]
struct Row {
    beta:         f64,
    alpha:        f64,
    mean_cos:     f64,
    std_cos:      f64,
    chi:          f64,
    c_spec:       f64,
    s_bar:        f64,
    delta_bar:    f64,
    acc_rate:     f64,
    final_dw:     f64,  // Final step sizes for diagnostics
    final_dth:    f64,
}

fn run_single(beta: f64, alpha: f64, cfg: &MediumConfig, seed_base: u64) -> Row {
    let links_per = cfg.n_nodes * (cfg.n_nodes - 1) / 2;
    let n_tri = cfg.n_nodes * (cfg.n_nodes - 1) * (cfg.n_nodes - 2) / 6;
    
    let mut master = ChaCha20Rng::seed_from_u64(seed_base);
    
    let mut stats_cos = OnlineStats::default();
    let mut stats_se  = OnlineStats::default();
    let mut stats_tri = OnlineStats::default();
    let mut stats_stot = OnlineStats::default();
    let mut total_acc = 0.0;
    let mut final_dw = 0.0;
    let mut final_dth = 0.0;

    for rep in 0..cfg.n_rep {
        let mut rng = ChaCha20Rng::seed_from_u64(master.next_u64() ^ rep as u64);
        let mut g = Graph::complete_random_with(&mut rng, cfg.n_nodes);
        
        // Get phase-appropriate initial steps
        let (init_dw, init_dth) = get_initial_steps(beta, alpha);
        
        let mut tuner_w = AdaptiveTuner::new(init_dw, cfg.tune_win, cfg.tune_tgt, cfg.tune_band);
        let mut tuner_th = AdaptiveTuner::new(init_dth, cfg.tune_win, cfg.tune_tgt, cfg.tune_band);

        let mut sum_w = g.sum_weights();
        let mut sum_cos = g.links_cos_sum();
        let mut accepted_count = 0;

        for step in 1..=cfg.n_steps {
            let StepInfo { accepted, delta_w, delta_cos } =
                g.metropolis_step(beta, alpha, tuner_w.delta, tuner_th.delta, &mut rng);

            if accepted {
                sum_w += delta_w;
                sum_cos += delta_cos;
                accepted_count += 1;
            }
            tuner_w.update(accepted);
            tuner_th.update(accepted);

            if step > cfg.equil_steps && step % cfg.sample_every == 0 {
                let avg_cos = sum_cos / g.m() as f64;
                let s_entropy = g.entropy_action();
                let tri_sum = g.triangle_sum_norm();
                let s_total = beta * s_entropy + alpha * tri_sum * n_tri as f64;
                
                stats_cos.push(avg_cos);
                stats_se.push(s_entropy);
                stats_tri.push(tri_sum);
                stats_stot.push(s_total);
            }
        }
        
        total_acc += accepted_count as f64 / cfg.n_steps as f64;
        final_dw = tuner_w.delta;
        final_dth = tuner_th.delta;
        
        // Log if acceptance is still poor after tuning
        let final_rate = accepted_count as f64 / cfg.n_steps as f64;
        if final_rate < 0.15 || final_rate > 0.50 {
            eprintln!(
                "β={:.1} α={:.1} rep={}: final acc={:.1}% δw={:.3} δθ={:.3}", 
                beta, alpha, rep, final_rate * 100.0, tuner_w.delta, tuner_th.delta
            );
        }
    }

    let chi = links_per as f64 * stats_cos.var();
    let c_spec = stats_stot.var() / links_per as f64;
    let s_bar = stats_se.mean() / links_per as f64;
    let delta_bar = stats_tri.mean() / n_tri as f64;
    let acc_rate = total_acc / cfg.n_rep as f64;

    Row {
        beta,
        alpha,
        mean_cos: stats_cos.mean(),
        std_cos: stats_cos.std(),
        chi,
        c_spec,
        s_bar,
        delta_bar,
        acc_rate,
        final_dw,
        final_dth,
    }
}

fn main() {
    // Choose configuration
    let use_focused = std::env::args().any(|arg| arg == "--focused");
    let cfg = if use_focused {
        println!("Using FOCUSED configuration on critical region");
        MediumConfig::focused()
    } else {
        println!("Using STANDARD medium configuration");
        MediumConfig::default()
    };
    
    println!("\nConfiguration:\n{cfg:#?}");
    println!("\nTarget acceptance: {:.0}% ± {:.0}%", 
             cfg.tune_tgt * 100.0, cfg.tune_band * 100.0);
    println!("Estimated runtime: 10-15 minutes on 4+ cores\n");

    let bar = ProgressBar::new((cfg.beta_vals.len() * cfg.alpha_vals.len()) as u64);
    bar.set_style(ProgressStyle::with_template(
        " {bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}] eta: {eta}"
    ).unwrap());

    let rows = Mutex::new(Vec::new());
    let start_time = std::time::Instant::now();

    // Parallel execution
    cfg.beta_vals.par_iter().enumerate().for_each(|(b_idx, &beta)| {
        for (a_idx, &alpha) in cfg.alpha_vals.iter().enumerate() {
            let seed = ((b_idx as u64) << 40) | ((a_idx as u64) << 20);
            let row = run_single(beta, alpha, &cfg, seed);
            
            rows.lock().unwrap().push(row);
            bar.inc(1);
        }
    });
    bar.finish();

    let elapsed = start_time.elapsed();
    println!("\nCompleted in {:.1} minutes", elapsed.as_secs_f64() / 60.0);

    // Sort results
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by(|a, b| a.beta.partial_cmp(&b.beta).unwrap()
        .then(a.alpha.partial_cmp(&b.alpha).unwrap()));

    // Save to CSV
    let mut wtr = WriterBuilder::new()
        .from_path("medium_scan_improved.csv")
        .expect("cannot create file");

    wtr.write_record(&[
        "beta", "alpha", "mean_cos", "std_cos", "susceptibility", "C", 
        "S_bar", "Delta_bar", "acc_rate", "final_dw", "final_dth"
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
            r.acc_rate.to_string(),
            r.final_dw.to_string(),
            r.final_dth.to_string(),
        ]).unwrap();
    }
    wtr.flush().unwrap();

    println!("\nResults saved → medium_scan_improved.csv");
    
    // Analysis
    let max_c_row = rows.iter().max_by(|a, b| a.c_spec.partial_cmp(&b.c_spec).unwrap()).unwrap();
    let max_chi_row = rows.iter().max_by(|a, b| a.chi.partial_cmp(&b.chi).unwrap()).unwrap();
    
    println!("\nPeak specific heat: C = {:.3} at β={:.2}, α={:.2}", 
             max_c_row.c_spec, max_c_row.beta, max_c_row.alpha);
    println!("Peak susceptibility: χ = {:.3} at β={:.2}, α={:.2}", 
             max_chi_row.chi, max_chi_row.beta, max_chi_row.alpha);
    
    // Acceptance rate analysis
    let avg_acc = rows.iter().map(|r| r.acc_rate).sum::<f64>() / rows.len() as f64;
    let good_acc = rows.iter().filter(|r| r.acc_rate >= 0.2 && r.acc_rate <= 0.5).count();
    
    println!("\nAcceptance rate statistics:");
    println!("  Average: {:.1}%", avg_acc * 100.0);
    println!("  Points in target range (20-50%): {}/{}", good_acc, rows.len());
    
    // Show worst acceptance rates
    let mut worst: Vec<_> = rows.iter()
        .filter(|r| r.acc_rate < 0.15 || r.acc_rate > 0.60)
        .collect();
    
    if !worst.is_empty() {
        println!("\n⚠ Points with poor acceptance:");
        worst.sort_by(|a, b| a.acc_rate.partial_cmp(&b.acc_rate).unwrap());
        for r in worst.iter().take(5) {
            println!("  β={:.1} α={:.1}: {:.1}% (δw={:.3}, δθ={:.3})",
                     r.beta, r.alpha, r.acc_rate * 100.0, r.final_dw, r.final_dth);
        }
    } else {
        println!("\n✓ All points achieved reasonable acceptance rates!");
    }
    
    // Step size adaptation summary
    println!("\nStep size ranges:");
    let dw_range = rows.iter().map(|r| r.final_dw);
    let dth_range = rows.iter().map(|r| r.final_dth);
    println!("  δw:  {:.3} - {:.3}", 
             dw_range.clone().fold(f64::INFINITY, f64::min),
             dw_range.fold(0.0, f64::max));
    println!("  δθ: {:.3} - {:.3}", 
             dth_range.clone().fold(f64::INFINITY, f64::min),
             dth_range.fold(0.0, f64::max));
}