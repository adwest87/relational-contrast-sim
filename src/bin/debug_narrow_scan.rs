// src/bin/debug_narrow_scan.rs
//! Narrow scan focused on susceptibility ridge with debugging and larger system
//! Includes action monitoring, alternative C calculation, and detailed diagnostics

use std::{fs::File, io::BufReader, path::PathBuf};
use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use scan::graph::{Graph, StepInfo};
use std::sync::Mutex;

#[derive(Clone, Debug)]
struct DebugConfig {
    n_nodes:      usize,
    n_steps:      usize,
    equil_steps:  usize,
    sample_every: usize,
    n_rep:        usize,
    tune_win:     usize,
    tune_tgt:     f64,
    tune_band:    f64,
    // Debug options
    log_large_actions: bool,
    action_threshold: f64,
    log_equilibration: bool,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            n_nodes:      48,      // Larger system for better finite-size behavior
            n_steps:      80_000,  // More steps for better statistics
            equil_steps:  20_000,  // Longer equilibration
            sample_every: 10,
            n_rep:        5,       // More replicas for error estimation
            tune_win:     200,
            tune_tgt:     0.30,
            tune_band:    0.05,
            // Debug settings
            log_large_actions: true,
            action_threshold: 1e4,
            log_equilibration: false,
        }
    }
}

#[derive(Parser)]
struct Cli {
    /// CSV with ridge points (default: susceptibility ridge)
    #[arg(long, default_value = "ridge_points_chi.csv")]
    ridge: PathBuf,
    
    /// Output file
    #[arg(long, default_value = "debug_narrow_results.csv")]
    output: PathBuf,
    
    /// Enable verbose debugging
    #[arg(long, short)]
    verbose: bool,
    
    /// Use smaller test system (24 nodes)
    #[arg(long)]
    small: bool,
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
    c_spec:       f64,      // From action variance
    c_energy:     f64,      // From energy fluctuations (more stable)
    s_bar:        f64,
    delta_bar:    f64,
    acc_rate:     f64,
    // New diagnostics
    max_action:   f64,
    min_weight:   f64,
    n_large_actions: usize,
    equilibrated: bool,
}

// Statistics collector with variance tracking
#[derive(Default, Clone)]
struct OnlineStats {
    n:    u64,
    mean: f64,
    m2:   f64,
    min:  f64,
    max:  f64,
}

impl OnlineStats {
    fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
    
    fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
        self.min = self.min.min(x);
        self.max = self.max.max(x);
    }
    
    fn mean(&self) -> f64 { self.mean }
    fn var(&self) -> f64 { if self.n > 1 { self.m2 / (self.n - 1) as f64 } else { 0.0 } }
    fn std(&self) -> f64 { self.var().sqrt() }
    fn count(&self) -> u64 { self.n }
}

// Smart adaptive tuner
struct AdaptiveTuner {
    delta: f64,
    attempts: usize,
    accepted: usize,
    window: usize,
    target: f64,
    band: f64,
    min_delta: f64,
    max_delta: f64,
    history: Vec<f64>,  // Track acceptance history
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
            min_delta: initial_delta * 0.05,
            max_delta: initial_delta * 20.0,
            history: Vec::with_capacity(100),
        }
    }
    
    fn update(&mut self, accepted: bool) {
        self.attempts += 1;
        if accepted { self.accepted += 1; }
        
        if self.attempts >= self.window {
            let rate = self.accepted as f64 / self.attempts as f64;
            self.history.push(rate);
            
            // Adaptive scaling based on deviation
            let error = (rate - self.target).abs();
            let scale = 1.0 + error.min(0.3);
            
            if rate > self.target + self.band {
                self.delta *= scale;
            } else if rate < self.target - self.band {
                self.delta /= scale;
            }
            
            self.delta = self.delta.clamp(self.min_delta, self.max_delta);
            self.accepted = 0;
            self.attempts = 0;
        }
    }
    
    fn current_rate(&self) -> f64 {
        if self.attempts > 0 {
            self.accepted as f64 / self.attempts as f64
        } else if !self.history.is_empty() {
            self.history[self.history.len() - 1]
        } else {
            0.0
        }
    }
}

// Get initial step sizes for the critical region
fn get_critical_region_steps(beta: f64, alpha: f64) -> (f64, f64) {
    // We're focusing on β ∈ [2.5, 3.5], α ∈ [1.0, 2.0]
    // This is near the susceptibility peak, so use moderate steps
    
    let delta_w = if alpha < 1.2 {
        0.20  // Slightly larger for lower α
    } else if alpha < 1.8 {
        0.15  // Critical region
    } else {
        0.12  // More ordered
    };
    
    let delta_theta = if alpha < 1.2 {
        0.35
    } else if alpha < 1.8 {
        0.25
    } else {
        0.20
    };
    
    (delta_w, delta_theta)
}

// Run a single (β, α) point with enhanced debugging
fn run_point_debug(
    beta: f64, 
    alpha: f64, 
    cfg: &DebugConfig, 
    seed: u64,
    verbose: bool
) -> Measurement {
    let links_per = cfg.n_nodes * (cfg.n_nodes - 1) / 2;
    let n_tri = cfg.n_nodes * (cfg.n_nodes - 1) * (cfg.n_nodes - 2) / 6;
    
    if verbose {
        eprintln!("\n=== Running β={:.2} α={:.2} ===", beta, alpha);
    }
    
    let mut master = ChaCha20Rng::seed_from_u64(seed);
    
    // Collect statistics across replicas
    let mut all_measurements = Vec::new();
    
    for rep in 0..cfg.n_rep {
        let mut rng = ChaCha20Rng::seed_from_u64(master.next_u64() ^ rep as u64);
        let mut g = Graph::complete_random_with(&mut rng, cfg.n_nodes);
        
        let (init_dw, init_dth) = get_critical_region_steps(beta, alpha);
        let mut tuner_w = AdaptiveTuner::new(init_dw, cfg.tune_win, cfg.tune_tgt, cfg.tune_band);
        let mut tuner_th = AdaptiveTuner::new(init_dth, cfg.tune_win, cfg.tune_tgt, cfg.tune_band);
        
        let mut sum_w = g.sum_weights();
        let mut sum_cos = g.links_cos_sum();
        
        // Track observables
        let mut stats_cos = OnlineStats::new();
        let mut stats_se = OnlineStats::new();
        let mut stats_tri = OnlineStats::new();
        let mut stats_action = OnlineStats::new();
        let mut stats_energy = OnlineStats::new();  // For alternative C calculation
        
        let mut n_large_actions = 0;
        let mut accepted_total = 0;
        let mut equilibrated = false;
        let mut equil_check_actions = Vec::new();
        
        // Main MC loop
        for step in 1..=cfg.n_steps {
            let StepInfo { accepted, delta_w, delta_cos } =
                g.metropolis_step(beta, alpha, tuner_w.delta, tuner_th.delta, &mut rng);
            
            if accepted {
                sum_w += delta_w;
                sum_cos += delta_cos;
                accepted_total += 1;
            }
            tuner_w.update(accepted);
            tuner_th.update(accepted);
            
            // Measure after equilibration
            if step > cfg.equil_steps {
                if step % cfg.sample_every == 0 {
                    let avg_cos = sum_cos / g.m() as f64;
                    let s_entropy = g.entropy_action();
                    let tri_sum = g.triangle_sum_norm();
                    let s_total = beta * s_entropy + alpha * tri_sum * n_tri as f64;
                    let energy = s_entropy;  // Energy is just entropy part
                    
                    // Check for large actions
                    if cfg.log_large_actions && s_total.abs() > cfg.action_threshold {
                        n_large_actions += 1;
                        if verbose {
                            eprintln!("  Large action: S={:.2e} at step {} (β={:.2} α={:.2})", 
                                     s_total, step, beta, alpha);
                        }
                    }
                    
                    stats_cos.push(avg_cos);
                    stats_se.push(s_entropy);
                    stats_tri.push(tri_sum);
                    stats_action.push(s_total);
                    stats_energy.push(energy);
                }
            } else if cfg.log_equilibration && step % 1000 == 0 {
                // Check equilibration progress
                let s_total = g.action(alpha, beta);
                equil_check_actions.push(s_total);
                
                if equil_check_actions.len() >= 10 {
                    // Simple equilibration check: is variance stabilizing?
                    let recent: Vec<_> = equil_check_actions.iter().rev().take(5).copied().collect();
                    let recent_var = variance(&recent);
                    let older: Vec<_> = equil_check_actions.iter().rev().skip(5).take(5).copied().collect();
                    let older_var = variance(&older);
                    
                    if recent_var < older_var * 1.5 {
                        equilibrated = true;
                    }
                }
            }
        }
        
        // Calculate observables for this replica
        let chi = links_per as f64 * stats_cos.var();
        let c_action = stats_action.var() / links_per as f64;
        let c_energy = beta * beta * stats_energy.var() / links_per as f64;
        let min_weight = g.min_weight();
        let avg_w = sum_w / g.m() as f64;
        
        // Create stats for weights
        let mut stats_w = OnlineStats::new();
        for link in &g.links {
            stats_w.push(link.w());
        }
        
        all_measurements.push((
            avg_w,               // 0: mean_w
            stats_w.std(),       // 1: std_w
            stats_cos.mean(),    // 2: mean_cos
            stats_cos.std(),     // 3: std_cos
            chi,                 // 4: chi
            c_action,            // 5: c_spec
            c_energy,            // 6: c_energy
            stats_se.mean() / links_per as f64,  // 7: s_bar
            stats_tri.mean() / n_tri as f64,     // 8: delta_bar
            accepted_total as f64 / cfg.n_steps as f64,  // 9: acc_rate
            stats_action.max,    // 10: max_action
            min_weight,          // 11: min_weight
            n_large_actions,     // 12: n_large_actions
            equilibrated,        // 13: equilibrated
        ));
        
        if verbose {
            eprintln!("  Rep {}: acc={:.1}% C_action={:.1} C_energy={:.1} max_S={:.1}", 
                     rep, 
                     100.0 * accepted_total as f64 / cfg.n_steps as f64,
                     c_action, c_energy, stats_action.max);
        }
    }
    
    // Average over replicas
    let n = all_measurements.len() as f64;
    Measurement {
        beta,
        alpha,
        mean_w:       all_measurements.iter().map(|m| m.0).sum::<f64>() / n,
        std_w:        all_measurements.iter().map(|m| m.1).sum::<f64>() / n,
        mean_cos:     all_measurements.iter().map(|m| m.2).sum::<f64>() / n,
        std_cos:      all_measurements.iter().map(|m| m.3).sum::<f64>() / n,
        chi:          all_measurements.iter().map(|m| m.4).sum::<f64>() / n,
        c_spec:       all_measurements.iter().map(|m| m.5).sum::<f64>() / n,
        c_energy:     all_measurements.iter().map(|m| m.6).sum::<f64>() / n,
        s_bar:        all_measurements.iter().map(|m| m.7).sum::<f64>() / n,
        delta_bar:    all_measurements.iter().map(|m| m.8).sum::<f64>() / n,
        acc_rate:     all_measurements.iter().map(|m| m.9).sum::<f64>() / n,
        max_action:   all_measurements.iter().map(|m| m.10).fold(0.0, f64::max),
        min_weight:   all_measurements.iter().map(|m| m.11).sum::<f64>() / n,
        n_large_actions: all_measurements.iter().map(|m| m.12).sum::<usize>(),
        equilibrated: all_measurements.iter().all(|m| m.13),
    }
}

// Helper function for variance
fn variance(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

fn main() {
    let args = Cli::parse();
    let mut cfg = DebugConfig::default();
    
    if args.small {
        cfg.n_nodes = 24;
        cfg.n_steps = 40_000;
        cfg.equil_steps = 10_000;
        println!("Using small system (24 nodes) for testing");
    }
    
    cfg.log_equilibration = args.verbose;
    
    println!("Debug Narrow Scan Configuration:");
    println!("  System size: {} nodes ({} links)", cfg.n_nodes, cfg.n_nodes * (cfg.n_nodes - 1) / 2);
    println!("  Steps: {} (equil: {})", cfg.n_steps, cfg.equil_steps);
    println!("  Replicas: {}", cfg.n_rep);
    println!("  Debug: large action threshold = {:.0e}", cfg.action_threshold);
    
    // Read ridge points
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
    
    println!("\nRunning {} (β,α) points near susceptibility ridge", points.len());
    
    let bar = ProgressBar::new(points.len() as u64);
    bar.set_style(ProgressStyle::with_template(
        " {bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}] eta: {eta}"
    ).unwrap());
    
    let measurements = Mutex::new(Vec::new());
    
    // Parallel execution
    points.par_iter().enumerate().for_each(|(idx, &(beta, alpha))| {
        let m = run_point_debug(beta, alpha, &cfg, idx as u64, args.verbose);
        measurements.lock().unwrap().push(m);
        bar.inc(1);
    });
    
    bar.finish();
    
    // Sort and save results
    let mut measurements = measurements.into_inner().unwrap();
    measurements.sort_by(|a, b| a.beta.partial_cmp(&b.beta).unwrap()
        .then(a.alpha.partial_cmp(&b.alpha).unwrap()));
    
    // Write detailed CSV
    let mut wtr = WriterBuilder::new()
        .from_path(&args.output)
        .expect("cannot create output file");
    
    wtr.write_record(&[
        "beta", "alpha", "mean_w", "std_w", "mean_cos", "std_cos", "susceptibility", 
        "C_action", "C_energy", "S_bar", "Delta_bar", "acc_rate",
        "max_action", "min_weight", "n_large_actions", "equilibrated"
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
            m.c_energy.to_string(),
            m.s_bar.to_string(),
            m.delta_bar.to_string(),
            m.acc_rate.to_string(),
            m.max_action.to_string(),
            m.min_weight.to_string(),
            m.n_large_actions.to_string(),
            m.equilibrated.to_string(),
        ]).unwrap();
    }
    
    wtr.flush().unwrap();
    println!("\nResults saved → {}", args.output.display());
    
    // Analysis summary
    println!("\n=== ANALYSIS SUMMARY ===");
    
    // Find peaks
    let max_chi = measurements.iter()
        .max_by(|a, b| a.chi.partial_cmp(&b.chi).unwrap())
        .unwrap();
    let max_c_energy = measurements.iter()
        .max_by(|a, b| a.c_energy.partial_cmp(&b.c_energy).unwrap())
        .unwrap();
    
    println!("\nPeak susceptibility:");
    println!("  χ = {:.3} at β={:.2}, α={:.2}", max_chi.chi, max_chi.beta, max_chi.alpha);
    
    println!("\nPeak specific heat (energy method):");
    println!("  C = {:.3} at β={:.2}, α={:.2}", max_c_energy.c_energy, max_c_energy.beta, max_c_energy.alpha);
    
    // Check for numerical issues
    let problematic: Vec<_> = measurements.iter()
        .filter(|m| m.n_large_actions > 10 || m.c_spec > 10000.0)
        .collect();
    
    if !problematic.is_empty() {
        println!("\n⚠️  Points with numerical issues:");
        for m in problematic.iter().take(5) {
            println!("  β={:.2} α={:.2}: {} large actions, C_action={:.0}", 
                     m.beta, m.alpha, m.n_large_actions, m.c_spec);
        }
        println!("\n💡 C_energy values are more stable than C_action");
    }
    
    // Acceptance rate summary
    let avg_acc = measurements.iter().map(|m| m.acc_rate).sum::<f64>() / measurements.len() as f64;
    println!("\nAcceptance rate: {:.1}% average", avg_acc * 100.0);
    
    // Equilibration check
    let n_equilibrated = measurements.iter().filter(|m| m.equilibrated).count();
    println!("Equilibration: {}/{} points confirmed equilibrated", n_equilibrated, measurements.len());
    
    // Suggest next steps
    println!("\n=== NEXT STEPS ===");
    if max_chi.chi > 10.0 {
        println!("✓ Clear critical point detected at β={:.2}, α={:.2}", max_chi.beta, max_chi.alpha);
        println!("  → Run finite-size scaling with n_nodes = 24, 48, 96");
    } else {
        println!("? Weak critical signal - consider:");
        println!("  → Expanding search region");
        println!("  → Increasing system size further");
    }
}

// Create the ridge points file based on susceptibility analysis
pub fn create_ridge_points_file() {
    let ridge_points = vec![
        // Main ridge around the peak
        (2.5, 1.8), (2.6, 1.7), (2.7, 1.6), (2.8, 1.55), (2.9, 1.5),
        (3.0, 1.5), (3.1, 1.5), (3.2, 1.55), (3.3, 1.6), (3.4, 1.7), (3.5, 1.8),
        // Cross-sections perpendicular to ridge
        (2.8, 1.4), (2.9, 1.4), (3.0, 1.4), (3.1, 1.4), (3.2, 1.4),
        (2.8, 1.6), (2.9, 1.6), (3.0, 1.6), (3.1, 1.6), (3.2, 1.6),
        // Additional points near the peak
        (2.9, 1.45), (3.0, 1.45), (3.1, 1.45),
        (2.9, 1.55), (3.0, 1.55), (3.1, 1.55),
        // Extend slightly beyond
        (2.4, 1.9), (2.5, 1.85), (3.5, 1.85), (3.6, 1.9),
    ];
    
    let mut wtr = WriterBuilder::new()
        .has_headers(false)
        .from_path("ridge_points_chi.csv")
        .expect("cannot create ridge points file");
    
    for (beta, alpha) in ridge_points {
        wtr.write_record(&[beta.to_string(), alpha.to_string()]).unwrap();
    }
    
    wtr.flush().unwrap();
    println!("Created ridge_points_chi.csv with {} points", 28);
}