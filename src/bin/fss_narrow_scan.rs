// src/bin/fss_narrow_scan.rs - Narrow scan with configurable system size

use std::{fs::File, io::BufReader, path::PathBuf};

use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use indicatif::{ProgressBar, ProgressStyle};
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use scan::graph::{Graph, StepInfo};

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "pairs.csv")]
    pairs: PathBuf,
    
    #[arg(long, default_value = "fss_results.csv")]
    output: PathBuf,
    
    /// Number of nodes (system size)
    #[arg(long, short, default_value = "48")]
    nodes: usize,
    
    /// Number of MC steps
    #[arg(long, default_value = "200000")]
    steps: usize,
    
    /// Number of replicas
    #[arg(long, short, default_value = "5")]
    replicas: usize,
    
    #[arg(long)]
    debug: bool,
}

/// Configuration that scales with system size
struct MCConfig {
    n_nodes:      usize,
    n_steps:      usize,
    equil_steps:  usize,
    sample_every: usize,
    n_rep:        usize,
    tune_win:     usize,
    tune_tgt:     f64,
    tune_band:    f64,
}

impl MCConfig {
    fn new(n_nodes: usize, n_steps: Option<usize>, n_rep: usize) -> Self {
        // Scale steps with sqrt(48/N) to maintain statistics
        let scale = (48.0 / n_nodes as f64).sqrt();
        let base_steps = 200_000;
        let base_equil = 80_000;
        
        Self {
            n_nodes,
            n_steps: n_steps.unwrap_or((base_steps as f64 * scale) as usize),
            equil_steps: (base_equil as f64 * scale) as usize,
            sample_every: 10,
            n_rep,
            tune_win: 200,
            tune_tgt: 0.30,
            tune_band: 0.05,
        }
    }
}

#[derive(Debug)]
struct ResultRow {
    beta:           f64,
    alpha:          f64,
    mean_w:         f64,
    std_w:          f64,
    mean_cos:       f64,
    std_cos:        f64,
    chi:            f64,
    mean_action:    f64,
    std_action:     f64,
    binder:         f64,  // Binder cumulant
    autocorr_time:  f64,
    acceptance:     f64,
}

#[derive(Default, Clone)]
struct OnlineStats {
    n: u64,
    mean: f64,
    m2: f64,
    m4_sum: f64,  // For 4th moment
}

impl OnlineStats {
    fn push(&mut self, x: f64) {
        self.n += 1;
        let delta  = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2   += delta * delta2;
        self.m4_sum += x.powi(4);
    }
    
    fn mean(&self) -> f64 { self.mean }
    fn var(&self)  -> f64 { if self.n > 1 { self.m2 / (self.n - 1) as f64 } else { 0.0 } }
    fn std(&self)  -> f64 { self.var().sqrt() }
    fn moment4(&self) -> f64 { if self.n > 0 { self.m4_sum / self.n as f64 } else { 0.0 } }
    
    fn binder_cumulant(&self) -> f64 {
        let m2 = self.var() + self.mean().powi(2);
        let m4 = self.moment4();
        if m2 > 0.0 { 1.0 - m4 / (3.0 * m2 * m2) } else { 0.0 }
    }
}

struct Tuner {
    delta: f64,
    attempts: usize,
    accepted: usize,
    win: usize,
    tgt: f64,
    band: f64,
    total_attempts: usize,
    total_accepted: usize,
}

impl Tuner {
    fn new(delta: f64, win: usize, tgt: f64, band: f64) -> Self {
        Self { 
            delta, attempts: 0, accepted: 0, win, tgt, band,
            total_attempts: 0, total_accepted: 0,
        }
    }
    
    fn update(&mut self, acc: bool) {
        self.attempts += 1;
        self.total_attempts += 1;
        if acc { 
            self.accepted += 1;
            self.total_accepted += 1;
        }
        
        if self.attempts == self.win {
            let r = self.accepted as f64 / self.win as f64;
            if r > self.tgt + self.band { 
                self.delta *= 1.1;
            } else if r < self.tgt - self.band { 
                self.delta *= 0.9;
            }
            self.attempts = 0;
            self.accepted = 0;
        }
    }
    
    fn acceptance_rate(&self) -> f64 {
        if self.total_attempts == 0 { 0.0 }
        else { self.total_accepted as f64 / self.total_attempts as f64 }
    }
}

struct TimeSeries {
    data: Vec<f64>,
}

impl TimeSeries {
    fn new() -> Self { Self { data: Vec::new() } }
    fn push(&mut self, x: f64) { self.data.push(x); }
    
    fn autocorrelation_time(&self) -> f64 {
        if self.data.len() < 100 { return 1.0; }
        
        let mean = self.data.iter().sum::<f64>() / self.data.len() as f64;
        let var = self.data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / self.data.len() as f64;
        
        if var == 0.0 { return 1.0; }
        
        let mut sum = 0.5;
        for t in 1..50.min(self.data.len() / 4) {
            let mut c_t = 0.0;
            for i in 0..self.data.len() - t {
                c_t += (self.data[i] - mean) * (self.data[i + t] - mean);
            }
            c_t /= (self.data.len() - t) as f64 * var;
            
            if c_t < 0.1 { break; }
            sum += c_t;
        }
        
        2.0 * sum
    }
}

fn run_single(beta: f64, alpha: f64, mc: &MCConfig, seed_modifier: u64, debug: bool) -> ResultRow {
    let links_per = mc.n_nodes * (mc.n_nodes - 1) / 2;
    
    let mut master = ChaCha20Rng::seed_from_u64(seed_modifier);
    let mut stats_w   = OnlineStats::default();
    let mut stats_cos = OnlineStats::default();
    let mut stats_action = OnlineStats::default();
    let mut action_series = TimeSeries::new();
    let mut total_acceptance = 0.0;
    let mut n_acceptance_samples = 0;

    for rep in 0..mc.n_rep {
        let mut rng = ChaCha20Rng::seed_from_u64(master.next_u64() ^ rep as u64);
        let mut g = Graph::complete_random_with(&mut rng, mc.n_nodes);
        
        let mut tuner_z  = Tuner::new(0.50, mc.tune_win, mc.tune_tgt, mc.tune_band);
        let mut tuner_th = Tuner::new(0.20, mc.tune_win, mc.tune_tgt, mc.tune_band);

        let mut sum_w   = g.sum_weights();
        let mut sum_cos = g.links_cos_sum();

        for step in 1..=mc.n_steps {
            let StepInfo { accepted, delta_w, delta_cos } =
                g.metropolis_step(beta, alpha, tuner_z.delta, tuner_th.delta, &mut rng);

            if accepted { 
                sum_w += delta_w; 
                sum_cos += delta_cos; 
            }
            
            tuner_z.update(accepted);
            tuner_th.update(accepted);

            if step > mc.equil_steps && step % mc.sample_every == 0 {
                let avg_w   = sum_w   / g.m() as f64;
                let avg_cos = sum_cos / g.m() as f64;
                let action = g.action(alpha, beta);
                
                stats_w.push(avg_w);
                stats_cos.push(avg_cos);
                stats_action.push(action);
                action_series.push(action);
            }
        }
        
        total_acceptance += tuner_z.acceptance_rate();
        n_acceptance_samples += 1;
        
        if debug && rep == 0 {
            eprintln!("N={} β={:.2} α={:.2} rep={}: action={:.1} accept={:.2}", 
                     mc.n_nodes, beta, alpha, rep, 
                     g.action(alpha, beta), 
                     tuner_z.acceptance_rate());
        }
    }

    let chi = links_per as f64 * stats_cos.var();
    let binder = stats_cos.binder_cumulant();
    let autocorr_time = action_series.autocorrelation_time();
    let acceptance = total_acceptance / n_acceptance_samples as f64;
    
    ResultRow {
        beta,
        alpha,
        mean_w:        stats_w.mean(),
        std_w:         stats_w.std(),
        mean_cos:      stats_cos.mean(),
        std_cos:       stats_cos.std(),
        chi,
        mean_action:   stats_action.mean(),
        std_action:    stats_action.std(),
        binder,
        autocorr_time,
        acceptance,
    }
}

fn main() {
    let args = Cli::parse();
    let mc = MCConfig::new(args.nodes, Some(args.steps), args.replicas);

    println!("FSS narrow scan - N={} nodes", mc.n_nodes);
    println!("Steps: {} (equil: {})", mc.n_steps, mc.equil_steps);
    println!("Replicas: {}", mc.n_rep);

    // Read (β, α) pairs
    let file = File::open(&args.pairs).expect("cannot open pairs file");
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(BufReader::new(file));

    let pairs: Vec<(f64, f64)> = rdr.records()
        .map(|r| {
            let rec = r.expect("bad CSV");
            let β: f64 = rec[0].parse().unwrap();
            let α: f64 = rec[1].parse().unwrap();
            (β, α)
        })
        .collect();

    println!("Running {} (β,α) points", pairs.len());

    let bar = ProgressBar::new(pairs.len() as u64);
    bar.set_style(ProgressStyle::with_template(
        " {bar:40.cyan/blue} {pos}/{len} [{elapsed_precise}]"
    ).unwrap());

    // Parallel execution
    let rows: Vec<ResultRow> = pairs
        .par_iter()
        .enumerate()
        .map(|(idx, &(β, α))| {
            let row = run_single(β, α, &mc, (idx as u64) << 32 | (args.nodes as u64), args.debug);
            bar.inc(1);
            row
        })
        .collect();

    bar.finish();

    // Write results
    let mut wtr = WriterBuilder::new()
        .from_path(&args.output)
        .expect("cannot create result file");

    wtr.write_record([
        "n_nodes", "beta", "alpha", "mean_w", "std_w", "mean_cos", "std_cos", 
        "susceptibility", "mean_action", "std_action", "binder", "autocorr_time", "acceptance"
    ]).unwrap();

    let mut rows = rows;
    rows.sort_by(|a, b| a.beta.partial_cmp(&b.beta).unwrap()
                      .then(a.alpha.partial_cmp(&b.alpha).unwrap()));

    for r in &rows {
        wtr.write_record([
            mc.n_nodes.to_string(),
            r.beta.to_string(),
            r.alpha.to_string(),
            r.mean_w.to_string(),
            r.std_w.to_string(),
            r.mean_cos.to_string(),
            r.std_cos.to_string(),
            r.chi.to_string(),
            r.mean_action.to_string(),
            r.std_action.to_string(),
            r.binder.to_string(),
            r.autocorr_time.to_string(),
            r.acceptance.to_string(),
        ]).unwrap();
    }
    
    wtr.flush().unwrap();
    println!("Done → {}", args.output.display());
}