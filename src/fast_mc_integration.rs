// Integration module for using the optimized FastGraph in simulations

use crate::graph_fast::{FastGraph, BatchedObservables, QuickObservables};
use crate::observables::Observables;
use rand_pcg::Pcg64;
use rand::SeedableRng;
use std::time::Instant;

/// Fast Monte Carlo runner with all optimizations
pub struct FastMCRunner {
    graph: FastGraph,
    rng: Pcg64,
    observable_calc: BatchedObservables,
    step_count: usize,
    equilibrated: bool,
}

impl FastMCRunner {
    /// Create new fast MC runner
    pub fn new(n: usize, seed: u64) -> Self {
        let graph = FastGraph::new(n, seed);
        let rng = Pcg64::seed_from_u64(seed);
        
        Self {
            graph,
            rng,
            observable_calc: BatchedObservables::new(),
            step_count: 0,
            equilibrated: false,
        }
    }
    
    /// Run equilibration with autocorrelation tracking
    pub fn equilibrate(
        &mut self, 
        alpha: f64, 
        beta: f64,
        delta_z: f64,
        delta_theta: f64,
        min_steps: usize,
    ) -> f64 {
        println!("Starting equilibration (min {} steps)...", min_steps);
        let start = Instant::now();
        
        // Track energy for autocorrelation
        for step in 0..min_steps {
            let info = self.graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut self.rng);
            
            // Update autocorrelation estimate every 10 steps
            if step % 10 == 0 {
                let energy = beta * self.graph.entropy_action() + alpha * self.graph.triangle_sum();
                self.graph.update_autocorrelation(energy);
            }
            
            // Progress report
            if step > 0 && step % 10000 == 0 {
                let tau = self.graph.get_autocorr_tau();
                let interval = self.graph.get_measurement_interval();
                println!("  Step {}: τ ≈ {:.1}, measurement interval = {}", step, tau, interval);
            }
        }
        
        self.equilibrated = true;
        let elapsed = start.elapsed();
        let steps_per_sec = min_steps as f64 / elapsed.as_secs_f64();
        
        println!("Equilibration complete in {:.2}s ({:.0} steps/sec)", 
                 elapsed.as_secs_f64(), steps_per_sec);
        println!("Final autocorrelation time: {:.1}", self.graph.get_autocorr_tau());
        println!("Optimal measurement interval: {}", self.graph.get_measurement_interval());
        
        steps_per_sec
    }
    
    /// Run production with optimized measurement frequency
    pub fn run_production(
        &mut self,
        alpha: f64,
        beta: f64,
        delta_z: f64,
        delta_theta: f64,
        n_measurements: usize,
    ) -> Vec<QuickObservables> {
        if !self.equilibrated {
            panic!("Must equilibrate before production run!");
        }
        
        let measurement_interval = self.graph.get_measurement_interval();
        let total_steps = n_measurements * measurement_interval;
        
        println!("Starting production run...");
        println!("  {} measurements every {} steps = {} total steps", 
                 n_measurements, measurement_interval, total_steps);
        
        let start = Instant::now();
        let mut measurements = Vec::with_capacity(n_measurements);
        let mut accepts = 0;
        
        for i in 0..n_measurements {
            // Run MC steps between measurements
            for _ in 0..measurement_interval {
                let info = self.graph.metropolis_step(alpha, beta, delta_z, delta_theta, &mut self.rng);
                if info.accept {
                    accepts += 1;
                }
                self.step_count += 1;
            }
            
            // Measure observables (with rotation for expensive ones)
            let obs = self.observable_calc.measure(&self.graph, alpha, beta);
            measurements.push(obs);
            
            // Progress report
            if (i + 1) % 100 == 0 {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = (i + 1) as f64 * measurement_interval as f64 / elapsed;
                println!("  Measurement {}/{} ({:.0} steps/sec)", i + 1, n_measurements, rate);
            }
        }
        
        let elapsed = start.elapsed();
        let accept_rate = accepts as f64 / total_steps as f64;
        let steps_per_sec = total_steps as f64 / elapsed.as_secs_f64();
        
        println!("Production complete in {:.2}s", elapsed.as_secs_f64());
        println!("  Accept rate: {:.1}%", accept_rate * 100.0);
        println!("  Performance: {:.0} steps/sec", steps_per_sec);
        println!("  Effective samples: {} (τ = {:.1})", n_measurements, self.graph.get_autocorr_tau());
        
        measurements
    }
    
    /// Convert QuickObservables to full Observables struct
    pub fn to_full_observables(&self, quick: &QuickObservables, alpha: f64, beta: f64) -> Observables {
        let n = self.graph.n() as f64;
        let m = self.graph.m() as f64;
        
        // Calculate variance from susceptibility
        let link_variance = quick.susceptibility / n;
        
        // Binder cumulant approximation
        let binder_cumulant = 1.0 - link_variance / (3.0 * quick.mean_w * quick.mean_w);
        
        Observables {
            mean_w: quick.mean_w,
            mean_cos: quick.mean_cos,
            entropy: self.graph.entropy_action(),
            triangle_sum: self.graph.triangle_sum(),
            susceptibility: quick.susceptibility,
            specific_heat: 0.0,  // Placeholder
            binder_cumulant,
            magnetization: quick.mean_cos.abs(),
            link_variance,
            correlation_length: 0.0,  // Placeholder
            spectral_gap: 0.0,  // Placeholder
        }
    }
}

/// Benchmark comparison of optimizations
pub fn benchmark_optimizations(n: usize, steps: usize) {
    use crate::graph::Graph;
    use rand::thread_rng;
    
    println!("\n=== Performance Benchmark: Original vs Optimized ===");
    println!("System size: N = {}", n);
    println!("MC steps: {}\n", steps);
    
    let alpha = 1.5;
    let beta = 2.9;
    let delta_z = 0.1;
    let delta_theta = 0.1;
    
    // Original implementation
    println!("Original implementation:");
    let start = Instant::now();
    let mut orig_graph = Graph::complete_random(n);
    let mut orig_rng = thread_rng();
    let mut orig_accepts = 0;
    
    for _ in 0..steps {
        let info = orig_graph.metropolis_step(beta, alpha, delta_z, delta_theta, &mut orig_rng);
        if info.accepted {
            orig_accepts += 1;
        }
    }
    
    let orig_time = start.elapsed();
    let orig_rate = steps as f64 / orig_time.as_secs_f64();
    println!("  Time: {:.2}s", orig_time.as_secs_f64());
    println!("  Rate: {:.0} steps/sec", orig_rate);
    println!("  Accept: {:.1}%", 100.0 * orig_accepts as f64 / steps as f64);
    
    // Optimized implementation
    println!("\nOptimized implementation:");
    let start = Instant::now();
    let mut fast_runner = FastMCRunner::new(n, 12345);
    let mut fast_accepts = 0;
    
    for _ in 0..steps {
        let info = fast_runner.graph.metropolis_step(
            alpha, beta, delta_z, delta_theta, &mut fast_runner.rng
        );
        if info.accept {
            fast_accepts += 1;
        }
    }
    
    let fast_time = start.elapsed();
    let fast_rate = steps as f64 / fast_time.as_secs_f64();
    println!("  Time: {:.2}s", fast_time.as_secs_f64());
    println!("  Rate: {:.0} steps/sec", fast_rate);
    println!("  Accept: {:.1}%", 100.0 * fast_accepts as f64 / steps as f64);
    
    // Summary
    println!("\nSpeedup: {:.1}x", fast_rate / orig_rate);
    
    // Breakdown of improvements
    println!("\nOptimization breakdown (estimated):");
    println!("  - Pcg64 vs ChaCha20 RNG: ~2.0x");
    println!("  - Precomputed trig values: ~1.3x");
    println!("  - Cache-friendly layout: ~1.2x");
    println!("  - Inline and fast-math: ~1.1x");
    println!("  - Combined effect: ~{:.1}x", fast_rate / orig_rate);
}

/// Example usage of fast MC
pub fn example_fast_simulation() {
    let n = 48;
    let seed = 42;
    let alpha = 1.5;
    let beta = 2.9;
    
    // Create fast runner
    let mut runner = FastMCRunner::new(n, seed);
    
    // Equilibrate with autocorrelation tracking
    runner.equilibrate(alpha, beta, 0.1, 0.1, 50000);
    
    // Run production with optimal measurement frequency
    let measurements = runner.run_production(alpha, beta, 0.1, 0.1, 1000);
    
    // Analyze results
    let mean_chi = measurements.iter().map(|o| o.susceptibility).sum::<f64>() / measurements.len() as f64;
    let mean_w = measurements.iter().map(|o| o.mean_w).sum::<f64>() / measurements.len() as f64;
    
    println!("\nResults:");
    println!("  <χ> = {:.2}", mean_chi);
    println!("  <w> = {:.6}", mean_w);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fast_runner() {
        let mut runner = FastMCRunner::new(24, 12345);
        runner.equilibrate(1.0, 1.0, 0.1, 0.1, 1000);
        let measurements = runner.run_production(1.0, 1.0, 0.1, 0.1, 100);
        
        assert_eq!(measurements.len(), 100);
        assert!(measurements[0].mean_w > 0.0);
    }
    
    #[test]
    fn test_benchmark() {
        benchmark_optimizations(24, 10000);
    }
}