use scan::graph::Graph;
use rand_pcg::Pcg64;
use rand::SeedableRng;
use std::collections::HashMap;

fn main() {
    println!("=== Low Temperature Degeneracy Test ===\n");
    
    let n = 48;
    let alpha = 1.48;
    
    // Test at increasingly low temperatures
    let betas = vec![5.0, 10.0, 20.0, 50.0, 100.0];
    
    for &beta in &betas {
        println!("\nβ = {} (T = {:.4}):", beta, 1.0/beta);
        
        let mut rng = Pcg64::seed_from_u64(42 + (beta * 1000.0) as u64);
        let mut graph = Graph::complete_random_with(&mut rng, n);
        
        // Equilibrate at low temperature with smaller steps
        let delta = 0.2 / (beta as f64).sqrt();
        println!("  Equilibrating with δ = {:.4}...", delta);
        
        let eq_steps = 50000;
        let mut accepts = 0;
        for _ in 0..eq_steps {
            let info = graph.metropolis_step(beta, alpha, delta, delta, &mut rng);
            if info.accepted {
                accepts += 1;
            }
        }
        
        let accept_rate = accepts as f64 / eq_steps as f64;
        println!("  Acceptance rate: {:.1}%", accept_rate * 100.0);
        
        // Collect energy samples
        let mut energies = Vec::new();
        let mut min_energy = f64::INFINITY;
        let mut max_energy = f64::NEG_INFINITY;
        
        for _ in 0..10000 {
            for _ in 0..10 {
                graph.metropolis_step(beta, alpha, delta, delta, &mut rng);
            }
            let e = graph.action(alpha, beta);
            energies.push(e);
            min_energy = min_energy.min(e);
            max_energy = max_energy.max(e);
        }
        
        // Calculate entropy from energy distribution
        let mut histogram: HashMap<i64, usize> = HashMap::new();
        let bin_size = 0.001;
        
        for &e in &energies {
            let bin = ((e - min_energy) / bin_size).round() as i64;
            *histogram.entry(bin).or_insert(0) += 1;
        }
        
        // Shannon entropy
        let mut entropy = 0.0;
        let total = energies.len() as f64;
        for &count in histogram.values() {
            if count > 0 {
                let p = count as f64 / total;
                entropy -= p * p.ln();
            }
        }
        
        // Statistics
        let mean_e = energies.iter().sum::<f64>() / energies.len() as f64;
        let var_e = energies.iter().map(|&e| (e - mean_e).powi(2)).sum::<f64>() / energies.len() as f64;
        let specific_heat = beta * beta * var_e / n as f64;
        
        println!("  Energy range: [{:.6}, {:.6}]", min_energy, max_energy);
        println!("  Energy spread: {:.6}", max_energy - min_energy);
        println!("  Occupied bins: {}", histogram.len());
        println!("  Shannon entropy: {:.4}", entropy);
        println!("  Specific heat: {:.4}", specific_heat);
        
        // Check for discrete energy levels
        if histogram.len() < 20 {
            println!("  Energy level structure:");
            let mut levels: Vec<_> = histogram.iter().collect();
            levels.sort_by_key(|&(bin, _)| bin);
            
            for (i, (&bin, &count)) in levels.iter().take(10).enumerate() {
                let e = min_energy + bin as f64 * bin_size;
                let prob = count as f64 / total;
                println!("    Level {}: E = {:.6}, P = {:.4}", i, e, prob);
            }
        }
        
        // Degeneracy estimate
        let effective_states = entropy.exp();
        println!("  Effective # of states: {:.0}", effective_states);
        println!("  Degeneracy per site: {:.3}", (effective_states.ln() / n as f64).exp());
    }
    
    println!("\n=== ANALYSIS ===");
    println!("If entropy remains finite as T→0, the system has:");
    println!("- Extensive ground state degeneracy");
    println!("- No unique ground state");
    println!("- Possible spin liquid behavior");
}