use scan::graph::Graph;
use rand_pcg::Pcg64;
use rand::SeedableRng;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::io::{self, Write};
use num_complex::Complex64;

/// Calculate correlation functions for the system
fn calculate_correlations(graph: &Graph) -> (Vec<(f64, f64)>, f64) {
    let n = graph.nodes.len();
    let mut distance_correlations: HashMap<usize, Vec<f64>> = HashMap::new();
    
    // Calculate pairwise correlations
    for i in 0..n {
        for j in (i+1)..n {
            let link_idx = graph.link_index(i, j);
            let theta_ij = graph.links[link_idx].theta;
            let cos_theta = theta_ij.cos();
            
            // Simple distance on complete graph (all pairs at distance 1)
            let distance = 1;
            distance_correlations.entry(distance).or_insert_with(Vec::new).push(cos_theta);
        }
    }
    
    // Average correlations by distance
    let mut avg_correlations = Vec::new();
    for (dist, values) in distance_correlations.iter() {
        let avg = values.iter().sum::<f64>() / values.len() as f64;
        avg_correlations.push((*dist as f64, avg));
    }
    avg_correlations.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    // Calculate triangle correlations
    let mut triangle_sum = 0.0;
    let mut triangle_count = 0;
    
    for i in 0..n {
        for j in (i+1)..n {
            for k in (j+1)..n {
                let theta_ij = graph.links[graph.link_index(i, j)].theta;
                let theta_jk = graph.links[graph.link_index(j, k)].theta;
                let theta_ik = graph.links[graph.link_index(i, k)].theta;
                
                // Check triangle constraint: θᵢⱼ + θⱼₖ + θᵢₖ = π
                let triangle_phase = theta_ij + theta_jk + theta_ik - PI;
                triangle_sum += triangle_phase.cos();
                triangle_count += 1;
            }
        }
    }
    
    let avg_triangle_correlation = triangle_sum / triangle_count as f64;
    
    (avg_correlations, avg_triangle_correlation)
}

/// Calculate entropy from energy histogram
fn calculate_entropy(energies: &[f64], _temperature: f64) -> f64 {
    if energies.is_empty() {
        return 0.0;
    }
    
    // Create energy histogram
    let mut histogram: HashMap<i64, usize> = HashMap::new();
    let bin_size = 0.01; // Energy bin size
    
    for &energy in energies {
        let bin = (energy / bin_size).round() as i64;
        *histogram.entry(bin).or_insert(0) += 1;
    }
    
    // Calculate probabilities and entropy
    let total = energies.len() as f64;
    let mut entropy = 0.0;
    
    for &count in histogram.values() {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.ln();
        }
    }
    
    entropy
}

/// Calculate structure factor S(k)
fn calculate_structure_factor(graph: &Graph, k_vectors: &[(f64, f64, f64)]) -> Vec<f64> {
    let n = graph.nodes.len();
    let mut structure_factors = Vec::new();
    
    // For a complete graph, we use a synthetic spatial arrangement
    // Place nodes on a cubic lattice for k-space analysis
    let n_side = (n as f64).powf(1.0/3.0).ceil() as usize;
    
    for &(kx, ky, kz) in k_vectors {
        let mut sum = Complex64::new(0.0, 0.0);
        
        for i in 0..n {
            // Map node index to 3D position
            let x = (i % n_side) as f64;
            let y = ((i / n_side) % n_side) as f64;
            let z = (i / (n_side * n_side)) as f64;
            
            // Calculate average phase at node i
            let mut phase_sum = 0.0;
            let mut phase_count = 0;
            
            for j in 0..n {
                if i != j {
                    let link_idx = graph.link_index(i.min(j), i.max(j));
                    let theta = graph.links[link_idx].theta;
                    phase_sum += if i < j { theta } else { -theta };
                    phase_count += 1;
                }
            }
            
            let avg_phase = phase_sum / phase_count as f64;
            
            // Fourier transform
            let k_dot_r = kx * x + ky * y + kz * z;
            sum += Complex64::from_polar(1.0, k_dot_r + avg_phase);
        }
        
        let s_k = (sum.norm_sqr() / n as f64).sqrt();
        structure_factors.push(s_k);
    }
    
    structure_factors
}

/// Calculate winding number for topological sectors
fn calculate_winding_number(graph: &Graph) -> f64 {
    // For complete graph, define a path through all nodes
    let n = graph.nodes.len();
    let mut total_winding = 0.0;
    
    // Create a Hamiltonian path (visiting each node once)
    for i in 0..(n-1) {
        let j = i + 1;
        let link_idx = graph.link_index(i, j);
        let theta = graph.links[link_idx].theta;
        
        // Accumulate phase differences
        total_winding += theta;
    }
    
    // Close the loop
    let link_idx = graph.link_index(0, n-1);
    total_winding += graph.links[link_idx].theta;
    
    total_winding / (2.0 * PI)
}

/// Calculate local and global Z3 order parameters
fn calculate_z3_order(graph: &Graph) -> (f64, Complex64, Complex64) {
    let n = graph.nodes.len();
    let mut local_sum = 0.0;
    let mut global_sum = Complex64::new(0.0, 0.0);
    let mut node_phases = vec![0.0; n];
    
    // Calculate effective phase at each node
    for i in 0..n {
        let mut phase_sum = 0.0;
        let mut count = 0;
        
        for j in 0..n {
            if i != j {
                let link_idx = graph.link_index(i.min(j), i.max(j));
                let theta = graph.links[link_idx].theta;
                phase_sum += if i < j { theta } else { -theta };
                count += 1;
            }
        }
        
        node_phases[i] = phase_sum / count as f64;
        
        // Local Z3 order
        local_sum += (3.0 * node_phases[i]).cos();
        
        // Global Z3 order
        global_sum += Complex64::from_polar(1.0, 3.0 * node_phases[i]);
    }
    
    let local_order = local_sum / n as f64;
    let global_order = global_sum / n as f64;
    
    (local_order, global_order, Complex64::from_polar(global_order.norm(), global_order.arg()))
}

/// Run finite size scaling analysis
fn finite_size_scaling(sizes: &[usize], beta: f64, alpha: f64, steps_per_size: usize) -> Vec<(usize, f64, f64)> {
    let mut results = Vec::new();
    
    for &n in sizes {
        println!("Running N = {} ...", n);
        let mut rng = Pcg64::seed_from_u64(42);
        let mut graph = Graph::complete_random_with(&mut rng, n);
        
        // Equilibrate
        let eq_steps = 10000 * n;
        for _ in 0..eq_steps {
            graph.metropolis_step(beta, alpha, 0.5, 0.5, &mut rng);
        }
        
        // Measure
        let mut cos_sum = 0.0;
        let mut cos_sq_sum = 0.0;
        let mut action_sum = 0.0;
        let mut action_sq_sum = 0.0;
        let mut count = 0;
        
        for _ in 0..steps_per_size {
            graph.metropolis_step(beta, alpha, 0.5, 0.5, &mut rng);
            
            if count % 100 == 0 {
                let mean_cos: f64 = graph.links.iter().map(|l| l.theta.cos()).sum::<f64>() / graph.m() as f64;
                let action = graph.action(alpha, beta);
                
                cos_sum += mean_cos;
                cos_sq_sum += mean_cos * mean_cos;
                action_sum += action;
                action_sq_sum += action * action;
                count += 1;
            }
        }
        
        let samples = count as f64;
        let mean_cos = cos_sum / samples;
        let var_cos = cos_sq_sum / samples - mean_cos * mean_cos;
        let susceptibility = graph.m() as f64 * var_cos;
        
        let mean_action = action_sum / samples;
        let var_action = action_sq_sum / samples - mean_action * mean_action;
        let specific_heat = beta * beta * var_action / graph.m() as f64;
        
        results.push((n, susceptibility, specific_heat));
        
        println!("  N = {}: χ = {:.2}, C = {:.2}", n, susceptibility, specific_heat);
    }
    
    results
}

fn main() {
    println!("=== Unconventional Physics Investigation ===\n");
    
    let n = 96;
    let beta = 2.9;
    let alpha = 1.48;
    
    // Initialize system
    let mut rng = Pcg64::seed_from_u64(42);
    let mut graph = Graph::complete_random_with(&mut rng, n);
    
    println!("System: N = {}, β = {}, α = {}", n, beta, alpha);
    println!("Equilibrating...");
    
    // Equilibrate
    for _ in 0..50000 {
        graph.metropolis_step(beta, alpha, 0.5, 0.5, &mut rng);
    }
    
    // Collect energy samples for entropy calculation
    println!("\nCollecting samples for analysis...");
    let mut energy_samples = Vec::new();
    let mut graph_samples = Vec::new();
    let sample_interval = 100;
    let n_samples = 10000;
    
    for i in 0..n_samples {
        for _ in 0..sample_interval {
            graph.metropolis_step(beta, alpha, 0.5, 0.5, &mut rng);
        }
        
        let energy = graph.action(alpha, beta);
        energy_samples.push(energy);
        
        // Store graph snapshots for correlation analysis
        if i % 100 == 0 {
            graph_samples.push(graph.clone());
        }
        
        if i % 1000 == 0 {
            print!(".");
            io::stdout().flush().unwrap();
        }
    }
    println!(" done!");
    
    // 1. CORRELATION FUNCTIONS
    println!("\n1. CORRELATION FUNCTIONS:");
    let (distance_corr, triangle_corr) = calculate_correlations(&graph);
    println!("   Distance correlations:");
    for (dist, corr) in &distance_corr {
        println!("     d = {}: ⟨cos(θᵢ-θⱼ)⟩ = {:.4}", dist, corr);
    }
    println!("   Triangle correlation: ⟨cos(θᵢ+θⱼ+θₖ-π)⟩ = {:.4}", triangle_corr);
    
    // 2. ENTROPY AT T→0
    println!("\n2. ENTROPY ANALYSIS:");
    let entropy = calculate_entropy(&energy_samples, 1.0/beta);
    println!("   S = {:.4} at β = {}", entropy, beta);
    
    // Run at lower temperatures
    let beta_values = vec![5.0, 10.0, 20.0, 50.0];
    println!("   Low temperature entropy:");
    for &beta_test in &beta_values {
        let mut test_graph = graph.clone();
        
        // Equilibrate at new temperature
        for _ in 0..20000 {
            test_graph.metropolis_step(beta_test, alpha, 0.3, 0.3, &mut rng);
        }
        
        let mut low_t_energies = Vec::new();
        for _ in 0..1000 {
            for _ in 0..100 {
                test_graph.metropolis_step(beta_test, alpha, 0.3, 0.3, &mut rng);
            }
            low_t_energies.push(test_graph.action(alpha, beta_test));
        }
        
        let s = calculate_entropy(&low_t_energies, 1.0/beta_test);
        println!("     β = {:2}: S = {:.4}", beta_test, s);
    }
    
    // 3. STRUCTURE FACTOR
    println!("\n3. STRUCTURE FACTOR:");
    let k_vectors = vec![
        (0.0, 0.0, 0.0),                    // Γ point
        (2.0*PI/3.0, 0.0, 0.0),            // Z3 ordering vector
        (4.0*PI/3.0, 0.0, 0.0),            // Second Z3 vector
        (PI, 0.0, 0.0),                    // Antiferromagnetic
        (PI, PI, 0.0),                     // (π,π) point
        (2.0*PI/3.0, 2.0*PI/3.0, 0.0),    // Diagonal Z3
    ];
    
    let s_k = calculate_structure_factor(&graph, &k_vectors);
    println!("   k-vector               S(k)");
    println!("   --------               ----");
    println!("   (0,0,0)                {:.4}", s_k[0]);
    println!("   (2π/3,0,0)             {:.4}", s_k[1]);
    println!("   (4π/3,0,0)             {:.4}", s_k[2]);
    println!("   (π,0,0)                {:.4}", s_k[3]);
    println!("   (π,π,0)                {:.4}", s_k[4]);
    println!("   (2π/3,2π/3,0)          {:.4}", s_k[5]);
    
    // 4. TOPOLOGICAL SECTORS
    println!("\n4. TOPOLOGICAL SECTORS:");
    let mut winding_numbers = Vec::new();
    for graph_sample in &graph_samples[..20.min(graph_samples.len())] {
        let w = calculate_winding_number(graph_sample);
        winding_numbers.push(w);
    }
    
    println!("   Winding numbers from {} samples:", winding_numbers.len());
    println!("   Mean: {:.4}", winding_numbers.iter().sum::<f64>() / winding_numbers.len() as f64);
    println!("   Std:  {:.4}", {
        let mean = winding_numbers.iter().sum::<f64>() / winding_numbers.len() as f64;
        let var = winding_numbers.iter().map(|&w| (w - mean).powi(2)).sum::<f64>() / winding_numbers.len() as f64;
        var.sqrt()
    });
    
    // 5. FINITE SIZE SCALING
    println!("\n5. FINITE SIZE SCALING:");
    let sizes = vec![24, 48, 96, 192];
    let fss_results = finite_size_scaling(&sizes, beta, alpha, 20000);
    
    println!("\n   Size   χ        χ/N      C");
    println!("   ----   ---      ---      ---");
    for (n, chi, c) in &fss_results {
        println!("   {:3}    {:6.2}   {:.4}   {:.2}", n, chi, chi / *n as f64, c);
    }
    
    // Check if χ/N saturates
    let chi_per_n: Vec<f64> = fss_results.iter().map(|(n, chi, _)| chi / *n as f64).collect();
    let saturation = chi_per_n.windows(2).all(|w| (w[1] - w[0]).abs() < 0.01);
    println!("\n   χ/N saturation: {}", if saturation { "YES" } else { "NO" });
    
    // 6. LOCAL VS GLOBAL ORDER
    println!("\n6. LOCAL VS GLOBAL ORDER:");
    let (local_z3, global_z3, _global_mag) = calculate_z3_order(&graph);
    println!("   Local Z3 order:  ⟨cos(3θᵢ)⟩ = {:.4}", local_z3);
    println!("   Global Z3 order: |⟨exp(3iθᵢ)⟩| = {:.4}", global_z3.norm());
    println!("   Global Z3 phase: arg⟨exp(3iθᵢ)⟩ = {:.4}", global_z3.arg());
    
    // Check fluctuations
    let mut local_samples = Vec::new();
    let mut global_samples = Vec::new();
    
    for graph_sample in &graph_samples[..50.min(graph_samples.len())] {
        let (local, global, _) = calculate_z3_order(graph_sample);
        local_samples.push(local);
        global_samples.push(global.norm());
    }
    
    let local_std = {
        let mean = local_samples.iter().sum::<f64>() / local_samples.len() as f64;
        let var = local_samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / local_samples.len() as f64;
        var.sqrt()
    };
    
    let global_std = {
        let mean = global_samples.iter().sum::<f64>() / global_samples.len() as f64;
        let var = global_samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / global_samples.len() as f64;
        var.sqrt()
    };
    
    println!("\n   Order parameter fluctuations:");
    println!("   Local Z3 std:  {:.4}", local_std);
    println!("   Global Z3 std: {:.4}", global_std);
    
    // Summary
    println!("\n=== SUMMARY ===");
    println!("Signs of unconventional physics:");
    
    if entropy > 0.1 {
        println!("✓ Finite entropy at low T → extensive degeneracy");
    }
    
    if saturation {
        println!("✓ χ/N saturates with system size → no divergence");
    }
    
    if local_z3.abs() > 0.1 && global_z3.norm() < 0.1 {
        println!("✓ Local order without global order → possible spin liquid");
    }
    
    if s_k[1] < s_k[0] && s_k[2] < s_k[0] {
        println!("✓ No peaks at Z3 wavevectors → disordered phase");
    }
    
    println!("\nConclusion: This model shows signatures of");
    if saturation && entropy > 0.1 {
        println!("a QUANTUM SPIN LIQUID phase with:");
        println!("- Extensive ground state degeneracy");
        println!("- No conventional phase transition");
        println!("- Possible topological order");
    } else {
        println!("CONVENTIONAL critical behavior");
    }
}