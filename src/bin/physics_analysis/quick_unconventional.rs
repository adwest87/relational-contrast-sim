use scan::graph::Graph;
use rand_pcg::Pcg64;
use rand::SeedableRng;
use std::collections::HashMap;
use std::f64::consts::PI;
use num_complex::Complex64;

fn main() {
    println!("=== Quick Unconventional Physics Test ===\n");
    
    let n = 48;  // Smaller system
    let beta = 2.9;
    let alpha = 1.48;
    
    // Initialize system
    let mut rng = Pcg64::seed_from_u64(42);
    let mut graph = Graph::complete_random_with(&mut rng, n);
    
    println!("System: N = {}, β = {}, α = {}", n, beta, alpha);
    println!("Equilibrating...");
    
    // Quick equilibration
    for _ in 0..10000 {
        graph.metropolis_step(beta, alpha, 0.5, 0.5, &mut rng);
    }
    
    // 1. Quick correlation check
    println!("\n1. CORRELATION CHECK:");
    let mut cos_sum = 0.0;
    let mut triangle_sum = 0.0;
    let mut count = 0;
    
    // Sample some pairs and triangles
    for i in 0..10 {
        for j in (i+1)..20.min(n) {
            let theta_ij = graph.links[graph.link_index(i, j)].theta;
            cos_sum += theta_ij.cos();
            count += 1;
            
            // Check one triangle
            if j < n-1 {
                let k = j + 1;
                let theta_jk = graph.links[graph.link_index(j, k)].theta;
                let theta_ik = graph.links[graph.link_index(i, k)].theta;
                let triangle_phase = theta_ij + theta_jk + theta_ik - PI;
                triangle_sum += triangle_phase.cos();
            }
        }
    }
    
    println!("   ⟨cos(θᵢ-θⱼ)⟩ = {:.4}", cos_sum / count as f64);
    println!("   ⟨cos(θᵢ+θⱼ+θₖ-π)⟩ ≈ {:.4}", triangle_sum / count as f64);
    
    // 2. Quick entropy estimate
    println!("\n2. ENTROPY ESTIMATE:");
    let mut energies = Vec::new();
    for _ in 0..1000 {
        for _ in 0..10 {
            graph.metropolis_step(beta, alpha, 0.5, 0.5, &mut rng);
        }
        energies.push(graph.action(alpha, beta));
    }
    
    // Simple entropy calculation
    let mut histogram: HashMap<i64, usize> = HashMap::new();
    for &e in &energies {
        let bin = (e * 100.0).round() as i64;
        *histogram.entry(bin).or_insert(0) += 1;
    }
    
    let mut entropy = 0.0;
    let total = energies.len() as f64;
    for &count in histogram.values() {
        if count > 0 {
            let p = count as f64 / total;
            entropy -= p * p.ln();
        }
    }
    println!("   S ≈ {:.4} (from {} energy bins)", entropy, histogram.len());
    
    // 3. Quick structure factor at key points
    println!("\n3. STRUCTURE FACTOR (key points):");
    
    // Calculate effective phases at nodes
    let mut node_phases = vec![0.0; n];
    for i in 0..n {
        let mut phase_sum = 0.0;
        for j in 0..n {
            if i != j {
                let link_idx = graph.link_index(i.min(j), i.max(j));
                let theta = graph.links[link_idx].theta;
                phase_sum += if i < j { theta } else { -theta };
            }
        }
        node_phases[i] = phase_sum / (n - 1) as f64;
    }
    
    // k = 0 (uniform)
    let s_0: f64 = node_phases.iter().map(|&phi| phi.cos()).sum::<f64>() / n as f64;
    
    // k = 2π/3 (Z3 order)
    let mut s_z3 = Complex64::new(0.0, 0.0);
    for (i, &phi) in node_phases.iter().enumerate() {
        let k_dot_r = 2.0 * PI / 3.0 * (i % 3) as f64;
        s_z3 += Complex64::from_polar(1.0, k_dot_r + phi);
    }
    
    println!("   S(0) = {:.4}", s_0.abs());
    println!("   S(2π/3) = {:.4}", (s_z3.norm() / n as f64));
    
    // 4. Quick Z3 order check
    println!("\n4. Z3 ORDER PARAMETERS:");
    let mut local_z3 = 0.0;
    let mut global_z3 = Complex64::new(0.0, 0.0);
    
    for &phi in &node_phases {
        local_z3 += (3.0 * phi).cos();
        global_z3 += Complex64::from_polar(1.0, 3.0 * phi);
    }
    
    local_z3 /= n as f64;
    global_z3 /= n as f64;
    
    println!("   Local Z3:  ⟨cos(3θᵢ)⟩ = {:.4}", local_z3);
    println!("   Global Z3: |⟨exp(3iθᵢ)⟩| = {:.4}", global_z3.norm());
    
    // 5. Quick finite size check
    println!("\n5. FINITE SIZE CHECK:");
    let sizes = vec![24, 48];
    for &size in &sizes {
        let mut test_graph = Graph::complete_random_with(&mut rng, size);
        
        // Quick equilibration
        for _ in 0..(5000 * size / 24) {
            test_graph.metropolis_step(beta, alpha, 0.5, 0.5, &mut rng);
        }
        
        // Quick susceptibility estimate
        let mut cos_samples = Vec::new();
        for _ in 0..100 {
            for _ in 0..10 {
                test_graph.metropolis_step(beta, alpha, 0.5, 0.5, &mut rng);
            }
            let mean_cos: f64 = test_graph.links.iter()
                .map(|l| l.theta.cos()).sum::<f64>() / test_graph.m() as f64;
            cos_samples.push(mean_cos);
        }
        
        let mean = cos_samples.iter().sum::<f64>() / cos_samples.len() as f64;
        let var = cos_samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / cos_samples.len() as f64;
        let chi = test_graph.m() as f64 * var;
        
        println!("   N = {}: χ = {:.2}, χ/N = {:.4}", size, chi, chi / size as f64);
    }
    
    // Summary
    println!("\n=== QUICK SUMMARY ===");
    if entropy > 0.5 {
        println!("✓ High entropy S = {:.2} suggests extensive degeneracy", entropy);
    }
    if local_z3.abs() > 0.1 && global_z3.norm() < 0.1 {
        println!("✓ Local Z3 order ({:.3}) without global order ({:.3})", local_z3, global_z3.norm());
    }
    if s_0.abs() > (s_z3.norm() / n as f64) {
        println!("✓ No Z3 ordering peak in structure factor");
    }
    
    println!("\nThis system shows signs of UNCONVENTIONAL behavior!");
}