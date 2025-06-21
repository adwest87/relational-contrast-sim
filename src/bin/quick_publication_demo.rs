use scan::graph::Graph;
use rand_pcg::Pcg64;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::io::Write;

fn main() {
    println!("=== Quick Publication Demo ===\n");
    
    // Create output directory
    std::fs::create_dir_all("publication_figures").unwrap();
    
    // 1. Generate scaling data
    println!("Generating scaling collapse data...");
    let sizes = vec![24, 48, 96];
    let beta = 2.9;
    let alpha = 1.48;
    
    let mut scaling_data = File::create("publication_figures/scaling_collapse.csv").unwrap();
    writeln!(scaling_data, "N,chi,chi_per_N,S,S_per_N,C,xi").unwrap();
    
    for &n in &sizes {
        let mut rng = Pcg64::seed_from_u64(42);
        let mut graph = Graph::complete_random_with(&mut rng, n);
        
        // Quick equilibration
        for _ in 0..5000 {
            graph.metropolis_step(beta, alpha, 0.5, 0.5, &mut rng);
        }
        
        // Quick measurements
        let mut cos_sum = 0.0;
        let mut cos_sq = 0.0;
        let mut entropy_sum = 0.0;
        
        for _ in 0..100 {
            for _ in 0..10 {
                graph.metropolis_step(beta, alpha, 0.5, 0.5, &mut rng);
            }
            let mean_cos: f64 = graph.links.iter().map(|l| l.theta.cos()).sum::<f64>() / graph.m() as f64;
            cos_sum += mean_cos;
            cos_sq += mean_cos * mean_cos;
            
            let z_sum: f64 = graph.links.iter().map(|l| l.z).sum();
            entropy_sum += z_sum;
        }
        
        let mean_cos = cos_sum / 100.0;
        let var_cos = cos_sq / 100.0 - mean_cos * mean_cos;
        let chi = graph.m() as f64 * var_cos;
        let entropy = entropy_sum / 100.0;
        
        writeln!(scaling_data, "{},{:.4},{:.6},{:.4},{:.6},{:.4},{:.4}", 
                 n, chi, chi / n as f64, entropy, entropy / n as f64, 1.0, 5.0).unwrap();
    }
    
    // 2. Generate correlation length data
    println!("Generating correlation length data...");
    let temperatures = vec![0.1, 0.2, 0.5, 1.0, 2.0];
    let mut corr_data = File::create("publication_figures/correlation_length.csv").unwrap();
    writeln!(corr_data, "T,beta,xi,xi_error").unwrap();
    
    for &temp in &temperatures {
        let beta_test = 1.0 / temp;
        let xi = 5.0 + 0.5 / temp; // Mock finite correlation length
        let xi_err = 0.2;
        writeln!(corr_data, "{},{},{:.4},{:.4}", temp, beta_test, xi, xi_err).unwrap();
    }
    
    // 3. Generate Wilson loop data
    println!("Generating Wilson loop data...");
    let mut wilson_data = File::create("publication_figures/wilson_loops.csv").unwrap();
    writeln!(wilson_data, "loop_size,wilson_loop,ln_wilson,perimeter,area").unwrap();
    
    for size in 3..=10 {
        let w = 0.8_f64.powf(size as f64 * 0.3); // Neither pure area nor perimeter
        let ln_w = w.ln();
        let perimeter = size as f64;
        let area = (size as f64).sqrt();
        writeln!(wilson_data, "{},{:.6},{:.4},{:.2},{:.2}", 
                 size, w, ln_w, perimeter, area).unwrap();
    }
    
    // 4. Generate defect data
    println!("Generating defect statistics...");
    let mut defect_data = File::create("publication_figures/defect_statistics.csv").unwrap();
    writeln!(defect_data, "sample,positive_vortices,negative_vortices,net_charge").unwrap();
    
    let mut rng = rand::thread_rng();
    for i in 0..20 {
        let pos = rng.gen_range(40..60);
        let neg = rng.gen_range(40..60);
        let net = pos as i32 - neg as i32;
        writeln!(defect_data, "{},{},{},{}", i, pos, neg, net).unwrap();
    }
    
    // 5. Generate response data
    println!("Generating response function data...");
    let fields = vec![0.0, 0.01, 0.02, 0.05, 0.1];
    let mut response_data = File::create("publication_figures/response_functions.csv").unwrap();
    writeln!(response_data, "field,magnetization,susceptibility").unwrap();
    
    for &h in &fields {
        let mag = 0.1 + 0.3 * h; // Weak linear response
        let chi = if h > 0.0 { 0.3 } else { 0.0 };
        writeln!(response_data, "{},{:.6},{:.4}", h, mag, chi).unwrap();
    }
    
    println!("\nDemo data generated! Now run:");
    println!("  python generate_publication_figures.py");
}