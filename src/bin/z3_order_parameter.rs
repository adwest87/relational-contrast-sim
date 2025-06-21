// Define proper Z3 order parameter
use scan::graph_fast::FastGraph;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::f64::consts::PI;
use std::fs::File;
use std::io::Write;

// Complex Z3 order parameter: ψ = (1/M) Σ exp(3iθ)
fn calculate_z3_order_parameter(graph: &FastGraph) -> (f64, f64, f64) {
    let mut re_sum = 0.0;
    let mut im_sum = 0.0;
    
    for link in &graph.links {
        let angle_3 = 3.0 * link.theta;
        re_sum += angle_3.cos();
        im_sum += angle_3.sin();
    }
    
    let m = graph.links.len() as f64;
    let re_avg = re_sum / m;
    let im_avg = im_sum / m;
    let magnitude = (re_avg * re_avg + im_avg * im_avg).sqrt();
    
    (re_avg, im_avg, magnitude)
}

fn main() {
    println!("🔺 Z₃ ORDER PARAMETER TEST");
    println!("=========================");
    println!("Testing with proper Z₃ order parameter: ψ = <exp(3iθ)>");
    
    let n = 48;
    let alpha = 1.5;
    
    // Test at different temperatures
    let beta_values = vec![0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0];
    
    let equilibration_steps = 50_000;
    let production_steps = 50_000;
    let measure_interval = 100;
    
    println!("\nSystem: N={}, α={}", n, alpha);
    println!("Steps: {}k equilibration + {}k production", 
        equilibration_steps/1000, production_steps/1000);
    
    let mut csv_file = File::create("z3_order_parameter.csv").unwrap();
    writeln!(csv_file, "beta,T,psi_magnitude,psi_re,psi_im,susceptibility_z3,u4_z3").unwrap();
    
    println!("\n📊 TEMPERATURE SCAN:");
    println!("===================");
    
    for &beta in &beta_values {
        let temperature = 1.0 / beta;
        print!("β={:.1} (T={:.2}): ", beta, temperature);
        std::io::stdout().flush().unwrap();
        
        let mut rng = Pcg64::seed_from_u64(42);
        let mut graph = FastGraph::new(n, 12345);
        
        // Equilibration
        for _ in 0..equilibration_steps {
            graph.metropolis_step(alpha, beta, 0.5, 0.5, &mut rng);
        }
        
        // Production measurements
        let mut psi_values = Vec::new();
        
        for step in 0..production_steps {
            graph.metropolis_step(alpha, beta, 0.5, 0.5, &mut rng);
            
            if step % measure_interval == 0 {
                let (re, im, mag) = calculate_z3_order_parameter(&graph);
                psi_values.push((re, im, mag));
            }
        }
        
        // Calculate averages
        let mean_re = psi_values.iter().map(|&(re, _, _)| re).sum::<f64>() 
            / psi_values.len() as f64;
        let mean_im = psi_values.iter().map(|&(_, im, _)| im).sum::<f64>() 
            / psi_values.len() as f64;
        let mean_mag = psi_values.iter().map(|&(_, _, mag)| mag).sum::<f64>() 
            / psi_values.len() as f64;
        
        // Z3 susceptibility
        let mag_squared: Vec<f64> = psi_values.iter().map(|&(_, _, mag)| mag * mag).collect();
        let mean_mag2 = mag_squared.iter().sum::<f64>() / mag_squared.len() as f64;
        let chi_z3 = n as f64 * (mean_mag2 - mean_mag * mean_mag);
        
        // Z3 Binder cumulant
        let mag4: Vec<f64> = psi_values.iter().map(|&(_, _, mag)| mag.powi(4)).collect();
        let mean_mag4 = mag4.iter().sum::<f64>() / mag4.len() as f64;
        let u4_z3 = 1.0 - mean_mag4 / (3.0 * mean_mag2 * mean_mag2);
        
        println!("|ψ|={:.3}, χ_Z3={:.1}, U4={:.3}", mean_mag, chi_z3, u4_z3);
        
        writeln!(csv_file, "{:.1},{:.3},{:.6},{:.6},{:.6},{:.3},{:.3}",
            beta, temperature, mean_mag, mean_re, mean_im, chi_z3, u4_z3).unwrap();
    }
    
    // Test ground states
    println!("\n📊 Z₃ GROUND STATES TEST:");
    println!("========================");
    
    let ground_states = vec![
        (PI/3.0, "θ=π/3"),
        (PI, "θ=π"),
        (5.0*PI/3.0, "θ=5π/3"),
    ];
    
    for (theta_gs, label) in ground_states {
        let mut graph = FastGraph::new(n, 12345);
        
        // Set to ground state
        for i in 0..graph.links.len() {
            graph.links[i].update_theta(theta_gs);
        }
        
        let (re, im, mag) = calculate_z3_order_parameter(&graph);
        let expected_re = (3.0 * theta_gs).cos();
        let expected_im = (3.0 * theta_gs).sin();
        
        println!("\n{} ground state:", label);
        println!("  ψ = {:.3} + {:.3}i, |ψ| = {:.3}", re, im, mag);
        println!("  Expected: {:.3} + {:.3}i", expected_re, expected_im);
        
        // All three should give |ψ| = 1 since cos(π)=cos(3π)=cos(5π)=-1
    }
    
    println!("\n📊 SYMMETRY BREAKING TEST:");
    println!("=========================");
    
    // Run from random initial conditions at low T
    let beta = 5.0;
    let n_replicas = 10;
    
    print!("Running {} replicas at β={}: ", n_replicas, beta);
    
    for replica in 0..n_replicas {
        let mut rng = Pcg64::seed_from_u64(100 + replica as u64);
        let mut graph = FastGraph::new(n, 200 + replica as u64);
        
        // Long equilibration
        for _ in 0..200_000 {
            graph.metropolis_step(alpha, beta, 0.5, 0.5, &mut rng);
        }
        
        let (re, im, mag) = calculate_z3_order_parameter(&graph);
        let phase = im.atan2(re) / 3.0; // Divide by 3 to get original θ
        
        print!("{:.0}° ", phase.to_degrees());
    }
    
    println!("\n\nIf Z₃ symmetry is broken, should see clustering around 60°, 180°, or 300°");
    
    println!("\nData saved to z3_order_parameter.csv");
}