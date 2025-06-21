use scan::graph_fast::{FastGraph, BatchedObservables};
use scan::error_analysis::{ErrorAnalysis, ChiSquaredTest, FiniteSizeError, ErrorBudget, print_error_budget_table};
use rand::SeedableRng;
use rand_pcg::Pcg64;

fn main() {
    println!("=== Error Analysis Demonstration ===\n");
    
    // Parameters
    let n = 20;
    let alpha = 1.5;
    let beta = 2.0;
    let equilibration_steps = 10000;
    let measurement_steps = 50000;
    let measure_interval = 10;
    
    println!("System size: N = {}", n);
    println!("Parameters: α = {}, β = {}", alpha, beta);
    println!("Equilibration: {} steps", equilibration_steps);
    println!("Measurements: {} steps\n", measurement_steps);
    
    // Initialize system
    let mut graph = FastGraph::new(n, 42);
    let mut rng = Pcg64::seed_from_u64(42);
    let mut observables = BatchedObservables::new();
    
    // Equilibrate
    println!("Equilibrating...");
    for _ in 0..equilibration_steps {
        graph.metropolis_step(alpha, beta, 0.2, 0.5, &mut rng);
    }
    
    // Reset accumulators after equilibration
    observables.reset_accumulators();
    
    // Collect time series data
    println!("Collecting measurements...");
    let mut energy_series = Vec::new();
    let mut magnetization_series = Vec::new();
    let mut specific_heat_series = Vec::new();
    let mut susceptibility_series = Vec::new();
    let mut binder_series = Vec::new();
    
    for step in 0..measurement_steps {
        graph.metropolis_step(alpha, beta, 0.2, 0.5, &mut rng);
        
        if step % measure_interval == 0 {
            let obs = observables.measure(&graph, alpha, beta);
            
            // Collect raw data
            let energy = graph.action(alpha, beta);
            let sum_cos: f64 = graph.links.iter().map(|l| l.cos_theta).sum();
            let magnetization = sum_cos / graph.n() as f64;
            
            energy_series.push(energy);
            magnetization_series.push(magnetization);
            
            // Collect derived observables
            if observables.sample_count() > 100 {
                specific_heat_series.push(obs.specific_heat);
                susceptibility_series.push(obs.susceptibility);
                binder_series.push(obs.binder_cumulant);
            }
            
            // Progress update
            if step % 5000 == 0 && step > 0 {
                println!("  Step {}: τ_int ≈ {:.1}", step, graph.get_autocorr_tau());
            }
        }
    }
    
    println!("\nAnalyzing errors...\n");
    
    // Create error analyses for each observable
    let energy_analysis = ErrorAnalysis::new(energy_series.clone());
    let mag_analysis = ErrorAnalysis::new(magnetization_series.clone());
    let specific_heat_analysis = ErrorAnalysis::new(specific_heat_series);
    let susceptibility_analysis = ErrorAnalysis::new(susceptibility_series);
    let binder_analysis = ErrorAnalysis::new(binder_series);
    
    // Calculate finite-size errors
    // For spin liquids, we expect unusual finite-size scaling
    let nu = 1.0;  // Effective correlation length exponent (mean-field-like)
    
    let energy_fse = FiniteSizeError::new(n, nu, 0.0);  // Energy is extensive
    let mag_fse = FiniteSizeError::new(n, nu, 0.0);     // Magnetization ~ O(1)
    let specific_heat_fse = FiniteSizeError::new(n, nu, 0.0);  // C/N ~ O(1)
    let susceptibility_fse = FiniteSizeError::new(n, nu, 2.0);  // χ ~ N^(γ/ν)
    let binder_fse = FiniteSizeError::new(n, nu, 0.0);  // Dimensionless
    
    // Create error budgets
    let mut budgets = Vec::new();
    
    // Energy per site
    let energy_per_site = energy_analysis.mean() / n as f64;
    budgets.push(ErrorBudget::new(
        "Energy/N".to_string(),
        energy_per_site,
        &energy_analysis,
        Some(energy_fse.absolute_error(energy_per_site)),
    ));
    
    // Magnetization
    budgets.push(ErrorBudget::new(
        "Magnetization".to_string(),
        mag_analysis.mean(),
        &mag_analysis,
        Some(mag_fse.absolute_error(mag_analysis.mean())),
    ));
    
    // Specific heat
    budgets.push(ErrorBudget::new(
        "Specific Heat".to_string(),
        specific_heat_analysis.mean(),
        &specific_heat_analysis,
        Some(specific_heat_fse.absolute_error(specific_heat_analysis.mean())),
    ));
    
    // Susceptibility
    budgets.push(ErrorBudget::new(
        "Susceptibility".to_string(),
        susceptibility_analysis.mean(),
        &susceptibility_analysis,
        Some(susceptibility_fse.absolute_error(susceptibility_analysis.mean())),
    ));
    
    // Binder cumulant
    budgets.push(ErrorBudget::new(
        "Binder Cumulant".to_string(),
        binder_analysis.mean(),
        &binder_analysis,
        Some(binder_fse.absolute_error(binder_analysis.mean())),
    ));
    
    // Print error budget table
    print_error_budget_table(&budgets);
    
    // Additional analysis: Test for equilibration using chi-squared test
    println!("\n=== Equilibration Test ===");
    
    // Split energy series into blocks and test for stationarity
    let block_size = energy_series.len() / 10;
    let mut block_means = Vec::new();
    let mut block_errors = Vec::new();
    
    for i in 0..10 {
        let start = i * block_size;
        let end = (i + 1) * block_size;
        let block = &energy_series[start..end];
        
        let block_mean = block.iter().sum::<f64>() / block.len() as f64;
        let block_var = block.iter()
            .map(|&x| (x - block_mean).powi(2))
            .sum::<f64>() / (block.len() - 1) as f64;
        let block_err = (block_var / block.len() as f64).sqrt();
        
        block_means.push(block_mean);
        block_errors.push(block_err);
    }
    
    // Expected values: overall mean for each block
    let overall_mean = energy_series.iter().sum::<f64>() / energy_series.len() as f64;
    let expected = vec![overall_mean; 10];
    
    let chi2_test = ChiSquaredTest::new(&block_means, &expected, &block_errors);
    
    println!("Block analysis for equilibration:");
    println!("  χ² = {:.2}, dof = {}, p-value = {:.3}", 
             chi2_test.chi2(), chi2_test.dof(), chi2_test.p_value());
    println!("  χ²/dof = {:.2}", chi2_test.chi2_per_dof());
    
    if chi2_test.p_value() > 0.05 {
        println!("  ✓ System appears to be well equilibrated (p > 0.05)");
    } else {
        println!("  ⚠ System may not be fully equilibrated (p < 0.05)");
    }
    
    // Demonstrate autocorrelation analysis
    println!("\n=== Autocorrelation Analysis ===");
    
    let observables_info = vec![
        ("Energy", &energy_analysis),
        ("Magnetization", &mag_analysis),
        ("Specific Heat", &specific_heat_analysis),
        ("Susceptibility", &susceptibility_analysis),
    ];
    
    for (name, analysis) in &observables_info {
        let errors = analysis.errors();
        println!("\n{}:", name);
        println!("  Integrated autocorrelation time: τ_int = {:.1}", errors.tau_int);
        println!("  Effective sample size: N_eff = {:.0}", errors.n_eff);
        println!("  Statistical efficiency: {:.1}%", 100.0 * errors.n_eff / (energy_series.len() as f64));
        println!("  Relative error: {:.2}%", 100.0 * errors.relative_error);
    }
    
    println!("\n=== Summary ===");
    println!("Total MC steps: {}", measurement_steps);
    println!("Measurement interval: {}", measure_interval);
    println!("Total measurements: {}", energy_series.len());
    println!("\nRecommendations:");
    
    let max_tau = observables_info.iter()
        .map(|(_, a)| a.errors().tau_int)
        .fold(0.0, f64::max);
    
    let recommended_interval = (2.0_f64 * max_tau).ceil() as usize;
    println!("  - Optimal measurement interval: {} steps", recommended_interval);
    println!("  - Current statistical efficiency: {:.1}%", 
             100.0 * measure_interval as f64 / recommended_interval as f64);
    
    if recommended_interval > measure_interval {
        println!("  - Consider increasing measurement interval to {} for better efficiency", 
                 recommended_interval);
    }
    
    println!("\nNote: For quantum spin liquid systems, finite-size effects may");
    println!("behave anomalously due to extensive ground state degeneracy.");
}