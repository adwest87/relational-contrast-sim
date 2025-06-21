use scan::error_analysis::{ErrorAnalysis, ChiSquaredTest, FiniteSizeError, ErrorBudget, print_error_budget_table};

#[test]
fn test_error_analysis_basic() {
    // Create synthetic data with known properties
    let mut data = Vec::new();
    let mean = 10.0;
    let noise = 0.5;
    
    // Generate correlated data
    let mut value = mean;
    for _ in 0..1000 {
        // AR(1) process with correlation
        value = 0.8 * value + 0.2 * mean + noise * (rand::random::<f64>() - 0.5);
        data.push(value);
    }
    
    let analysis = ErrorAnalysis::new(data);
    let errors = analysis.errors();
    
    // Check that tau_int > 0.5 (indicates correlation)
    assert!(errors.tau_int > 0.5, "Should detect autocorrelation");
    
    // Check effective sample size is reduced
    assert!(errors.n_eff < 1000.0, "Effective samples should be less than total");
    
    // Check errors are positive
    assert!(errors.stat_error > 0.0, "Statistical error should be positive");
    assert!(errors.jack_error > 0.0, "Jackknife error should be positive");
}

#[test]
fn test_chi_squared_goodness_of_fit() {
    // Test with data that should pass chi-squared test
    let observed = vec![10.0, 11.0, 9.5, 10.5, 10.2];
    let expected = vec![10.0; 5];
    let errors = vec![0.5; 5];
    
    let chi2_test = ChiSquaredTest::new(&observed, &expected, &errors);
    
    // Chi-squared should be reasonable
    assert!(chi2_test.chi2() > 0.0, "Chi-squared should be positive");
    assert!(chi2_test.chi2_per_dof() < 3.0, "Chi-squared per dof should be reasonable");
    
    // p-value should indicate good fit
    assert!(chi2_test.p_value() > 0.05, "p-value should indicate good fit");
}

#[test]
fn test_finite_size_error() {
    let n = 20;
    let nu = 1.0;  // Correlation length exponent
    let scaling_dim = 2.0;  // Scaling dimension
    
    let fse = FiniteSizeError::new(n, nu, scaling_dim);
    
    // Check relative error scales as N^(-dim/nu)
    let rel_err = fse.relative_error();
    let expected = (n as f64).powf(-scaling_dim / nu);
    
    assert!((rel_err - expected).abs() < 1e-10, 
            "Finite size error formula incorrect");
    
    // Check absolute error
    let observable = 100.0;
    let abs_err = fse.absolute_error(observable);
    assert_eq!(abs_err, observable * rel_err, 
               "Absolute error calculation incorrect");
}

#[test]
fn test_error_budget_integration() {
    // Create a simple time series
    let data: Vec<f64> = (0..100).map(|i| 10.0 + 0.1 * (i as f64).sin()).collect();
    let analysis = ErrorAnalysis::new(data);
    
    // Create error budget
    let budget = ErrorBudget::new(
        "Test Observable".to_string(),
        10.0,
        &analysis,
        Some(0.5),  // Systematic error
    );
    
    // Check that total error combines stat and sys correctly
    let expected_total = (budget.stat_error.powi(2) + budget.systematic_error.powi(2)).sqrt();
    assert!((budget.total_error - expected_total).abs() < 1e-10,
            "Total error not computed correctly");
    
    // Test formatting
    let row = budget.format_row();
    assert!(row.contains("Test Observable"), "Should contain observable name");
    assert!(row.contains("10."), "Should contain value");
}

#[test] 
fn test_autocorrelation_time_estimation() {
    // Test with uncorrelated data
    let uncorr_data: Vec<f64> = (0..1000).map(|_| rand::random()).collect();
    let uncorr_analysis = ErrorAnalysis::new(uncorr_data);
    
    // tau_int should be close to 0.5 for uncorrelated data
    assert!(uncorr_analysis.errors().tau_int < 1.0, 
            "Uncorrelated data should have tau_int ≈ 0.5");
    
    // Test with highly correlated data
    let mut corr_data = Vec::new();
    let mut value = 0.0;
    for _ in 0..1000 {
        value = 0.95 * value + 0.05 * rand::random::<f64>();
        corr_data.push(value);
    }
    let corr_analysis = ErrorAnalysis::new(corr_data);
    
    // tau_int should be much larger for correlated data
    assert!(corr_analysis.errors().tau_int > 5.0,
            "Highly correlated data should have large tau_int");
}

#[test]
fn test_error_budget_table() {
    // Create multiple error budgets
    let mut budgets = Vec::new();
    
    for i in 0..3 {
        let data: Vec<f64> = (0..100).map(|_| i as f64 + rand::random::<f64>()).collect();
        let analysis = ErrorAnalysis::new(data);
        
        budgets.push(ErrorBudget::new(
            format!("Observable {}", i + 1),
            i as f64,
            &analysis,
            Some(0.1 * i as f64),
        ));
    }
    
    // This should not panic
    print_error_budget_table(&budgets);
}