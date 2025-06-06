use rc_sim::graph::Graph;

#[test]
fn test_metropolis_acceptance_rate() {
    let mut g = Graph::complete_random(8);
    let beta  = 1.0;
    let alpha = 1.0;
    let delta_w     = 0.1;
    let delta_theta = 0.2;
    let n_steps = 1000;
    let mut accepted = 0;
    for _ in 0..n_steps {
        if g.metropolis_step(beta, alpha, delta_w, delta_theta) {
            accepted += 1;
        }
    }
    let acc_rate = accepted as f64 / n_steps as f64;
    // For a well-chosen delta, should be neither 0 nor 1
    assert!(
        (0.01..=1.0).contains(&acc_rate),
        "Acceptance rate {} out of plausible range", acc_rate
    );
}

