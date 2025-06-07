//! Unit‑test: basic sanity check on Metropolis acceptance rate.

use scan::graph::{Graph, StepInfo};

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[test]
fn test_metropolis_acceptance_rate() {
    // -----------------------------------------------------------
    // Deterministic RNG so the test is repeatable.
    // -----------------------------------------------------------
    let mut rng = ChaCha20Rng::seed_from_u64(0xDEADBEEF);

    // Build graph with the same RNG.
    let mut g = Graph::complete_random_with(&mut rng, 8);

    // Simulation parameters.
    let beta        = 1.0;
    let alpha       = 1.0;
    let delta_w     = 0.10;
    let delta_theta = 0.20;
    let n_steps     = 1_000;

    // Counters.
    let mut accepted = 0usize;

    // -----------------------------------------------------------
    // Run Metropolis sweeps.
    // -----------------------------------------------------------
    for _ in 0..n_steps {
        let StepInfo { accepted: acc, .. } =
            g.metropolis_step(beta, alpha, delta_w, delta_theta, &mut rng);

        if acc { accepted += 1; }
    }

    let acc_rate = accepted as f64 / n_steps as f64;

    // For a sensible (δw, δθ) we expect a rate strictly between 0 % and 100 %.
    // The chosen bounds are generous enough to cope with RNG variance while
    // still catching pathological behaviour.
    assert!(
        (0.01..=0.99).contains(&acc_rate),
        "Acceptance rate {acc_rate:.3} is outside plausible range"
    );
}
