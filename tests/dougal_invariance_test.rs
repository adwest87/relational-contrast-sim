use rand::Rng;
use crate::graph::Graph;

#[test]
fn test_dougal_invariant_action() {
    let mut g = Graph::complete_random(4);
    let original_i = g.invariant_action();

    // Choose a random positive scaling factor λ
    let lambda = rand::thread_rng().gen_range(0.2..5.0);
    g.rescale(lambda);

    let scaled_i = g.invariant_action();

    assert!(
        (scaled_i - original_i).abs() < 1e-12,
        "Invariant action changed: before {}, after {}", original_i, scaled_i
    );
}

