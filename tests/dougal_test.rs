use scan::graph::Graph;

#[test]
fn test_entropy_scaling_law() {
    let mut g = Graph::complete_random(5);   // any size ≥2
    let s0 = g.entropy_action();
    let w0: f64 = g.links.iter().map(|l| l.w()).sum();

    // Rescale
    let lambda = 2.5_f64;
    g.rescale(lambda);
    let s1 = g.entropy_action();

    let expected = lambda * s0 + lambda * w0 * lambda.ln();

    assert!(
        (s1 - expected).abs() < 1e-10,
        "Scaling law failed:   S(λw) = {},  expected {}",
        s1, expected
    );
}

