use scan::graph::Graph;

#[test]
fn test_entropy_on_two_nodes() {
    // Build a 2-node graph manually so we know the answer exactly
    let mut g = Graph::complete_random(2);
    g.links[0].w = 0.5;
    g.links[0].tensor = [[[0.0; 3]; 3]; 3];
    let w = g.links[0].w;


    let expected = w * w.ln();             // only one link
    let action   = g.entropy_action();

    // use approx equality with small tolerance
    assert!((action - expected).abs() < 1e-12,
            "Entropy action incorrect: expected {}, got {}", expected, action);
}

