use scan::graph::Graph;

#[test]
fn test_entropy_on_two_nodes() {
    // Build a 2-node graph manually so we know the answer exactly
    let mut g = Graph::complete_random(2);
    
    // Set a specific weight using z-variable
    // To set w = 0.5, we need z = -ln(0.5) = ln(2)
    g.links[0].z = 2.0_f64.ln();  // This gives w = exp(-ln(2)) = 0.5
    g.links[0].tensor = [[[0.0; 3]; 3]; 3];
    
    let w = g.links[0].w();
    assert!((w - 0.5).abs() < 1e-10, "Weight setting failed: got {}", w);
    
    let expected = w * w.ln();             // only one link
    let action   = g.entropy_action();

    // use approx equality with small tolerance
    assert!((action - expected).abs() < 1e-12,
            "Entropy action incorrect: expected {}, got {}", expected, action);
}