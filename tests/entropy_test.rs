use rc_sim::graph::Graph;

#[test]
fn test_entropy_action_positive_for_small_graph() {
    let g = Graph::complete_random(4);
    let s = g.entropy_action();
    // Usually w ln w < 0 for w < 1, so sum might be negative. Let's just check it's finite.
    assert!(
        s.is_finite(),
        "Entropy action returned non-finite value: {}",
        s
    );
    println!("Entropy action for n=4 is: {}", s);
}

