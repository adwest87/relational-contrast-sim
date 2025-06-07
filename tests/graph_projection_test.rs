use scan::graph::Graph;

#[test]
fn test_graph_projection_reduces_norm() {
    let mut g = Graph::complete_random(4);            // small graph
    let (before, after) = g.project_all();

    assert!(
        after < before,
        "Projection did not reduce total norm: before = {}, after = {}", before, after
    );
}

