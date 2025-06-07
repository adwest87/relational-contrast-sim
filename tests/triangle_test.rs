use crate::graph::Graph;

#[test]
fn test_triangle_action_identity() {
    let g = Graph::complete_random(5); // 5 nodes ⇒ 10 triangles
    let n_tri = g.triangles().count();
    let alpha = 1.23;

    let s_tri = g.triangle_action(alpha);

    // For identity holonomy, trace = 3 per triangle
    let expected = alpha * 3.0 * n_tri as f64;

    assert!(
        (s_tri - expected).abs() < 1e-10,
        "Triangle action incorrect: expected {}, got {}", expected, s_tri
    );
}

