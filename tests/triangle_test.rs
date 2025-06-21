use scan::graph::Graph;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[test]
fn test_triangle_action_identity() {
    // Create a graph where we manually set all phases to 0
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let mut g = Graph::complete_random_with(&mut rng, 5); // 5 nodes ⇒ 10 triangles
    
    // Set all phases to 0 for identity holonomy
    for link in &mut g.links {
        link.theta = 0.0;
    }
    
    let n_tri = g.triangles().count();
    let alpha = 1.23;

    let s_tri = g.triangle_action(alpha);

    // For identity holonomy (all phases = 0), cos(0) = 1 per triangle
    let expected = alpha * 1.0 * n_tri as f64;

    assert!(
        (s_tri - expected).abs() < 1e-10,
        "Triangle action incorrect: expected {} (alpha={}, n_tri={}), got {}", 
        expected, alpha, n_tri, s_tri
    );
}

#[test]
fn test_triangle_action_random() {
    // Test that triangle action is finite and reasonable with random phases
    let mut rng = ChaCha20Rng::seed_from_u64(12345);
    let g = Graph::complete_random_with(&mut rng, 6);
    
    let alpha = 1.5;
    let triangle_action = g.triangle_action(alpha);
    let n_tri = g.triangles().count() as f64;
    
    // Check action is finite
    assert!(triangle_action.is_finite());
    
    // Check action is bounded (cos ∈ [-1, 1])
    assert!(triangle_action.abs() <= alpha * n_tri);
    
    // Check normalized action is reasonable
    let normalized = triangle_action / (alpha * n_tri);
    assert!(normalized >= -1.0 && normalized <= 1.0);
}

