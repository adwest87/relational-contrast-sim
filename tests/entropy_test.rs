use rc_sim::graph::{Graph, Link, Node};

#[test]
fn test_entropy_on_two_nodes() {
    // Build a 2-node graph manually so we know the answer exactly
    let nodes = vec![Node { id: 0 }, Node { id: 1 }];
    let w = 0.5_f64;                       // weight
    let links = vec![Link {
        i: 0,
        j: 1,
        w,
        tensor: [[[0.0; 3]; 3]; 3],        // dummy tensor, not used here
        holonomy: [[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]],
    }];
    let g = Graph { nodes, links };

    let expected = w * w.ln();             // only one link
    let action   = g.entropy_action();

    // use approx equality with small tolerance
    assert!((action - expected).abs() < 1e-12,
            "Entropy action incorrect: expected {}, got {}", expected, action);
}

