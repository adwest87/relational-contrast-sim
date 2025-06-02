use rc_sim::projector::aib_project;

#[test]
fn test_zero_tensor_returns_zero() {
    let zero_tensor = [[[0.0; 3]; 3]; 3];
    let projected = aib_project(zero_tensor);

    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                assert!((projected[a][b][c]).abs() < 1e-10,
                        "Expected zero at [{}][{}][{}], got {}", a, b, c, projected[a][b][c]);
            }
        }
    }
}

#[test]
fn test_single_entry_tensor() {
    let mut t = [[[0.0; 3]; 3]; 3];
    t[0][0][0] = 1.0; // one non-zero value

    let projected = aib_project(t);

    // Check it's still finite and not NaN
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                assert!(
                    projected[a][b][c].is_finite(),
                    "Component [{}][{}][{}] is not finite",
                    a, b, c
                );
            }
        }
    }
}

// ------------------------------------------------------------------
// new test: axial (epsilon) tensor should project to zero
// ------------------------------------------------------------------
#[test]
fn test_axial_tensor_projects_to_zero() {
    // Build a tensor with T_abc = ε_abc
    let mut t = [[[0.0; 3]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                t[a][b][c] = match (a, b, c) {
                    (0, 1, 2) | (1, 2, 0) | (2, 0, 1) =>  1.0,  // +1 permutations
                    (0, 2, 1) | (2, 1, 0) | (1, 0, 2) => -1.0,  // −1 permutations
                    _ => 0.0,
                };
            }
        }
    }

    let projected = aib_project(t);

    // Every component should be (almost) zero after projection
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                assert!(
                    projected[a][b][c].abs() < 1e-10,
                    "Component [{}][{}][{}] not zero: {}", a, b, c, projected[a][b][c]
                );
            }
        }
    }
}
