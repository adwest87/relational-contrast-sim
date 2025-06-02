pub fn aib_project(t: [[[f64; 3]; 3]; 3]) -> [[[f64; 3]; 3]; 3] {
    let mut result = t;

    // Step 1: Axial (A)
    let mut axial = [[[0.0; 3]; 3]; 3];
    let mut axial_sum = 0.0;
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                let eps = levi_civita(a, b, c);
                axial_sum += eps * t[a][b][c];
            }
        }
    }
    axial_sum /= 6.0;
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                let eps = levi_civita(a, b, c);
                axial[a][b][c] = eps * axial_sum;
            }
        }
    }

    // Step 2: Isotropic (D)
    let mut v = [0.0; 3];
    for a in 0..3 {
        v[a] = t[a][a][0] + t[a][a][1] + t[a][a][2];
    }
    let mut isotropic = [[[0.0; 3]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                isotropic[a][b][c] = (kronecker(a, b) * v[c]
                    + kronecker(b, c) * v[a]
                    + kronecker(c, a) * v[b])
                    / 3.0;
            }
        }
    }

    // Step 3: Cyclic (B)
    let mut cyclic = [[[0.0; 3]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                let t1 = t[a][b][c];
                let t2 = t[b][c][a];
                let t3 = t[c][a][b];
                cyclic[a][b][c] = (t1 + t2 + t3) / 3.0;
            }
        }
    }

    // Subtract A, D, B
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                result[a][b][c] = t[a][b][c]
                                - axial[a][b][c]
                                - isotropic[a][b][c]
                                - cyclic[a][b][c];
            }
        }
    }

    result
}

fn kronecker(i: usize, j: usize) -> f64 {
    if i == j { 1.0 } else { 0.0 }
}

fn levi_civita(i: usize, j: usize, k: usize) -> f64 {
    match (i, j, k) {
        (0, 1, 2) | (1, 2, 0) | (2, 0, 1) => 1.0,
        (0, 2, 1) | (2, 1, 0) | (1, 0, 2) => -1.0,
        _ => 0.0,
    }
}
