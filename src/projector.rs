pub fn aib_project(t: [[[f64; 3]; 3]; 3]) -> [[[f64; 3]; 3]; 3] {
    // ---------- Step 1: axial A ----------
    let mut axial = [[[0.0; 3]; 3]; 3];
    let mut axial_sum = 0.0;
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                axial_sum += levi_civita(a, b, c) * t[a][b][c];
            }
        }
    }
    axial_sum /= 6.0;
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                axial[a][b][c] = levi_civita(a, b, c) * axial_sum;
            }
        }
    }

    // ---------- Step 2: isotropic D ----------
    let mut v = [0.0; 3];
    for a in 0..3 {
        v[a] = t[a][a][0] + t[a][a][1] + t[a][a][2];
    }
    let mut iso = [[[0.0; 3]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                iso[a][b][c] = (kronecker(a, b) * v[c]
                    + kronecker(b, c) * v[a]
                    + kronecker(c, a) * v[b]) / 3.0;
            }
        }
    }

    // ---------- Step 3: cyclic B (only after A & D removed) ----------
    let mut tmp = [[[0.0; 3]; 3]; 3]; // T − A − D
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                tmp[a][b][c] = t[a][b][c] - axial[a][b][c] - iso[a][b][c];
            }
        }
    }

    let mut cyclic = [[[0.0; 3]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                cyclic[a][b][c] = (tmp[a][b][c] + tmp[b][c][a] + tmp[c][a][b]) / 3.0;
            }
        }
    }

    // ---------- Final projection ----------
    let mut result = [[[0.0; 3]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                result[a][b][c] = t[a][b][c] - axial[a][b][c] - iso[a][b][c] - cyclic[a][b][c];
            }
        }
    }
    result
}
pub fn frobenius_norm(t: &[[[f64; 3]; 3]; 3]) -> f64 {
    let mut sum = 0.0;
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                sum += t[a][b][c] * t[a][b][c];
            }
        }
    }
    sum.sqrt()
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
pub fn count_nonzero_components(t: &[[[f64; 3]; 3]; 3]) -> usize {
    let mut count = 0;
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                if t[a][b][c].abs() > 1e-10 {
                    count += 1;
                }
            }
        }
    }
    count
}
pub fn flatten(t: &[[[f64; 3]; 3]; 3]) -> [f64; 27] {
    let mut v = [0.0; 27];
    let mut k = 0;
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                v[k] = t[a][b][c];
                k += 1;
            }
        }
    }
    v
}
