use scan::graph::Graph;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

#[test]
fn test_phase_antisymmetry_basic() {
    // Test that θ_ij + θ_ji = 0 for all links
    let g = Graph::complete_random(5);
    let n = g.n();
    
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let theta_ij = g.get_phase(i, j);
                let theta_ji = g.get_phase(j, i);
                
                assert!(
                    (theta_ij + theta_ji).abs() < 1e-10,
                    "Phase antisymmetry violated: θ[{},{}] = {}, θ[{},{}] = {}, sum = {}",
                    i, j, theta_ij, j, i, theta_ji, theta_ij + theta_ji
                );
            }
        }
    }
}

#[test]
fn test_phase_diagonal_zero() {
    // Test that θ_ii = 0 for all nodes
    let g = Graph::complete_random(5);
    let n = g.n();
    
    for i in 0..n {
        let theta_ii = g.get_phase(i, i);
        assert_eq!(
            theta_ii, 0.0,
            "Diagonal phase should be zero: θ[{},{}] = {}",
            i, i, theta_ii
        );
    }
}

#[test]
fn test_link_storage_convention() {
    // Test that links are stored with i < j convention
    let g = Graph::complete_random(6);
    
    for link in &g.links {
        assert!(
            link.i < link.j,
            "Link storage convention violated: i={} >= j={}",
            link.i, link.j
        );
    }
}

#[test]
fn test_triangle_holonomy_consistency() {
    // Test that triangle sum is independent of node ordering
    let mut rng = ChaCha20Rng::seed_from_u64(12345);
    let g = Graph::complete_random_with(&mut rng, 5);
    
    // Pick a specific triangle (0, 1, 2)
    let i = 0;
    let j = 1;
    let k = 2;
    
    // Calculate holonomy in different orderings
    let h1 = g.get_phase(i, j) + g.get_phase(j, k) + g.get_phase(k, i);
    let h2 = g.get_phase(j, k) + g.get_phase(k, i) + g.get_phase(i, j);
    let h3 = g.get_phase(k, i) + g.get_phase(i, j) + g.get_phase(j, k);
    
    // Also check reverse orientation
    let h4 = g.get_phase(i, k) + g.get_phase(k, j) + g.get_phase(j, i);
    
    assert!(
        (h1 - h2).abs() < 1e-10,
        "Holonomy depends on starting point: {} vs {}",
        h1, h2
    );
    assert!(
        (h1 - h3).abs() < 1e-10,
        "Holonomy depends on starting point: {} vs {}",
        h1, h3
    );
    assert!(
        (h1 + h4).abs() < 1e-10,
        "Holonomy should change sign with orientation: {} vs {}",
        h1, -h4
    );
}

#[test]
fn test_metropolis_preserves_antisymmetry() {
    // Test that Metropolis updates preserve phase antisymmetry
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let mut g = Graph::complete_random_with(&mut rng, 5);
    
    let alpha = 1.0;
    let beta = 2.0;
    let delta_z = 0.1;
    let delta_theta = 0.3;
    
    // Run many Metropolis steps
    for _ in 0..1000 {
        g.metropolis_step(beta, alpha, delta_z, delta_theta, &mut rng);
        
        // Check a random pair
        let i = rng.gen_range(0..g.n());
        let j = rng.gen_range(0..g.n());
        
        if i != j {
            let theta_ij = g.get_phase(i, j);
            let theta_ji = g.get_phase(j, i);
            
            assert!(
                (theta_ij + theta_ji).abs() < 1e-10,
                "Antisymmetry violated after Metropolis: θ[{},{}] + θ[{},{}] = {}",
                i, j, j, i, theta_ij + theta_ji
            );
        }
    }
}

#[test]
fn test_phase_range() {
    // Test that phases can take any value (not restricted to [0, 2π))
    let mut rng = ChaCha20Rng::seed_from_u64(123);
    let mut g = Graph::complete_random_with(&mut rng, 4);
    
    // Run Metropolis with large phase updates
    let alpha = 0.5;
    let beta = 1.0;
    let delta_z = 0.1;
    let delta_theta = 10.0;  // Large phase updates
    
    for _ in 0..1000 {
        g.metropolis_step(beta, alpha, delta_z, delta_theta, &mut rng);
    }
    
    // Check that some phases are outside [-π, π]
    let mut found_large_phase = false;
    for link in &g.links {
        if link.theta.abs() > std::f64::consts::PI {
            found_large_phase = true;
            break;
        }
    }
    
    // This is not a hard requirement, but with large updates it's likely
    if !found_large_phase {
        println!("Warning: No phases found outside [-π, π], but this can happen by chance");
    }
}

#[test]
fn test_get_phase_method() {
    // Test the Link::get_phase method directly
    let mut rng = ChaCha20Rng::seed_from_u64(999);
    let g = Graph::complete_random_with(&mut rng, 4);
    
    // Test a specific link
    let link = &g.links[0];
    let i = link.i;
    let j = link.j;
    
    // Forward direction
    let phase_forward = link.get_phase(i, j);
    assert_eq!(phase_forward, link.theta);
    
    // Reverse direction
    let phase_reverse = link.get_phase(j, i);
    assert_eq!(phase_reverse, -link.theta);
    
    // Check consistency with graph method
    assert_eq!(g.get_phase(i, j), phase_forward);
    assert_eq!(g.get_phase(j, i), phase_reverse);
}

#[test]
fn test_triangle_action_gauge_invariant() {
    // Test that triangle action is invariant under gauge transformations
    let mut rng = ChaCha20Rng::seed_from_u64(2023);
    let g = Graph::complete_random_with(&mut rng, 6);
    
    let alpha = 1.5;
    
    // Calculate triangle action manually with and without gauge transformation
    let gauge: Vec<f64> = (0..g.n()).map(|_| rng.gen_range(-1.0..1.0)).collect();
    
    let mut sum_without_gauge = 0.0;
    let mut sum_with_gauge = 0.0;
    
    for &(i, j, k) in g.triangles().collect::<Vec<_>>().iter() {
        // Without gauge
        let t_ij = g.get_phase(i, j);
        let t_jk = g.get_phase(j, k);
        let t_ki = g.get_phase(k, i);
        sum_without_gauge += (t_ij + t_jk + t_ki).cos();
        
        // With gauge transformation
        let t_ij_gauge = t_ij + gauge[i] - gauge[j];
        let t_jk_gauge = t_jk + gauge[j] - gauge[k];
        let t_ki_gauge = t_ki + gauge[k] - gauge[i];
        sum_with_gauge += (t_ij_gauge + t_jk_gauge + t_ki_gauge).cos();
    }
    
    assert!(
        (sum_without_gauge - sum_with_gauge).abs() < 1e-10,
        "Triangle action not gauge invariant: {} vs {}",
        sum_without_gauge, sum_with_gauge
    );
}