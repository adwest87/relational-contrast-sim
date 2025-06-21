use scan::graph::Graph;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn main() {
    println!("=== U(1) Phase Antisymmetry Demo ===\n");
    
    let mut rng = ChaCha20Rng::seed_from_u64(12345);
    let n = 5;
    let g = Graph::complete_random_with(&mut rng, n);
    
    println!("Graph with {} nodes ({} links)\n", n, g.m());
    
    // Show phase antisymmetry for some links
    println!("Phase antisymmetry θ_ij = -θ_ji:");
    println!("--------------------------------");
    
    // Check first few links
    for (idx, link) in g.links.iter().take(5).enumerate() {
        let i = link.i;
        let j = link.j;
        let theta_ij = g.get_phase(i, j);
        let theta_ji = g.get_phase(j, i);
        
        println!("Link {}: ({}, {})", idx, i, j);
        println!("  θ[{},{}] = {:+.4}", i, j, theta_ij);
        println!("  θ[{},{}] = {:+.4}", j, i, theta_ji);
        println!("  Sum = {:+.4e}", theta_ij + theta_ji);
        println!();
    }
    
    // Show triangle holonomy calculation
    println!("Triangle holonomy examples:");
    println!("---------------------------");
    
    // First triangle (0, 1, 2)
    let (i, j, k) = (0, 1, 2);
    
    println!("Triangle ({}, {}, {}):", i, j, k);
    let theta_ij = g.get_phase(i, j);
    let theta_jk = g.get_phase(j, k); 
    let theta_ki = g.get_phase(k, i);
    let holonomy = theta_ij + theta_jk + theta_ki;
    
    println!("  θ[{},{}] = {:+.4}", i, j, theta_ij);
    println!("  θ[{},{}] = {:+.4}", j, k, theta_jk);
    println!("  θ[{},{}] = {:+.4}", k, i, theta_ki);
    println!("  Holonomy = {:+.4}", holonomy);
    println!("  cos(holonomy) = {:+.4}", holonomy.cos());
    println!();
    
    // Same triangle, different orientation
    println!("Same triangle, opposite orientation:");
    let theta_ik = g.get_phase(i, k);
    let theta_kj = g.get_phase(k, j);
    let theta_ji = g.get_phase(j, i);
    let holonomy_rev = theta_ik + theta_kj + theta_ji;
    
    println!("  θ[{},{}] = {:+.4}", i, k, theta_ik);
    println!("  θ[{},{}] = {:+.4}", k, j, theta_kj);
    println!("  θ[{},{}] = {:+.4}", j, i, theta_ji);
    println!("  Holonomy = {:+.4}", holonomy_rev);
    println!("  cos(holonomy) = {:+.4}", holonomy_rev.cos());
    println!();
    
    println!("Note: Holonomies differ by sign, but cos() is the same!");
    println!("This ensures the triangle action is well-defined.\n");
    
    // Verify all links maintain antisymmetry
    println!("Verifying antisymmetry for all {} links...", g.m());
    let mut max_error: f64 = 0.0;
    for link in &g.links {
        let sum = g.get_phase(link.i, link.j) + g.get_phase(link.j, link.i);
        max_error = max_error.max(sum.abs());
    }
    println!("Maximum antisymmetry error: {:.2e}", max_error);
    
    if max_error < 1e-10 {
        println!("✓ Perfect antisymmetry maintained!");
    } else {
        println!("✗ Antisymmetry violation detected!");
    }
}