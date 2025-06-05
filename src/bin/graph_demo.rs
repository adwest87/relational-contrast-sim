use rc_sim::graph::Graph;

fn main() {
    let mut g = Graph::complete_random(4);
    let (before, after) = g.project_all();
    println!("Graph with {} links", g.links.len());
    println!("Total Frobenius norm:");
    println!("  Before projection: {:.5}", before);
    println!("  After  projection: {:.5}", after);
    println!("Entropy action Σ w ln w = {:.5}", g.entropy_action());
    println!("\nDougal scaling check (λ = 2.0)");
    let lambda = 2.0;
    let s_before = g.entropy_action();
    let sum_w: f64 = g.links.iter().map(|l| l.w).sum();
    g.rescale_weights(lambda);
    let s_after = g.entropy_action();
    let expected  = lambda * s_before + lambda * sum_w * lambda.ln();
    println!("  S(λw)        = {:.6}", s_after);
    println!("  Expected     = {:.6}", expected);
    println!("Dougal-invariant action  I  = {:.5}", g.invariant_action());
    println!("Triangle term  S_Δ = {:.5}", g.triangle_action(1.0));
}

