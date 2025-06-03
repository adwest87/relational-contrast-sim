use rc_sim::graph::Graph;

fn main() {
    let mut g = Graph::complete_random(4);
    let (before, after) = g.project_all();
    println!("Graph with {} links", g.links.len());
    println!("Total Frobenius norm:");
    println!("  Before projection: {:.5}", before);
    println!("  After  projection: {:.5}", after);
    println!("Entropy action Σ w ln w = {:.5}", g.entropy_action());
}

