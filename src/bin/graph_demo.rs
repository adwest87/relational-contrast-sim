use rc_sim::graph::Graph;

fn main() {
    let g = Graph::complete_random(4);
    println!("Graph with {} nodes and {} links:", g.n(), g.m());
    for link in &g.links {
        println!("({},{})  w = {:.3}", link.i, link.j, link.w);
    }
}

