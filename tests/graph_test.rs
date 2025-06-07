use scan::graph::Graph;

#[test]
fn test_complete_graph_sizes() {
    let n = 4;
    let g = Graph::complete_random(n);

    // A complete undirected graph has n(n-1)/2 links
    assert_eq!(g.n(), n, "Wrong number of nodes");
    assert_eq!(g.m(), n * (n - 1) / 2, "Wrong number of links");
}

