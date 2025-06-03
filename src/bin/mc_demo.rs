use rc_sim::graph::Graph;

fn main() {
    let mut g   = Graph::complete_random(8);
    let beta  = 1.0;
    let delta_w     = 0.1;
    let delta_theta = 0.2;
    let mut accepted = 0;
    let n_steps = 5000;

    for step in 1..=n_steps {
        if g.metropolis_step(beta, delta_w, delta_theta) {   // ← three args
            accepted += 1;
        }
        if step % 1000 == 0 {
            println!(
                "Step {:5}  action = {:8.4},  acceptance so far = {:.3}",
                step,
                g.action(),
                accepted as f64 / step as f64
            );
        }
    }
    println!("Done. Final acceptance rate = {:.3}", accepted as f64 / n_steps as f64);
}

