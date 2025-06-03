use rc_sim::graph::Graph;

fn main() {
    // Simulation parameters
    let mut g          = Graph::complete_random(8);
    let beta           = 1.0;
    let delta_w        = 0.10;
    let delta_theta    = 0.20;
    let n_steps: usize = 10_000;
    let report_every   = 1_000;

    // Counters
    let mut accepted = 0;

    println!(
        "# step  accept%   <w>      <cosθ>    S_entropy   S_triangle   action"
    );

    for step in 1..=n_steps {
        if g.metropolis_step(beta, delta_w, delta_theta) {
            accepted += 1;
        }

        if step % report_every == 0 {
            // ---- observables ---------------------------------------
            let avg_w: f64 = g.sum_weights() / g.m() as f64;
            let avg_cos: f64 =
                g.links.iter().map(|l| l.theta.cos()).sum::<f64>() / g.m() as f64;
            let s_entropy = g.entropy_action();
            let s_tri     = g.triangle_action(1.0); // α = 1
            let total_a   = g.action();
            let acc_rate  = accepted as f64 / step as f64;

            println!(
                "{:>6}  {:>7.3}  {:>7.4}  {:>8.4}  {:>11.4}  {:>11.4}  {:>8.2}",
                step,
                acc_rate,
                avg_w,
                avg_cos,
                s_entropy,
                s_tri,
                total_a,
            );
        }
    }
}
