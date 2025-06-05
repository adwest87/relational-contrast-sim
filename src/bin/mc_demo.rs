use rc_sim::graph::Graph;
use std::fs::File;
use std::io::Write;
use rc_sim::measure::Recorder;


/// Simple proportional tuner for proposal widths
struct Tuner {
    delta: f64,
    acc_count: usize,
    window: usize,
    target: f64,
    band: f64,
}
impl Tuner {
    fn new(delta: f64, window: usize, target: f64, band: f64) -> Self {
        Self { delta, acc_count: 0, window, target, band }
    }
    /// Call after each proposal with `accepted == true` if accepted
    fn update(&mut self, accepted: bool) {
        if accepted { self.acc_count += 1; }
        if self.acc_count + 1 == self.window { // window reached
            let acc_rate = self.acc_count as f64 / self.window as f64;
            if acc_rate > self.target + self.band {        // too easy → enlarge step
                self.delta *= 1.1;
            } else if acc_rate < self.target - self.band { // too hard → shrink step
                self.delta *= 0.9;
            }
            self.acc_count = 0; // reset window
        }
    }
}


fn main() {
    // Simulation parameters
    let mut g          = Graph::complete_random(8);
    let beta           = 1.0;
    let mut tuner_w    = Tuner::new(0.10, 200, 0.30, 0.05);   // weight tuner
    let mut tuner_th = Tuner::new(0.20, 200, 0.30, 0.05);   // start δθ=0.20
    let n_steps: usize = 100_000;
    let report_every   = 1_000;
    let equil_steps = 20_000; // or 100_000 depending on your total sweeps


    // ----------------------------------------------------------
    // CSV output
    // ----------------------------------------------------------
    let mut csv = File::create("mc_observables.csv")
        .expect("cannot create mc_observables.csv");
    let mut recorder = Recorder::default();

    writeln!(
        csv,
        "step,accept_rate,delta_w,delta_theta,avg_w,avg_cos_theta,S_entropy,S_triangle,action"
    ).unwrap();


    let mut theta_csv = File::create("theta_final.csv")
        .expect("cannot create theta_final.csv");
    writeln!(theta_csv, "link_i,link_j,theta").unwrap();


    // Counters
    let mut accepted_count = 0;


    println!(
        "# step  accept%   <w>      <cosθ>    S_entropy   S_triangle   action"
    );

    for step in 1..=n_steps {
        let accepted = g.metropolis_step(beta, tuner_w.delta, tuner_th.delta);
        if accepted {
            accepted_count += 1;
        }
        tuner_w.update(accepted);
        tuner_th.update(accepted);

        if step >= equil_steps {
            recorder.push(&g.links);
        }

        if step % report_every == 0 {
            // ---- observables ---------------------------------------
            let avg_w: f64 = g.sum_weights() / g.m() as f64;
            let avg_cos: f64 = g.links.iter().map(|l| l.theta.cos()).sum::<f64>() / g.m() as f64;
            let s_entropy = g.entropy_action();
            let s_tri     = g.triangle_action(1.0); // α = 1
            let total_a   = g.action();
            let acc_rate = accepted_count as f64 / step as f64;

            println!(
                "{:>6} {acc:>5.2}%  δw={:>6.3}  δθ={:>6.3}  ⟨w⟩={:>6.3}  ⟨cosθ⟩={:>6.3}  SΔ={:>8.2}  Sₑ={:>8.2}  A={:>8.2}",
                step,
                tuner_w.delta,
                tuner_th.delta,
                avg_w,
                avg_cos,
                s_tri,
                s_entropy,
                total_a,
                acc = 100.0 * acc_rate,
            );


            writeln!(
                csv,
                "{},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5},{:.5}",
                step,
                acc_rate,
                tuner_w.delta,
                tuner_th.delta,
                avg_w,
                avg_cos,
                s_entropy,
                s_tri,
                total_a
            ).unwrap();


        }
    }

    // --------------------------------------------
    // Save final θ for every link
    // --------------------------------------------
    for link in &g.links {
        writeln!(theta_csv, "{},{},{}", link.i, link.j, link.theta).unwrap();
    }
    println!("Saved final U(1) phases to theta_final.csv");

    use std::fs::File;
    use std::io::Write;

    let mut file = File::create("timeseries_cos.csv").expect("cannot create file");
    for val in &recorder.cos_theta {
        writeln!(file, "{val}").unwrap();
    }
}