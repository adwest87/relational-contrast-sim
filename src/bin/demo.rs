use scan::projector::aib_project;
use rand::Rng;               // add rand = "0.8" in Cargo.toml

fn main() {
    // Make a random tensor
    let mut rng = rand::thread_rng();
    let mut t = [[[0.0; 3]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                t[a][b][c] = rng.gen_range(-1.0..1.0);
            }
        }
    }

    println!("Raw tensor:");
    print_tensor(&t);

    // Project it
    let p = aib_project(t);

    println!("\nProjected tensor:");
    print_tensor(&p);
}

fn print_tensor(t: &[[[f64; 3]; 3]; 3]) {
    for a in 0..3 {
        for b in 0..3 {
            for c in 0..3 {
                print!("{:6.3} ", t[a][b][c]);
            }
            println!();
        }
        println!();
    }
}

