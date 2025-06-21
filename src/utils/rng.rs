use rand_chacha::ChaCha20Rng;
use rand::SeedableRng;        // <── NEW  (comes from the main rand crate)
use rand::RngCore;            // <── NEW  (needed inside the function)

/// Per‑thread deterministic RNG
pub fn thread_rng(master: u64, thread_id: usize) -> ChaCha20Rng {
    let mut x = master ^ ((thread_id as u64).wrapping_mul(0x9E3779B97F4A7C15));
    x = x ^ (x >> 30).wrapping_mul(0xBF58476D1CE4E5B9);
    x = x ^ (x >> 27).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31);
    ChaCha20Rng::seed_from_u64(x)
}

