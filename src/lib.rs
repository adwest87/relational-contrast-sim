pub mod projector;
pub mod graph;
pub mod measure;
pub mod observables;
pub mod finite_size;
pub mod graph_fast;
pub mod fast_mc_integration;
pub mod importance_sampling;
pub mod importance_mc_integration;
pub mod error_analysis;
pub mod graph_ultra_optimized;
pub mod minimal_correct_physics;

#[cfg(target_arch = "aarch64")]
pub mod graph_m1_optimized;

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub mod graph_m1_accelerate;

#[cfg(target_os = "macos")]
pub mod graph_metal;

// Utility modules
pub mod utils {
    pub mod config;
    pub mod output;
    pub mod ridge;
    pub mod rng;
}

// Re-exports for backward compatibility
pub use utils::config;
pub use utils::output;
pub use utils::ridge;
pub use utils::rng;
