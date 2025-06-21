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

#[cfg(target_arch = "aarch64")]
pub mod graph_m1_optimized;

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub mod graph_m1_accelerate;

#[cfg(target_os = "macos")]
pub mod graph_metal;