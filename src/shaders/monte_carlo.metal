// Metal compute shaders for Monte Carlo simulations on M1 GPU
// Optimized for Apple Silicon unified memory architecture

#include <metal_stdlib>
#include <metal_compute>
using namespace metal;

// Constants matching Rust side
constant int WARP_SIZE = 32;
constant int MAX_THREADS = 1024;

// Link structure matching Rust layout
struct MetalLink {
    uint i;
    uint j;
    float z;
    float theta;
    float cos_theta;
    float sin_theta;
    float exp_neg_z;
    float w_cos;
    float w_sin;
    float padding[7];
};

// Random number generator state (PCG-like)
struct RNGState {
    uint64_t state;
    uint64_t inc;
};

// Batched observables for reduction
struct Observables {
    float sum_cos;
    float sum_sin;
    float sum_w;
    float sum_w_cos;
    float sum_w_sin;
    float triangle_sum;
    uint accept_count;
};

// Fast inline RNG (PCG algorithm adapted for GPU)
inline uint32_t pcg32_random(thread RNGState* rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

inline float uniform_float(thread RNGState* rng) {
    return pcg32_random(rng) * (1.0f / 4294967296.0f);
}

inline float uniform_range(thread RNGState* rng, float min, float max) {
    return min + uniform_float(rng) * (max - min);
}

// Metropolis update kernel for links
kernel void metropolis_update(
    device MetalLink* links [[buffer(0)]],
    device const float* params [[buffer(1)]], // [alpha, beta, delta_z, delta_theta]
    device RNGState* rng_states [[buffer(2)]],
    device atomic_uint* accept_counter [[buffer(3)]],
    constant uint& n_links [[buffer(4)]],
    constant uint& n_nodes [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n_links) return;
    
    // Load parameters
    float alpha = params[0];
    float beta = params[1];
    float delta_z = params[2];
    float delta_theta = params[3];
    
    // Get thread-local RNG
    thread RNGState rng = rng_states[tid];
    
    // Load link data
    MetalLink link = links[tid];
    
    // Decide update type
    bool do_z_update = (delta_z > 0.0f) && (uniform_float(&rng) < 0.5f);
    
    if (do_z_update) {
        // Z-update
        float old_z = link.z;
        float old_exp_neg_z = link.exp_neg_z;
        float new_z = max(0.001f, old_z + uniform_range(&rng, -delta_z, delta_z));
        float new_exp_neg_z = exp(-new_z);
        
        // Calculate energy change (entropy only for z-updates)
        float delta_entropy = -new_z * new_exp_neg_z - (-old_z * old_exp_neg_z);
        float delta_s = beta * delta_entropy;
        
        // Metropolis acceptance
        bool accept = (delta_s <= 0.0f) || (uniform_float(&rng) < exp(-delta_s));
        
        if (accept) {
            link.z = new_z;
            link.exp_neg_z = new_exp_neg_z;
            link.w_cos = new_exp_neg_z * link.cos_theta;
            link.w_sin = new_exp_neg_z * link.sin_theta;
            atomic_fetch_add_explicit(accept_counter, 1, memory_order_relaxed);
        }
    } else {
        // Theta-update
        float old_theta = link.theta;
        float new_theta = old_theta + uniform_range(&rng, -delta_theta, delta_theta);
        
        // For theta updates, we need triangle sum change
        // This is expensive on GPU without shared memory optimization
        // For now, accept based on local change only
        float delta_s = 0.0f; // Simplified - would need triangle computation
        
        bool accept = (delta_s <= 0.0f) || (uniform_float(&rng) < exp(-delta_s));
        
        if (accept) {
            link.theta = new_theta;
            link.cos_theta = cos(new_theta);
            link.sin_theta = sin(new_theta);
            link.w_cos = link.exp_neg_z * link.cos_theta;
            link.w_sin = link.exp_neg_z * link.sin_theta;
            atomic_fetch_add_explicit(accept_counter, 1, memory_order_relaxed);
        }
    }
    
    // Write back updated link and RNG state
    links[tid] = link;
    rng_states[tid] = rng;
}

// Batched Metropolis update kernel - each thread processes multiple MC steps
kernel void metropolis_update_batched(
    device MetalLink* links [[buffer(0)]],
    device const float* params [[buffer(1)]], // [alpha, beta, delta_z, delta_theta]
    device RNGState* rng_states [[buffer(2)]],
    device atomic_uint* accept_counter [[buffer(3)]],
    constant uint& n_links [[buffer(4)]],
    constant uint& n_nodes [[buffer(5)]],
    constant uint& steps_per_thread [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n_links) return;
    
    // Load parameters once
    float alpha = params[0];
    float beta = params[1];
    float delta_z = params[2];
    float delta_theta = params[3];
    
    // Load RNG state and link data once
    thread RNGState rng = rng_states[tid];
    thread MetalLink link = links[tid];
    
    // Local accept counter
    uint local_accepts = 0;
    
    // Process multiple MC steps in a loop
    for (uint step = 0; step < steps_per_thread; step++) {
        // Decide update type
        bool do_z_update = (delta_z > 0.0f) && (uniform_float(&rng) < 0.5f);
        
        if (do_z_update) {
            // Z-update
            float old_z = link.z;
            float old_exp_neg_z = link.exp_neg_z;
            float new_z = max(0.001f, old_z + uniform_range(&rng, -delta_z, delta_z));
            float new_exp_neg_z = exp(-new_z);
            
            // Calculate energy change (entropy only for z-updates)
            float delta_entropy = -new_z * new_exp_neg_z - (-old_z * old_exp_neg_z);
            float delta_s = beta * delta_entropy;
            
            // Metropolis acceptance
            bool accept = (delta_s <= 0.0f) || (uniform_float(&rng) < exp(-delta_s));
            
            if (accept) {
                link.z = new_z;
                link.exp_neg_z = new_exp_neg_z;
                link.w_cos = new_exp_neg_z * link.cos_theta;
                link.w_sin = new_exp_neg_z * link.sin_theta;
                local_accepts++;
            }
        } else {
            // Theta-update
            float old_theta = link.theta;
            float new_theta = old_theta + uniform_range(&rng, -delta_theta, delta_theta);
            
            // For theta updates, we need triangle sum change
            // This is expensive on GPU without shared memory optimization
            // For now, accept based on local change only
            float delta_s = 0.0f; // Simplified - would need triangle computation
            
            bool accept = (delta_s <= 0.0f) || (uniform_float(&rng) < exp(-delta_s));
            
            if (accept) {
                link.theta = new_theta;
                link.cos_theta = cos(new_theta);
                link.sin_theta = sin(new_theta);
                link.w_cos = link.exp_neg_z * link.cos_theta;
                link.w_sin = link.exp_neg_z * link.sin_theta;
                local_accepts++;
            }
        }
    }
    
    // Write back updated link and RNG state once
    links[tid] = link;
    rng_states[tid] = rng;
    
    // Update global accept counter once
    if (local_accepts > 0) {
        atomic_fetch_add_explicit(accept_counter, local_accepts, memory_order_relaxed);
    }
}

// Triangle sum kernel using threadgroup memory
kernel void triangle_sum_kernel(
    device const MetalLink* links [[buffer(0)]],
    device float* partial_sums [[buffer(1)]],
    constant uint& n_nodes [[buffer(2)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    // Each thread handles one triangle
    uint n_triangles = n_nodes * (n_nodes - 1) * (n_nodes - 2) / 6;
    
    float local_sum = 0.0f;
    
    // Grid-stride loop for load balancing
    for (uint tri_idx = tid; tri_idx < n_triangles; tri_idx += group_size * 32768) {
        // Decode triangle indices (i < j < k)
        uint remaining = tri_idx;
        uint i = 0;
        
        // Find i using quadratic formula approximation
        while (remaining >= (n_nodes - i - 1) * (n_nodes - i - 2) / 2) {
            remaining -= (n_nodes - i - 1) * (n_nodes - i - 2) / 2;
            i++;
        }
        
        uint j = i + 1;
        while (remaining >= n_nodes - j - 1) {
            remaining -= n_nodes - j - 1;
            j++;
        }
        
        uint k = j + 1 + remaining;
        
        // Calculate link indices
        uint idx_ij = i * n_nodes - i * (i + 1) / 2 + j - i - 1;
        uint idx_jk = j * n_nodes - j * (j + 1) / 2 + k - j - 1;
        uint idx_ik = i * n_nodes - i * (i + 1) / 2 + k - i - 1;
        
        // Load theta values and compute triangle contribution
        float theta_sum = links[idx_ij].theta + links[idx_jk].theta + links[idx_ik].theta;
        local_sum += 3.0f * cos(theta_sum);
    }
    
    // Reduce within threadgroup
    shared_mem[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction
    for (uint stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            shared_mem[lid] += shared_mem[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write partial sum
    if (lid == 0) {
        partial_sums[gid] = shared_mem[0];
    }
}

// Batched RNG initialization kernel
kernel void init_rng_states(
    device RNGState* rng_states [[buffer(0)]],
    constant uint64_t& base_seed [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    // Initialize each thread with unique stream
    rng_states[tid].state = base_seed + tid * 2654435769ULL;
    rng_states[tid].inc = (tid * 2 + 1) | 1;
    
    // Warm up the generator
    RNGState local_rng = rng_states[tid];
    for (int i = 0; i < 10; i++) {
        pcg32_random(&local_rng);
    }
    rng_states[tid] = local_rng;
}

// Observable reduction kernel - each threadgroup writes to its own slot
kernel void reduce_observables(
    device const MetalLink* links [[buffer(0)]],
    device float3* partial_results [[buffer(1)]],  // cos, w, w_cos per threadgroup
    constant uint& n_links [[buffer(2)]],
    threadgroup float* shared_cos [[threadgroup(0)]],
    threadgroup float* shared_w [[threadgroup(1)]],
    threadgroup float* shared_w_cos [[threadgroup(2)]],
    uint tid [[thread_position_in_grid]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    float local_cos = 0.0f;
    float local_w = 0.0f;
    float local_w_cos = 0.0f;
    
    // Grid-stride loop
    for (uint i = tid; i < n_links; i += group_size * 32768) {
        MetalLink link = links[i];
        local_cos += link.cos_theta;
        local_w += link.exp_neg_z;
        local_w_cos += link.w_cos;
    }
    
    // Store in shared memory
    shared_cos[lid] = local_cos;
    shared_w[lid] = local_w;
    shared_w_cos[lid] = local_w_cos;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction
    for (uint stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            shared_cos[lid] += shared_cos[lid + stride];
            shared_w[lid] += shared_w[lid + stride];
            shared_w_cos[lid] += shared_w_cos[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write partial result for this threadgroup
    if (lid == 0) {
        partial_results[gid] = float3(shared_cos[0], shared_w[0], shared_w_cos[0]);
    }
}

// Optimized entropy calculation - each threadgroup writes to its own slot
kernel void entropy_sum_kernel(
    device const MetalLink* links [[buffer(0)]],
    device float* partial_results [[buffer(1)]],
    constant uint& n_links [[buffer(2)]],
    threadgroup float* shared_mem [[threadgroup(0)]],
    uint tid [[thread_position_in_grid]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]]
) {
    float local_sum = 0.0f;
    
    // Process links in chunks
    for (uint i = tid; i < n_links; i += group_size * 32768) {
        float z = links[i].z;
        float exp_neg_z = links[i].exp_neg_z;
        local_sum += -z * exp_neg_z;
    }
    
    // Threadgroup reduction
    shared_mem[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = group_size / 2; stride > 0; stride /= 2) {
        if (lid < stride) {
            shared_mem[lid] += shared_mem[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (lid == 0) {
        partial_results[gid] = shared_mem[0];
    }
}