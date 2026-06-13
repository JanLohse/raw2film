@group(0) @binding(0) var input_texture: texture_2d<f32>;

// Intermediate Atomic Counter Buffer for Pass 1
struct HistogramCounts {
    r: array<atomic<u32>, 256>,
    g: array<atomic<u32>, 256>,
    b: array<atomic<u32>, 256>,
}
@group(0) @binding(1) var<storage, read_write> global_counts: HistogramCounts;

// Intermediate Processed Heights Buffer for Pass 2 -> Pass 3
struct FinalHeights {
    r: array<u32, 256>,
    g: array<u32, 256>,
    b: array<u32, 256>,
}
@group(0) @binding(2) var<storage, read_write> final_heights: FinalHeights;

// The mix_table lookup array (Flattened 2x2x2x4 into a 1D array of vec4<u32> or vec4<f32>)
// Indexing: (is_r * 4) + (is_g * 2) + is_b
@group(0) @binding(3) var<storage, read> mix_table: array<vec4<u32>, 8>;

// Final output texture/buffer
@group(0) @binding(4) var output_texture: texture_storage_2d<rgba8uint, write>;

// Uniforms for heights and dimensions
struct Params {
    height: u32,
    img_width: u32,
    img_height: u32,
}
@group(0) @binding(5) var<uniform> params: Params;


@compute @workgroup_size(16, 16)
fn pass1_accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.img_width || id.y >= params.img_height) {
        return;
    }

    // Read the texture pixel (assumed unnormalized or converted to 0.0-1.0 range)
    let pixel = textureLoad(input_texture, vec2<i32>(id.xy), 0);

    // Convert float color [0.0, 1.0] to uint index [0, 255]
    let r_idx = u32(clamp(pixel.r * 255.0, 0.0, 255.0));
    let g_idx = u32(clamp(pixel.g * 255.0, 0.0, 255.0));
    let b_idx = u32(clamp(pixel.b * 255.0, 0.0, 255.0));

    // Atomically increment the frequencies
    atomicAdd(&global_counts.r[r_idx], 1u);
    atomicAdd(&global_counts.g[g_idx], 1u);
    atomicAdd(&global_counts.b[b_idx], 1u);
}

// Shared memory for finding maximums and doing cross-thread smoothing
var<workgroup> shared_r: array<f32, 256>;
var<workgroup> shared_g: array<f32, 256>;
var<workgroup> shared_b: array<f32, 256>;
var<workgroup> global_max: f32;

@compute @workgroup_size(256)
fn pass2_process(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let idx = local_id.x;

    // 1. Read atomic values into float arrays
    let r_val = f32(atomicLoad(&global_counts.r[idx]));
    let g_val = f32(atomicLoad(&global_counts.g[idx]));
    let b_val = f32(atomicLoad(&global_counts.b[idx]));

    shared_r[idx] = r_val;
    shared_g[idx] = g_val;
    shared_b[idx] = b_val;

    if (idx == 0u) { global_max = 1.0; }
    workgroupBarrier();

    // 2. Find maximum raw value across all bins (Parallel reduction can be used here,
    // but for 256 elements a quick atomic max or single thread loop is fine.
    // For simplicity here, we let Thread 0 calculate it quickly)
    if (idx == 0u) {
        var m = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) {
            m = max(m, max(shared_r[i], max(shared_g[i], shared_b[i])));
        }
        if (m > 0.0) { global_max = m; }
    }
    workgroupBarrier();

    // 3. Apply Log transform: log1p(val / max) -> log(1.0 + (val / max))
    let max_v = global_max;
    shared_r[idx] = log(1.0 + (shared_r[idx] / max_v));
    shared_g[idx] = log(1.0 + (shared_g[idx] / max_v));
    shared_b[idx] = log(1.0 + (shared_b[idx] / max_v));
    workgroupBarrier();

    // 4. Smooth using 1-pixel moving average kernel
    let left = select(idx - 1u, idx, idx == 0u);
    let right = select(idx + 1u, idx, idx == 255u);

    let smooth_r = (shared_r[left] + shared_r[idx] + shared_r[right]) / 3.0;
    let smooth_g = (shared_g[left] + shared_g[idx] + shared_g[right]) / 3.0;
    let smooth_b = (shared_b[left] + shared_b[idx] + shared_b[right]) / 3.0;

    // Store back into shared memory to find the secondary maximum for scaling
    shared_r[idx] = smooth_r;
    shared_g[idx] = smooth_g;
    shared_b[idx] = smooth_b;
    workgroupBarrier();

    // 5. Find max of smoothed values to scale to vertical height bounds
    if (idx == 0u) {
        var m = 0.0;
        for (var i = 0u; i < 256u; i = i + 1u) {
            m = max(m, max(shared_r[i], max(shared_g[i], shared_b[i])));
        }
        global_max = select(m, 1.0, m == 0.0);
    }
    workgroupBarrier();

    // 6. Scale to final height bounds and store to global storage
    let final_m = global_max;
    let h_f32 = f32(params.height);

    final_heights.r[idx] = u32((shared_r[idx] * h_f32) / final_m);
    final_heights.g[idx] = u32((shared_g[idx] * h_f32) / final_m);
    final_heights.b[idx] = u32((shared_b[idx] * h_f32) / final_m);
}

@compute @workgroup_size(16, 16)
fn pass3_render(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x; // Bin column (0 to 255)
    let y = id.y; // Height row (0 to height-1)

    if (x >= 256u || y >= params.height) {
        return;
    }

    // Read calculated pixel limit heights for this specific bin
    let final_h_r = final_heights.r[x];
    let final_h_g = final_heights.g[x];
    let final_h_b = final_heights.b[x];

    let r_lim = params.height - final_h_r;
    let g_lim = params.height - final_h_g;
    let b_lim = params.height - final_h_b;

    // Compute boolean flags (0 or 1)
    let is_r = select(0u, 1u, y >= r_lim);
    let is_g = select(0u, 1u, y >= g_lim);
    let is_b = select(0u, 1u, y >= b_lim);

    // Flattened 3D lookup table index: (is_r * 4) + (is_g * 2) + is_b
    let table_idx = (is_r * 4u) + (is_g * 2u) + is_b;
    let color_out = mix_table[table_idx];

    // Write final pixel to texture coordinate (x, y)
    textureStore(output_texture, vec2<u32>(x, y), color_out);
}
