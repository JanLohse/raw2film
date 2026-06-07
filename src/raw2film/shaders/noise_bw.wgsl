struct Params {
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    seed: u32,
};

@group(0) @binding(0)
var output_tex: texture_storage_2d<rgba16float, write>;

@group(0) @binding(1)
var<uniform> params: Params;

fn pcg_3d(p: vec3<u32>) -> vec3<u32> {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> vec3<u32>(16u);
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

// 2. Generates uniform random float numbers in the range [0.0, 1.0)
// Optimized to only convert the first two channels since monochromatic needs fewer dimensions
fn rand_uniform_2d(seed: vec3<u32>) -> vec2<f32> {
    let hashed = pcg_3d(seed);
    return vec2<f32>(hashed.xy) * (1.0 / f32(0xffffffffu));
}

// 3. Yields 1 Gaussian sample using Box-Muller, duplicated across RGB
fn sample_gaussian_mono(seed: vec3<u32>) -> vec4<f32> {
    let u = rand_uniform_2d(seed);

    // Guard against log(0)
    let u1 = max(u.x, 1e-7);
    let u2 = u.y;

    // Box-Muller transform gives us a single Gaussian noise value
    let r = sqrt(-2.0 * log(u1));
    let theta = 2.0 * 3.14159265359 * u2;
    let noise_val = r * cos(theta);

    // Assign the same noise value to R, G, and B channels
    return vec4<f32>(noise_val, noise_val, noise_val, 1.0);
}

@compute
@workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let tex_coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let seed = vec3<u32>(gid.x, gid.y, params.seed);
    let noise = sample_gaussian_mono(seed);

    textureStore(output_tex, tex_coords, noise);
}
