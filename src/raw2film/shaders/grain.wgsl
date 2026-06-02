struct Params {
    xp_min: f32,
    xp_max: f32,
    inv_range: f32,
    seed: u32,
};

@group(0) @binding(0)
var input_tex: texture_2d<f32>;

@group(0) @binding(1)
var output_tex: texture_storage_2d<rgba32float, write>;

@group(0) @binding(2)
var lut_tex: texture_2d<f32>;

@group(0) @binding(3)
var lut_sampler: sampler;

@group(0) @binding(4)
var<uniform> params: Params;

fn safe_log10_vec3(v: vec3<f32>) -> vec3<f32> {
    let eps = 1e-6;
    return log2(max(v, vec3<f32>(eps))) / log2(10.0);
}

fn pcg_3d(p: vec3<u32>) -> vec3<u32> {
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> vec3<u32>(16u);
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

// 2. Generates two uniform random float numbers in the range [0.0, 1.0)
fn rand_uniform_2d(seed: vec3<u32>) -> vec2<f32> {
    let hashed = pcg_3d(seed);
    // Convert 32-bit uint bits uniformly into a float interval [0, 1)
    return vec2<f32>(hashed.xy) * (1.0 / f32(0xffffffffu));
}

// 3. Applies Box-Muller Transform to yield a Gaussian distributed sample
// Returns two independent Gaussian samples: x and y components.
fn sample_gaussian(seed: vec3<u32>) -> vec2<f32> {
    let u = rand_uniform_2d(seed);

    // Guard against log(0) which causes NaN/Inf
    let u1 = max(u.x, 1e-7);
    let u2 = u.y;

    let r = sqrt(-2.0 * log(u1));
    let theta = 2.0 * 3.14159265359 * u2;

    // Both outputs are independent, valid standard Gaussian variables
    return vec2<f32>(r * cos(theta), r * sin(theta));
}

@compute
@workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let tex_coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let pixel = textureLoad(input_tex, tex_coords, 0);

    let log_pixel = safe_log10_vec3(pixel.rgb);

    let normalized_pos = clamp(
        (log_pixel - params.xp_min) * params.inv_range,
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );
    let seed = vec3<u32>(gid.x, gid.y, params.seed);
    let noise = sample_gaussian(seed).x;

    let out_r = textureSampleLevel(lut_tex, lut_sampler, vec2<f32>(normalized_pos.r, 0.5), 0.0).r * noise;
    let out_g = textureSampleLevel(lut_tex, lut_sampler, vec2<f32>(normalized_pos.g, 0.5), 0.0).g * noise;
    let out_b = textureSampleLevel(lut_tex, lut_sampler, vec2<f32>(normalized_pos.b, 0.5), 0.0).b * noise;

    let out_color = vec4<f32>(pixel.r + out_r, pixel.g + out_g, pixel.b + out_b, pixel.a);

    textureStore(output_tex, tex_coords, out_color);
}
