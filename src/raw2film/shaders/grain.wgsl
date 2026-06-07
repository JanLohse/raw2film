struct Params {
    xp_min: f32,
    xp_max: f32,
    inv_range: f32,
    _pad: u32,
};

struct Kernel {
    weights: array<vec4<f32>>,
};

@group(0) @binding(0)
var input_tex: texture_2d<f32>;

@group(0) @binding(1)
var output_tex: texture_storage_2d<rgba16float, write>;

@group(0) @binding(2)
var lut_tex: texture_2d<f32>;

@group(0) @binding(3)
var lut_sampler: sampler;

@group(0) @binding(4)
var<uniform> params: Params;

@group(0) @binding(5)
var noise_tex: texture_2d<f32>;

@group(0) @binding(6)
var<storage, read> kernel : Kernel;

@group(0) @binding(7)
var<uniform> kernel_size : vec2<u32>;

@compute
@workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let dims = textureDimensions(input_tex);
    let pixel_coords = vec2<i32>(gid.xy);

    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let pixel = textureLoad(input_tex, pixel_coords, 0);

    // Convolution smoothing on noise to get grain texture:
    let k_size = vec2<i32>(kernel_size);
    let k_center = k_size / 2;

    let start_coord = pixel_coords - k_center;

    var sum = vec4<f32>(0.0);
    var weight_idx = 0u;

    // Convert dimensions to i32 safely to fix the clamp type mismatch
    let dims_i = vec2<i32>(dims);
    let max_coord = dims_i - vec2<i32>(1);

    for (var ky: i32 = 0; ky < k_size.y; ky++) {
        let src_y = clamp(start_coord.y + ky, 0, max_coord.y);

        for (var kx: i32 = 0; kx < k_size.x; kx++) {
            let src_x = clamp(start_coord.x + kx, 0, max_coord.x);

            let grain_pixel = textureLoad(noise_tex, vec2<i32>(src_x, src_y), 0);
            let weight = kernel.weights[weight_idx];

            sum += grain_pixel * weight;
            weight_idx++;
        }
    }

    // Calculate scaling factors:
    let normalized_pos = clamp(
        (pixel.rgb - params.xp_min) * params.inv_range,
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );

    let noise_r = textureSampleLevel(lut_tex, lut_sampler, vec2<f32>(normalized_pos.r, 0.5), 0.0).r * sum.r;
    let noise_g = textureSampleLevel(lut_tex, lut_sampler, vec2<f32>(normalized_pos.g, 0.5), 0.0).g * sum.g;
    let noise_b = textureSampleLevel(lut_tex, lut_sampler, vec2<f32>(normalized_pos.b, 0.5), 0.0).b * sum.b;

    // Combine for final output using correct variable names
    let out_color = vec4<f32>(pixel.r + noise_r, pixel.g + noise_g, pixel.b + noise_b, pixel.a);

    textureStore(output_tex, pixel_coords, out_color);
}
