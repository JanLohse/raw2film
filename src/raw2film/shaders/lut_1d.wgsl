struct Params {
    xp_min: f32,
    xp_max: f32,
    inv_range: f32,
    _pad: u32,
};

struct ColorMaskingMatrix {
    m00: f32, m01: f32, m02: f32, _pad0: u32,
    m10: f32, m11: f32, m12: f32, _pad1: u32,
    m20: f32, m21: f32, m22: f32, _pad2: u32,
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
var<uniform> color_matrix: ColorMaskingMatrix;

fn safe_log10_vec3(v: vec3<f32>) -> vec3<f32> {
    let eps = 1e-6;
    return log2(max(v, vec3<f32>(eps))) / log2(10.0);
}

fn apply_color_masking_matrix(rgb: vec3<f32>) -> vec3<f32> {
    let r = rgb.r * color_matrix.m00 + rgb.g * color_matrix.m01 + rgb.b * color_matrix.m02;
    let g = rgb.r * color_matrix.m10 + rgb.g * color_matrix.m11 + rgb.b * color_matrix.m12;
    let b = rgb.r * color_matrix.m20 + rgb.g * color_matrix.m21 + rgb.b * color_matrix.m22;
    return vec3<f32>(r, g, b);
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

    // Apply color masking matrix after log conversion
    let masked_pixel = apply_color_masking_matrix(log_pixel);

    let normalized_pos = clamp(
        (masked_pixel - params.xp_min) * params.inv_range,
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );

    let out_r = textureSampleLevel(lut_tex, lut_sampler, vec2<f32>(normalized_pos.r, 0.5), 0.0).r;
    let out_g = textureSampleLevel(lut_tex, lut_sampler, vec2<f32>(normalized_pos.g, 0.5), 0.0).g;
    let out_b = textureSampleLevel(lut_tex, lut_sampler, vec2<f32>(normalized_pos.b, 0.5), 0.0).b;

    let out_color = vec4<f32>(out_r, out_g, out_b, pixel.a);

    textureStore(output_tex, tex_coords, out_color);
}
