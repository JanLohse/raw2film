struct CurveParams {
    xp_min: f32,
    xp_max: f32,
    inv_range: f32,
    apply_log: u32,
};

struct DyeInhibitionParams {
    row0: vec4<f32>,
    row1: vec4<f32>,
    row2: vec4<f32>,
    d_max: vec3<f32>,
    positive_film: u32,
};

@group(0) @binding(0)
var input_tex: texture_2d<f32>;

@group(0) @binding(1)
var output_tex: texture_storage_2d<rgba16float, write>;

@group(0) @binding(2)
var density_lut: texture_2d<f32>;

@group(0) @binding(3)
var lut_sampler: sampler;

@group(0) @binding(4)
var<uniform> density_params: CurveParams;

@group(0) @binding(5)
var<uniform> adjusted_params: CurveParams;

@group(0) @binding(6)
var adjusted_lut: texture_2d<f32>;

@group(0) @binding(7)
var<uniform> dye_inhibition: DyeInhibitionParams;

fn safe_log10_vec3(v: vec3<f32>) -> vec3<f32> {
    let eps = 1e-6;
    return log2(max(v, vec3<f32>(eps))) / log2(10.0);
}

fn sample_density(normalized_pos: vec3<f32>) -> vec3<f32> {
    let r = textureSampleLevel(
        density_lut, lut_sampler, vec2<f32>(normalized_pos.r, 0.5), 0.0
    ).r;
    let g = textureSampleLevel(
        density_lut, lut_sampler, vec2<f32>(normalized_pos.g, 0.5), 0.0
    ).g;
    let b = textureSampleLevel(
        density_lut, lut_sampler, vec2<f32>(normalized_pos.b, 0.5), 0.0
    ).b;
    return vec3<f32>(r, g, b);
}

fn sample_adjusted_density(normalized_pos: vec3<f32>) -> vec3<f32> {
    let r = textureSampleLevel(
        adjusted_lut, lut_sampler, vec2<f32>(normalized_pos.r, 0.5), 0.0
    ).r;
    let g = textureSampleLevel(
        adjusted_lut, lut_sampler, vec2<f32>(normalized_pos.g, 0.5), 0.0
    ).g;
    let b = textureSampleLevel(
        adjusted_lut, lut_sampler, vec2<f32>(normalized_pos.b, 0.5), 0.0
    ).b;
    return vec3<f32>(r, g, b);
}

@compute
@workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(input_tex);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let tex_coords = vec2<i32>(i32(gid.x), i32(gid.y));
    let pixel = textureLoad(input_tex, tex_coords, 0);

    var working_pixel = pixel.rgb;
    if (density_params.apply_log != 0u) {
        working_pixel = safe_log10_vec3(working_pixel);
    }

    let density_pos = clamp(
        (working_pixel - density_params.xp_min) * density_params.inv_range,
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );
    let initial_density = sample_density(density_pos);

    var inhibition_source = max(initial_density, vec3<f32>(0.0));
    if (dye_inhibition.positive_film != 0u) {
        inhibition_source = max(dye_inhibition.d_max - initial_density, vec3<f32>(0.0));
    }

    let adjusted_exposure = working_pixel - vec3<f32>(
        dot(dye_inhibition.row0.xyz, inhibition_source),
        dot(dye_inhibition.row1.xyz, inhibition_source),
        dot(dye_inhibition.row2.xyz, inhibition_source)
    );
    let adjusted_pos = clamp(
        (adjusted_exposure - adjusted_params.xp_min) * adjusted_params.inv_range,
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );
    let output_color = sample_adjusted_density(adjusted_pos);

    textureStore(output_tex, tex_coords, vec4<f32>(output_color, pixel.a));
}
