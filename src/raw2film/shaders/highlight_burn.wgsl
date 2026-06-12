struct Params {
    highlight_burn: f32,
    d_ref: f32,
};

// ==========================================
// PASS 1: DOWNSAMPLE & FILTRATION
// ==========================================
@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var smp_linear: sampler;
@group(0) @binding(2) var lowres_mask_write: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params_p1: Params;

@compute @workgroup_size(16, 16)
fn downsample_func(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(lowres_mask_write);
    if (id.x >= size.x || id.y >= size.y) { return; }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let color = textureSampleLevel(input_tex, smp_linear, uv, 0.0);

    let func_val = max(color.g - params_p1.d_ref, 0.0);

    textureStore(lowres_mask_write, id.xy, vec4<f32>(func_val, 0.0, 0.0, 1.0));
}

// ==========================================
// PASS 2: PRECISION SCIPY BLUR & COMPOSITION
// ==========================================
@group(0) @binding(0) var orig_highres_tex: texture_2d<f32>;
@group(0) @binding(1) var lowres_mask_read: texture_2d<f32>;
@group(0) @binding(2) var smp_bilinear: sampler;
@group(0) @binding(3) var output_highres_write: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params_p2: Params;

@compute @workgroup_size(16, 16)
fn final_burn(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output_highres_write);
    if (id.x >= size.x || id.y >= size.y) { return; }

    let uv = (vec2<f32>(id.xy) + 0.5) / vec2<f32>(size);
    let tex_el_size = 1.0 / vec2<f32>(textureDimensions(lowres_mask_read));

    // Exact 1D SciPy Gaussian weights for sigma=3, truncate=2
    let weights = array<f32, 13>(
        0.01854402, 0.03416694, 0.05633176, 0.08310854, 0.10971929, 0.12961803,
        0.13702282, 0.12961803, 0.10971929, 0.08310854, 0.05633176, 0.03416694,
        0.01854402
    );

    var blur_accum = 0.0;

    // Outer-product 2D loop matching the radius of 6
    for (var dy: i32 = -6; dy <= 6; dy++) {
        for (var dx: i32 = -6; dx <= 6; dx++) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * tex_el_size;
            let sample_val = textureSampleLevel(lowres_mask_read, smp_bilinear, uv + offset, 0.0).r;

            // Map [-6, 6] loop indices to [0, 12] array indices
            let weight = weights[dx + 6] * weights[dy + 6];
            blur_accum += sample_val * weight;
        }
    }

    // Fetch original pixel and apply the burn
    let orig_color = textureLoad(orig_highres_tex, id.xy, 0);
    var final_rgb = orig_color.rgb - (params_p2.highlight_burn * blur_accum);
    final_rgb = max(final_rgb, vec3<f32>(0.0));

    textureStore(output_highres_write, id.xy, vec4<f32>(final_rgb, orig_color.a));
}
