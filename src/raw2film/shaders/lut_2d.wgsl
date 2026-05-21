@group(0) @binding(0)
var input_tex: texture_2d<f32>;

@group(0) @binding(1)
var output_tex: texture_storage_2d<rgba32float, write>;

@group(0) @binding(2)
var lut_tex: texture_2d<f32>;

fn lut_fetch(x: i32, y: i32) -> vec3<f32> {
    return textureLoad(
        lut_tex,
        vec2<i32>(x, y),
        0
    ).rgb;
}

@compute
@workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id)
    gid: vec3<u32>
) {

    let x = gid.x;
    let y = gid.y;

    let dims = textureDimensions(input_tex);
    if (x >= dims.x || y >= dims.y) {
        return;
    }

    let pixel = textureLoad(
        input_tex,
        vec2<i32>(i32(x), i32(y)),
        0
    );

    var r = pixel.r;
    var g = pixel.g;
    var b = pixel.b;

    let S = r + g + b;

    if (S < 1e-12) {

        textureStore(
            output_tex,
            vec2<i32>(i32(x), i32(y)),
            vec4<f32>(0.0, 0.0, 0.0, pixel.a)
        );

        return;
    }

    let lut_size = textureDimensions(lut_tex).x;

    let scaling = f32(lut_size - 1u);
    let max_idx = i32(lut_size) - 2;

    let inv_sum = scaling / S;

    r *= inv_sum;
    g *= inv_sum;

    var r_ind = i32(floor(r));
    var g_ind = i32(floor(g));

    r_ind = clamp(r_ind, 0, max_idx);
    g_ind = clamp(g_ind, 0, max_idx);

    let r_factor = fract(r);
    let g_factor = fract(g);

    let factor_sum = r_factor + g_factor;

    var result: vec3<f32>;

    if (factor_sum <= 1.0) {

        let s_factor = 1.0 - factor_sum;

        let r_val = lut_fetch(r_ind + 1, g_ind);
        let g_val = lut_fetch(r_ind, g_ind + 1);
        let s_val = lut_fetch(r_ind, g_ind);

        result = (r_val * r_factor + g_val * g_factor + s_val * s_factor) * S;

    } else {

        let s_factor = factor_sum - 1.0;

        let r_factor2 = 1.0 - g_factor;
        let g_factor2 = 1.0 - r_factor;

        let r_val = lut_fetch(r_ind + 1, g_ind);
        let g_val = lut_fetch(r_ind, g_ind + 1);
        let s_val = lut_fetch(r_ind + 1, g_ind + 1);

        result = (r_val * r_factor2 + g_val * g_factor2 + s_val * s_factor) * S;
    }

    textureStore(
        output_tex,
        vec2<i32>(i32(x), i32(y)),
        vec4<f32>(result, pixel.a)
    );
}
