@group(0) @binding(0)
var input_tex: texture_2d<f32>;

@group(0) @binding(1)
var output_tex: texture_storage_2d<rgba32float, write>;

@group(0) @binding(2)
var lut_tex: texture_2d<f32>;

fn lut_fetch(
    x: i32,
    y: i32
) -> vec3<f32> {

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

    var X = pixel.r;
    var Y = pixel.g;
    var Z = pixel.b;

    let S = X + Y + Z;

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

    X *= inv_sum;
    Y *= inv_sum;

    var xi = i32(floor(X));

    var yi = i32(floor(Y));

    xi = clamp(xi, 0, max_idx);
    yi = clamp(yi, 0, max_idx);

    let xf =
        fract(X);

    let yf =
        fract(Y);

    let factor_sum = xf + yf;

    var result: vec3<f32>;

    if (factor_sum <= 1.0) {

        let s_factor =
            1.0 - factor_sum;

        let r_val =
            lut_fetch(xi + 1, yi);

        let g_val =
            lut_fetch(xi, yi + 1);

        let s_val =
            lut_fetch(xi, yi);

        result =
            (
                r_val * xf +
                g_val * yf +
                s_val * s_factor
            ) * S;

    } else {

        let s_factor =
            factor_sum - 1.0;

        let xf2 = 1.0 - yf;

        let yf2 = 1.0 - xf;

        let r_val = lut_fetch(xi + 1, yi);

        let g_val = lut_fetch(xi, yi + 1);

        let s_val = lut_fetch(xi + 1, yi + 1);

        result =
            (
                r_val * xf2 +
                g_val * yf2 +
                s_val * s_factor
            ) * S;
    }

    textureStore(
        output_tex,
        vec2<i32>(i32(x), i32(y)),
        vec4<f32>(result, pixel.a)
    );
}
