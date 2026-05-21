struct Params {
    xp_min: f32,
    xp_max: f32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0)
var input_tex: texture_2d<f32>;

@group(0) @binding(1)
var output_tex: texture_storage_2d<rgba32float, write>;

@group(0) @binding(2)
var lut_tex: texture_2d<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

fn interp_channel(
    value: f32,
    channel: u32
) -> f32 {

    let xp_min = params.xp_min;
    let xp_max = params.xp_max;
    let lut_size = textureDimensions(lut_tex).x;

    if (value <= xp_min) {
        return textureLoad(
            lut_tex,
            vec2<i32>(0, 0),
            0
        )[channel];
    }

    if (value >= xp_max) {
        return textureLoad(
            lut_tex,
            vec2<i32>(i32(lut_size - 1u), 0),
            0
        )[channel];
    }

    let bin_width =
        (xp_max - xp_min) /
        f32(lut_size - 1u);

    let pos =
        (value - xp_min) /
        bin_width;

    let idx = u32(floor(pos));

    let f =  pos - f32(idx);

    let y0 = textureLoad(
        lut_tex,
        vec2<i32>(i32(idx), 0),
        0
    )[channel];

    let y1 = textureLoad(
        lut_tex,
        vec2<i32>(i32(idx + 1u), 0),
        0
    )[channel];

    return y0 + f * (y1 - y0);
}

fn safe_log10(x: f32) -> f32 {
    // avoid -inf / NaN
    let eps = 1e-6;
    return log2(max(x, eps)) / log2(10.0);
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

    let log_pixel = vec3<f32>(
        safe_log10(pixel.r),
        safe_log10(pixel.g),
        safe_log10(pixel.b)
    );

    let out_color = vec4<f32>(
        interp_channel(log_pixel.r, 0u),
        interp_channel(log_pixel.g, 1u),
        interp_channel(log_pixel.b, 2u),
        pixel.a
    );

    textureStore(
        output_tex,
        vec2<i32>(i32(x), i32(y)),
        out_color
    );
}
