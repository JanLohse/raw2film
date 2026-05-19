struct Params {
    xp_min: f32,
    xp_max: f32,
    channels: u32,
    _pad: u32,
}

@group(0) @binding(0)
var input_tex: texture_2d<f32>;

@group(0) @binding(1)
var lut_tex: texture_2d<f32>;

@group(0) @binding(2)
var out_tex: texture_storage_2d<rgba32float, write>;

@group(0) @binding(3)
var<uniform> params: Params;

fn safe_log10(x: f32) -> f32 {
    let v = max(x, 1e-16);
    return log(v) / log(10.0);
}


fn lut_interp(x: f32, ch: u32) -> f32 {

    let xmin = params.xp_min;
    let xmax = params.xp_max;

    let dims = textureDimensions(lut_tex);

    let size = dims.x;

    let xc = clamp(x, xmin, xmax);

    let pos =
        (xc - xmin) /
        (xmax - xmin) *
        f32(size - 1u);

    let idx = u32(floor(pos));

    let idx1 = min(idx + 1u, size - 1u);

    let f = pos - f32(idx);

    let y0 = textureLoad(
        lut_tex,
        vec2<i32>(i32(idx), i32(ch)),
        0
    ).r;

    let y1 = textureLoad(
        lut_tex,
        vec2<i32>(i32(idx1), i32(ch)),
        0
    ).r;

    return y0 + f * (y1 - y0);
}


@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id)
    gid: vec3<u32>
) {

    let dims = textureDimensions(out_tex);

    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let coord = vec2<i32>(gid.xy);

    let color = textureLoad(
        input_tex,
        coord,
        0
    );

    let r = lut_interp(safe_log10(color.r), 0u);
    let g = lut_interp(safe_log10(color.g), 1u);
    let b = lut_interp(safe_log10(color.b), 2u);

    textureStore(
        out_tex,
        coord,
        vec4<f32>(r, g, b, color.a)
    );
}
