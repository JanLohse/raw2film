const SCALE = 0.25;

@group(0) @binding(0)
var input_tex: texture_2d<f32>;

@group(0) @binding(1)
var output_tex: texture_storage_2d<rgba32float, write>;

@group(0) @binding(2)
var lut_tex: texture_3d<f32>;

fn lut_fetch(r: i32, g: i32, b: i32) -> vec3<f32> {
    let c = textureLoad(
        lut_tex,
        vec3<i32>(b, g, r),
        0
    );

    return c.rgb;
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

    let lut_size = textureDimensions(lut_tex).x;

    let pixel = textureLoad(
        input_tex,
        vec2<i32>(i32(x), i32(y)),
        0
    );

    let scale = f32(lut_size - 1u) * SCALE;

    let rf = pixel.r * scale;
    let gf = pixel.g * scale;
    let bf = pixel.b * scale;

    var r0 = i32(floor(rf));
    var g0 = i32(floor(gf));
    var b0 = i32(floor(bf));

    var dr: f32;
    var dg: f32;
    var db: f32;

    let max_idx = i32(lut_size) - 2;

    if (r0 >= max_idx + 1) {
        r0 = max_idx;
        dr = 1.0;
    } else {
        dr = rf - f32(r0);
    }

    if (g0 >= max_idx + 1) {
        g0 = max_idx;
        dg = 1.0;
    } else {
        dg = gf - f32(g0);
    }

    if (b0 >= max_idx + 1) {
        b0 = max_idx;
        db = 1.0;
    } else {
        db = bf - f32(b0);
    }

    let r1 = r0 + 1;
    let g1 = g0 + 1;
    let b1 = b0 + 1;

    let c000 = lut_fetch(r0, g0, b0);

    var c: vec3<f32>;

    if (dr >= dg) {

        if (dg >= db) {

            // dr >= dg >= db

            let c100 = lut_fetch(r1, g0, b0);
            let c110 = lut_fetch(r1, g1, b0);
            let c111 = lut_fetch(r1, g1, b1);

            c =
                c000 +
                dr * (c100 - c000) +
                dg * (c110 - c100) +
                db * (c111 - c110);

        } else if (dr >= db) {

            // dr >= db > dg

            let c100 = lut_fetch(r1, g0, b0);
            let c101 = lut_fetch(r1, g0, b1);
            let c111 = lut_fetch(r1, g1, b1);

            c =
                c000 +
                dr * (c100 - c000) +
                db * (c101 - c100) +
                dg * (c111 - c101);

        } else {

            // db > dr >= dg

            let c001 = lut_fetch(r0, g0, b1);
            let c101 = lut_fetch(r1, g0, b1);
            let c111 = lut_fetch(r1, g1, b1);

            c =
                c000 +
                db * (c001 - c000) +
                dr * (c101 - c001) +
                dg * (c111 - c101);
        }

    } else {

        if (db >= dg) {

            // db >= dg > dr

            let c001 = lut_fetch(r0, g0, b1);
            let c011 = lut_fetch(r0, g1, b1);
            let c111 = lut_fetch(r1, g1, b1);

            c =
                c000 +
                db * (c001 - c000) +
                dg * (c011 - c001) +
                dr * (c111 - c011);

        } else if (db >= dr) {

            // dg > db >= dr

            let c010 = lut_fetch(r0, g1, b0);
            let c011 = lut_fetch(r0, g1, b1);
            let c111 = lut_fetch(r1, g1, b1);

            c =
                c000 +
                dg * (c010 - c000) +
                db * (c011 - c010) +
                dr * (c111 - c011);

        } else {

            // dg > dr > db

            let c010 = lut_fetch(r0, g1, b0);
            let c110 = lut_fetch(r1, g1, b0);
            let c111 = lut_fetch(r1, g1, b1);

            c =
                c000 +
                dg * (c010 - c000) +
                dr * (c110 - c010) +
                db * (c111 - c110);
        }
    }

    textureStore(
        output_tex,
        vec2<i32>(i32(x), i32(y)),
        vec4<f32>(c, pixel.a)
    );
}
