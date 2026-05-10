struct Params {
    width: u32,
    height: u32,

    lut_size: u32,

    // padding/alignment
    _pad0: u32,
};

@group(0) @binding(0)
var input_img: texture_2d<f32>;

@group(0) @binding(1)
var lut_tex: texture_3d<f32>;

@group(0) @binding(2)
var output_img: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(3)
var<uniform> params: Params;

fn lut(
    r: u32,
    g: u32,
    b: u32,
) -> vec3<f32> {

    // numpy LUT[r,g,b]
    // texture coords are effectively [b,g,r]

    return textureLoad(
        lut_tex,
        vec3<i32>(
            i32(b),
            i32(g),
            i32(r)
        ),
        0
    ).rgb;
}

@compute @workgroup_size(8, 8)
fn main(
    @builtin(global_invocation_id)
    gid: vec3<u32>
) {
    // bounds check
    if (gid.x >= params.width ||
        gid.y >= params.height) {
        return;
    }

    let coord = vec2<i32>(
        i32(gid.x),
        i32(gid.y)
    );

    // input image is float32 in [0,1]
    let rgb = textureLoad(
        input_img,
        coord,
        0
    ).rgb;

    let size = params.lut_size;

    // ---- LUT coordinate space ----
    //
    // Example:
    // size = 33
    // scaled coords range in [0, 32]
    //
    let max_coord = f32(size - 1u);

    let rf = rgb.r * max_coord;
    let gf = rgb.g * max_coord;
    let bf = rgb.b * max_coord;

    // integer cube corner
    let r0 = u32(floor(rf));
    let g0 = u32(floor(gf));
    let b0 = u32(floor(bf));

    // upper corner
    let r1 = min(r0 + 1u, size - 1u);
    let g1 = min(g0 + 1u, size - 1u);
    let b1 = min(b0 + 1u, size - 1u);

    // local tetrahedral coords
    let dr = rf - f32(r0);
    let dg = gf - f32(g0);
    let db = bf - f32(b0);

    // base corner
    let c000 = lut(r0, g0, b0);

    var c: vec3<f32>;

    // tetrahedral interpolation
    //
    // exact same branch topology
    // as your CPU implementation
    //
    if (dr >= dg) {

        if (dg >= db) {

            // dr >= dg >= db

            let c100 = lut(r1, g0, b0);
            let c110 = lut(r1, g1, b0);
            let c111 = lut(r1, g1, b1);

            c =
                c000
                + dr * (c100 - c000)
                + dg * (c110 - c100)
                + db * (c111 - c110);

        } else if (dr >= db) {

            // dr >= db > dg

            let c100 = lut(r1, g0, b0);
            let c101 = lut(r1, g0, b1);
            let c111 = lut(r1, g1, b1);

            c =
                c000
                + dr * (c100 - c000)
                + db * (c101 - c100)
                + dg * (c111 - c101);

        } else {

            // db > dr >= dg

            let c001 = lut(r0, g0, b1);
            let c101 = lut(r1, g0, b1);
            let c111 = lut(r1, g1, b1);

            c =
                c000
                + db * (c001 - c000)
                + dr * (c101 - c001)
                + dg * (c111 - c101);
        }

    } else {

        if (db >= dg) {

            // db >= dg > dr

            let c001 = lut(r0, g0, b1);
            let c011 = lut(r0, g1, b1);
            let c111 = lut(r1, g1, b1);

            c =
                c000
                + db * (c001 - c000)
                + dg * (c011 - c001)
                + dr * (c111 - c011);

        } else if (db >= dr) {

            // dg > db >= dr

            let c010 = lut(r0, g1, b0);
            let c011 = lut(r0, g1, b1);
            let c111 = lut(r1, g1, b1);

            c =
                c000
                + dg * (c010 - c000)
                + db * (c011 - c010)
                + dr * (c111 - c011);

        } else {

            // dg > dr > db

            let c010 = lut(r0, g1, b0);
            let c110 = lut(r1, g1, b0);
            let c111 = lut(r1, g1, b1);

            c =
                c000
                + dg * (c010 - c000)
                + dr * (c110 - c010)
                + db * (c111 - c110);
        }
    }

    // numerical safety
    c = clamp(
        c,
        vec3<f32>(0.0),
        vec3<f32>(1.0)
    );

    textureStore(
        output_img,
        coord,
        vec4<f32>(c, 1.0)
    );
}
