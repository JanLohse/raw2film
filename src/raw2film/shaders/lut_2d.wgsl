struct Params {
    width: u32,
    height: u32,
    lutSize: u32,
};

@group(0) @binding(0)
var inputTex : texture_2d<f32>;

@group(0) @binding(1)
var lutTex : texture_2d<f32>;

@group(0) @binding(2)
var outputTex : texture_storage_2d<rgba32float, write>;

@group(0) @binding(3)
var<uniform> params : Params;


// Fetch LUT texel
fn lut_fetch(x: i32, y: i32) -> vec4<f32> {
    return textureLoad(lutTex, vec2<i32>(x, y), 0);
}


@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {

    let px = gid.x;
    let py = gid.y;

    if (px >= params.width || py >= params.height) {
        return;
    }

    // -------------------------------------------------------------------------
    // Load XYZ
    // -------------------------------------------------------------------------

    let xyz = textureLoad(inputTex, vec2<i32>(i32(px), i32(py)), 0);

    var X = xyz.r;
    var Y = xyz.g;
    var Z = xyz.b;

    let S = X + Y + Z;

    // Zero output for degenerate pixels
    if (S < 1e-12) {
        textureStore(
            outputTex,
            vec2<i32>(i32(px), i32(py)),
            vec4<f32>(0.0)
        );
        return;
    }

    // -------------------------------------------------------------------------
    // Chromaticity mapping
    // -------------------------------------------------------------------------

    let scaling = f32(params.lutSize - 1u);
    let invSum = scaling / S;

    let rx = X * invSum;
    let gy = Y * invSum;

    // Integer cell coordinates
    var rInd = i32(floor(rx));
    var gInd = i32(floor(gy));

    let maxIndex = i32(params.lutSize) - 2;

    rInd = clamp(rInd, 0, maxIndex);
    gInd = clamp(gInd, 0, maxIndex);

    // Fractional coordinates inside cell
    let rFactor = fract(rx);
    let gFactor = fract(gy);

    let factorSum = rFactor + gFactor;

    var outColor : vec4<f32>;

    // -------------------------------------------------------------------------
    // Lower-left triangle
    // -------------------------------------------------------------------------

    if (factorSum <= 1.0) {

        let sFactor = 1.0 - factorSum;

        let rVal = lut_fetch(rInd + 1, gInd);
        let gVal = lut_fetch(rInd, gInd + 1);
        let sVal = lut_fetch(rInd, gInd);

        outColor =
            rVal * rFactor +
            gVal * gFactor +
            sVal * sFactor;

    }

    // -------------------------------------------------------------------------
    // Upper-right triangle
    // -------------------------------------------------------------------------

    else {

        let sFactor  = factorSum - 1.0;
        let rFactor2 = 1.0 - gFactor;
        let gFactor2 = 1.0 - rFactor;

        let rVal = lut_fetch(rInd + 1, gInd);
        let gVal = lut_fetch(rInd, gInd + 1);
        let sVal = lut_fetch(rInd + 1, gInd + 1);

        outColor =
            rVal * rFactor2 +
            gVal * gFactor2 +
            sVal * sFactor;
    }

    // Rescale by tristimulus sum
    outColor *= S;

    textureStore(
        outputTex,
        vec2<i32>(i32(px), i32(py)),
        outColor
    );
}
