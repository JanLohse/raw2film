const SCALE = 0.25;

@group(0) @binding(0)
var input_tex: texture_2d<f32>;

@group(0) @binding(1)
var output_tex: texture_storage_2d<rgba32float, write>;

@group(0) @binding(2)
var lut_tex: texture_3d<f32>;
@group(0) @binding(3)
var lut_sampler: sampler;

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

    let scaled_rgb = clamp(pixel.rgb * SCALE, vec3<f32>(0.0), vec3<f32>(1.0));

    let lut_size = f32(textureDimensions(lut_tex).x);
    let scale_factor = (lut_size - 1.0) / lut_size;
    let offset = 0.5 / lut_size;

    let uvw = scaled_rgb * scale_factor + offset;

    let sampled_color = textureSampleLevel(
        lut_tex,
        lut_sampler,
        vec3<f32>(uvw.b, uvw.g, uvw.r),
        0.0
    ).rgb;

    textureStore(
        output_tex,
        tex_coords,
        vec4<f32>(sampled_color, pixel.a)
    );
}
