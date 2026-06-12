@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba8unorm, write>;

struct Uniforms {
    scale_x: f32,
    scale_y: f32,
    offset_x: f32,
    offset_y: f32,
}
@group(0) @binding(3) var<uniform> u_transforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dst_size = textureDimensions(dst_tex);

    // Ensure we don't write out of the destination bounds
    if (id.x >= dst_size.x || id.y >= dst_size.y) {
        return;
    }

    // FIX: Add 0.5 to sample from the center of the destination pixel
    let dst_coords = vec2<f32>(f32(id.x) + 0.5, f32(id.y) + 0.5);

    // Map destination pixel back to normalized source UV coordinates (0.0 to 1.0)
    let src_uv = vec2<f32>(
        (dst_coords.x - u_transforms.offset_x) * u_transforms.scale_x,
        (dst_coords.y - u_transforms.offset_y) * u_transforms.scale_y
    );

    var color: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // Only sample if the mapped UV actually falls inside the source image bounds
    if (src_uv.x >= 0.0 && src_uv.x <= 1.0 && src_uv.y >= 0.0 && src_uv.y <= 1.0) {
        color = textureSampleLevel(src_tex, src_sampler, src_uv, 0.0);
    }

    let write_coords = vec2<i32>(i32(id.x), i32(id.y));
    textureStore(dst_tex, write_coords, color);
}
