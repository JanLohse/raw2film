@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var dst_tex: texture_storage_2d<rgba8unorm, write>;

struct Uniforms {
    offset_x: i32,
    offset_y: i32,
}
@group(0) @binding(2) var<uniform> u_offsets: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let src_size = textureDimensions(src_tex);

    // Check bounds using u32 to keep it clean
    if (id.x >= src_size.x || id.y >= src_size.y) {
        return;
    }

    // Explicitly cast components to i32 for textureLoad
    let src_coords = vec2<i32>(i32(id.x), i32(id.y));
    let color: vec4<f32> = textureLoad(src_tex, src_coords, 0);

    // Calculate where it belongs in the destination texture
    let dst_coords = vec2<i32>(
        i32(id.x) + u_offsets.offset_x,
        i32(id.y) + u_offsets.offset_y
    );

    textureStore(dst_tex, dst_coords, color);
}
