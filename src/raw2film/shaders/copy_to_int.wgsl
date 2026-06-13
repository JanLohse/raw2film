@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba8unorm, write>;

struct Uniforms {
    scale_x: f32,
    scale_y: f32,
    offset_x: f32,
    offset_y: f32,
    canvas_min_x: f32,
    canvas_min_y: f32,
    canvas_max_x: f32,
    canvas_max_y: f32,
    canvas_color: vec3<f32>, // Automatically maps to the 3 color floats + 1 padding float
}
@group(0) @binding(3) var<uniform> u_transforms: Uniforms;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dst_size = textureDimensions(dst_tex);

    // Ensure we don't write out of the destination bounds
    if (id.x >= dst_size.x || id.y >= dst_size.y) {
        return;
    }

    let dst_coords = vec2<f32>(f32(id.x) + 0.5, f32(id.y) + 0.5);

    // Map destination pixel back to normalized source UV coordinates (0.0 to 1.0)
    let src_uv = vec2<f32>(
        (dst_coords.x - u_transforms.offset_x) * u_transforms.scale_x,
        (dst_coords.y - u_transforms.offset_y) * u_transforms.scale_y
    );

    var color: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    // 1. Check if destination coordinate is within the scaled inner image bounds
    if (src_uv.x >= 0.0 && src_uv.x <= 1.0 && src_uv.y >= 0.0 && src_uv.y <= 1.0) {
        color = textureSampleLevel(src_tex, src_sampler, src_uv, 0.0);
    }
    // 2. Check if we missed the image, but are still inside the canvas bounds area
    else if (dst_coords.x >= u_transforms.canvas_min_x && dst_coords.x <= u_transforms.canvas_max_x &&
             dst_coords.y >= u_transforms.canvas_min_y && dst_coords.y <= u_transforms.canvas_max_y) {
        // Fill canvas with custom color and fully opaque alpha channel (1.0)
        color = vec4<f32>(u_transforms.canvas_color, 1.0);
    }
    // 3. Otherwise, color defaults to transparent vec4(0.0) for everything else

    let write_coords = vec2<i32>(i32(id.x), i32(id.y));
    textureStore(dst_tex, write_coords, color);
}
