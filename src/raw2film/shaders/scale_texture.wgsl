@group(0) @binding(0) var source_texture: texture_2d<u32>;
@group(0) @binding(1) var target_texture: texture_storage_2d<rgba8unorm, write>;

struct ScaleParams {
    target_width: u32,
    target_height: u32,
}
@group(0) @binding(2) var<uniform> params: ScaleParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.target_width || id.y >= params.target_height) {
        return;
    }

    // Calculate normalized texture coordinates [0.0, 1.0] for the target canvas
    let uv = vec2<f32>(id.xy) / vec2<f32>(f32(params.target_width), f32(params.target_height));

    // Map normalized coordinates back to the source histogram texture space (256 x height)
    let src_size = vec2<f32>(textureDimensions(source_texture));
    let src_coords = vec2<i32>(uv * src_size);

    // Read the uint color from the fixed histogram texture
    let hist_color = textureLoad(source_texture, src_coords, 0);

    // Convert the color from uint8 [0, 255] to unorm float [0.0, 1.0] for the canvas
    let final_color = vec4<f32>(hist_color) / 255.0;

    // Write to the UI target canvas texture
    textureStore(target_texture, id.xy, final_color);
}
