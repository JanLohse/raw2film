struct Kernel {
    weights: array<vec4<f32>>,
};

@group(0) @binding(0)
var src : texture_2d<f32>;

@group(0) @binding(1)
var dst : texture_storage_2d<rgba32float, write>;

@group(0) @binding(2)
var<storage, read> kernel : Kernel;

@group(0) @binding(3)
var<uniform> kernel_size : vec2<u32>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid : vec3<u32>
) {
    let image_size = vec2<i32>(textureDimensions(src));
    let pixel_coords = vec2<i32>(gid.xy);

    // Early exit if out of bounds
    if (pixel_coords.x >= image_size.x || pixel_coords.y >= image_size.y) {
        return;
    }

    let k_size = vec2<i32>(kernel_size);
    let k_center = k_size / 2;

    // Calculate the top-left starting coordinate in the source image for this pixel
    let start_coord = pixel_coords - k_center;

    var sum = vec4<f32>(0.0);
    var weight_idx = 0u;

    // Pre-calculate image limits for clamping
    let max_coord = image_size - vec2<i32>(1);

    for (var ky: i32 = 0; ky < k_size.y; ky++) {
        // Pre-clamp Y coordinate for the entire row iteration
        let src_y = clamp(start_coord.y + ky, 0, max_coord.y);

        for (var kx: i32 = 0; kx < k_size.x; kx++) {
            // Clamp X coordinate
            let src_x = clamp(start_coord.x + kx, 0, max_coord.x);

            let pixel = textureLoad(src, vec2<i32>(src_x, src_y), 0);
            let weight = kernel.weights[weight_idx];

            sum += pixel * weight;
            weight_idx++;
        }
    }

    textureStore(dst, pixel_coords, sum);
}
