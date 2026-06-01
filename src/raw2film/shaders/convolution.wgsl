@group(0) @binding(0)
var src : texture_2d<f32>;

@group(0) @binding(1)
var dst : texture_storage_2d<rgba32float, write>;

@group(0) @binding(2)
var kernel : texture_2d<f32>;

fn clamp_coord(coord: vec2<i32>, image_width: i32, image_height: i32) -> vec2<i32> {
    return vec2<i32>(
        clamp(coord.x, 0, image_width - 1),
        clamp(coord.y, 0, image_height - 1)
    );
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid : vec3<u32>
) {
    let x = gid.x;
    let y = gid.y;

    let image_size = textureDimensions(src);
    let image_width = image_size.x;
    let image_height = image_size.y;

    let kernel_size = textureDimensions(kernel);
    let kernel_width = kernel_size.x;
    let kernel_height = kernel_size.y;

    if (x >= image_width || y >= image_height) {
        return;
    }

    let kernel_cx = i32(kernel_width) / 2;
    let kernel_cy = i32(kernel_height) / 2;

    var sum = vec4<f32>(0.0);

    for (var ky: u32 = 0u; ky < kernel_height; ky++) {
        for (var kx: u32 = 0u; kx < kernel_width; kx++) {

            let dx = i32(kx) - kernel_cx;
            let dy = i32(ky) - kernel_cy;

            let src_coord = clamp_coord(
                vec2<i32>(
                    i32(x) + dx,
                    i32(y) + dy
                ),
                i32(image_width),
                i32(image_height),
            );

            let pixel = textureLoad(
                src,
                src_coord,
                0
            );

            let weight = textureLoad(
                kernel,
                vec2<i32>(i32(kx), i32(ky)),
                0
            );

            sum += pixel * weight;
        }
    }

    textureStore(
        dst,
        vec2<i32>(i32(x), i32(y)),
        sum
    );
}
