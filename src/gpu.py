from pathlib import Path

import numpy as np
import wgpu
from wgpu.utils import get_default_device

# ------------------------------------------------------------
# GPU LUT APPLY
# ------------------------------------------------------------


def apply_lut_gpu(
    image: np.ndarray,
    lut: np.ndarray,
) -> np.ndarray:

    device = get_default_device()

    h, w, _ = image.shape

    image_tex = image_to_gpu(device, image)
    lut_tex = lut_to_gpu(device, lut)

    out_tex = apply_lut_shader(
        device=device,
        image_tex=image_tex,
        lut_tex=lut_tex,
        width=w,
        height=h,
        lut_size=lut.shape[0],
    )

    return texture_to_numpy(
        device=device,
        texture=out_tex,
        width=w,
        height=h,
    )


# ------------------------------------------------------------
# IMAGE -> GPU
# ------------------------------------------------------------


def image_to_gpu(
    device,
    image: np.ndarray,
):

    queue = device.queue

    h, w, _ = image.shape

    # RGBA32F
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :3] = image.astype(np.float32)

    tex = device.create_texture(
        size=(w, h, 1),
        dimension="2d",
        format="rgba32float",
        usage=(wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING),
    )

    queue.write_texture(
        {
            "texture": tex,
            "origin": (0, 0, 0),
        },
        rgba,
        {
            "bytes_per_row": w * 4 * 4,
            "rows_per_image": h,
        },
        (w, h, 1),
    )

    return tex


# ------------------------------------------------------------
# LUT -> GPU
# ------------------------------------------------------------


def lut_to_gpu(
    device,
    lut: np.ndarray,
):

    queue = device.queue

    size = lut.shape[0]

    # uint8 -> normalized float texture
    rgba = np.ones((size, size, size, 4), dtype=np.uint8) * 255
    rgba[..., :3] = lut

    tex = device.create_texture(
        size=(size, size, size),
        dimension="3d",
        format="rgba8unorm",
        usage=(wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING),
    )

    queue.write_texture(
        {
            "texture": tex,
            "origin": (0, 0, 0),
        },
        rgba,
        {
            "bytes_per_row": size * 4,
            "rows_per_image": size,
        },
        (size, size, size),
    )

    return tex


# ------------------------------------------------------------
# SHADER EXECUTION
# ------------------------------------------------------------


def apply_lut_shader(
    device,
    image_tex,
    lut_tex,
    width: int,
    height: int,
    lut_size: int,
):

    queue = device.queue

    # --------------------------------------------------------
    # output texture
    # --------------------------------------------------------

    out_tex = device.create_texture(
        size=(width, height, 1),
        dimension="2d",
        format="rgba8unorm",
        usage=(wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC),
    )

    # --------------------------------------------------------
    # uniforms
    # --------------------------------------------------------

    params_dtype = np.dtype(
        [
            ("width", np.uint32),
            ("height", np.uint32),
            ("lut_size", np.uint32),
            ("_pad0", np.uint32),
        ]
    )

    params = np.array(
        [
            (
                width,
                height,
                lut_size,
                0,
            )
        ],
        dtype=params_dtype,
    )

    param_buffer = device.create_buffer_with_data(
        data=params.tobytes(),
        usage=wgpu.BufferUsage.UNIFORM,
    )

    # --------------------------------------------------------
    # shader
    # --------------------------------------------------------

    shader_path = Path(__file__).parent / "raw2film/shaders/lut_tetrahedral.wgsl"

    shader_code = shader_path.read_text()

    shader = device.create_shader_module(code=shader_code)

    # --------------------------------------------------------
    # bind group layout
    # --------------------------------------------------------

    bind_group_layout = device.create_bind_group_layout(
        entries=[
            {
                "binding": 0,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "texture": {
                    "sample_type": "unfilterable-float",
                    "view_dimension": "2d",
                },
            },
            {
                "binding": 1,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "texture": {
                    "sample_type": "float",
                    "view_dimension": "3d",
                },
            },
            {
                "binding": 2,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "storage_texture": {
                    "access": "write-only",
                    "format": "rgba8unorm",
                    "view_dimension": "2d",
                },
            },
            {
                "binding": 3,
                "visibility": wgpu.ShaderStage.COMPUTE,
                "buffer": {
                    "type": "uniform",
                },
            },
        ]
    )

    bind_group = device.create_bind_group(
        layout=bind_group_layout,
        entries=[
            {
                "binding": 0,
                "resource": image_tex.create_view(),
            },
            {
                "binding": 1,
                "resource": lut_tex.create_view(),
            },
            {
                "binding": 2,
                "resource": out_tex.create_view(),
            },
            {
                "binding": 3,
                "resource": {
                    "buffer": param_buffer,
                    "offset": 0,
                    "size": params.nbytes,
                },
            },
        ],
    )

    # --------------------------------------------------------
    # pipeline
    # --------------------------------------------------------

    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )

    pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={
            "module": shader,
            "entry_point": "main",
        },
    )

    # --------------------------------------------------------
    # dispatch
    # --------------------------------------------------------

    wg_x = (width + 7) // 8
    wg_y = (height + 7) // 8

    encoder = device.create_command_encoder()

    compute_pass = encoder.begin_compute_pass()

    compute_pass.set_pipeline(pipeline)
    compute_pass.set_bind_group(0, bind_group)

    compute_pass.dispatch_workgroups(
        wg_x,
        wg_y,
        1,
    )

    compute_pass.end()

    queue.submit([encoder.finish()])

    return out_tex


# ------------------------------------------------------------
# GPU -> NUMPY
# ------------------------------------------------------------


def texture_to_numpy(
    device,
    texture,
    width: int,
    height: int,
):

    queue = device.queue

    # WebGPU requires 256-byte alignment
    bytes_per_row = ((width * 4 + 255) // 256) * 256

    buffer_size = bytes_per_row * height

    read_buffer = device.create_buffer(
        size=buffer_size,
        usage=(wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ),
    )

    encoder = device.create_command_encoder()

    encoder.copy_texture_to_buffer(
        {
            "texture": texture,
        },
        {
            "buffer": read_buffer,
            "bytes_per_row": bytes_per_row,
            "rows_per_image": height,
        },
        (width, height, 1),
    )

    queue.submit([encoder.finish()])

    read_buffer.map_sync(
        wgpu.MapMode.READ,
        0,
        buffer_size,
    )

    mapped = read_buffer.read_mapped()

    raw = np.frombuffer(
        mapped,
        dtype=np.uint8,
    ).reshape(height, bytes_per_row)

    # remove row padding
    raw = raw[:, : width * 4]

    image = raw.reshape(height, width, 4)

    read_buffer.unmap()

    return image[..., :3].copy()
