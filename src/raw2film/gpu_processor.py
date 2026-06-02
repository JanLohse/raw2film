import struct
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import wgpu
from PIL import Image, ImageCms
from spectral_film_lut.color_space import GAMMA_KEYS
from spectral_film_lut.config import DEFAULT_DTYPE
from spectral_film_lut.film_spectral import FilmSpectral
from spectral_film_lut.utils import create_lut
from wgpu import TextureUsage, get_default_device

from raw2film import effects
from raw2film.effects import (
    compute_halation_kernel,
    mtf_kernel,
)
from raw2film.raw_conversion import CANVAS_MODES, crop_rotate_zoom, raw_to_linear
from raw2film.utils import (
    load_metadata,
    resolution_scaling,
)

# TODO: fix BW for convolution effects


class GpuTexture:
    def __init__(self, device, size, format=wgpu.TextureFormat.rgba32float):
        self.device = device
        self.size = size

        self.texture = device.create_texture(
            size={"width": size[0], "height": size[1], "depth_or_array_layers": 1},
            format=format,
            usage=(
                TextureUsage.TEXTURE_BINDING
                | TextureUsage.COPY_DST
                | TextureUsage.RENDER_ATTACHMENT
                | TextureUsage.STORAGE_BINDING
                | TextureUsage.COPY_SRC
            ),
        )

        self.view = self.texture.create_view()

    def upload(self, queue, data: np.ndarray):
        """Upload RGBA float texture."""
        data = np.ascontiguousarray(data)
        queue.write_texture(
            {"texture": self.texture},
            data,
            {"bytes_per_row": data.strides[0], "rows_per_image": data.shape[0]},
            {
                "width": data.shape[1],
                "height": data.shape[0],
                "depth_or_array_layers": 1,
            },
        )


class GpuProcessor:
    def __init__(self, cameras, lenses):
        self.device = get_default_device()
        self.queue = self.device.queue

        # Load shaders
        lut_1d_shader_path = Path(__file__).parent / "shaders/lut_1d.wgsl"
        lut_1d_shader_code = lut_1d_shader_path.read_text()
        lut_1d_shader = self.device.create_shader_module(code=lut_1d_shader_code)

        lut_2d_shader_path = Path(__file__).parent / "shaders/lut_2d.wgsl"
        lut_2d_shader_code = lut_2d_shader_path.read_text()
        lut_2d_shader = self.device.create_shader_module(code=lut_2d_shader_code)

        lut_3d_shader_path = Path(__file__).parent / "shaders/lut_3d.wgsl"
        lut_3d_shader_code = lut_3d_shader_path.read_text()
        lut_3d_shader = self.device.create_shader_module(code=lut_3d_shader_code)

        copy_to_int_shader_path = Path(__file__).parent / "shaders/copy_to_int.wgsl"
        copy_to_int_shader_code = copy_to_int_shader_path.read_text()
        copy_to_int_shader = self.device.create_shader_module(
            code=copy_to_int_shader_code
        )

        convolution_shader_path = Path(__file__).parent / "shaders/convolution.wgsl"
        convolution_shader_code = convolution_shader_path.read_text()
        convolution_shader = self.device.create_shader_module(
            code=convolution_shader_code
        )

        # create pipelines
        self.pipeline_lut_1d = self.device.create_compute_pipeline(
            layout="auto", compute={"module": lut_1d_shader, "entry_point": "main"}
        )
        self.pipeline_lut_2d = self.device.create_compute_pipeline(
            layout="auto", compute={"module": lut_2d_shader, "entry_point": "main"}
        )
        self.pipeline_lut_3d = self.device.create_compute_pipeline(
            layout="auto", compute={"module": lut_3d_shader, "entry_point": "main"}
        )
        self.pipeline_copy_to_int = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": copy_to_int_shader, "entry_point": "main"},
        )
        self.pipeline_convolution = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": convolution_shader, "entry_point": "main"},
        )

        # Init gpu textures
        self.tex_input = None
        self.tex_a = None
        self.tex_b = None
        self.tex_int_out = None

        self.tex_lut_1d = None
        self.tex_lut_2d = None
        self.tex_lut_3d = None

        self.buffer_params_lut_1d = None
        self.buffer_mtf_kernel = None
        self.buffer_mtf_kernel_size = None
        self.buffer_halation_kernel = None
        self.buffer_halation_kernel_size = None

        # Comparison dicts
        self.image_param_dict = None
        self.input_param_dict = None
        self.curve_param_dict = None
        self.output_param_dict = None
        self.mtf_param_dict = None
        self.halation_param_dict = None

        self.width = None
        self.height = None

        self.lut_sampler = self.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
        )

        # Init lens correction data
        self.cameras = cameras
        self.lenses = lenses

    @lru_cache
    def load_raw_image_cached(self, src, cam=None, lens=None, half_size=True):
        return self.load_raw_image(src, cam, lens, half_size)

    def load_raw_image(self, src, cam=None, lens=None, half_size=True):
        image = raw_to_linear(src, half_size=half_size)

        if cam is not None and lens is not None:
            cam = self.cameras[cam]
            lens = self.lenses[lens]

            image = effects.lens_correction(image, load_metadata(src), cam, lens)

        image = image.astype(DEFAULT_DTYPE)

        return image

    def _ensure_image_texture(self, image: np.ndarray):
        h, w = image.shape[:2]

        if self.tex_input is None or self.tex_input.size != (w, h):
            self.tex_input = GpuTexture(self.device, (w, h))
            self.tex_a = GpuTexture(self.device, (w, h))
            self.tex_b = GpuTexture(self.device, (w, h))
            self.tex_int_out = GpuTexture(
                self.device, (w, h), format=wgpu.TextureFormat.rgba8unorm
            )

        self.tex_input.upload(self.queue, image)

    def _ensure_lut_1d(self, lut: np.ndarray):
        size = lut.shape[1]
        if self.tex_lut_1d is None or size != self.tex_lut_1d.size[0]:
            self.tex_lut_1d = self.device.create_texture(
                size=(size, 1, 1),
                format=wgpu.TextureFormat.rgba16float,
                usage=(wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST),
            )
            self.tex_lut_1d_view = self.tex_lut_1d.create_view()

            self.lut_sampler = self.device.create_sampler(
                mag_filter=wgpu.FilterMode.linear,
                min_filter=wgpu.FilterMode.linear,
                address_mode_u=wgpu.AddressMode.clamp_to_edge,
                address_mode_v=wgpu.AddressMode.clamp_to_edge,
            )

        lut_rgba = np.ones((size, 4), dtype=np.float32)
        lut_rgba[..., 0:3] = lut[1:, ...].T

        lut_rgba_16 = lut_rgba.astype(np.float16)

        xp_min = lut[0, 0]
        xp_max = lut[0, -1]
        denom = xp_max - xp_min
        inv_range = 1.0 / denom if denom != 0.0 else 0.0

        params_lut_1d = struct.pack("ffff", xp_min, xp_max, inv_range, 0.0)

        self.buffer_params_lut_1d = self.device.create_buffer_with_data(
            data=params_lut_1d,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self.queue.write_texture(
            {"texture": self.tex_lut_1d},
            lut_rgba_16,
            {
                "bytes_per_row": size * 4 * 2,
                "rows_per_image": 1,
            },
            {
                "width": size,
                "height": 1,
                "depth_or_array_layers": 1,
            },
        )

    def _ensure_lut_2d(self, lut: np.ndarray):
        size = lut.shape[0]
        if self.tex_lut_2d is None or size != self.tex_lut_2d.size[0]:
            self.tex_lut_2d = self.device.create_texture(
                size=(size, size, 1),
                format=wgpu.TextureFormat.rgba32float,
                usage=(wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST),
            )
            self.tex_lut_2d_view = self.tex_lut_2d.create_view()

        lut_rgba = np.ones((size, size, 4), dtype=np.float32)

        lut_rgba[..., 0:3] = lut

        self.queue.write_texture(
            {"texture": self.tex_lut_2d},
            lut_rgba,
            {
                "bytes_per_row": size * 4 * 4,
                "rows_per_image": size,
            },
            {
                "width": size,
                "height": size,
                "depth_or_array_layers": 1,
            },
        )

    def _ensure_lut_3d(self, lut: np.ndarray):
        size = lut.shape[0]
        if self.tex_lut_3d is None or size != self.tex_lut_3d.size[0]:
            self.tex_lut_3d = self.device.create_texture(
                size=(size, size, size),
                dimension=wgpu.TextureDimension.d3,
                format=wgpu.TextureFormat.rgba32float,
                usage=(wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST),
            )
            self.tex_lut_3d_view = self.tex_lut_3d.create_view()

        lut_rgba = np.ones((size, size, size, 4), dtype=np.float32)
        lut_rgba[..., 0:3] = lut

        self.queue.write_texture(
            {
                "texture": self.tex_lut_3d,
            },
            lut_rgba,
            {
                "bytes_per_row": size * 4 * 4,
                "rows_per_image": size,
            },
            {
                "width": size,
                "height": size,
                "depth_or_array_layers": size,
            },
        )

    def _ensure_mtf_kernel(self, kernel: np.ndarray):
        h, w = kernel.shape[:2]

        kernel_rgba = np.ones((h, w, 4), dtype=np.float32)

        if kernel.ndim == 2:
            kernel_rgba[..., :3] = kernel[..., None]
        else:
            kernel_rgba[..., :3] = kernel

        flat = kernel_rgba.reshape(-1, 4)

        if self.buffer_mtf_kernel is None or self.buffer_mtf_kernel.size < flat.nbytes:
            self.buffer_mtf_kernel = self.device.create_buffer(
                size=flat.nbytes,
                usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
            )

        self.queue.write_buffer(
            self.buffer_mtf_kernel,
            0,
            flat.tobytes(),
        )

        self.kernel_size_data = np.array(
            [h, w],
            dtype=np.uint32,
        )

        if self.buffer_mtf_kernel_size is None:
            self.buffer_mtf_kernel_size = self.device.create_buffer(
                size=16,  # uniform alignment
                usage=(wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST),
            )

        self.queue.write_buffer(
            self.buffer_mtf_kernel_size,
            0,
            self.kernel_size_data.tobytes(),
        )

    def _ensure_halation_kernel(self, kernel: np.ndarray):
        h, w = kernel.shape[:2]

        kernel_rgba = np.ones((h, w, 4), dtype=np.float32)

        if kernel.ndim == 2:
            kernel_rgba[..., :3] = kernel[..., None]
        elif kernel.ndim == 3 and kernel.shape[2] == 3:
            kernel_rgba[..., :3] = kernel
        elif kernel.ndim == 3 and kernel.shape[2] == 4:
            kernel_rgba = kernel.astype(np.float32, copy=False)
        else:
            raise ValueError(f"Unexpected kernel shape {kernel.shape}")

        flat = kernel_rgba.reshape(-1, 4)

        if (
            self.buffer_halation_kernel is None
            or self.buffer_halation_kernel.size < flat.nbytes
        ):
            self.buffer_halation_kernel = self.device.create_buffer(
                size=flat.nbytes,
                usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
            )

        self.queue.write_buffer(
            self.buffer_halation_kernel,
            0,
            flat.tobytes(),
        )

        self.kernel_size_data = np.array(
            [h, w],
            dtype=np.uint32,
        )

        if self.buffer_halation_kernel_size is None:
            self.buffer_halation_kernel_size = self.device.create_buffer(
                size=16,  # uniform alignment
                usage=(wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST),
            )

        self.queue.write_buffer(
            self.buffer_halation_kernel_size,
            0,
            self.kernel_size_data.tobytes(),
        )

    def load_image_texture(
        self,
        src: str,
        cam: str | None,
        lens: str | None,
        lens_correction: bool,
        frame_width: int | float,
        frame_height: int | float,
        rotation: float,
        zoom: float,
        rotate_times: int,
        flip: bool,
        resolution: None | tuple[int, int] = None,
        half_size: bool = True,
        cache: bool = True,
    ):
        new_param_dict = {
            "src": src,
            "cam": cam,
            "lens": lens,
            "lens_correction": lens_correction,
            "frame_width": frame_width,
            "frame_height": frame_height,
            "rotation": rotation,
            "zoom": zoom,
            "rotate_times": rotate_times,
            "flip": flip,
            "resolution": resolution,
            "half_size": half_size,
        }

        if new_param_dict == self.image_param_dict:
            return

        if not lens_correction:
            cam, lens = None, None

        if cache:
            image = self.load_raw_image_cached(src, cam, lens, half_size)
        else:
            image = self.load_raw_image(src, cam, lens, half_size)

        image = crop_rotate_zoom(
            image, frame_width, frame_height, rotation, zoom, rotate_times, flip
        )

        if resolution is not None:
            image = resolution_scaling(image, resolution)

        image = np.dstack((image, np.ones_like(image[..., :1])))

        self._ensure_image_texture(image)

        self.image_param_dict = new_param_dict

        self.height, self.width = image.shape[:2]

    def load_mtf_kernel(self, negative_film: FilmSpectral, scale: float):
        new_param_dict = {
            "negative_film": negative_film.name,
            "scale": scale,
        }

        if new_param_dict == self.mtf_param_dict:
            return

        mtf_kernel_np = mtf_kernel(negative_film, scale)

        self._ensure_mtf_kernel(mtf_kernel_np)

        self.mtf_param_dict = new_param_dict

    def load_halation_kernel(
        self,
        scale: float,
        halation_size: float = 1.0,
        halation_red_factor: float = 1.0,
        halation_green_factor: float = 0.4,
        halation_blue_factor: float = 0.0,
        halation_intensity: float = 1.0,
        bw: bool = False,
    ):
        new_param_dict = {
            "scale": scale,
            "halation_size": halation_size,
            "halation_red_factor": halation_red_factor,
            "halation_green_factor": halation_green_factor,
            "halation_blue_factor": halation_blue_factor,
            "halation_intensity": halation_intensity,
            "bw": bw,
        }

        if new_param_dict == self.halation_param_dict:
            return

        kernel = compute_halation_kernel(
            scale,
            halation_size,
            halation_red_factor,
            halation_green_factor,
            halation_blue_factor,
            halation_intensity,
            bw=bw,
        )

        self._ensure_halation_kernel(kernel)

        self.halation_param_dict = new_param_dict

    def load_input_lut(
        self,
        negative_film: FilmSpectral,
        exp_kelvin: float | int,
        tint: float | int,
        exp_comp: float | int,
    ):
        new_param_dict = {
            "negative_film": negative_film.name,
            "exp_kelvin": exp_kelvin,
            "tint": tint,
            "exp_comp": exp_comp,
        }

        if new_param_dict == self.input_param_dict:
            return

        input_lut = negative_film.get_input_lut(exp_kelvin, tint, exp_comp)

        self._ensure_lut_2d(input_lut)

        self.input_param_dict = new_param_dict

    def load_density_curve(self, negative_film: FilmSpectral, push_pull: float | int):
        new_param_dict = {"negative_film": negative_film.name, "push_pull": push_pull}

        if new_param_dict == self.curve_param_dict:
            return

        density_curve = negative_film.get_density_curve(push_pull=push_pull)

        self._ensure_lut_1d(density_curve)

        self.curve_param_dict = new_param_dict

    def load_output_lut(
        self,
        negative_film: FilmSpectral,
        print_film: FilmSpectral | None = None,
        red_light: float = 0.0,
        green_light: float = 0.0,
        blue_light: float = 0.0,
        projector_kelvin: int = 6500,
        shadow_comp: float = 0.0,
        sat_adjust: float = 1.0,
        gamma_func: GAMMA_KEYS = "sRGB",
        inversion_gamma: float = 4.0,
        idealized_curve: bool = False,
        inversion: bool = False,
        white_balance: bool = False,
        white_clip: bool = False,
        icc_transform=None,
    ):
        new_param_dict = {
            "negative_film": negative_film.name,
            "print_film": print_film.name if print_film is not None else None,
            "red_light": red_light,
            "green_light": green_light,
            "blue_light": blue_light,
            "projector_kelvin": projector_kelvin,
            "shadow_comp": shadow_comp,
            "sat_adjust": sat_adjust,
            "gamma_func": gamma_func,
            "inversion_gamma": inversion_gamma,
            "idealized_curve": idealized_curve,
            "inversion": inversion,
            "white_balance": white_balance,
            "white_clip": white_clip,
            "icc_transform": icc_transform,
        }

        if new_param_dict == self.output_param_dict:
            return

        lut = create_lut(
            negative_film,
            print_film,
            mode="print",
            input_colorspace=None,
            adx_coding=False,
            cube=False,
            red_light=red_light,
            green_light=green_light,
            blue_light=blue_light,
            projector_kelvin=projector_kelvin,
            shadow_comp=shadow_comp,
            sat_adjust=sat_adjust,
            gamma_func=gamma_func,
            inversion_gamma=inversion_gamma,
            idealized_curve=idealized_curve,
            inversion=inversion,
            white_balance=white_balance,
            white_clip=white_clip,
            linear_scaling=4.0,
        )

        if icc_transform is not None:
            lut = (lut * 255).astype(np.uint8)
            lut_shape = lut.shape
            lut = lut.reshape(lut_shape[0], -1, lut_shape[-1])
            lut = Image.fromarray(lut)
            ImageCms.applyTransform(lut, icc_transform, inPlace=True)
            lut = np.array(lut, np.uint8)
            lut = lut.reshape(lut_shape)
            lut = (lut / 255.0).astype(np.float32)

        self._ensure_lut_3d(lut)

        self.output_param_dict = new_param_dict

    def _bind_lut_2d(self, tex_a, tex_b):
        return self.device.create_bind_group(
            layout=self.pipeline_lut_2d.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": tex_a.view},
                {"binding": 1, "resource": tex_b.view},
                {"binding": 2, "resource": self.tex_lut_2d_view},
            ],
        )

    def _bind_lut_1d(self, tex_a, tex_b):
        return self.device.create_bind_group(
            layout=self.pipeline_lut_1d.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": tex_a.view},
                {"binding": 1, "resource": tex_b.view},
                {"binding": 2, "resource": self.tex_lut_1d_view},
                {
                    "binding": 3,
                    "resource": self.lut_sampler,
                },
                {
                    "binding": 4,
                    "resource": {
                        "buffer": self.buffer_params_lut_1d,
                        "offset": 0,
                        "size": self.buffer_params_lut_1d.size,
                    },
                },
            ],
        )

    def _bind_lut_3d(self, tex_a, tex_b):
        return self.device.create_bind_group(
            layout=self.pipeline_lut_3d.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": tex_a.view},
                {"binding": 1, "resource": tex_b.view},
                {"binding": 2, "resource": self.tex_lut_3d_view},
            ],
        )

    def _bind_convolution(self, tex_a, tex_b, buffer, buffer_size):
        return self.device.create_bind_group(
            layout=self.pipeline_convolution.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": tex_a.view},
                {"binding": 1, "resource": tex_b.view},
                {
                    "binding": 2,
                    "resource": {
                        "buffer": buffer,
                    },
                },
                {
                    "binding": 3,
                    "resource": {
                        "buffer": buffer_size,
                    },
                },
            ],
        )

    def _dispatch(self, pipeline, bind_group, size):
        encoder = self.device.create_command_encoder()

        pass_enc = encoder.begin_compute_pass()
        pass_enc.set_pipeline(pipeline)
        pass_enc.set_bind_group(0, bind_group)

        # Workgroup sizing (assumes 8x8)
        gx = (size[0] + 7) // 8
        gy = (size[1] + 7) // 8

        pass_enc.dispatch_workgroups(gx, gy, 1)
        pass_enc.end()

        self.queue.submit([encoder.finish()])

    def read_texture(self, texture, width, height):
        # rgba8unorm uses 4 bytes per pixel (1 byte per channel)
        bytes_per_pixel = 4
        row_bytes = width * bytes_per_pixel
        padded_row_bytes = ((row_bytes + 255) // 256) * 256
        buffer_size = padded_row_bytes * height

        readback_buffer = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

        encoder = self.device.create_command_encoder()

        encoder.copy_texture_to_buffer(
            {
                "texture": texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "buffer": readback_buffer,
                "offset": 0,
                "bytes_per_row": padded_row_bytes,
                "rows_per_image": height,
            },
            {
                "width": width,
                "height": height,
                "depth_or_array_layers": 1,
            },
        )

        self.queue.submit([encoder.finish()])
        readback_buffer.map_sync(wgpu.MapMode.READ)
        data = readback_buffer.read_mapped()
        readback_buffer.unmap()

        array = np.frombuffer(data, dtype=np.uint8)

        array = array.reshape((height, padded_row_bytes))

        array = array.reshape((height, padded_row_bytes // 4, 4))[:, :width, :]

        return array

    def copy_to_dst(self, src_texture, dst_texture):
        src_w, src_h, _ = src_texture.size
        dst_w, dst_h, _ = dst_texture.size
        offset_x = (dst_w - src_w) // 2
        offset_y = (dst_h - src_h) // 2

        encoder = self.device.create_command_encoder()

        view = dst_texture.create_view()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": view,
                    "clear_value": (0.0, 0.0, 0.0, 0.0),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )
        render_pass.end()

        offset_data = struct.pack("ii", offset_x, offset_y)
        uniform_buffer = self.device.create_buffer_with_data(
            data=offset_data, usage=wgpu.BufferUsage.UNIFORM
        )

        bind_group = self.device.create_bind_group(
            layout=self.pipeline_copy_to_int.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": src_texture.create_view()},
                {"binding": 1, "resource": view},
                {
                    "binding": 2,
                    "resource": {
                        "buffer": uniform_buffer,
                        "offset": 0,
                        "size": len(offset_data),
                    },
                },
            ],
        )

        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline_copy_to_int)
        compute_pass.set_bind_group(0, bind_group)

        dispatch_x = (src_w + 15) // 16
        dispatch_y = (src_h + 15) // 16
        compute_pass.dispatch_workgroups(dispatch_x, dispatch_y)
        compute_pass.end()

        self.device.queue.submit([encoder.finish()])

    def process(
        self,
        src: str,
        negative_film: FilmSpectral,
        grain_size: float,
        grain_sigma: float,
        dst_texture=None,
        lens_correction: bool = True,
        print_film: FilmSpectral | None = None,
        exp_comp: float = 0.0,
        red_light: float = 0.0,
        green_light: float = 0.0,
        blue_light: float = 0.0,
        projector_kelvin: int = 6500,
        shadow_comp: float = 0.0,
        sat_adjust: float = 1.0,
        gamma_func: GAMMA_KEYS = "sRGB",
        exp_kelvin: int = 6500,
        tint: float = 0.0,
        inversion_gamma: float = 4.0,
        idealized_curve: bool = False,
        inversion: bool = False,
        push_pull: float = 0.0,
        white_balance: bool = False,
        white_clip: bool = False,
        icc_transform=None,
        resolution: None | tuple[int, int] = None,
        frame_width: int | float = 36,
        frame_height: int | float = 24,
        rotation: float = 0.0,
        zoom: float = 1.0,
        rotate_times: int = 0,
        flip: bool = False,
        cam: str | None = None,
        lens: str | None = None,
        half_res: bool = False,
        canvas_mode: CANVAS_MODES = "No",
        canvas_scale: float = 1.0,
        canvas_ratio: float = 1.0,
        halation_intensity: float = 1.0,
        halation: bool = True,
        halation_size: float = 1.0,
        halation_green_factor: float = 0.4,
        sharpness: bool = True,
        chroma_nr: int = 0,
        grain: int = 2,
        highlight_burn: float = 0.0,
        burn_scale: float = 50.0,
        half_size: bool = True,
        cache: bool = True,
        **_,
    ) -> np.ndarray | None:
        start = time.time()
        # Update textures
        self.load_image_texture(
            src,
            cam,
            lens,
            lens_correction,
            frame_width,
            frame_height,
            rotation,
            zoom,
            rotate_times,
            flip,
            resolution,
            half_size,
            cache,
        )
        self.load_input_lut(negative_film, exp_kelvin, tint, exp_comp)
        self.load_density_curve(negative_film, push_pull)
        self.load_output_lut(
            negative_film,
            print_film,
            red_light,
            green_light,
            blue_light,
            projector_kelvin,
            shadow_comp,
            sat_adjust,
            gamma_func,
            inversion_gamma,
            idealized_curve,
            inversion,
            white_balance,
            white_clip,
            icc_transform,
        )

        scale = max(self.width, self.height) / max(
            frame_width, frame_height
        )  # pixels per mm
        ping_pong_tex = [self.tex_a, self.tex_b]
        idx = 1

        print(f"loading {time.time() - start:.4f}s")
        start = time.time()
        self._dispatch(
            self.pipeline_lut_2d,
            self._bind_lut_2d(self.tex_input, ping_pong_tex[1 - idx]),
            (self.width, self.height),
        )
        idx = 1 - idx

        if halation:
            self.load_halation_kernel(
                scale,
                halation_size=halation_size,
                halation_green_factor=halation_green_factor,
                halation_intensity=halation_intensity,
                bw=negative_film.density_measure == "bw",
            )

            self._dispatch(
                self.pipeline_convolution,
                self._bind_convolution(
                    ping_pong_tex[idx],
                    ping_pong_tex[1 - idx],
                    self.buffer_halation_kernel,
                    self.buffer_halation_kernel_size,
                ),
                (self.width, self.height),
            )
            idx = 1 - idx

        self._dispatch(
            self.pipeline_lut_1d,
            self._bind_lut_1d(ping_pong_tex[idx], ping_pong_tex[1 - idx]),
            (self.width, self.height),
        )
        idx = 1 - idx

        if sharpness and negative_film.mtf is not None:
            self.load_mtf_kernel(negative_film, scale)

            self._dispatch(
                self.pipeline_convolution,
                self._bind_convolution(
                    ping_pong_tex[idx],
                    ping_pong_tex[1 - idx],
                    self.buffer_mtf_kernel,
                    self.buffer_mtf_kernel_size,
                ),
                (self.width, self.height),
            )
            idx = 1 - idx

        self._dispatch(
            self.pipeline_lut_3d,
            self._bind_lut_3d(ping_pong_tex[idx], ping_pong_tex[1 - idx]),
            (self.width, self.height),
        )
        idx = 1 - idx

        # TODO: grain
        # TODO: chroma NR
        # TODO: highlight burn
        # TODO: canvas_mode
        # TODO: half res

        print(f"shaders {time.time() - start:.4f}s")
        start = time.time()
        if dst_texture is None:
            self.copy_to_dst(ping_pong_tex[idx].texture, self.tex_int_out.texture)
            print(f"to int {time.time() - start:.4f}s")
            start = time.time()
            image = self.read_texture(
                self.tex_int_out.texture, self.width, self.height
            )[..., 0:3]

            print(f"to numpy {time.time() - start:.4f}s")
            return image
        else:
            self.copy_to_dst(ping_pong_tex[idx].texture, dst_texture)
            print(f"to int {time.time() - start:.4f}s")

            return None
