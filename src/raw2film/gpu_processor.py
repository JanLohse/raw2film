import struct
from functools import lru_cache
from pathlib import Path

import numpy as np
import wgpu
from PIL import Image, ImageCms
from spectral_film_lut.color_space import GAMMA_KEYS
from spectral_film_lut.film_spectral import FilmSpectral
from spectral_film_lut.utils import create_lut
from wgpu import TextureUsage, get_default_device

from raw2film import effects
from raw2film.raw_conversion import CANVAS_MODES, crop_rotate_zoom, raw_to_linear
from raw2film.utils import (
    load_metadata,
    resolution_scaling,
)


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
        self.lut_1d_shader = self.device.create_shader_module(code=lut_1d_shader_code)

        lut_2d_shader_path = Path(__file__).parent / "shaders/lut_2d.wgsl"
        lut_2d_shader_code = lut_2d_shader_path.read_text()
        self.lut_2d_shader = self.device.create_shader_module(code=lut_2d_shader_code)

        lut_3d_shader_path = Path(__file__).parent / "shaders/lut_3d.wgsl"
        lut_3d_shader_code = lut_3d_shader_path.read_text()
        self.lut_3d_shader = self.device.create_shader_module(code=lut_3d_shader_code)

        # create pipelines
        self.pipeline_lut_1d = self.device.create_compute_pipeline(
            layout="auto", compute={"module": self.lut_1d_shader, "entry_point": "main"}
        )
        self.pipeline_lut_2d = self.device.create_compute_pipeline(
            layout="auto", compute={"module": self.lut_2d_shader, "entry_point": "main"}
        )
        self.pipeline_lut_3d = self.device.create_compute_pipeline(
            layout="auto", compute={"module": self.lut_3d_shader, "entry_point": "main"}
        )

        # Init gpu textures
        self.tex_input = None
        self.tex_a = None
        self.tex_b = None

        self.tex_lut_1d = None
        self.tex_lut_2d = None
        self.tex_lut_3d = None

        self.buffer_params_lut_1d = None

        # Comparison dicts
        self.image_param_dict = None
        self.input_param_dict = None
        self.curve_param_dict = None
        self.output_param_dict = None

        self.width = None
        self.height = None

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

        return image

    def _ensure_image_texture(self, image: np.ndarray):
        h, w = image.shape[:2]

        if self.tex_input is None or self.tex_input.size != (w, h):
            self.tex_input = GpuTexture(self.device, (w, h))
            self.tex_a = GpuTexture(self.device, (w, h))
            self.tex_b = GpuTexture(self.device, (w, h))

        self.tex_input.upload(self.queue, image)

    def _ensure_lut_1d(self, lut: np.ndarray):
        size = lut.shape[1]
        if self.tex_lut_1d is None or size != self.tex_lut_1d.size[0]:
            self.tex_lut_1d = self.device.create_texture(
                size=(size, 1, 1),
                format=wgpu.TextureFormat.rgba32float,
                usage=(wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST),
            )
        lut_rgba = np.ones((size, 4), dtype=np.float32)
        lut_rgba[..., 0:3] = lut[1:, ...].T

        params_lut_1d = struct.pack("ff2f", lut[0, 0], lut[0, -1], 0.0, 0.0)

        self.buffer_params_lut_1d = self.device.create_buffer_with_data(
            data=params_lut_1d,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self.queue.write_texture(
            {"texture": self.tex_lut_1d},
            lut_rgba,
            {
                "bytes_per_row": size * 4 * 4,
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

        self._ensure_image_texture(image)

        self.image_param_dict = new_param_dict

        self.height, self.width = image.shape[:2]

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

    def _bind_lut_2d(self):
        return self.device.create_bind_group(
            layout=self.pipeline_lut_2d.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": self.tex_input.view},
                {"binding": 1, "resource": self.tex_a.view},
                {"binding": 2, "resource": self.tex_lut_2d.create_view()},
            ],
        )

    def _bind_lut_1d(self):
        return self.device.create_bind_group(
            layout=self.pipeline_lut_1d.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": self.tex_a.view},
                {"binding": 1, "resource": self.tex_b.view},
                {"binding": 2, "resource": self.tex_lut_1d.create_view()},
                {
                    "binding": 3,
                    "resource": {
                        "buffer": self.buffer_params_lut_1d,
                        "offset": 0,
                        "size": self.buffer_params_lut_1d.size,
                    },
                },
            ],
        )

    def _bind_lut_3d(self):
        return self.device.create_bind_group(
            layout=self.pipeline_lut_3d.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": self.tex_b.view},
                {"binding": 1, "resource": self.tex_a.view},
                {"binding": 2, "resource": self.tex_lut_3d.create_view()},
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
        import numpy as np
        import wgpu

        bytes_per_pixel = 16  # rgba32float
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

        # Reshape: height rows, each row has padded_row_bytes // 4 float32 values
        array = np.frombuffer(data, dtype=np.float32)
        array = array.reshape((height, padded_row_bytes // 4))
        # Each pixel is 4 floats (RGBA), so reshape and crop to actual width
        array = array.reshape((height, padded_row_bytes // 16, 4))[:, :width, :]

        return array

    def process(
        self,
        src: str,
        negative_film: FilmSpectral,
        grain_size: float,
        grain_sigma: float,
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
    ):
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

        self._dispatch(
            self.pipeline_lut_2d, self._bind_lut_2d(), (self.width, self.height)
        )
        self._dispatch(
            self.pipeline_lut_1d, self._bind_lut_1d(), (self.width, self.height)
        )
        self._dispatch(
            self.pipeline_lut_3d, self._bind_lut_3d(), (self.width, self.height)
        )

        image = self.read_texture(self.tex_a.texture, self.width, self.height)[..., 0:3]

        image = (np.clip(image, 0.0, 1.0) * (2**8 - 1)).astype(np.uint8)

        return image
