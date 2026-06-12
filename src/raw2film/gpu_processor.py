import random
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
from spectral_film_lut.grain_generation import grain_kernel
from spectral_film_lut.utils import create_lut
from wgpu import TextureUsage, get_default_device

from raw2film import effects
from raw2film.effects import (
    chroma_nr_filter,
    compute_halation_kernel,
    mtf_kernel,
)
from raw2film.raw_conversion import CANVAS_MODES, crop_rotate_zoom, raw_to_linear
from raw2film.utils import (
    MIX_TABLE,
    load_metadata,
    resolution_scaling,
)


class GpuTexture:
    def __init__(self, device, size, format=wgpu.TextureFormat.rgba16float):
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

        noise_shader_path = Path(__file__).parent / "shaders/noise.wgsl"
        noise_shader_code = noise_shader_path.read_text()
        noise_shader = self.device.create_shader_module(code=noise_shader_code)

        noise_bw_shader_path = Path(__file__).parent / "shaders/noise_bw.wgsl"
        noise_bw_shader_code = noise_bw_shader_path.read_text()
        noise_bw_shader = self.device.create_shader_module(code=noise_bw_shader_code)

        grain_shader_path = Path(__file__).parent / "shaders/grain.wgsl"
        grain_shader_code = grain_shader_path.read_text()
        grain_shader = self.device.create_shader_module(code=grain_shader_code)

        histogram_shader_path = Path(__file__).parent / "shaders/histogram.wgsl"
        histogram_shader_code = histogram_shader_path.read_text()
        histogram_shader = self.device.create_shader_module(code=histogram_shader_code)

        scale_shader_path = Path(__file__).parent / "shaders/scale_texture.wgsl"
        scale_shader_code = scale_shader_path.read_text()
        scale_shader = self.device.create_shader_module(code=scale_shader_code)

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
        self.pipeline_noise = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": noise_shader, "entry_point": "main"},
        )
        self.pipeline_noise_bw = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": noise_bw_shader, "entry_point": "main"},
        )
        self.pipeline_grain = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": grain_shader, "entry_point": "main"},
        )
        self.pipeline_histogram_1 = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": histogram_shader, "entry_point": "pass1_accumulate"},
        )
        self.pipeline_histogram_2 = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": histogram_shader, "entry_point": "pass2_process"},
        )
        self.pipeline_histogram_3 = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": histogram_shader, "entry_point": "pass3_render"},
        )
        self.pipeline_scale_texture = self.device.create_compute_pipeline(
            layout="auto", compute={"module": scale_shader, "entry_point": "main"}
        )

        # Init gpu textures
        self.tex_input = None
        self.tex_a = None
        self.tex_b = None
        self.tex_temp = None
        self.tex_int_out = None

        self.tex_lut_1d = None
        self.tex_lut_2d = None
        self.tex_lut_3d = None
        self.tex_lut_grain = None

        self.buffer_params_lut_1d = None
        self.buffer_params_grain = None
        self.buffer_mtf_kernel = None
        self.buffer_mtf_kernel_size = None
        self.buffer_halation_kernel = None
        self.buffer_halation_kernel_size = None
        self.buffer_grain_kernel = None
        self.buffer_grain_kernel_size = None

        # Comparison dicts
        self.image_param_dict = None
        self.input_param_dict = None
        self.curve_param_dict = None
        self.output_param_dict = None
        self.mtf_param_dict = None
        self.halation_param_dict = None
        self.grain_kernel_param_dict = None

        # Add these definitions to your existing __init__ method:
        # --------------------------------------------------------
        self.tex_hist_out = None
        self.buffer_hist_counts = None
        self.buffer_hist_heights = None
        self.buffer_hist_mix_table = None
        self.buffer_hist_params = None

        # A cached copy of the flat mix table to avoid re-uploading if unchanged
        self.mix_table = None

        self.width = None
        self.height = None

        self.lut_1d_sampler = self.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
        )
        self.lut_3d_sampler = self.device.create_sampler(
            mag_filter=wgpu.FilterMode.linear,
            min_filter=wgpu.FilterMode.linear,
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            address_mode_w=wgpu.AddressMode.clamp_to_edge,
        )

        self.cameras = cameras
        self.lenses = lenses

        self._create_histogram_buffers()

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
        start = time.time()
        np.clip(image, 0, 65504, out=image)
        print(f"clipping: {time.time() - start:.4f}s")

        return image

    def _ensure_image_texture(self, image: np.ndarray):
        h, w = image.shape[:2]

        if self.tex_input is None or self.tex_input.size != (w, h):
            self.tex_input = GpuTexture(
                self.device, (w, h), format=wgpu.TextureFormat.rgba32float
            )
            self.tex_a = GpuTexture(self.device, (w, h))
            self.tex_b = GpuTexture(self.device, (w, h))
            self.tex_temp = GpuTexture(self.device, (w, h))
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
                format=wgpu.TextureFormat.rgba16float,
                usage=(wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST),
            )
            self.tex_lut_3d_view = self.tex_lut_3d.create_view()

        lut_rgba = np.ones((size, size, size, 4), dtype=np.float32)
        lut_rgba[..., 0:3] = lut

        lut_rgba_16 = lut_rgba.astype(np.float16)

        self.queue.write_texture(
            {
                "texture": self.tex_lut_3d,
            },
            lut_rgba_16,
            {
                "bytes_per_row": size * 4 * 2,
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

    def _ensure_grain_kernel(self, kernel: np.ndarray):
        h, w = kernel.shape[:2]

        kernel_rgba = np.ones((h, w, 4), dtype=np.float32)

        if kernel.ndim == 2:
            kernel_rgba[..., :3] = kernel[..., None]
        else:
            kernel_rgba[..., :3] = kernel

        flat = kernel_rgba.reshape(-1, 4)

        if (
            self.buffer_grain_kernel is None
            or self.buffer_grain_kernel.size < flat.nbytes
        ):
            self.buffer_grain_kernel = self.device.create_buffer(
                size=flat.nbytes,
                usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST),
            )

        self.queue.write_buffer(
            self.buffer_grain_kernel,
            0,
            flat.tobytes(),
        )

        self.kernel_size_data = np.array(
            [h, w],
            dtype=np.uint32,
        )

        if self.buffer_grain_kernel_size is None:
            self.buffer_grain_kernel_size = self.device.create_buffer(
                size=16,  # uniform alignment
                usage=(wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST),
            )

        self.queue.write_buffer(
            self.buffer_grain_kernel_size,
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

    def _ensure_grain_lut(self, lut: np.ndarray):
        size = lut.shape[1]
        if self.tex_lut_grain is None or size != self.tex_lut_grain.size[0]:
            self.tex_lut_grain = self.device.create_texture(
                size=(size, 1, 1),
                format=wgpu.TextureFormat.rgba16float,
                usage=(wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST),
            )
            self.tex_lut_grain_view = self.tex_lut_grain.create_view()

        lut_rgba = np.ones((size, 4), dtype=np.float32)
        lut_rgba[..., 0:3] = lut[1:, ...].T

        lut_rgba_16 = lut_rgba.astype(np.float16)

        xp_min = lut[0, 0]
        xp_max = lut[0, -1]
        denom = xp_max - xp_min
        inv_range = 1.0 / denom if denom != 0.0 else 0.0

        params_lut_1d = struct.pack(
            "ffff",
            xp_min,
            xp_max,
            inv_range,
            random.randint(0, 100000000),
        )

        self.buffer_params_grain = self.device.create_buffer_with_data(
            data=params_lut_1d,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        self.queue.write_texture(
            {"texture": self.tex_lut_grain},
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

    def _create_histogram_buffers(self, mix_table: np.ndarray = MIX_TABLE):
        self.tex_hist_out = GpuTexture(
            self.device, (256, 256), format=wgpu.TextureFormat.rgba8uint
        )

        # Buffer 1: Atomic collection buckets (3 channels * 256 bins * 4 bytes)
        self.buffer_hist_counts = self.device.create_buffer(
            size=3 * 256 * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
        )

        # Buffer 2: Scaled vertical line markers (3 channels * 256 bins * 4 bytes)
        self.buffer_hist_heights = self.device.create_buffer(
            size=3 * 256 * 4, usage=wgpu.BufferUsage.STORAGE
        )

        self._cached_mix_table = mix_table.copy()
        flat_table = mix_table.reshape((8, 4)).astype(np.uint32)
        self.buffer_hist_mix_table = self.device.create_buffer_with_data(
            data=flat_table, usage=wgpu.BufferUsage.STORAGE
        )

    def _ensure_histogram_buffers(self):
        param_data = np.array([256, self.width, self.height, 0], dtype=np.uint32)
        self.buffer_hist_params = self.device.create_buffer_with_data(
            data=param_data, usage=wgpu.BufferUsage.UNIFORM
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
        chroma_nr: int = 0,
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
            "chroma_nr": chroma_nr,
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

        if chroma_nr:
            image = chroma_nr_filter(image, chroma_nr)

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

    def load_grain(
        self,
        negative_film: FilmSpectral,
        scale: float,
        grain_size_mm: float = 0.01,
        grain_sigma: float = 0.4,
        bw_grain: bool = False,
    ):
        input_lut = negative_film.get_grain_curve(scale, adx=False, bw_grain=bw_grain)

        self._ensure_grain_lut(input_lut)

        new_kernel_dict = {
            "scale": scale,
            "grain_size_mm": grain_size_mm,
            "grain_sigma": grain_sigma,
            "bw_grain": bw_grain,
        }

        if new_kernel_dict == self.grain_kernel_param_dict:
            return

        kernel = grain_kernel(
            1 / scale, grain_size_mm=grain_size_mm, grain_sigma=grain_sigma
        )

        if kernel is None:
            kernel = np.ones((1, 1), dtype=DEFAULT_DTYPE)

        self._ensure_grain_kernel(kernel)

        self.grain_kernel_param_dict = new_kernel_dict

    def load_density_curve(
        self,
        negative_film: FilmSpectral,
        push_pull: float | int,
        color_masking: float | None = None,
    ):
        new_param_dict = {
            "negative_film": negative_film.name,
            "push_pull": push_pull,
            "color_masking": color_masking,
        }

        if new_param_dict == self.curve_param_dict:
            return

        density_curve = negative_film.get_density_curve(
            push_pull=push_pull, color_masking=color_masking
        )

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
        color_masking: float | None = None,
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
            "color_masking": color_masking,
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
            color_masking=color_masking,
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
                    "resource": self.lut_1d_sampler,
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
                {
                    "binding": 3,
                    "resource": self.lut_3d_sampler,
                },
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

    def _bind_noise(self, tex_out, pipeline):
        return self.device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": tex_out.view},
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.buffer_params_grain,
                        "offset": 0,
                        "size": self.buffer_params_grain.size,
                    },
                },
            ],
        )

    def _bind_grain(self, tex_a, tex_b, tex_noise):
        return self.device.create_bind_group(
            layout=self.pipeline_grain.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": tex_a.view},
                {"binding": 1, "resource": tex_b.view},
                {"binding": 2, "resource": self.tex_lut_grain_view},
                {
                    "binding": 3,
                    "resource": self.lut_1d_sampler,
                },
                {
                    "binding": 4,
                    "resource": {
                        "buffer": self.buffer_params_grain,
                        "offset": 0,
                        "size": self.buffer_params_grain.size,
                    },
                },
                {"binding": 5, "resource": tex_noise.view},
                {"binding": 6, "resource": {"buffer": self.buffer_grain_kernel}},
                {"binding": 7, "resource": {"buffer": self.buffer_grain_kernel_size}},
            ],
        )

    def _bind_histogram_pass1(self, source_tex):
        return self.device.create_bind_group(
            layout=self.pipeline_histogram_1.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": source_tex.view},
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.buffer_hist_counts,
                        "offset": 0,
                        "size": self.buffer_hist_counts.size,
                    },
                },
                {
                    "binding": 5,
                    "resource": {
                        "buffer": self.buffer_hist_params,
                        "offset": 0,
                        "size": self.buffer_hist_params.size,
                    },
                },
            ],
        )

    def _bind_histogram_pass2(self):
        return self.device.create_bind_group(
            layout=self.pipeline_histogram_2.get_bind_group_layout(0),
            entries=[
                {
                    "binding": 1,
                    "resource": {
                        "buffer": self.buffer_hist_counts,
                        "offset": 0,
                        "size": self.buffer_hist_counts.size,
                    },
                },
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self.buffer_hist_heights,
                        "offset": 0,
                        "size": self.buffer_hist_heights.size,
                    },
                },
                {
                    "binding": 5,
                    "resource": {
                        "buffer": self.buffer_hist_params,
                        "offset": 0,
                        "size": self.buffer_hist_params.size,
                    },
                },
            ],
        )

    def _bind_histogram_pass3(self):
        return self.device.create_bind_group(
            layout=self.pipeline_histogram_3.get_bind_group_layout(0),
            entries=[
                {
                    "binding": 2,
                    "resource": {
                        "buffer": self.buffer_hist_heights,
                        "offset": 0,
                        "size": self.buffer_hist_heights.size,
                    },
                },
                {
                    "binding": 3,
                    "resource": {
                        "buffer": self.buffer_hist_mix_table,
                        "offset": 0,
                        "size": self.buffer_hist_mix_table.size,
                    },
                },
                {"binding": 4, "resource": self.tex_hist_out.view},
                {
                    "binding": 5,
                    "resource": {
                        "buffer": self.buffer_hist_params,
                        "offset": 0,
                        "size": self.buffer_hist_params.size,
                    },
                },
            ],
        )

    def generate_histogram(self, source_tex):
        """
        Executes the complete multi-pass pipeline chain to draw the output histogram.
        """
        self._ensure_histogram_buffers()

        zero_data = np.zeros(3 * 256, dtype=np.uint32)
        self.queue.write_buffer(self.buffer_hist_counts, 0, zero_data)

        # Open single primary command stream record for all 3 passes
        encoder = self.device.create_command_encoder()

        # Pass 1: Frequencies Accumulation (16x16 workgroups)
        self._dispatch(
            pipeline=self.pipeline_histogram_1,
            bind_group=self._bind_histogram_pass1(source_tex),
            size=(self.width, self.height),
            wg_size=(16, 16),
            encoder=encoder,
        )

        # Pass 2: Log Transformation & Smoothing (1x1 execution layout)
        self._dispatch(
            pipeline=self.pipeline_histogram_2,
            bind_group=self._bind_histogram_pass2(),
            size=(256, 1),
            wg_size=(256, 1),
            encoder=encoder,
        )

        # Pass 3: Grid Rasterization & Blending (16x16 workgroups over 256x256 target)
        self._dispatch(
            pipeline=self.pipeline_histogram_3,
            bind_group=self._bind_histogram_pass3(),
            size=(256, 256),
            wg_size=(16, 16),
            encoder=encoder,
        )

        self.queue.submit([encoder.finish()])
        return self.tex_hist_out

    def _dispatch_scale_histogram(self, target_texture):
        """Scales the internal histogram texture to fit an arbitrary target canvas."""
        target_w, target_h = target_texture.size[0], target_texture.size[1]

        # Create transient uniform buffer
        scale_params = np.array([target_w, target_h], dtype=np.uint32)
        buffer_scale_params = self.device.create_buffer_with_data(
            data=scale_params, usage=wgpu.BufferUsage.UNIFORM
        )

        # Build layout entries smoothly
        bind_group = self.device.create_bind_group(
            layout=self.pipeline_scale_texture.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": self.tex_hist_out.view},
                {"binding": 1, "resource": target_texture.create_view()},
                {
                    "binding": 2,
                    "resource": {
                        "buffer": buffer_scale_params,
                        "offset": 0,
                        "size": buffer_scale_params.size,
                    },
                },
            ],
        )

        # Recycle the primary pattern dispatcher completely
        self._dispatch(
            pipeline=self.pipeline_scale_texture,
            bind_group=bind_group,
            size=(target_w, target_h),
            wg_size=(16, 16),
        )

    def _dispatch(self, pipeline, bind_group, size, wg_size=(8, 8), encoder=None):
        """
        A unified, flexible dispatch helper that handles arbitrary workgroup sizes
        and optional multi-pass command encoding chaining.
        """
        # If no encoder is passed, we manage the lifecycle (single-pass mode)
        submit_instantly = False
        if encoder is None:
            encoder = self.device.create_command_encoder()
            submit_instantly = True

        pass_enc = encoder.begin_compute_pass()
        pass_enc.set_pipeline(pipeline)
        pass_enc.set_bind_group(0, bind_group)

        # Dynamic workgroup ceiling division math
        gx = (size[0] + (wg_size[0] - 1)) // wg_size[0]
        gy = (size[1] + (wg_size[1] - 1)) // wg_size[1]

        pass_enc.dispatch_workgroups(gx, gy, 1)
        pass_enc.end()

        if submit_instantly:
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
        histogram_texture=None,
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
        color_masking: float | None = None,
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
            chroma_nr,
        )
        self.load_input_lut(negative_film, exp_kelvin, tint, exp_comp)
        self.load_density_curve(negative_film, push_pull, color_masking)
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
            color_masking,
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

        if grain and negative_film.rms_density is not None:
            bw_grain = grain == 1
            self.load_grain(
                negative_film, scale, grain_size / 1000, grain_sigma, bw_grain
            )

            noise_pipeline = (
                self.pipeline_noise if not bw_grain else self.pipeline_noise_bw
            )

            self._dispatch(
                noise_pipeline,
                self._bind_noise(self.tex_temp, noise_pipeline),
                (self.width, self.height),
            )

            self._dispatch(
                self.pipeline_grain,
                self._bind_grain(
                    ping_pong_tex[idx], ping_pong_tex[1 - idx], self.tex_temp
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

        # TODO: highlight burn
        # TODO: canvas_mode
        # TODO: half res
        # TODO: auto downscale when needed

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

            if histogram_texture is not None:
                self.generate_histogram(ping_pong_tex[idx])

                self._dispatch_scale_histogram(histogram_texture)

            return None
