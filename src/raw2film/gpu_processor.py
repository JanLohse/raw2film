from functools import lru_cache
from pathlib import Path

import numpy as np
from spectral_film_lut.color_space import GAMMA_KEYS
from spectral_film_lut.film_spectral import FilmSpectral
from spectral_film_lut.utils import create_lut, log_clip, multi_channel_interp
from spectral_film_lut.xy_lut import apply_2d_lut
from wgpu import get_default_device

from raw2film import effects
from raw2film.raw_conversion import crop_rotate_zoom, raw_to_linear
from raw2film.utils import (
    apply_lut_tetrahedral_float,
    load_metadata,
    resolution_scaling,
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

        lut_3d_shader_path = Path(__file__).parent / "shaders/lut_tetrahedral.wgsl"
        lut_3d_shader_code = lut_3d_shader_path.read_text()
        self.lut_3d_shader = self.device.create_shader_module(code=lut_3d_shader_code)

        # Init gpu textures
        self.tex_input = None
        self.tex_a = None
        self.tex_b = None
        self.tex_output = None

        self.tex_lut_1d = None
        self.tex_lut_2d = None
        self.tex_lut_3d = None

        # Comparison dicts
        self.image_param_dict = None
        self.input_param_dict = None
        self.curve_param_dict = None
        self.output_param_dict = None

        # Init lens correction data
        self.cameras = cameras
        self.lenses = lenses

    @lru_cache
    def load_raw_image(self, src, cam=None, lens=None):
        image = raw_to_linear(src)

        if cam is not None and lens is not None:
            cam = self.cameras[cam]
            lens = self.lenses[lens]

            image = effects.lens_correction(image, load_metadata(src), cam, lens)

        return image

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
        }

        if new_param_dict == self.image_param_dict:
            return

        if lens_correction:
            image = self.load_raw_image(src, cam, lens)
        else:
            image = self.load_raw_image(src)

        image = crop_rotate_zoom(
            image, frame_width, frame_height, rotation, zoom, rotate_times, flip
        )

        if resolution is not None:
            image = resolution_scaling(image, resolution)

        self.tex_input = image  # TODO

        self.image_param_dict = new_param_dict

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

        self.tex_lut_2d = input_lut  # TODO

        self.input_param_dict = new_param_dict

    def load_density_curve(self, negative_film: FilmSpectral, push_pull: float | int):
        new_param_dict = {"negative_film": negative_film.name, "push_pull": push_pull}

        if new_param_dict == self.curve_param_dict:
            return

        density_curve = negative_film.get_density_curve(push_pull=push_pull)
        density_curve[1:] /= 4.0

        self.tex_lut_1d = density_curve

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
        lut = (lut * (2**8 - 1)).astype(np.uint8)  # TODO

        self.tex_lut_3d = lut  # TODO

        self.output_param_dict = new_param_dict

    def process(
        self,
        src: str,
        negative_film: FilmSpectral,
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
        icc_transform=None,  # TODO
        lut_size: int = 33,  # TODO
        resolution: None | tuple[int, int] = None,
        frame_width: int | float = 36,
        frame_height: int | float = 24,
        rotation: float = 0.0,
        zoom: float = 1.0,
        rotate_times: int = 0,
        flip: bool = False,
        cam: str | None = None,
        lens: str | None = None,
        **_,
    ):
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
        )

        self.tex_a = apply_2d_lut(self.tex_input, self.tex_lut_2d)

        log_clip(self.tex_a)

        self.tex_b = multi_channel_interp(self.tex_a, self.tex_lut_1d)

        self.tex_output = apply_lut_tetrahedral_float(self.tex_b, self.tex_lut_3d)

        return self.tex_output
