from functools import lru_cache

import cv2 as cv
import numpy as np
from PIL import Image, ImageCms
from spectral_film_lut.color_space import GAMMA_KEYS
from spectral_film_lut.config import DEFAULT_DTYPE
from spectral_film_lut.film_spectral import FilmSpectral
from spectral_film_lut.utils import create_lut, log_clip, multi_channel_interp
from spectral_film_lut.xy_lut import apply_2d_lut
from wgpu import get_default_device

from raw2film import effects
from raw2film.effects import add_canvas, add_canvas_uniform, chroma_nr_filter
from raw2film.raw_conversion import CANVAS_MODES, crop_rotate_zoom, raw_to_linear
from raw2film.utils import (
    apply_lut_tetrahedral,
    load_metadata,
    resolution_scaling,
)


class CpuProcessor:
    def __init__(self, cameras, lenses):
        self.device = get_default_device()
        self.queue = self.device.queue

        # Init textures
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

        self.tex_input = np.ascontiguousarray(image)

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

        self.tex_lut_2d = input_lut

        self.input_param_dict = new_param_dict

    def load_density_curve(self, negative_film: FilmSpectral, push_pull: float | int):
        new_param_dict = {"negative_film": negative_film.name, "push_pull": push_pull}

        if new_param_dict == self.curve_param_dict:
            return

        density_curve = negative_film.get_density_curve(push_pull=push_pull)
        # density_curve[1:] /= 4.0

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

        self.tex_lut_3d = lut

        self.output_param_dict = new_param_dict

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

        # process image
        image = apply_2d_lut(self.tex_input, self.tex_lut_2d)

        if chroma_nr:
            image = chroma_nr_filter(image, chroma_nr)

        scale = max(image.shape) / max(frame_width, frame_height)  # pixels per mm

        if halation:
            image = effects.halation(
                image,
                scale,
                halation_size=halation_size,
                halation_green_factor=halation_green_factor,
                halation_intensity=halation_intensity,
            )

        log_clip(image)

        image = multi_channel_interp(image, self.tex_lut_1d)

        if sharpness and negative_film.mtf is not None:
            image = effects.film_sharpness(image, negative_film, scale)

        if grain and negative_film.rms_density is not None:
            image = effects.apply_grain(
                image,
                negative_film,
                scale,
                grain_size_mm=grain_size / 1000,
                grain_sigma=grain_sigma,
                bw_grain=grain == 1,
                adx=False,
            )
            image = np.clip(image, 0, None)

        if highlight_burn and (
            print_film is not None
            or negative_film.density_measure in ["status_m", "bw"]
        ):
            image = effects.burn(image, negative_film, highlight_burn, burn_scale)

        image = apply_lut_tetrahedral(image, self.tex_lut_3d, 0.25)

        image = (image * (2**8 - 1)).astype(np.uint8)

        # post-processing (canvas, scaling)
        if canvas_mode != "No":
            if "white" in canvas_mode:
                canvas_color = (255, 255, 255)
            elif "black" in canvas_mode:
                canvas_color = (0, 0, 0)
            else:
                canvas_color = (128, 128, 128)
            if "Proportional" in canvas_mode:
                canvas_ratio = image.shape[1] / image.shape[0]
                image = add_canvas(image, canvas_ratio, canvas_scale, canvas_color)
            elif "Fixed" in canvas_mode:
                image = add_canvas(image, canvas_ratio, canvas_scale, canvas_color)
            elif "Uniform" in canvas_mode:
                image = add_canvas_uniform(image, canvas_scale, canvas_color)
            if resolution is not None:
                image = resolution_scaling(image, resolution)

        if half_res:
            image = cv.resize(
                image,
                None,
                fx=2,
                fy=2,
                interpolation=cv.INTER_CUBIC,
            )

        return image
