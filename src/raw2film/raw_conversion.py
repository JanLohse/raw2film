"""
The main processing pipeline for RAW images.
"""

from functools import cache
from typing import Literal

import cv2 as cv
import numpy as np
import rawpy
from spectral_film_lut.color_space import GAMMA_KEYS
from spectral_film_lut.film_spectral import FilmSpectral
from spectral_film_lut.utils import (
    apply_lut_tetrahedral_int,
    create_lut,
    film_conversion,
)

from raw2film import effects
from raw2film.color_processing import calc_exposure
from raw2film.effects import add_canvas, add_canvas_uniform, chroma_nr_filter

CANVAS_MODES = Literal[
    "No",
    "Proportional white",
    "Proportional black",
    "Uniform white",
    "Uniform black",
    "Fixed white",
    "Fixed black",
]
"""Available canvas modes."""


def raw_to_linear(src, half_size=True):
    """Load linear raw data using rawpy."""
    # convert raw file to linear data
    with rawpy.imread(src) as raw:
        # noinspection PyUnresolvedReferences
        rgb = raw.postprocess(
            output_color=rawpy.ColorSpace(5),
            gamma=(1, 1),
            output_bps=16,
            no_auto_bright=True,
            use_camera_wb=False,
            use_auto_wb=False,
            half_size=half_size,
            demosaic_algorithm=rawpy.DemosaicAlgorithm(2),
            four_color_rgb=False,
        )

    return rgb


def crop_rotate_zoom(
    image: np.ndarray,
    frame_width: int | float = 36,
    frame_height: int | float = 24,
    rotation: float = 0.0,
    zoom: float = 1.0,
    rotate_times: int = 0,
    flip: bool = False,
):
    """Apply cropping, rotation, and a zoom to an image."""
    image = effects.crop_image(image, 1, aspect=frame_width / frame_height, flip=flip)
    if rotation:
        image = effects.rotate(image, rotation)
    image = effects.crop_image(image, zoom, aspect=frame_width / frame_height)
    image = np.rot90(image, k=rotate_times)

    return image


@cache
def create_lut_cached(*args, **kwargs):
    """Cache LUTs for specific settings."""
    return create_lut(*args, **kwargs)


def process_image(
    image: np.ndarray,
    negative_film: FilmSpectral,
    grain_size: float,
    grain_sigma: float,
    frame_width: int | float = 36,
    frame_height: int | float = 24,
    print_film: FilmSpectral | None = None,
    halation: bool = True,
    sharpness: bool = True,
    grain: int = 2,
    resolution: None | tuple[int, int] = None,
    metadata: dict | None = None,
    canvas_mode: CANVAS_MODES = "No",
    canvas_scale: float = 1.0,
    canvas_ratio: float = 1.0,
    highlight_burn: float = 0.0,
    burn_scale: float = 50.0,
    chroma_nr: int = 0,
    double_upscale: bool = False,
    exp_comp: float = 0.0,
    rotation: float = 0.0,
    zoom: float = 1.0,
    rotate_times: int = 0,
    flip: bool = False,
    red_light: float = 0.0,
    green_light: float = 0.0,
    blue_light: float = 0.0,
    halation_size: float = 1.0,
    halation_green_factor: float = 0.4,
    projector_kelvin: int = 6500,
    halation_intensity: float = 1.0,
    shadow_comp: float = 0.0,
    white_comp: bool = True,
    sat_adjust: float = 1.0,
    gamma_func: GAMMA_KEYS = "sRGB",
    exp_kelvin: int = 6500,
    tint: float = 0.0,
    inversion_gamma: float = 4.0,
    idealized_curve: bool = False,
    inversion: bool = False,
    **_,
):
    """
    The full image processing pipeline that converts from linear XYZ to a display
    referred image with film emulation applied.
    """
    exp_comp += calc_exposure(image, metadata=metadata)

    image = crop_rotate_zoom(
        image, frame_width, frame_height, rotation, zoom, rotate_times, flip
    )

    if chroma_nr:
        image = chroma_nr_filter(image, chroma_nr)

    if resolution is not None:
        h, w = image.shape[:2]
        h_factor = resolution[0] / h
        w_factor = resolution[1] / w
        scaling_factor = min(h_factor, w_factor)
        if scaling_factor < 1:
            image = cv.resize(
                image,
                (int(w * scaling_factor), int(h * scaling_factor)),
                interpolation=cv.INTER_AREA,
            )
        elif scaling_factor > 1:
            image = cv.resize(
                image,
                (int(w * scaling_factor), int(h * scaling_factor)),
                interpolation=cv.INTER_LANCZOS4,
            )

    scale = max(image.shape) / max(frame_width, frame_height)  # pixels per mm

    if halation:

        def halation_func(x: np.ndarray) -> np.ndarray:
            return effects.halation(
                x,
                scale,
                halation_size=halation_size,
                halation_green_factor=halation_green_factor,
                halation_intensity=halation_intensity,
            )
    else:
        halation_func = None

    image = film_conversion(
        image,
        negative_film,
        mode="negative",
        adx_coding=True,  # TODO: remove unnecessary ADX endcoding
        input_colorspace=None,
        exp_kelvin=exp_kelvin,
        tint=tint,
        exp_comp=exp_comp,
        halation_func=halation_func,
    )

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
            adx=True,  # TODO: remove unnecessary ADX endcoding
        )

    if highlight_burn and (
        print_film is not None or negative_film.density_measure in ["status_m", "bw"]
    ):
        image = effects.burn(image, negative_film, highlight_burn, burn_scale)

    image = np.clip(image * 2, 0, 1)
    image *= 2**16 - 1
    image = image.astype(np.uint16)

    lut = create_lut_cached(
        negative_film,
        print_film,
        mode="print",
        input_colorspace=None,
        adx_coding=True,  # TODO: remove unnecessary ADX endcoding
        cube=False,
        adx_scaling=2,
        red_light=red_light,
        green_light=green_light,
        blue_light=blue_light,
        projector_kelvin=projector_kelvin,
        shadow_comp=shadow_comp,
        white_comp=white_comp,
        sat_adjust=sat_adjust,
        gamma_func=gamma_func,
        inversion_gamma=inversion_gamma,
        idealized_curve=idealized_curve,
        inversion=inversion,
    )
    lut = (lut * (2**16 - 1)).astype(np.uint16)
    if image.shape[-1] == 1:
        image = image.repeat(3, -1)

    image = apply_lut_tetrahedral_int(image, lut)

    if canvas_mode != "No":
        # TODO: fix canvas preview
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

    if double_upscale:
        image = cv.resize(
            image,
            None,
            fx=2,
            fy=2,
            interpolation=cv.INTER_CUBIC,
        )

    return image
