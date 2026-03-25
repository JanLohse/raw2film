import time
from contextlib import nullcontext
from functools import cache

import cv2 as cv
import numpy as np
import rawpy
from spectral_film_lut.film_spectral import FilmSpectral
from spectral_film_lut.utils import (
    CUDA_AVAILABLE,
    apply_lut_tetrahedral_int,
    create_lut,
    to_numpy,
    xp,
)

from raw2film import effects
from raw2film.color_processing import calc_exposure
from raw2film.effects import add_canvas, add_canvas_uniform, chroma_nr_filter


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
    image,
    frame_width=36,
    frame_height=24,
    rotation=0,
    zoom=1,
    rotate_times=0,
    flip=False,
    **kwargs,
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
    image,
    negative_film,
    grain_size,
    grain_sigma,
    frame_width=36,
    frame_height=24,
    print_film=None,
    halation=True,
    sharpness=True,
    grain=2,
    resolution: None | tuple[int, int] = None,
    metadata=None,
    measure_time=False,
    semaphore=None,
    canvas_mode="No",
    highlight_burn=0,
    burn_scale=50,
    chroma_nr=0,
    double_upscale=False,
    **kwargs,
):
    """
    The full image processing pipeline that converts from linear XYZ to a display
    referred image with film emulation applied.
    """
    if measure_time:
        kwargs["measure_time"] = True
        start = time.time()
    exp_comp = calc_exposure(image, metadata=metadata, **kwargs)
    if "exp_comp" in kwargs:
        kwargs["exp_comp"] += exp_comp
    else:
        kwargs["exp_comp"] = exp_comp

    image = crop_rotate_zoom(image, frame_width, frame_height, **kwargs)

    if chroma_nr:
        # TODO: support cuda
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

    lock = semaphore if semaphore is not None and CUDA_AVAILABLE else nullcontext()
    with lock:
        image = xp.asarray(image)
        scale = max(image.shape) / max(frame_width, frame_height)  # pixels per mm

        if halation:

            def halation_func(x):
                return effects.halation(x, scale, **kwargs)
        else:
            halation_func = None

        transform = FilmSpectral.generate_conversion(
            negative_film,
            mode="negative",
            adx=False,
            input_colourspace=None,
            halation_func=halation_func,
            **kwargs,
        )
        image = transform(image)

        if sharpness and negative_film.mtf is not None:
            start_sub = time.time()
            image = effects.film_sharpness(image, negative_film, scale)
            if measure_time:
                print(
                    f"{'sharpness':28} {time.time() - start_sub:.4f}s {image.dtype} "
                    f"{image.shape} {type(image)}"
                )

        if grain and negative_film.rms_density is not None:
            start_sub = time.time()
            image = effects.apply_grain(
                image,
                negative_film,
                scale,
                grain_size_mm=grain_size / 1000,
                grain_sigma=grain_sigma,
                bw_grain=grain == 1,
            )
            if measure_time:
                print(
                    f"{'grain':28} {time.time() - start_sub:.4f}s {image.dtype} "
                    f"{image.shape} {type(image)}"
                )

        if highlight_burn and (
            print_film is not None
            or negative_film.density_measure in ["status_m", "bw"]
        ):
            start_sub = time.time()
            image = effects.burn(image, negative_film, highlight_burn, burn_scale)
            if measure_time:
                print(
                    f"{'burn':28} {time.time() - start_sub:.4f}s {image.dtype} "
                    f"{image.shape} {type(image)}"
                )

        image = xp.clip(image * 2, 0, 1)
        image *= 2**16 - 1
        image = image.astype(xp.uint16)

    if measure_time:
        start_sub = time.time()
    if "exp_comp" in kwargs:
        kwargs["exp_comp"] = round(kwargs["exp_comp"], ndigits=1)
    lut = create_lut_cached(
        negative_film,
        print_film,
        mode="print",
        input_colourspace=None,
        adx=False,
        cube=False,
        adx_scaling=2,
        **kwargs,
    )
    lut = (lut * (2**16 - 1)).astype(xp.uint16)
    if measure_time:
        print(f"{'create lut':28} {time.time() - start_sub:.4f}s")
        start_sub = time.time()
    if image.shape[-1] == 1:
        image = image.repeat(3, -1)

    height, width, _ = image.shape
    if CUDA_AVAILABLE:
        image = apply_lut_tetrahedral_int(to_numpy(image), lut)
        # TODO fix image = to_numpy(run_lut_cuda(xp.asarray(image), xp.asarray(lut)))
    else:
        image = apply_lut_tetrahedral_int(image, lut)
    if measure_time:
        print(f"{'apply lut':28} {time.time() - start_sub:.4f}s")
        print(f"{'total':28} {time.time() - start:.4f}s")

    if canvas_mode and canvas_mode is not None and canvas_mode != "No":
        if "white" in canvas_mode:
            kwargs["canvas_color"] = [255, 255, 255]
        elif "black" in canvas_mode:
            kwargs["canvas_color"] = [0, 0, 0]
        else:
            kwargs["canvas_color"] = [128, 128, 128]
        if "canvas_scale" not in kwargs:
            kwargs["canvas_scale"] = 1
        if "Proportional" in canvas_mode:
            kwargs["canvas_ratio"] = image.shape[1] / image.shape[0]
            image = add_canvas(image, **kwargs)
        elif "Fixed" in canvas_mode and "canvas_ratio" in kwargs:
            image = add_canvas(image, **kwargs)
        elif "Uniform" in canvas_mode:
            image = add_canvas_uniform(image, **kwargs)

    if double_upscale:
        image = cv.resize(
            image,
            None,
            fx=2,
            fy=2,
            interpolation=cv.INTER_CUBIC,
        )

    return image
