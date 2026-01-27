import math
from contextlib import nullcontext
from functools import cache

import cv2 as cv
import rawpy
from spectral_film_lut.film_spectral import FilmSpectral
from spectral_film_lut.utils import *

from raw2film import effects
from raw2film.color_processing import calc_exposure
from raw2film.effects import add_canvas, add_canvas_uniform, chroma_nr_filter


def raw_to_linear(src, half_size=True):
    # convert raw file to linear data
    with rawpy.imread(src) as raw:
        # noinspection PyUnresolvedReferences
        rgb = raw.postprocess(output_color=rawpy.ColorSpace(5), gamma=(1, 1), output_bps=16, no_auto_bright=True,
                              use_camera_wb=False, use_auto_wb=False, half_size=half_size,
                              demosaic_algorithm=rawpy.DemosaicAlgorithm(2), four_color_rgb=False, )

    return rgb


def crop_rotate_zoom(image, frame_width=36, frame_height=24, rotation=0, zoom=1, rotate_times=0, flip=False, **kwargs):
    image = effects.crop_image(image, 1, aspect=frame_width / frame_height, flip=flip)
    if rotation:
        image = effects.rotate(image, rotation)
    image = effects.crop_image(image, zoom, aspect=frame_width / frame_height)
    image = np.rot90(image, k=rotate_times)

    return image

@cache
def create_lut_cached(*args, **kwargs):
    return create_lut(*args, **kwargs)


def process_image(image, negative_film, grain_size, grain_sigma, frame_width=36, frame_height=24, fast_mode=False,
                  print_film=None, halation=True, sharpness=True, grain=2, resolution=None, metadata=None,
                  measure_time=False, semaphore=None, canvas_mode="No", highlight_burn=0, burn_scale=50, chroma_nr=0,
                  **kwargs):
    if measure_time:
        kwargs['measure_time'] = True
        start = time.time()
    exp_comp = calc_exposure(image, metadata=metadata, **kwargs)
    if "exp_comp" in kwargs:
        kwargs["exp_comp"] += exp_comp
    else:
        kwargs["exp_comp"] = exp_comp

    image = crop_rotate_zoom(image, frame_width, frame_height, **kwargs)

    if not fast_mode and chroma_nr:
        # TODO: support cuda
        image = chroma_nr_filter(image, chroma_nr)

    if resolution is not None:
        h, w = image.shape[:2]
        scaling_factor = resolution / max(w, h)
        if scaling_factor < 1:
            image = cv.resize(image, (int(w * scaling_factor), int(h * scaling_factor)), interpolation=cv.INTER_AREA)
        elif scaling_factor > 1 and not fast_mode:
            image = cv.resize(image, (int(w * scaling_factor), int(h * scaling_factor)),
                              interpolation=cv.INTER_LANCZOS4)

    if fast_mode:
        if image.dtype != xp.uint16:
            image = xp.array(image, xp.float32)
            image_max = image.max()
            factor = 65535
            if image_max > 65535:
                adjustment = 65535. / image_max
                factor /= adjustment
                if "exp_comp" in kwargs:
                    kwargs["exp_comp"] -= math.log2(adjustment)
                else:
                    kwargs["exp_comp"] = math.log2(adjustment)
            image = xp.round(xp.sqrt(image / factor) * 65535).astype(xp.uint16)
            kwargs["gamma"] = 2
        mode = 'full'
    else:
        image = image.astype(xp.float32) / 65535
        mode = 'print'


    if not fast_mode:
        lock = semaphore if semaphore is not None and cuda_available else nullcontext()
        with lock:
            image = xp.asarray(image)
            scale = max(image.shape) / max(frame_width, frame_height)  # pixels per mm

            if halation:
                halation_func = lambda x: effects.halation(x, scale, **kwargs)
            else:
                halation_func = None

            transform = FilmSpectral.generate_conversion(negative_film, mode='negative', adx=False,
                                                         input_colourspace=None, halation_func=halation_func, **kwargs)
            image = transform(image)

            if sharpness and negative_film.mtf is not None:
                start_sub = time.time()
                image = effects.film_sharpness(image, negative_film, scale)
                if measure_time:
                    print(f"{'sharpness':28} {time.time() - start_sub:.4f}s {image.dtype} {image.shape} {type(image)}")

            if grain and negative_film.rms_density is not None:
                start_sub = time.time()
                image = effects.apply_grain(image, negative_film, scale, grain_size_mm=grain_size / 1000,
                                            grain_sigma=grain_sigma, bw_grain=grain == 1)
                if measure_time:
                    print(f"{'grain':28} {time.time() - start_sub:.4f}s {image.dtype} {image.shape} {type(image)}")

            if highlight_burn and (print_film is not None or negative_film.density_measure in ["status_m", "bw"]):
                start_sub = time.time()
                image = effects.burn(image, negative_film, highlight_burn, burn_scale)
                if measure_time:
                    print(f"{'burn':28} {time.time() - start_sub:.4f}s {image.dtype} {image.shape} {type(image)}")

            image = xp.clip(image * 2, 0, 1)
            image *= 2 ** 16 - 1
            image = image.astype(xp.uint16)

    if measure_time:
        start_sub = time.time()
    if "exp_comp" in kwargs:
        kwargs["exp_comp"]  = round(kwargs["exp_comp"], ndigits=1)
    lut = create_lut_cached(negative_film, print_film, mode=mode, input_colourspace=None, adx=False,
                     cube=False, adx_scaling=2, **kwargs)
    lut = (lut * (2 ** 16 - 1)).astype(xp.uint16)
    if measure_time:
        print(f"{'create lut':28} {time.time() - start_sub:.4f}s")
        start_sub = time.time()
    if image.shape[-1] == 1:
        image = image.repeat(3, -1)

    height, width, _ = image.shape
    if cuda_available:
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

    return image
