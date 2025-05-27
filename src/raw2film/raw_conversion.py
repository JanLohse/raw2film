from contextlib import nullcontext

import cv2 as cv
import ffmpeg
import rawpy
from raw2film import effects
from raw2film.color_processing import calc_exposure, xyz_to_srgb, xyz_to_displayP3
from raw2film.effects import add_canvas, add_canvas_uniform
from spectral_film_lut.film_spectral import FilmSpectral
from spectral_film_lut.utils import *


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


def process_image(image, negative_film, grain_size, frame_width=36, frame_height=24, fast_mode=False, print_film=None,
                  halation=True, sharpness=True, grain=True, resolution=None, metadata=None, measure_time=False,
                  full_cuda=True, semaphore=None, canvas_mode="No", highlight_burn=0, burn_scale=50, **kwargs):
    if measure_time:
        kwargs['measure_time'] = True
        start = time.time()
    exp_comp = calc_exposure(image, metadata=metadata, **kwargs)
    if "exp_comp" in kwargs:
        kwargs["exp_comp"] += exp_comp
    else:
        kwargs["exp_comp"] = exp_comp

    if fast_mode:
        if image.dtype != xp.uint16 and not cuda_available:
            image = image.astype(xp.float32)
            image_max = image.max()
            if image_max > 65535:
                factor = 65535. / image_max
                image *= factor
                if "exp_comp" in kwargs:
                    kwargs["exp_comp"] -= xp.log2(factor)
                else:
                    kwargs["exp_comp"] = xp.log2(factor)
            image = image.astype(xp.uint16)
        elif cuda_available:
            image = image.astype(xp.float32) / 65535
        mode = 'full'
    else:
        image = image.astype(xp.float32) / 65535
        mode = 'print'

    image = crop_rotate_zoom(image, frame_width, frame_height, **kwargs)

    if resolution is not None:
        h, w = image.shape[:2]
        scaling_factor = resolution / max(w, h)
        if scaling_factor < 1:
            image = cv.resize(image, (int(w * scaling_factor), int(h * scaling_factor)), interpolation=cv.INTER_AREA)
        elif scaling_factor > 1:
            image = cv.resize(image, (int(w * scaling_factor), int(h * scaling_factor)),
                              interpolation=cv.INTER_LANCZOS4)
    if fast_mode:
        h, w = image.shape[:2]
        h //= 2
        w //= 2
        image = cv.resize(image, (w, h), interpolation=cv.INTER_AREA)
        if not cuda_available:
            image = ((image / 65535) ** 0.25 * 65535).astype(xp.uint16)
            kwargs["gamma"] = 4
            kwargs["lut_size"] = 17
    else:
        lock = semaphore if semaphore is not None and cuda_available else nullcontext()
        with lock:
            image = xp.asarray(image)
            scale = max(image.shape) / max(frame_width, frame_height)  # pixels per mm

            if halation:
                halation_func = lambda x: effects.halation(x, scale, **kwargs)
            else:
                halation_func = None

            transform, d_factor = FilmSpectral.generate_conversion(negative_film, mode='negative',
                                                                   input_colourspace=None,
                                                                   halation_func=halation_func, **kwargs)
            image = transform(image)

            if sharpness:
                start_sub = time.time()
                image = effects.film_sharpness(image, negative_film, scale)
                if measure_time:
                    print(f"{'sharpness':28} {time.time() - start_sub:.4f}s {image.dtype} {image.shape} {type(image)}")

            if grain:
                start_sub = time.time()
                image = effects.grain(image, negative_film, scale, grain_size=grain_size, d_factor=d_factor)
                if measure_time:
                    print(f"{'grain':28} {time.time() - start_sub:.4f}s {image.dtype} {image.shape} {type(image)}")

            if highlight_burn:
                start_sub = time.time()
                image = effects.burn(image, negative_film, highlight_burn, burn_scale, d_factor)
                if measure_time:
                    print(f"{'burn':28} {time.time() - start_sub:.4f}s {image.dtype} {image.shape} {type(image)}")

            if not cuda_available or not full_cuda:
                image = xp.clip(image, 0, 1)
                image *= 2 ** 16 - 1
                image = to_numpy(image).astype(xp.uint16)

    start_sub = time.time()
    if cuda_available and full_cuda:
        if "output_colourspace" in kwargs and kwargs["output_colourspace"] == "Display P3":
            output_transform = xyz_to_displayP3
        else:
            output_transform = xyz_to_srgb

        if highlight_burn and fast_mode:
            transform, d_factor = FilmSpectral.generate_conversion(negative_film, print_film, mode="negative",
                                                                   input_colourspace=None,**kwargs)
            image = transform(image)
            image = effects.burn(image, negative_film, highlight_burn, burn_scale, d_factor)
            mode = "print"

        transform, d_factor = FilmSpectral.generate_conversion(negative_film, print_film, mode=mode,
                                                               input_colourspace=None,
                                                               output_transform=output_transform, **kwargs)
        image = transform(image)
    else:
        lut = create_lut(negative_film, print_film, name=str(time.time()), mode=mode, input_colourspace=None, **kwargs)

        if image.shape[-1] == 1:
            image = image.repeat(3, -1)

        height, width, _ = image.shape
        process = run_async(
            ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb48', s='{}x{}'.format(width, height)).filter('lut3d',
                                                                                                              file=lut).output(
                'pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1, loglevel='quiet'), pipe_stdin=True,
            pipe_stdout=True)
        process.stdin.write(image.tobytes())
        process.stdin.close()
        image = process.stdout.read(width * height * 3)
        process.wait()
        os.remove(lut)
        image = np.frombuffer(image, np.uint8).reshape([height, width, 3])
    if measure_time:
        print(f"{'lut':28} {time.time() - start_sub:.4f}s")
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
