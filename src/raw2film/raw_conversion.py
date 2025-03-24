import os
import time

import exiftool
import ffmpeg
import numpy as np
import rawpy
from raw2film import effects
from spectral_film_lut.film_spectral import FilmSpectral
from spectral_film_lut.utils import create_lut, run_async


def raw_to_linear(src, half_size=True, lens_correction=False, metadata=False):
    # convert raw file to linear data
    with rawpy.imread(src) as raw:
        # noinspection PyUnresolvedReferences
        rgb = raw.postprocess(output_color=rawpy.ColorSpace(5), gamma=(1, 1), output_bps=16, no_auto_bright=True,
                              use_camera_wb=False, use_auto_wb=False, half_size=half_size,
                              demosaic_algorithm=rawpy.DemosaicAlgorithm(11), four_color_rgb=True, )

    exp_comp = 0

    # TODO: lower resolution for fast mode

    if lens_correction or metadata:
        with exiftool.ExifToolHelper() as et:
            meta = et.get_metadata(src)[0]

    if lens_correction:
        rgb = effects.lens_correction(rgb.astype(np.float64), meta)
    if metadata:
        return rgb, meta
    else:
        return rgb


def crop_rotate_zoom(image, frame_width=36, frame_height=24, rotation=0, zoom=1, **kwargs):
    image = effects.crop_image(image, 1, aspect=frame_width / frame_height)
    if rotation:
        image = effects.rotate(image, rotation)
    image = effects.crop_image(image, zoom, aspect=frame_width / frame_height)

    return image


def process_image(image, negative_film, frame_width=36, frame_height=24, rotation=0, zoom=1, fast_mode=False,
                  print_film=None, **kwargs):
    # TODO: auto exposure

    # TODO: when lens correction and full mode dark image
    if fast_mode:
        if image.dtype != np.uint16:
            image_max = image.max()
            if image_max > 65535:
                factor = 65535. / image_max
                image *= factor
                if "exp_comp" in kwargs:
                    kwargs["exp_comp"] -= np.log2(factor)
                else:
                    kwargs["exp_comp"] = np.log2(factor)
            image = image.astype(np.uint16)
        mode = 'full'
    else:
        image = image.astype(np.float32) / 65535
        mode = 'print'

    if not fast_mode:
        image = crop_rotate_zoom(image, frame_width, frame_height, rotation, zoom)

        scale = max(image.shape) / max(frame_width, frame_height)  # pixels per mm

        image = effects.halation(image, scale)

        transform = FilmSpectral.generate_conversion(negative_film, mode='negative', input_colourspace=None, **kwargs)
        image = transform(image)

        image = effects.film_sharpness(image, negative_film, scale)

        image = effects.grain(image, negative_film, scale)

        image = np.clip(image, 0, 1)
        image *= 2 ** 16 - 1
        image = image.astype(np.uint16)

    lut = create_lut(negative_film, print_film, name=str(time.time()), mode=mode, input_colourspace=None, **kwargs)

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

    return np.frombuffer(image, np.uint8).reshape([height, width, 3])
