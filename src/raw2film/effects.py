import math
import time
from functools import cache

import cv2
import cv2 as cv
import lensfunpy
import numpy as np
from lensfunpy import util as lensfunpy_util
from numba import njit
from spectral_film_lut.utils import multi_channel_interp


def lens_correction(rgb, metadata, cam, lens):
    """Apply lens correction using lensfunpy."""
    # noinspection PyUnresolvedReferences
    rgb = rgb.astype(np.float64)
    if lens is None or cam is None:
        return rgb
    try:
        focal_length = metadata['EXIF:FocalLength']
        aperture = metadata['EXIF:FNumber']
    except KeyError:
        return rgb
    height, width = rgb.shape[0], rgb.shape[1]
    # noinspection PyUnresolvedReferences
    mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(focal_length, aperture, pixel_format=np.float64)
    undistorted_cords = mod.apply_geometry_distortion()
    rgb = np.clip(lensfunpy_util.remap(rgb, undistorted_cords), a_min=0, a_max=None)
    mod.apply_color_modification(rgb)
    return rgb


def rotate(rgb, degrees):
    if degrees:
        input_height, input_width = rgb.shape[:2]
        image_center = tuple(np.array(rgb.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, -degrees, 1.0)
        rgb = cv2.warpAffine(rgb, rot_mat, rgb.shape[1::-1], flags=cv2.INTER_LINEAR)
        aspect_ratio = input_height / input_width
        angle = math.fabs(degrees) * math.pi / 180

        if aspect_ratio < 1:
            total_height = input_height
            aspect_ratio = 1 / aspect_ratio
            switch = True
        else:
            switch = False
            total_height = input_width

        w = total_height / (aspect_ratio * math.sin(angle) + math.cos(angle))
        h = w * aspect_ratio
        if switch:
            w, h = h, w
        crop_height = int((rgb.shape[0] - h) // 2)
        crop_width = int((rgb.shape[1] - w) // 2)
        rgb = rgb[crop_height: rgb.shape[0] - crop_height, crop_width: rgb.shape[1] - crop_width]
    return rgb


def crop_image(rgb, zoom=1, aspect=1.5):
    """Crops rgb data to aspect ratio."""
    x, y, c = rgb.shape
    if x > y:
        if x > aspect * y:
            rgb = rgb[round(x / 2 - y * aspect / 2): round(x / 2 + y * aspect / 2), :, :]
        else:
            rgb = rgb[:, round(y / 2 - x / aspect / 2): round(y / 2 + x / aspect / 2), :]
    elif y > aspect * x:
        rgb = rgb[:, round(y / 2 - x * aspect / 2): round(y / 2 + x * aspect / 2), :]
    else:
        rgb = rgb[round(x / 2 - y / aspect / 2): round(x / 2 + y / aspect / 2), :, :]

    if zoom > 1:
        x, y, c = rgb.shape
        zoom_factor = (zoom - 1) / (2 * zoom)
        x = round(zoom_factor * x)
        y = round(zoom_factor * y)
        rgb = rgb[x: -x, y: -y, :]

    return rgb


def gaussian_filter(input, sigma=1.):
    """Compute gaussian filter"""
    return cv.GaussianBlur(input, ksize=(0, 0), sigmaX=sigma)


def gaussian_blur(rgb, sigma=1.):
    """Applies gaussian blur per channel of rgb image."""
    return cv.GaussianBlur(rgb, ksize=(0, 0), sigmaX=sigma)


def mtf_curve(mtf_data, lowest=1, highest=200):
    mtf_data = {k: v for k, v in mtf_data.items() if k >= lowest}
    if lowest not in mtf_data:
        mtf_data[lowest] = 1
    mtf_data = {k: v for k, v in sorted(mtf_data.items(), key=lambda item: item[0])}
    frequencies = np.array(list(mtf_data.keys()), dtype=np.float32)
    values = np.array(list(mtf_data.values()), dtype=np.float32)
    if max(frequencies) < highest:
        ax, bx = frequencies[-2:]
        ay, by = values[-2:]
        extrapolation = lambda x: ay * np.e ** (np.log(by / ay) / (bx - ax) * (x - ax))
        number = math.ceil((highest - max(frequencies)) / 30)
        extrapol_freq = (np.array(list(range(number))) + 1) / number * (highest - max(frequencies)) + max(frequencies)
        extrapol_val = np.vectorize(extrapolation)(extrapol_freq)
        frequencies = np.hstack((frequencies, extrapol_freq))
        values = np.hstack((values, extrapol_val))
    frequencies = np.log10(frequencies)
    values = np.log10(values)
    return lambda x: 10 ** np.interp(np.log10(x + 0.00001), frequencies, values, right=-1000000000000000)


def mtf_kernel(mtf_data, frequency, f_shift):
    highest = frequency.max()
    mtf = mtf_curve(mtf_data, highest=highest)
    factors = mtf(frequency)
    shift = f_shift * factors
    kernel = np.fft.ifft2(np.fft.ifftshift(shift)).real
    kernel /= np.sum(kernel)
    return kernel


def film_sharpness(rgb, stock, scale):
    size = int(scale // 2)
    if not size % 2:
        size += 1
    if size < 13:
        size = 13

    kernel = np.zeros((size, size))
    kernel[size // 2, size // 2] = 1
    f = np.fft.fft2(kernel)
    f_shift = np.fft.fftshift(f)

    frequency = np.abs(np.fft.fftfreq(size, 1 / scale)[:, None])
    frequency_x, frequency_y = np.meshgrid(frequency, frequency)
    frequency = np.fft.fftshift(np.sqrt(frequency_x ** 2 + frequency_y ** 2))

    if hasattr(stock, 'red_mtf') and hasattr(stock, 'green_mtf') and hasattr(stock, 'blue_mtf'):
        red_kernel = mtf_kernel(stock.red_mtf, frequency, f_shift)
        green_kernel = mtf_kernel(stock.green_mtf, frequency, f_shift)
        blue_kernel = mtf_kernel(stock.blue_mtf, frequency, f_shift)
        kernel = np.dstack((red_kernel, green_kernel, blue_kernel))
    else:
        kernel = mtf_kernel(stock.mtf, frequency, f_shift)

    rgb = cv.filter2D(rgb, ddepth=-1, kernel=kernel)

    return rgb

@njit
def exponential_blur_kernel(size):
    radius = size / 2
    size = 2 * math.floor(math.ceil(size) / 2) + 1
    center = math.ceil(size / 2)
    kernel = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            dist = (i + 1 - center) ** 2 + (j + 1 - center) ** 2
            if not dist:
                kernel[i, j] = 1
            else:
                kernel[i, j] = (1 / dist) * max((radius - np.sqrt(dist)) / radius, 0)
    kernel /= np.sum(kernel)

    return kernel


@cache
def gaussian_noise_cache(shape):
    return np.random.default_rng().standard_normal(shape, dtype=np.float32)


def gaussian_noise(shape):
    assert len(shape) == 3
    noise_size = ((max(shape[:2]) + 100) // 1024 + 1) * 1024
    noise_map = gaussian_noise_cache((noise_size, noise_size))
    offsets = np.random.randint([0, 0], [noise_size - shape[0] + 1, noise_size - shape[1] + 1], size=(shape[2], 2))
    noise = np.stack([noise_map[x:shape[0] + x, y:shape[1] + y] for x, y in offsets], axis=-1)
    return noise


def grain(rgb, stock, scale, grain_size=0.002, d_factor=6, **kwargs):
    # compute scaling factor of exposure rms in regard to measuring device size
    std_factor = math.sqrt(math.pi) * 0.024 * scale / d_factor
    noise = gaussian_noise(rgb.shape)
    xps = [(stock.red_rms_density + 0.25) / d_factor, (stock.green_rms_density + 0.25) / d_factor,
           (stock.blue_rms_density + 0.25) / d_factor]
    fps = [stock.red_rms * std_factor, stock.green_rms * std_factor, stock.blue_rms * std_factor]
    noise *= multi_channel_interp(rgb, xps, fps)
    factor = scale * grain_size * 2 * math.sqrt(math.pi)
    if factor > 1:
        noise = gaussian_blur(noise, scale * grain_size)
    rgb += noise
    return rgb


@njit
def apply_halation_inplace(rgb, blured, color_factors):
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            for c in range(rgb.shape[2]):
                rgb[i, j, c] += blured[i, j, c] * color_factors[c]
                rgb[i, j, c] /= (color_factors[c] + 1.0)

def halation(rgb, scale, halation_size=1, halation_red_factor=1., halation_green_factor=0.4, halation_blue_factor=0.,
             halation_intensity=1, **kwargs):
    kernel = exponential_blur_kernel(scale / 4 * halation_size)
    blured = cv2.filter2D(rgb, -1, kernel)
    color_factors = halation_intensity * np.array([halation_red_factor, halation_green_factor, halation_blue_factor],
                                                  dtype=np.float32)
    apply_halation_inplace(rgb, blured, color_factors)
    return rgb
