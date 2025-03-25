import math

import cv2
import cv2 as cv
import lensfunpy
import numpy as np
import torch
from lensfunpy import util as lensfunpy_util
from raw2film import utils, data


def lens_correction(rgb, metadata):
    """Apply lens correction using lensfunpy."""
    # noinspection PyUnresolvedReferences
    db = lensfunpy.Database()
    try:
        cam = db.find_cameras(metadata['EXIF:Make'], metadata['EXIF:Model'], loose_search=True)[0]
        lens = db.find_lenses(cam, metadata['EXIF:LensMake'], metadata['EXIF:LensModel'], loose_search=True)[0]
    except (KeyError, IndexError):
        cam, lens = utils.find_data(metadata)
        if lens and cam:
            cam = db.find_cameras(*cam, loose_search=True)[0]
            lens = db.find_lenses(cam, *lens, loose_search=True)[0]
        else:
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
        else:
            total_height = input_width

        w = total_height / (aspect_ratio * math.sin(angle) + math.cos(angle))
        h = w * aspect_ratio
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
    if max(mtf_data.keys()) < highest:
        ax, ay = list(mtf_data.items())[-1]
        bx, by = list(mtf_data.items())[-2]
        m = (np.log10(ay) - np.log10(by)) / (np.log10(ax) - np.log10(bx))
        mtf_data[highest] = highest ** m * (ay / ax ** m)
    frequencies = np.array(list(mtf_data.keys()), dtype=np.float32)
    values = np.array(list(mtf_data.values()), dtype=np.float32)
    return lambda x: np.interp(x, frequencies, values)


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

    rgb = cv.filter2D(rgb, -1, kernel)

    return rgb


def exponential_blur(rgb, size):
    radius = size / 2
    size = 2 * math.floor(math.ceil(size) / 2) + 1
    center = math.ceil(size / 2)
    kernel = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            dist = (i + 1 - center) ** 2 + (j + 1 - center) ** 2
            if not dist:
                dist = 1
            kernel[i, j] = (1 / dist) * max((radius - np.sqrt(dist)) / radius, 0)

    kernel /= np.sum(kernel)

    return cv.filter2D(rgb, -1, kernel)


def grain(rgb, stock, scale, grain_size=0.002, smoothing=False, **kwargs):
    # compute scaling factor of exposure rms in regard to measuring device size
    std_factor = math.sqrt(math.pi) * 0.024 * scale / 6
    noise = np.array(torch.empty(rgb.shape, dtype=torch.float32).normal_(), dtype=np.float32)
    # TODO: properly scale intensity with grain size
    red_rms = np.interp(rgb[..., 0], stock.red_rms_density, stock.red_rms * std_factor)
    green_rms = np.interp(rgb[..., 1], stock.green_rms_density, stock.green_rms * std_factor)
    blue_rms = np.interp(rgb[..., 2], stock.blue_rms_density, stock.blue_rms * std_factor)
    rms = np.stack([red_rms, green_rms, blue_rms], axis=-1, dtype=rgb.dtype)
    noise = np.multiply(noise, rms)
    if scale * grain_size * (1 + smoothing * 2) * 2 * math.sqrt(math.pi) > 1:
        noise = gaussian_blur(noise, scale * grain_size * (1 + smoothing * 2)) * (
                    scale * grain_size * 2 * math.sqrt(math.pi))
    rgb += noise
    return rgb


def halation(rgb, scale):
    # TODO: check scale and factors
    blured = exponential_blur(rgb, scale / 4)
    color_factors = np.dot(np.array([1.2, 0.5, 0], dtype=np.float32), data.REC709_TO_XYZ)
    rgb += np.multiply(blured, color_factors)
    rgb = np.divide(rgb, color_factors + 1)
    return rgb
