import math

import cv2 as cv
import lensfunpy
import numpy as np
from lensfunpy import util as lensfunpy_util
from raw2film import utils
from scipy import ndimage


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


def rotate(rotation, rgb):
    degrees = rotation % 360

    while degrees > 45:
        rgb = np.rot90(rgb, k=1)
        degrees -= 90
    if degrees:
        input_height = rgb.shape[0]
        input_width = rgb.shape[1]
        rgb = ndimage.rotate(rgb, angle=degrees, reshape=True)
        aspect_ratio = input_height / input_width
        rotated_ratio = rgb.shape[0] / rgb.shape[1]
        angle = math.fabs(degrees) * math.pi / 180

        if aspect_ratio < 1:
            total_height = input_height / rotated_ratio
        else:
            total_height = input_width

        w = total_height / (aspect_ratio * math.sin(angle) + math.cos(angle))
        h = w * aspect_ratio
        crop_height = int((rgb.shape[0] - h) // 2)
        crop_width = int((rgb.shape[1] - w) // 2)
        rgb = rgb[crop_height: rgb.shape[0] - crop_height, crop_width: rgb.shape[1] - crop_width]
    return rgb


def crop_image(zoom, rgb, aspect=1.5):
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


def mtf_curve(a=1., f=50.):
    assert a >= 1 and f > 0
    b = a / (math.sqrt((2 * a - 1) * f ** 2) - math.sqrt(a - 1) * f)
    c = - math.sqrt(a - 1)

    mtf = lambda x: a / (1 + (x * b + c) ** 2)

    return mtf


def film_sharpness(stock, rgb, scale):
    red_mtf = mtf_curve(stock['r_a'], stock['r_f'])
    green_mtf = mtf_curve(stock['g_a'], stock['g_f'])
    blue_mtf = mtf_curve(stock['b_a'], stock['b_f'])
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

    red_factors = np.vectorize(red_mtf)(frequency)
    green_factors = np.vectorize(green_mtf)(frequency)
    blue_factors = np.vectorize(blue_mtf)(frequency)

    red_shift = f_shift * red_factors
    green_shift = f_shift * green_factors
    blue_shift = f_shift * blue_factors

    red_kernel = np.fft.ifft2(np.fft.ifftshift(red_shift)).real
    green_kernel = np.fft.ifft2(np.fft.ifftshift(green_shift)).real
    blue_kernel = np.fft.ifft2(np.fft.ifftshift(blue_shift)).real

    red_kernel /= np.sum(red_kernel)
    green_kernel /= np.sum(green_kernel)
    blue_kernel /= np.sum(blue_kernel)

    kernel = np.dstack((red_kernel, green_kernel, blue_kernel))

    rgb = cv.filter2D(rgb, -1, kernel)

    return rgb


def exponential_blur(rgb, size):
    size = math.ceil(size)
    kernel = np.zeros((size, size))
    radius = math.floor(size / 2)

    for i in range(size):
        for j in range(size):
            dist = (i - size / 2) ** 2 + (j - size / 2) ** 2
            if not dist:
                dist = 1
            kernel[i, j] = (1 / dist) * max((radius - np.sqrt(dist)) / radius, 0)

    kernel /= np.sum(kernel)

    return cv.filter2D(rgb, -1, kernel)
