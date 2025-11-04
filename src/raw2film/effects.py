from functools import lru_cache

import cv2 as cv
import lensfunpy
from lensfunpy import util as lensfunpy_util
from spectral_film_lut import adx16_decode
from spectral_film_lut.grain_generation import *

if cuda_available:
    try:
        import torch

        torch_available = True
    except ImportError:
        torch_available = False
else:
    torch_available = False



def lens_correction(rgb, metadata, cam, lens):
    """Apply lens correction using lensfunpy."""
    # noinspection PyUnresolvedReferences
    rgb = rgb.astype(np.float64)
    if lens is None or cam is None:
        return rgb
    try:
        focal_length = metadata['EXIF:FocalLength']
        aperture = float(metadata['EXIF:FNumber'])
    except (KeyError, ValueError):
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


def crop_image(rgb, zoom=1, aspect=1.5, flip=False):
    """Crops rgb data to aspect ratio."""
    x, y, _ = rgb.shape
    if flip:
        aspect = 1 / aspect
    if x > y:
        if x > aspect * y:
            rgb = rgb[math.ceil(x / 2 - y * aspect / 2): math.ceil(x / 2 + y * aspect / 2), :, :]
        else:
            rgb = rgb[:, math.ceil(y / 2 - x / aspect / 2): math.ceil(y / 2 + x / aspect / 2), :]
    elif y > aspect * x:
        rgb = rgb[:, math.ceil(y / 2 - x * aspect / 2): math.ceil(y / 2 + x * aspect / 2), :]
    else:
        rgb = rgb[math.ceil(x / 2 - y / aspect / 2): math.ceil(x / 2 + y / aspect / 2), :, :]

    if zoom > 1:
        x, y, _ = rgb.shape
        zoom_factor = (zoom - 1) / (2 * zoom)
        x = math.ceil(zoom_factor * x)
        y = math.ceil(zoom_factor * y)
        rgb = rgb[x: -x, y: -y, :]

    return rgb


def gaussian_blur(rgb, sigma=1.):
    """Applies gaussian blur per channel of rgb image."""
    if cuda_available:
        return xdimage.gaussian_filter(rgb, sigma=(sigma, sigma, 0), mode='constant')
    else:
        return cv.GaussianBlur(rgb, ksize=(0, 0), sigmaX=sigma)


def mtf_curve(logf, vals):
    return lambda x: xp.interp(xp.log1p(x), logf, vals, left=1, right=0)


def compute_kernel_from_function(func, kernel_size_mm, pixel_size_mm):
    kernel_size = round(kernel_size_mm / pixel_size_mm)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Frequency grid
    fx = xp.fft.fftfreq(kernel_size, d=pixel_size_mm)
    fy = xp.fft.fftfreq(kernel_size, d=pixel_size_mm)
    FX, FY = xp.meshgrid(fx, fy)
    f = xp.sqrt(FX ** 2 + FY ** 2)  # radial frequency magnitude

    # Apply transfer function in frequency domain
    H = func(f)

    # Get spatial kernel by inverse FFT
    kernel = xp.fft.ifft2(H)
    kernel = xp.fft.fftshift(xp.abs(kernel))  # center it
    kernel /= xp.sum(kernel)

    return kernel


@lru_cache(maxsize=50)
def mtf_kernel(logf, vals, scale):
    mtf_func = mtf_curve(xp.asarray(logf), xp.asarray(vals))
    kernel = compute_kernel_from_function(mtf_func, 1., 1 / scale)
    return kernel

def film_sharpness(rgb, stock, scale):
    kernel = xp.stack([mtf_kernel(logf, vals, scale) for logf, vals in stock.mtf], axis=-1, dtype=xp.float32)
    size = kernel.shape[0]
    if len(kernel.shape) == 2 or size >= 13 or cuda_available:
        rgb = convolution_filter(rgb, kernel, padding=True)
    elif len(kernel.shape) == 3:
        for c in range(kernel.shape[-1]):
            rgb[..., c] = cv2.filter2D(rgb[..., c], -1, kernel[..., c])
    if len(rgb.shape) == 2:
        rgb = rgb[..., xp.newaxis]
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


def apply_grain(rgb, stock, scale, grain_size_mm=0.01, grain_sigma=0.4, bw_grain=False, **kwargs):
    grain = generate_grain(rgb.shape, scale, grain_size_mm, bw_grain, cached=True, grain_sigma=grain_sigma)
    grain_factors = stock.grain_transform(rgb, scale)
    grain = grain * grain_factors
    rgb += grain
    return rgb


@njit
def apply_halation_inplace(rgb, blured, color_factors):
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            for c in range(rgb.shape[2]):
                rgb[i, j, c] += blured[i, j, c] * color_factors[c]
                rgb[i, j, c] /= (color_factors[c] + 1.0)


@njit
def apply_halation_bw(rgb, blured, intensity):
    for i in range(rgb.shape[0]):
        for j in range(rgb.shape[1]):
            rgb[i, j] += blured[i, j] * intensity
            rgb[i, j] /= (intensity + 1.0)


def halation(rgb, scale, halation_size=1, halation_red_factor=1., halation_green_factor=0.4, halation_blue_factor=0.,
             halation_intensity=1, **kwargs):
    kernel = xp.asarray(exponential_blur_kernel(scale / 4 * halation_size), dtype=xp.float32)
    blured = convolution_filter(rgb, kernel)
    color_factors = halation_intensity * xp.array([halation_red_factor, halation_green_factor, halation_blue_factor],
                                                  dtype=np.float32)
    if cuda_available:
        rgb = (rgb + blured * color_factors) / (color_factors + 1)
    elif rgb.shape[-1] == 1:
        apply_halation_bw(rgb, blured, color_factors[0])
    else:
        apply_halation_inplace(rgb, blured, color_factors)
    return rgb


def add_canvas(image, canvas_ratio, canvas_scale, canvas_color, **kwargs):
    """Adds background canvas to image."""
    img_ratio = image.shape[1] / image.shape[0]
    if img_ratio > canvas_ratio:
        output_resolution = (
            int(image.shape[1] / canvas_ratio * canvas_scale), int(image.shape[1] * canvas_scale))
    else:
        output_resolution = (
            int(image.shape[0] * canvas_scale), int(image.shape[0] * canvas_ratio * canvas_scale))
    offset = np.subtract(output_resolution, image.shape[:2]) // 2
    canvas = np.tensordot(np.ones(output_resolution), canvas_color, axes=0)
    canvas[offset[0]:offset[0] + image.shape[0], offset[1]:offset[1] + image.shape[1]] = image
    return canvas.astype(dtype='uint8')


def add_canvas_uniform(image, canvas_scale, canvas_color, **kwargs):
    """Adds background canvas to image."""
    side_length = max(image.shape[:2])
    border_size = int(side_length * (canvas_scale - 1))
    output_resolution = (image.shape[0] + border_size, image.shape[1] + border_size)
    offset = np.subtract(output_resolution, image.shape[:2]) // 2
    canvas = np.tensordot(np.ones(output_resolution), canvas_color, axes=0)
    canvas[offset[0]:offset[0] + image.shape[0], offset[1]:offset[1] + image.shape[1]] = image
    return canvas.astype(dtype='uint8')


def down_up_blur(image, scale=50, func=None):
    scale = math.ceil(min(image.shape[:2]) / scale)
    # Downsample
    blurred_channels = []
    for c in range(image.shape[-1]):
        # Downsample
        if cuda_available:
            if torch_available:
                down = cupy_area_downsample(image[:, :, c], scale)
            else:
                down = xdimage.zoom(image[:, :, c], 1 / scale, order=1)
        else:
            down = cv.resize(image[:, :, c], (image.shape[1] // scale, image.shape[0] // scale), interpolation=cv.INTER_AREA)
        if func is not None:
            down = func(down)
        # Downsample channel
        blurred = xdimage.gaussian_filter(down, sigma=3)

        # Upsample back
        up = xdimage.zoom(blurred, scale, order=1)
        # Crop or pad to match original shape
        up_resized = xp.pad(up, [(0, max(x - y, 0)) for x, y in zip(image.shape, up.shape)], mode='edge')[:image.shape[0],
                     :image.shape[1]]
        blurred_channels.append(up_resized)

    # Stack back into (H, W, 3)
    return xp.stack(blurred_channels, axis=-1)


def cupy_area_downsample(image, factor):
    img_torch = torch.utils.dlpack.from_dlpack(xp.asarray(image)[None, None, ...].toDlpack())

    # Apply mean pooling
    downsampled = torch.nn.functional.avg_pool2d(img_torch, kernel_size=factor, stride=factor)

    # Convert back to CuPy (remove batch and channel dimensions)
    return xp.fromDlpack(torch.utils.dlpack.to_dlpack(downsampled))[0, 0]


def burn(image, negative_film, highlight_burn, burn_scale):
    highlight_burn *= 8000. / 65535.
    func = lambda x: xp.clip(adx16_decode(x) - negative_film.d_ref[1 if len(negative_film.d_ref) > 1 else 0], 0, None)
    if image.shape[-1] == 3:
        image = image - highlight_burn * down_up_blur(image[..., 1:2], burn_scale, func)
    else:
        image = image - highlight_burn * down_up_blur(image, burn_scale, func)

    return image
