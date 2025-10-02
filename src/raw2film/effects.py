from functools import cache

import cv2
import cv2 as cv
import lensfunpy
from lensfunpy import util as lensfunpy_util
from spectral_film_lut.utils import *

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
    return xp.asarray(kernel)


def film_sharpness(rgb, stock, scale):
    size = int(scale // 2)
    if not size % 2:
        size += 1

    kernel = np.zeros((size, size))
    kernel[size // 2, size // 2] = 1
    f = np.fft.fft2(kernel)
    f_shift = np.fft.fftshift(f)

    frequency = np.abs(np.fft.fftfreq(size, 1 / scale)[:, None])
    frequency_x, frequency_y = np.meshgrid(frequency.reshape(-1), frequency.reshape(-1))
    frequency = np.fft.fftshift(np.sqrt(frequency_x ** 2 + frequency_y ** 2))

    kernel = xp.stack([mtf_kernel(mtf, frequency, f_shift) for mtf in stock.mtf], axis=-1, dtype=xp.float32)

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


@cache
def gaussian_noise_cache(shape):
    return xp.random.default_rng().standard_normal(shape, dtype=np.float32)


def gaussian_noise(shape):
    noise_size = ((max(shape[:2]) + 100) // 1024 + 1) * 1024
    noise_map = gaussian_noise_cache((noise_size, noise_size))
    if cuda_available:
        offsets = xp.stack(
            [xp.random.randint(0, x, size=shape[2]) for x in [noise_size - shape[0] + 1, noise_size - shape[1] + 1]]).T
    else:
        offsets = xp.random.randint([0, 0], [noise_size - shape[0] + 1, noise_size - shape[1] + 1], size=(shape[2], 2))
    noise = xp.stack([noise_map[x:shape[0] + x, y:shape[1] + y] for x, y in offsets], axis=-1)
    return noise


def grain_kernel(pixel_size_mm, dye_size1_mm=0.0065, dye_size2_mm=0.015):
    # based on the paper:
    # Simulating Film Grain using the Noise-Power Spectrum by Ian Stephenson and Arthur Saunders
    kernel_size_mm = 4.24 * max(dye_size1_mm, dye_size2_mm)
    kernel_size = round(kernel_size_mm / pixel_size_mm)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 3:
        return None

    # Frequency grid (cycles per mm)
    fx = xp.fft.fftfreq(kernel_size, d=pixel_size_mm)
    fy = xp.fft.fftfreq(kernel_size, d=pixel_size_mm)
    FX, FY = xp.meshgrid(fx, fy)
    f = xp.sqrt(FX ** 2 + FY ** 2)  # radial frequency

    # Gaussian model for dye NPS: exp(-(pi*f*D)^2)
    nps1 = xp.exp(-(np.pi * f * dye_size1_mm) ** 2)
    nps2 = xp.exp(-(np.pi * f * dye_size2_mm) ** 2)

    # Total NPS (weighted sum)
    nps = (nps1 + nps2)
    # generate convolution kernel
    kernel = xp.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, kernel_size // 2] = 1
    kernel = xp.fft.fft2(kernel)
    kernel = kernel * np.sqrt(nps)
    kernel = xp.fft.ifft2(kernel)
    kernel = kernel.real.astype(xp.float32)

    # normalize kernel
    expected_var = xp.mean(nps)
    expected_std = xp.sqrt(expected_var / 2)
    kernel /= xp.sum(kernel) * xp.sqrt(expected_std)

    return kernel


def grain(rgb, stock, scale, grain_size=0.001, d_factor=6, bw_grain=False, **kwargs):
    # compute scaling factor of exposure rms in regard to measuring device size
    std_factor = math.sqrt(math.pi) * 0.024 * scale / d_factor
    shape = rgb.shape
    if bw_grain:
        shape = (shape[0], shape[1], 1)
    noise = gaussian_noise(shape)
    xps = [(rms_density + 0.25) / d_factor for rms_density in stock.rms_density]
    fps = [rms * std_factor for rms in stock.rms_curve]
    noise_factors = multi_channel_interp(rgb, xps, fps)
    noise = noise * noise_factors
    kernel = grain_kernel(1 / scale, 6.5 * grain_size, 15 * grain_size)
    if kernel is not None:
        noise = convolution_filter(noise, kernel)
    if len(noise.shape) == 2:
        noise = noise[..., xp.newaxis]
    if bw_grain:
        noise /= 1.5
    rgb += noise
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


def convolution_filter(rgb, kernel, padding=False):
    if not cuda_available:
        return cv2.filter2D(rgb, -1, kernel)
    else:
        if len(kernel.shape) == 2:
            kernel = kernel[..., xp.newaxis]
        if padding:
            pad_h = kernel.shape[0] // 2
            pad_w = kernel.shape[1] // 2
            rgb = xp.pad(rgb, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'reflect')
            return signal.oaconvolve(rgb, kernel, mode='valid', axes=(0, 1))
        else:
            return signal.oaconvolve(rgb, kernel, mode='same', axes=(0, 1))


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


def burn(image, negative_film, highlight_burn, burn_scale, d_factor):
    func = lambda x: xp.clip(x * d_factor - 0.25 - negative_film.d_ref[1 if len(negative_film.d_ref) > 1 else 0], 0,
        None)
    if image.shape[-1] == 3:
        image = image - highlight_burn * down_up_blur(image[..., 1:2], burn_scale, func) / d_factor
    else:
        image = image - highlight_burn * down_up_blur(image, burn_scale, func) / d_factor


    return image
