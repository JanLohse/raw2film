"""
Additional utility functions.
"""

import math
from functools import cache

import cv2 as cv
import exiftool
import numpy as np
from numba import njit, prange
from spectral_film_lut.config import DEFAULT_DTYPE

from raw2film import data


@cache
def load_metadata(src):
    """Loads and caches image exit data."""
    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(src)[0]
    return metadata


def find_data(metadata, db):
    """Search for camera and lens name in metadata"""
    cam, lens = None, None

    def check_tags(metadata, tags, endings):
        result = []
        for tag in tags:
            if tag in metadata:
                result.append(metadata[tag])
        for ending in endings:
            for key in metadata:
                if key.endswith(ending):
                    result.append(metadata[key])
        if not result:
            return [None]
        return result

    cam_makes = check_tags(metadata, ["EXIF:Make"], ["Make"])
    cam_models = check_tags(metadata, ["EXIF:Model"], ["Model"])
    lens_makes = check_tags(metadata, ["EXIF:LensMake"], ["LensMake"])
    lens_models = check_tags(
        metadata, ["EXIF:LensModel", "EXIF:LensType"], ["LensModel", "LensType", "Lens"]
    )

    if cam_makes != [None]:
        for cam_make in cam_makes:
            if cam_make is not None:
                cam_make = str(cam_make)
            for cam_model in cam_models:
                if cam_model is not None:
                    cam_model = str(cam_model)
                cam = db.find_cameras(cam_make, cam_model, loose_search=True)
                if cam:
                    cam = cam[0]
                    break
            else:
                continue
            break
        if cam is not None and cam:
            for lens_make in lens_makes:
                if lens_make is not None:
                    lens_make = str(lens_make)
                for lens_model in lens_models:
                    if lens_model is not None:
                        lens_model = str(lens_model)
                    lens = db.find_lenses(cam, lens_make, lens_model, loose_search=True)
                    if lens:
                        lens = lens[0]
                        break
                else:
                    continue
                break
        elif cam is not None:
            cam = None

    return cam, lens


def add_metadata(src, metadata, exp_comp):
    """Adds metadata to an image file."""
    metadata = {
        key: metadata[key]
        for key in metadata
        if key.startswith("EXIF") and key[5:] in data.METADATA_KEYS
    }
    metadata["EXIF:ExposureCompensation"] = exp_comp
    with exiftool.ExifToolHelper() as et:
        et.set_tags([src], metadata, "-overwrite_original")


@njit
def generate_histogram(image, black_value=39, white_value=222, height=100):
    """
    Generate an RGB histogram as an image-like numpy array with logarithmic y-scaling.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image (H x W x 3), dtype should be uint8 (0-255).
    height : int
        Height of the histogram image.

    Returns
    -------
    hist_img : np.ndarray
        Histogram as a numpy array of shape (height, 256, 3).
    """
    # Initialize bins
    hist_r = np.zeros(256, dtype=DEFAULT_DTYPE)
    hist_g = np.zeros(256, dtype=DEFAULT_DTYPE)
    hist_b = np.zeros(256, dtype=DEFAULT_DTYPE)

    h, w, _ = image.shape

    # Count frequencies
    for i in range(h):
        for j in range(w):
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            hist_r[r] += 1
            hist_g[g] += 1
            hist_b[b] += 1

    # Normalize histograms to 1
    max_val = max(hist_r.max(), hist_g.max(), hist_b.max())
    if max_val == 0:
        max_val = 1  # avoid division by zero

    hist_r /= max_val
    hist_g /= max_val
    hist_b /= max_val

    # Apply log transform (avoid log(0))
    for i in range(256):
        hist_r[i] = np.log1p(hist_r[i])
        hist_g[i] = np.log1p(hist_g[i])
        hist_b[i] = np.log1p(hist_b[i])

    # Smooth with a 1-pixel kernel (moving average over neighbors)
    smoothed_r = np.empty_like(hist_r)
    smoothed_g = np.empty_like(hist_g)
    smoothed_b = np.empty_like(hist_b)

    for i in range(256):
        left = i - 1 if i > 0 else i
        right = i + 1 if i < 255 else i
        smoothed_r[i] = (hist_r[left] + hist_r[i] + hist_r[right]) / 3
        smoothed_g[i] = (hist_g[left] + hist_g[i] + hist_g[right]) / 3
        smoothed_b[i] = (hist_b[left] + hist_b[i] + hist_b[right]) / 3

    hist_r, hist_g, hist_b = smoothed_r, smoothed_g, smoothed_b

    # normalize to fit in height
    max_val = max(hist_r.max(), hist_g.max(), hist_b.max())
    if max_val == 0:
        max_val = 1  # avoid division by zero

    hist_r = (hist_r * height) / max_val
    hist_g = (hist_g * height) / max_val
    hist_b = (hist_b * height) / max_val

    # Create histogram image
    hist_img = np.full((height, 256, 3), black_value, dtype=np.uint8)

    for x in range(256):
        r_val = hist_r[x]
        g_val = hist_g[x]
        b_val = hist_b[x]

        for y in range(height - r_val, height):
            hist_img[y, x, 0] = white_value
        for y in range(height - g_val, height):
            hist_img[y, x, 1] = white_value
        for y in range(height - b_val, height):
            hist_img[y, x, 2] = white_value

    return hist_img


def resolution_scaling(image: np.ndarray, resolution) -> np.ndarray:
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
    return image


@njit(parallel=True)
def apply_lut_tetrahedral_float(
    image: np.ndarray,  # float32 in [0, 1]
    lut: np.ndarray,  # uint8 LUT (size, size, size, 3)
    interpolation_depth: int = 16,
) -> np.ndarray:
    h, w, c = image.shape
    size = lut.shape[0]
    max_int = 2**interpolation_depth - 1
    k = interpolation_depth - int(math.log2(size - 1))
    scale = 1 << k

    out = np.empty((h, w, 3), dtype=np.uint8)

    for y in prange(h):
        for x in prange(w):
            r = int(image[y, x, 0] * max_int)
            g = int(image[y, x, 1] * max_int)
            b = int(image[y, x, 2] * max_int)

            r0 = r >> k
            g0 = g >> k
            b0 = b >> k

            dr = r & (scale - 1)
            dg = g & (scale - 1)
            db = b & (scale - 1)

            r1 = min(r0 + 1, size - 1)
            g1 = min(g0 + 1, size - 1)
            b1 = min(b0 + 1, size - 1)

            c000 = lut[r0, g0, b0]

            # Tetrahedral interpolation
            if dr >= dg:
                if dg >= db:
                    c100 = lut[r1, g0, b0]
                    c110 = lut[r1, g1, b0]
                    c111 = lut[r1, g1, b1]
                    c = (
                        c000 * scale
                        + dr * (c100 - c000)
                        + dg * (c110 - c100)
                        + db * (c111 - c110)
                    )
                elif dr >= db:
                    c100 = lut[r1, g0, b0]
                    c101 = lut[r1, g0, b1]
                    c111 = lut[r1, g1, b1]
                    c = (
                        c000 * scale
                        + dr * (c100 - c000)
                        + db * (c101 - c100)
                        + dg * (c111 - c101)
                    )
                else:
                    c001 = lut[r0, g0, b1]
                    c101 = lut[r1, g0, b1]
                    c111 = lut[r1, g1, b1]
                    c = (
                        c000 * scale
                        + db * (c001 - c000)
                        + dr * (c101 - c001)
                        + dg * (c111 - c101)
                    )
            else:
                if db >= dg:
                    c001 = lut[r0, g0, b1]
                    c011 = lut[r0, g1, b1]
                    c111 = lut[r1, g1, b1]
                    c = (
                        c000 * scale
                        + db * (c001 - c000)
                        + dg * (c011 - c001)
                        + dr * (c111 - c011)
                    )
                elif db >= dr:
                    c010 = lut[r0, g1, b0]
                    c011 = lut[r0, g1, b1]
                    c111 = lut[r1, g1, b1]
                    c = (
                        c000 * scale
                        + dg * (c010 - c000)
                        + db * (c011 - c010)
                        + dr * (c111 - c011)
                    )
                else:
                    c010 = lut[r0, g1, b0]
                    c110 = lut[r1, g1, b0]
                    c111 = lut[r1, g1, b1]
                    c = (
                        c000 * scale
                        + dg * (c010 - c000)
                        + dr * (c110 - c010)
                        + db * (c111 - c110)
                    )

            # Convert back to uint8 safely
            out[y, x, 0] = np.uint8(c[0] // scale)
            out[y, x, 1] = np.uint8(c[1] // scale)
            out[y, x, 2] = np.uint8(c[2] // scale)

    return out


@njit(parallel=True)
def apply_lut_tetrahedral(
    image: np.ndarray,  # float32 image in [0, 1], shape (H, W, 3)
    lut: np.ndarray,  # float32 LUT in [0, 1], shape (size, size, size, 3)
    scale: float = 1.0,
) -> np.ndarray:
    h, w, _ = image.shape
    size = lut.shape[0]

    out = np.empty((h, w, 3), dtype=np.float32)

    scale *= size - 1

    for y in prange(h):
        for x in range(w):
            # Continuous LUT coordinates
            r = image[y, x, 0] * scale
            g = image[y, x, 1] * scale
            b = image[y, x, 2] * scale

            # Base indices
            r0 = int(r)
            g0 = int(g)
            b0 = int(b)

            # Clamp upper edge
            if r0 >= size - 1:
                r0 = size - 2
                dr = 1.0
            else:
                dr = r - r0

            if g0 >= size - 1:
                g0 = size - 2
                dg = 1.0
            else:
                dg = g - g0

            if b0 >= size - 1:
                b0 = size - 2
                db = 1.0
            else:
                db = b - b0

            r1 = r0 + 1
            g1 = g0 + 1
            b1 = b0 + 1

            c000 = lut[r0, g0, b0]

            # Tetrahedral interpolation
            if dr >= dg:
                if dg >= db:
                    # dr >= dg >= db
                    c100 = lut[r1, g0, b0]
                    c110 = lut[r1, g1, b0]
                    c111 = lut[r1, g1, b1]

                    c = (
                        c000
                        + dr * (c100 - c000)
                        + dg * (c110 - c100)
                        + db * (c111 - c110)
                    )

                elif dr >= db:
                    # dr >= db > dg
                    c100 = lut[r1, g0, b0]
                    c101 = lut[r1, g0, b1]
                    c111 = lut[r1, g1, b1]

                    c = (
                        c000
                        + dr * (c100 - c000)
                        + db * (c101 - c100)
                        + dg * (c111 - c101)
                    )

                else:
                    # db > dr >= dg
                    c001 = lut[r0, g0, b1]
                    c101 = lut[r1, g0, b1]
                    c111 = lut[r1, g1, b1]

                    c = (
                        c000
                        + db * (c001 - c000)
                        + dr * (c101 - c001)
                        + dg * (c111 - c101)
                    )

            else:
                if db >= dg:
                    # db >= dg > dr
                    c001 = lut[r0, g0, b1]
                    c011 = lut[r0, g1, b1]
                    c111 = lut[r1, g1, b1]

                    c = (
                        c000
                        + db * (c001 - c000)
                        + dg * (c011 - c001)
                        + dr * (c111 - c011)
                    )

                elif db >= dr:
                    # dg > db >= dr
                    c010 = lut[r0, g1, b0]
                    c011 = lut[r0, g1, b1]
                    c111 = lut[r1, g1, b1]

                    c = (
                        c000
                        + dg * (c010 - c000)
                        + db * (c011 - c010)
                        + dr * (c111 - c011)
                    )

                else:
                    # dg > dr > db
                    c010 = lut[r0, g1, b0]
                    c110 = lut[r1, g1, b0]
                    c111 = lut[r1, g1, b1]

                    c = (
                        c000
                        + dg * (c010 - c000)
                        + dr * (c110 - c010)
                        + db * (c111 - c110)
                    )

            out[y, x] = c

    return out
