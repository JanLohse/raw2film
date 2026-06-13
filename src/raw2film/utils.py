"""
Additional utility functions.
"""

from functools import cache

import colour
import cv2 as cv
import exiftool
import numpy as np
from numba import njit, prange

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


def precompute_mix_table(red=None, green=None, blue=None):
    """Precomputes a (2, 2, 2, 4) sRGB lookup table using linear blending.

    Indices match: mix_table[red_active, green_active, blue_active]
    Output channel 0-2: RGB, Channel 3: Alpha (0 or 255)
    """
    if red is None or green is None or blue is None:
        hues = np.array((29.23, 142.50, 264.05)) / 360
        red, green, blue = [
            np.clip(colour.convert([0.6, 0.2, hues[i]], "Oklch", "sRGB"), 0, 1) * 255
            for i in range(3)
        ]

    # 1. Convert 8-bit sRGB input arrays to linear float32 [0, 1]
    r_lin = (red.astype(np.float32) / 255.0) ** 2.2
    g_lin = (green.astype(np.float32) / 255.0) ** 2.2
    b_lin = (blue.astype(np.float32) / 255.0) ** 2.2

    mix_table = np.zeros((2, 2, 2, 4), dtype=np.uint8)

    # 2. Populate combinations using linear blending physics
    for r in (0, 1):
        for g in (0, 1):
            for b in (0, 1):
                if r == 0 and g == 0 and b == 0:
                    # Background pixel remains fully transparent black
                    mix_table[0, 0, 0] = np.array([0, 0, 0, 0], dtype=np.uint8)
                    continue

                # Add up linear photons based on which channels are active
                lin_mix = (r * r_lin) + (g * g_lin) + (b * b_lin)

                # Clamp to prevent overflow, then re-encode to gamma sRGB
                lin_mix_clipped = np.clip(lin_mix, 0.0, 1.0)
                srgb_float = lin_mix_clipped ** (1.0 / 2.2)
                srgb_8bit = np.round(srgb_float * 255.0).astype(np.uint8)

                # Store RGB along with an opaque alpha channel
                mix_table[r, g, b, 0:3] = srgb_8bit
                mix_table[r, g, b, 3] = 255

    peak_rgb = (mix_table[1, 1, 1, :3] / 255.0) ** 2.2
    peak_white = peak_rgb.mean() ** (1.0 / 2.2) * 255.0
    mix_table[1, 1, 1, :3] = peak_white

    return mix_table


MIX_TABLE = precompute_mix_table()


@njit
def generate_histogram(image, mix_table=MIX_TABLE, height=100):
    """Generate an RGB histogram using a precomputed uint8 blending lookup table.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image (H x W x 3), dtype uint8.
    mix_table : np.ndarray
        Precomputed lookup matrix of shape (2, 2, 2, 4), dtype uint8.
    height : int
        Height of the histogram image.
    """
    hist_r = np.zeros(256, dtype=np.int32)
    hist_g = np.zeros(256, dtype=np.int32)
    hist_b = np.zeros(256, dtype=np.int32)

    h, w, _ = image.shape

    # Count frequencies
    for i in range(h):
        for j in range(w):
            hist_r[image[i, j, 0]] += 1
            hist_g[image[i, j, 1]] += 1
            hist_b[image[i, j, 2]] += 1

    # Normalize and log transform
    f_hist_r = hist_r.astype(np.float32)
    f_hist_g = hist_g.astype(np.float32)
    f_hist_b = hist_b.astype(np.float32)

    max_val = max(f_hist_r.max(), f_hist_g.max(), f_hist_b.max())
    if max_val == 0:
        max_val = 1

    for i in range(256):
        f_hist_r[i] = np.log1p(f_hist_r[i] / max_val)
        f_hist_g[i] = np.log1p(f_hist_g[i] / max_val)
        f_hist_b[i] = np.log1p(f_hist_b[i] / max_val)

    # Smooth with a 1-pixel moving average kernel
    smoothed_r = np.empty_like(f_hist_r)
    smoothed_g = np.empty_like(f_hist_g)
    smoothed_b = np.empty_like(f_hist_b)

    for i in range(256):
        left = i - 1 if i > 0 else i
        right = i + 1 if i < 255 else i
        smoothed_r[i] = (f_hist_r[left] + f_hist_r[i] + f_hist_r[right]) / 3
        smoothed_g[i] = (f_hist_g[left] + f_hist_g[i] + f_hist_g[right]) / 3
        smoothed_b[i] = (f_hist_b[left] + f_hist_b[i] + f_hist_b[right]) / 3

    # Final scale to vertical height bounds
    max_val = max(smoothed_r.max(), smoothed_g.max(), smoothed_b.max())
    if max_val == 0:
        max_val = 1

    final_h_r = ((smoothed_r * height) / max_val).astype(np.int32)
    final_h_g = ((smoothed_g * height) / max_val).astype(np.int32)
    final_h_b = ((smoothed_b * height) / max_val).astype(np.int32)

    # Allocating the output image array
    hist_img = np.zeros((height, 256, 4), dtype=np.uint8)

    for x in range(256):
        r_lim = height - final_h_r[x]
        g_lim = height - final_h_g[x]
        b_lim = height - final_h_b[x]

        for y in range(height):
            # Compute boolean flags (0 or 1) based on current row height
            is_r = 1 if y >= r_lim else 0
            is_g = 1 if y >= g_lim else 0
            is_b = 1 if y >= b_lim else 0

            # Direct lookup copy. If all are 0, it copies the empty background [0,0,0,0]
            hist_img[y, x] = mix_table[is_r, is_g, is_b]

    return hist_img


def resolution_scaling(image: np.ndarray, resolution) -> np.ndarray:
    h, w = image.shape[:2]
    h_factor = resolution[0] / h
    w_factor = resolution[1] / w
    scaling_factor = min(h_factor, w_factor)
    if scaling_factor < 1:
        image = cv.resize(
            image,
            (round(w * scaling_factor), round(h * scaling_factor)),
            interpolation=cv.INTER_AREA,
        )
    elif scaling_factor > 1:
        image = cv.resize(
            image,
            (round(w * scaling_factor), round(h * scaling_factor)),
            interpolation=cv.INTER_LANCZOS4,
        )

    return image


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
