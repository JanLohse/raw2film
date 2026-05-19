"""
The main processing pipeline for RAW images.
"""

from functools import cache
from typing import Literal

import numpy as np
import rawpy
from spectral_film_lut.config import DEFAULT_DTYPE
from spectral_film_lut.utils import (
    create_lut,
)

from raw2film import effects
from raw2film.color_processing import calc_exposure
from raw2film.utils import (
    load_metadata,
)

CANVAS_MODES = Literal[
    "No",
    "Proportional white",
    "Proportional black",
    "Uniform white",
    "Uniform black",
    "Fixed white",
    "Fixed black",
]
"""Available canvas modes."""


def raw_to_linear(src, half_size=True):
    """Load linear raw data using rawpy."""
    # convert raw file to linear data
    with rawpy.imread(src) as raw:
        # noinspection PyUnresolvedReferences
        rgb = raw.postprocess(
            output_color=rawpy.ColorSpace(5),
            gamma=(1, 1),
            output_bps=16,
            no_auto_bright=True,
            use_camera_wb=False,
            use_auto_wb=False,
            half_size=half_size,
            demosaic_algorithm=rawpy.DemosaicAlgorithm(2),
            four_color_rgb=False,
        )
    rgb = rgb.astype(DEFAULT_DTYPE) / 65535.0

    rgb *= 2 ** calc_exposure(rgb, metadata=load_metadata(src))

    return rgb


def crop_rotate_zoom(
    image: np.ndarray,
    frame_width: int | float = 36,
    frame_height: int | float = 24,
    rotation: float = 0.0,
    zoom: float = 1.0,
    rotate_times: int = 0,
    flip: bool = False,
):
    """Apply cropping, rotation, and a zoom to an image."""
    image = effects.crop_image(image, 1, aspect=frame_width / frame_height, flip=flip)
    if rotation:
        image = effects.rotate(image, rotation)
    image = effects.crop_image(image, zoom, aspect=frame_width / frame_height)
    image = np.rot90(image, k=rotate_times)

    return image


@cache
def create_lut_cached(*args, **kwargs):
    """Cache LUTs for specific settings."""
    return create_lut(*args, **kwargs)
