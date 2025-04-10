import numpy as np


def XYZ_to_kelvin(XYZ):
    x = XYZ[0] / np.sum(XYZ)
    y = XYZ[1] / np.sum(XYZ)
    n = (x - 0.3366) / (y - 0.1735)
    CCT = (-949.86315 + 6253.80338 * np.exp(-n / 0.92159) + 28.70599 * np.exp(-n / 0.20039) + 0.00004 * np.exp(
        -n / 0.07125))
    return CCT


def kelvin_to_XYZ(CCT):
    # This section is ripped from the Colour Science package:
    CCT_3 = CCT ** 3
    CCT_2 = CCT ** 2

    x = np.where(CCT <= 4000,
                 -0.2661239 * 10 ** 9 / CCT_3 - 0.2343589 * 10 ** 6 / CCT_2 + 0.8776956 * 10 ** 3 / CCT + 0.179910,
                 -3.0258469 * 10 ** 9 / CCT_3 + 2.1070379 * 10 ** 6 / CCT_2 + 0.2226347 * 10 ** 3 / CCT + 0.24039, )

    x_3 = x ** 3
    x_2 = x ** 2

    cnd_l = [CCT <= 2222, np.logical_and(CCT > 2222, CCT <= 4000), CCT > 4000]
    i = -1.1063814 * x_3 - 1.34811020 * x_2 + 2.18555832 * x - 0.20219683
    j = -0.9549476 * x_3 - 1.37418593 * x_2 + 2.09137015 * x - 0.16748867
    k = 3.0817580 * x_3 - 5.8733867 * x_2 + 3.75112997 * x - 0.37001483
    y = np.select(cnd_l, [i, j, k])

    XYZ = np.array([x / y, 1, (1 - x - y) / y])
    return XYZ


def encode_ARRILogC3(x):
    cut, a, b, c, d, e, f = 0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809

    return np.where(x > cut, (c / np.log(10)) * np.log(a * x + b) + d, e * x + f)


def calc_exposure(rgb, ref_exposure=0.18, metadata=None, **kwargs):
    """Calculates exposure value of the rgb image."""
    lum_mat = rgb[:, :, 1]

    factor = 3
    if metadata is not None:
        if 'EXIF:FNumber' in metadata and metadata['EXIF:FNumber']:
            factor = metadata['EXIF:FNumber'] ** 2 / metadata['EXIF:ISO'] / metadata['EXIF:ExposureTime']
        else:
            factor = 4 ** 2 / metadata['EXIF:ISO'] / metadata['EXIF:ExposureTime']
        factor = np.sqrt(factor) + 1

    log_lum = lum_mat ** (1 / factor)
    average_exposure = log_lum.mean() ** factor

    ref_exposure *= 65535
    exp_comp = np.log2(ref_exposure / average_exposure)

    return exp_comp


def gamut_compression(rgb, adaption=0.2):
    # compute achromaticity (max rgb value per pixel)
    achromatic = np.repeat(np.max(rgb, axis=2)[:, :, np.newaxis], 3, axis=2)

    # compute distance to gamut
    distance = (achromatic - rgb) / achromatic

    # smoothing parameter is a
    # precompute smooth compression function
    x = np.linspace(1 - adaption, 1 + adaption, 16)
    y = 1 - adaption + (x - 1 + adaption) / (np.sqrt(1 + ((x - 1) / adaption + 1) ** 2))
    # compress distance
    distance = np.interp(distance, np.concatenate((np.array([0]), x)), np.concatenate((np.array([0]), y)))

    rgb = achromatic - distance * achromatic
