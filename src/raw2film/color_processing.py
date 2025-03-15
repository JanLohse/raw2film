import numpy as np

from raw2film import data


def BT2020_to_kelvin(rgb):
    XYZ = np.dot(data.REC2020_TO_XYZ, rgb)
    x = XYZ[0] / np.sum(XYZ)
    y = XYZ[1] / np.sum(XYZ)
    n = (x - 0.3366) / (y - 0.1735)
    CCT = (-949.86315 + 6253.80338 * np.exp(-n / 0.92159) + 28.70599 * np.exp(-n / 0.20039)
           + 0.00004 * np.exp(-n / 0.07125))
    return CCT


def kelvin_to_BT2020(CCT):
    # This section is ripped from the Colour Science package:
    CCT_3 = CCT ** 3
    CCT_2 = CCT ** 2

    x = np.where(
        CCT <= 4000,
        -0.2661239 * 10 ** 9 / CCT_3
        - 0.2343589 * 10 ** 6 / CCT_2
        + 0.8776956 * 10 ** 3 / CCT
        + 0.179910,
        -3.0258469 * 10 ** 9 / CCT_3
        + 2.1070379 * 10 ** 6 / CCT_2
        + 0.2226347 * 10 ** 3 / CCT
        + 0.24039,
    )

    x_3 = x ** 3
    x_2 = x ** 2

    cnd_l = [CCT <= 2222, np.logical_and(CCT > 2222, CCT <= 4000), CCT > 4000]
    i = -1.1063814 * x_3 - 1.34811020 * x_2 + 2.18555832 * x - 0.20219683
    j = -0.9549476 * x_3 - 1.37418593 * x_2 + 2.09137015 * x - 0.16748867
    k = 3.0817580 * x_3 - 5.8733867 * x_2 + 3.75112997 * x - 0.37001483
    y = np.select(cnd_l, [i, j, k])

    XYZ = np.array([x / y, 1, (1 - x - y) / y])
    rgb = np.dot(data.XYZ_TO_REC2020, XYZ)
    return rgb


def encode_ARRILogC3(x):
    cut, a, b, c, d, e, f = 0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809

    return np.where(x > cut, (c / np.log(10)) * np.log(a * x + b) + d, e * x + f)

def calc_exposure(rgb, crop=.8):
    """Calculates exposure value of the rgb image."""
    lum_mat = np.dot(rgb, np.dot(np.asarray(data.REC2020_TO_REC709), np.asarray(np.array([.2127, .7152, .0722]))))

    if 0 < crop < 1:
        ratio = lum_mat.shape[0] / lum_mat.shape[1]
        if ratio > 1:
            width = int((lum_mat.shape[0] - ratio ** .5 / ratio * lum_mat.shape[0] * crop) / 2)
            height = int((lum_mat.shape[1] - lum_mat.shape[1] * crop) / 2)
        else:
            width = int((lum_mat.shape[0] - lum_mat.shape[0] * crop) / 2)
            ratio = 1 / ratio
            height = int((lum_mat.shape[1] - ratio ** .5 / ratio * lum_mat.shape[1] * crop) / 2)
        lum_mat = lum_mat[width: -width, height: -height]
    return np.average(np.log(lum_mat + np.ones_like(lum_mat) * 2 ** -16)) / np.log(2)