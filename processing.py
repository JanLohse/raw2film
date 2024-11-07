import math
import operator
import os
import sys
import time
import warnings
from multiprocessing import Pool, Semaphore
from pathlib import Path
from shutil import copy

import configargparse as argparse
import cv2 as cv
import exiftool
import ffmpeg
import imageio.v3 as imageio
import lensfunpy
import numpy as np
from PIL import Image
import torch

try:
    import cupy as cp

    is_cupy_available = True
except ModuleNotFoundError:
    import numpy as np

    is_cupy_available = False
import rawpy
from lensfunpy import util as lensfunpy_util
from scipy import ndimage

try:
    import cupy as cp
    from cupyx.scipy import ndimage as cdimage

    is_cupy_available = True
    cp.cuda.set_allocator(None)
    cp.cuda.set_pinned_memory_allocator(None)
except ModuleNotFoundError:
    import numpy as np
    from scipy import ndimage as cdimage

    is_cupy_available = False


class Raw2Film:
    METADATA_KEYS = [
        'GPSDateStamp', 'ModifyDate', 'FocalLengthIn35mmFormat', 'ShutterSpeedValue', 'FocalLength', 'Make',
        'Saturation', 'SubSecTimeOriginal', 'SubSecTimeDigitized', 'GPSImgDirectionRef', 'ExposureProgram',
        'GPSLatitudeRef', 'Software', 'GPSVersionID', 'ResolutionUnit', 'LightSource', 'FileSource', 'ExposureMode',
        'Compression', 'MaxApertureValue', 'OffsetTime', 'DigitalZoomRatio', 'Contrast', 'InteropIndex',
        'ThumbnailLength', 'DateTimeOriginal', 'OffsetTimeOriginal', 'SensingMethod', 'SubjectDistance',
        'CreateDate', 'ExposureCompensation', 'SensitivityType', 'ApertureValue', 'ExifImageWidth', 'SensorLeftBorder',
        'FocalPlaneYResolution', 'GPSImgDirection', 'ComponentsConfiguration', 'Flash', 'Model', 'ColorSpace',
        'LensModel', 'XResolution', 'GPSTimeStamp', 'ISO', 'CompositeImage', 'FocalPlaneXResolution', 'SubSecTime',
        'GPSAltitude', 'OffsetTimeDigitized', 'ExposureTime', 'LensMake', 'WhiteBalance', 'BrightnessValue',
        'GPSLatitude', 'YResolution', 'GPSLongitude', 'YCbCrPositioning', 'Copyright', 'SubjectDistanceRange',
        'SceneType', 'GPSAltitudeRef', 'FocalPlaneResolutionUnit', 'MeteringMode', 'GPSLongitudeRef', 'SensorTopBorder',
        'SceneCaptureType', 'FNumber', 'LightValue', 'BrightnessValue', 'SensorWidth', 'SensorHeight',
        'SensorBottomBorder', 'SensorRightBorder', 'ProcessingSoftware']
    EXTENSION_LIST = ('.rw2', '.dng', '.crw', '.cr2', '.cr3', '.nef', '.orf', '.ori', '.raf', '.rwl', '.pef', '.ptx')
    FORMATS = {'110': (17, 13),
               '135-half': (24, 18), '135': (36, 24),
               'xpan': (65, 24),
               '120-4.5': (56, 42), '120-6': (56, 56), '120': (70, 56), '120-9': (83, 56),
               '4x5': (127, 101.6), '5x7': (177.8, 127), '8x10': (254, 203.2), '11x14': (355.6, 279.4),
               'super16': (12.42, 7.44), 'scope': (24.89, 10.4275), 'flat': (24.89, 13.454), 'academy': (24.89, 18.7),
               '65mm': (48.56, 22.1), 'IMAX': (70.41, 52.63)}
    REC2020_TO_ARRIWCG = np.array([[1.0959, -.0751, -.0352],
                                   [-.1576, 0.8805, 0.0077],
                                   [0.0615, 0.1946, 1.0275]], dtype=np.float32)
    REC2020_TO_REC709 = np.array([[1.6605, -.1246, -.0182],
                                  [-.5879, 1.1330, -.1006],
                                  [-.0728, -.0084, 1.1187]], dtype=np.float32)
    REC709_TO_REC2020 = np.array([[.6274, .0691, .0164],
                                  [.3294, .9195, .0880],
                                  [.0433, .0114, .8956]], dtype=np.float32)
    REC2020_TO_XYZ = np.array([[0.6369580, 0.1446169, 0.1688810],
                               [0.2627002, 0.6779981, 0.0593017],
                               [0.0000000, 0.0280727, 1.0609851]], dtype=np.float32)
    XYZ_TO_REC2020 = np.array([[1.7166512, -0.3556708, -0.2533663],
                               [-0.6666844, 1.6164812, 0.0157685],
                               [0.0176399, -0.0427706, 0.9421031]], dtype=np.float32)
    CAMERA_DB = {"X100S": "Fujifilm : X100S",
                 "DMC-GX80": "Panasonic : DMC-GX80",
                 "DC-FZ10002": "Panasonic : DC-FZ10002"}
    LENS_DB = {"X100S": "Fujifilm : X100 & compatibles (Standard)",
               "LUMIX G 25/F1.7": "Panasonic : Lumix G 25mm f/1.7 Asph.",
               "LUMIX G VARIO 12-32/F3.5-5.6": "Panasonic : Lumix G Vario 12-32mm f/3.5-5.6 Asph. Mega OIS",
               "DC-FZ10002": "Leica : FZ1000 & compatibles"}

    def __init__(self, crop=True, resolution=True, halation=True, grain=True, organize=True, canvas=False, nd=0,
                 width=36, height=24, ratio=4 / 5, scale=1., color=None, artist="Jan Lohse", luts=None, tiff=False,
                 wb='standard', exp=0, zoom=1., correct=True, cores=None, sleep_time=0, rename=False, rotation=0,
                 cuda=False, keep_exp=False, gamma=1., bw_grain=False, **args):
        self.crop = crop
        self.resolution = resolution
        self.halation = halation
        self.grain = grain
        self.organize = organize
        self.canvas = canvas
        self.width = width
        self.height = height
        self.output_ratio = ratio
        self.output_scale = scale
        self.output_color = color
        self.artist = artist
        self.luts = luts
        self.wb = wb
        self.tiff = tiff
        self.exp = exp
        self.zoom = zoom
        self.nd = nd
        self.correct = correct
        self.sleep_time = sleep_time
        self.cores = cores
        self.rename = rename
        self.rotation = rotation
        self.cuda = cuda
        self.keep_exp = keep_exp
        self.gamma = gamma
        self.bw_grain = bw_grain

    def process_runner(self, starter: tuple[int, str]):
        run_count, src = starter
        try:
            if 0 < run_count < self.cores:
                time.sleep(self.sleep_time * run_count)
            self.process_image(src, run_count)
        except KeyboardInterrupt:
            return False
        return src

    def process_image(self, src: str, run_count):
        """Manages image processing pipeline."""

        rgb, metadata = self.raw_to_linear(src)

        if self.keep_exp:
            exp_comp, gamma = self.find_exp(src, self.exp)
        else:
            exp_comp, gamma = self.exp, self.gamma

        if self.correct:
            rgb = np.asarray(self.lens_correction(rgb, metadata))

        if not run_count or not self.cuda:
            rgb = self.film_emulation(rgb, metadata, exp_comp, gamma)
        else:
            with semaphore:
                rgb = self.film_emulation(rgb, metadata, exp_comp, gamma)

        Raw2Film.save_tiff(src, rgb)
        if self.tiff:
            return

        file_list = [self.apply_lut(src, i, metadata) for i in range(len(self.luts))]
        file_list = [self.convert_jpg(file) for file in file_list]
        os.remove(src.split('.')[0] + "_log.tiff")

        for file in file_list:
            self.add_metadata(file, metadata, exp_comp, gamma)

        if self.organize:
            self.organize_files(src, file_list, metadata)

    def raw_to_linear(self, src):
        """Takes raw file location and outputs linear rgb data and metadata."""
        # read metadata
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(src)[0]
        metadata['EXIF:Artist'] = self.artist
        metadata['EXIF:ProcessingSoftware'] = "raw2film"

        # convert raw file to linear data
        with rawpy.imread(src) as raw:
            # noinspection PyUnresolvedReferences
            rgb = raw.postprocess(output_color=rawpy.ColorSpace(6), gamma=(1, 1), output_bps=16, no_auto_bright=True,
                                  use_camera_wb=(self.wb == 'camera'), use_auto_wb=(self.wb == 'auto'),
                                  demosaic_algorithm=rawpy.DemosaicAlgorithm(11), four_color_rgb=True)
        rgb = rgb.astype(dtype='float64')
        rgb /= 2 ** 16 - 1

        return rgb, metadata

    @staticmethod
    def BT2020_to_kelvin(rgb):
        XYZ = np.dot(Raw2Film.REC2020_TO_XYZ, rgb)
        x = XYZ[0] / np.sum(XYZ)
        y = XYZ[1] / np.sum(XYZ)
        n = (x - 0.3366) / (y - 0.1735)
        CCT = (-949.86315 + 6253.80338 * np.exp(-n / 0.92159) + 28.70599 * np.exp(-n / 0.20039)
               + 0.00004 * np.exp(-n / 0.07125))
        return CCT

    def kelvin_to_BT2020(self, CCT):
        global cp
        if not self.cuda:
            cp = np

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
        rgb = np.dot(Raw2Film.XYZ_TO_REC2020, XYZ)
        return cp.asarray(rgb)

    # noinspection PyUnresolvedReferences
    def lens_correction(self, rgb, metadata):
        """Apply lens correction using lensfunpy."""
        db = lensfunpy.Database()
        try:
            cam = db.find_cameras(metadata['EXIF:Make'], metadata['EXIF:Model'], loose_search=True)[0]
            lens = db.find_lenses(cam, metadata['EXIF:LensMake'], metadata['EXIF:LensModel'], loose_search=True)[0]
        except (KeyError, IndexError):
            cam, lens = self.find_data(metadata)
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
        mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
        mod.initialize(focal_length, aperture, pixel_format=np.float64)
        undistorted_cords = mod.apply_geometry_distortion()
        rgb = np.clip(lensfunpy_util.remap(rgb, undistorted_cords), a_min=0, a_max=None)
        mod.apply_color_modification(rgb)
        return rgb

    def find_data(self, metadata):
        """Search for camera and lens name in metadata"""
        values = list(metadata.values())
        cam, lens = None, None
        for key in self.CAMERA_DB:
            if key in values:
                cam = self.CAMERA_DB[key].split(':')
        for key in self.LENS_DB:
            if key in values:
                lens = self.LENS_DB[key].split(':')
        return cam, lens

    def film_emulation(self, rgb, metadata, exp_comp=0., gamma=1.):
        global cp, cdimage
        if not self.cuda:
            cp = np
            cdimage = ndimage
        rgb = cp.asarray(rgb, dtype=cp.float32)

        if self.rotation:
            rgb = self.rotate(rgb)
        if self.wb == 'tungsten':
            daylight_rgb = self.kelvin_to_BT2020(5600)
            tungsten_rgb = self.kelvin_to_BT2020(4400)
            rgb = cp.dot(rgb, cp.diag(daylight_rgb / tungsten_rgb))

        lower, upper, max_amount = 2400, 8000, 1200
        if self.wb == 'standard':
            if self.cuda:
                image_kelvin = Raw2Film.BT2020_to_kelvin(cp.mean(rgb, axis=(0, 1)).get())
            else:
                image_kelvin = Raw2Film.BT2020_to_kelvin(cp.mean(rgb, axis=(0, 1)))
            value, target = image_kelvin, image_kelvin
            if image_kelvin <= lower:
                value, target = lower, lower + max_amount
            elif lower < image_kelvin < lower + 2 * max_amount:
                target = 0.5 * value + 0.5 * (lower + 2 * max_amount)
            elif upper - max_amount < image_kelvin < upper:
                target = 0.5 * value + 0.5 * (upper - max_amount)
            elif upper <= image_kelvin:
                value, target = upper, upper - max_amount
            rgb *= self.kelvin_to_BT2020(target) / self.kelvin_to_BT2020(value)

        # crop to specified aspect ratio
        if self.crop:
            rgb = self.crop_image(rgb, aspect=self.width / self.height)

        # adjust exposure
        if 'EXIF:FNumber' in metadata and metadata['EXIF:FNumber']:
            rgb *= metadata['EXIF:FNumber'] ** 2 / metadata['EXIF:ISO'] / metadata['EXIF:ExposureTime']
        else:
            rgb *= 4 ** 2 / metadata['EXIF:ISO'] / metadata['EXIF:ExposureTime']
        # adjust exposure if ND filter is used on Fuji X100 camera (sadly imprecise)
        if ('x100' in metadata['EXIF:Model'].lower() and metadata['EXIF:BrightnessValue'] > 3
                and metadata['Composite:LightValue'] - metadata['EXIF:BrightnessValue'] < 1.5):
            rgb *= 8
        exposure = self.calc_exposure(rgb)
        middle, max_under, max_over, slope, slope_offset = -3, -.75, .66, .9, .5
        lower_bound = -exposure + middle + max_under
        sloped = -slope * exposure + middle + slope_offset
        upper_bound = -exposure + middle + max_over
        adjustment = max(lower_bound, min(sloped, upper_bound))
        rgb *= 2 ** (adjustment + exp_comp)

        # texture
        scale = max(rgb.shape) / self.width  # pixels per mm

        if self.halation:
            blured = self.exponential_blur(rgb, scale / 4)
            color_factors = cp.dot(cp.array([1.2, 0.5, 0], dtype=cp.float32), self.REC709_TO_REC2020)
            rgb += cp.multiply(blured, color_factors)
            rgb = cp.divide(rgb, color_factors + 1)

        if self.resolution:
            rgb = cp.log(rgb + 2 ** -16) / cp.log(2)
            rgb = self.film_sharpness(rgb, scale)
            if not self.grain:
                rgb = cp.exp(rgb * cp.log(2)) - 2 ** -16

        if self.grain:
            if not self.resolution:
                rgb = cp.log(rgb + 2 ** -16) / cp.log(2)
            # compute scaling factor of exposure rms in regard to measuring device size
            std_factor = math.sqrt((math.pi * 0.024) ** 2 * scale ** 2)
            strength = 1.
            if not self.bw_grain:
                noise = np.dot(torch.empty(rgb.shape, dtype=torch.float32).normal_(), self.REC709_TO_REC2020)
                rough_noise = cp.multiply(noise, cp.array([9, 10, 23], dtype=cp.float32) * std_factor / 1000 * strength)
                clean_noise = cp.multiply(noise, cp.array([4, 5, 12], dtype=cp.float32) * std_factor / 1000 * strength)
            else:
                noise = torch.empty(rgb.shape[:2], dtype=torch.float32).normal_().numpy()
                rough_noise = noise * (10 * std_factor / 1000 / math.sqrt(3) * strength)
                clean_noise = noise * (5 * std_factor / 1000 / math.sqrt(3) * strength)
            rough_scale, clean_scale = 0.005, 0.002
            if scale * rough_scale * 2 * math.sqrt(math.pi) > 1:
                rough_noise = self.gaussian_blur(rough_noise, scale * rough_scale) * (
                        scale * rough_scale * 2 + math.sqrt(math.pi))
            if scale * clean_scale * 2 * math.sqrt(math.pi) > 1:
                clean_noise = self.gaussian_blur(clean_noise, scale * clean_scale) * (
                        scale * clean_scale * 2 + math.sqrt(math.pi))
            if self.bw_grain:
                rough_noise = np.repeat(rough_noise[:, :, np.newaxis], 3, axis=2)
                clean_noise = np.repeat(clean_noise[:, :, np.newaxis], 3, axis=2)
            noise_blending = cp.clip((1 / 11) * (rgb + 1) + 0.5, a_min=0, a_max=1) ** (1 / 3)
            rgb += rough_noise * (1 - noise_blending) + clean_noise * noise_blending
            rgb = cp.exp(rgb * cp.log(2)) - 2 ** -16

        # adjust gamma while preserving middle grey exposure
        if gamma != 1.:
            lum_mat = cp.dot(rgb, cp.dot(cp.asarray(self.REC2020_TO_REC709, dtype=cp.float32),
                                         cp.asarray(cp.array([.2127, .7152, .0722], dtype=cp.float32))))
            gamma_mat = 0.2 * (5 * cp.clip(lum_mat, a_min=0, a_max=None)) ** gamma

            rgb = cp.multiply(rgb, cp.dstack([cp.divide(gamma_mat, lum_mat)] * 3))

        if self.cuda:
            rgb = rgb.get()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp.get_default_memory_pool().free_all_blocks()

        return rgb

    def rotate(self, rgb):
        global cp, cdimage
        if not self.cuda:
            cp = np
            cdimage = ndimage
        degrees = self.rotation % 360

        while degrees > 45:
            rgb = cp.rot90(rgb, k=1)
            degrees -= 90
        if degrees:
            input_height = rgb.shape[0]
            input_width = rgb.shape[1]
            rgb = cdimage.rotate(rgb, angle=degrees, reshape=True)
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

    def crop_image(self, rgb, aspect=1.5):
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

        if self.zoom > 1:
            x, y, c = rgb.shape
            zoom_factor = (self.zoom - 1) / (2 * self.zoom)
            x = round(zoom_factor * x)
            y = round(zoom_factor * y)
            rgb = rgb[x: -x, y: -y, :]

        return rgb

    def calc_exposure(self, rgb, lum_vec=np.array([.2127, .7152, .0722]), crop=.8):
        """Calculates exposure value of the rgb image."""
        global cp
        if not self.cuda:
            cp = np
        lum_mat = cp.dot(rgb, cp.dot(cp.asarray(self.REC2020_TO_REC709), cp.asarray(cp.array([.2127, .7152, .0722]))))

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
        return cp.average(cp.log(lum_mat + cp.ones_like(lum_mat) * 2 ** -16)) / cp.log(2)

    def find_exp(self, src, fallback_exp=0.):
        """Search for exposure compensation in already processed file."""
        # search for candidate files
        options = []
        for path in Path().rglob(f'*{src.split(".")[0]}*.jpg'):
            path = str(path)
            split_path = path.split("\\")
            options.append((path, split_path[-1].split('.')[0], len(split_path)))

        # return fallback if no candidates found
        if not options:
            return fallback_exp, self.gamma

        # pick candidate with the shortest path and the shortest name
        path = sorted(options, key=operator.itemgetter(2, 1))[0][0]

        # return exposure compensation value of reference file
        with exiftool.ExifToolHelper() as et:
            meta = et.get_metadata(path)[0]
            exp_comp = meta['EXIF:ExposureCompensation']
            try:
                gamma = meta['EXIF:Gamma']
            except KeyError:
                gamma = self.gamma
        return exp_comp, gamma

    def gaussian_filter(self, input, sigma=1.):
        """Compute gaussian filter"""
        if self.cuda:
            return cdimage.gaussian_filter(input, sigma=sigma)
        else:
            return cv.GaussianBlur(input, ksize=(0, 0), sigmaX=sigma)

    def gaussian_blur(self, rgb, sigma=1.):
        """Applies gaussian blur per channel of rgb image."""
        if self.cuda:
            r, g, b = cp.dsplit(rgb, 3)
            r = self.gaussian_filter(r, sigma=sigma)
            g = self.gaussian_filter(g, sigma=sigma)
            b = self.gaussian_filter(b, sigma=sigma)
            return cp.dstack((r, g, b))
        else:
            return cv.GaussianBlur(rgb, ksize=(0, 0), sigmaX=sigma)

    @staticmethod
    def mtf_curve(x, a, b, c, d):
        gauss = lambda z: a * math.exp(- (x - b) ** 2 / (2 * (b / (2 * math.pi)) ** 2))
        linear = lambda z: c * z + d

        y_1 = gauss(x)
        y_2 = linear(x)

        base = 1000

        return y_1 + y_2 - math.log(base ** y_1 + base ** y_2, base)

    @staticmethod
    def film_sharpness(rgb, scale):
        red_mtf = lambda x: 10 ** Raw2Film.mtf_curve(math.log10(x + 0.000001), 0.019, 1.06, -1.41, 1.95)
        green_mtf = lambda x: 10 ** Raw2Film.mtf_curve(math.log10(x + 0.000001), 0.0547, 1.197, -1.06, 1.59)
        blue_mtf = lambda x: 10 ** Raw2Film.mtf_curve(math.log10(x + 0.000001), 0.059, 1.287, -1.3, 2.07)

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

    def exponential_blur(self, rgb, size):
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

    @staticmethod
    def save_tiff(src, rgb):
        # convert to arri wcg
        rgb = np.dot(rgb, Raw2Film.REC2020_TO_ARRIWCG)
        # compute achromaticity (max rgb value per pixel)
        achromatic = np.repeat(np.max(rgb, axis=2)[:, :, np.newaxis], 3, axis=2)

        # compute distance to gamut
        distance = (achromatic - rgb) / achromatic

        # smoothing parameter
        a = 0.2
        # precompute smooth compression function
        x = np.linspace(1 - a, 1 + a, 16)
        y = 1 - a + (x - 1 + a) / (np.sqrt(1 + ((x - 1) / a + 1) ** 2))
        # compress distance
        distance = np.interp(distance, np.concatenate((np.array([0]), x)), np.concatenate((np.array([0]), y)))

        rgb = achromatic - distance * achromatic
        # convert to arri log C

        rgb = Raw2Film.encode_ARRILogC3(rgb)

        rgb = (rgb * (2 ** 16 - 1)).astype(dtype='uint16')

        imageio.imwrite(src.split(".")[0] + "_log.tiff", rgb)

    @staticmethod
    def encode_ARRILogC3(x):
        cut, a, b, c, d, e, f = 0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809

        return np.where(x > cut, (c / np.log(10)) * np.log(a * x + b) + d, e * x + f)

    def apply_lut(self, src, index, metadata):
        """Loads tiff file and applies LUT, generates jpg."""
        file_name = src.split('.')[0]
        if self.rename:
            date_str = metadata['EXIF:DateTimeOriginal'].translate({ord(i): None for i in ' :'}) + src.split(".")[0][
                                                                                                   -3:]
            if len(self.luts) == 1:
                file_name = f"IMG_{date_str}"
            else:
                file_name = f"IMG_{index + 1}_BURST{date_str}"
                if index == 0:
                    file_name += "_COVER"
        else:
            if index:
                reference = Raw2Film.lut_name_ending(self.luts[0])
                for ending in Raw2Film.lut_name_ending(self.luts[index]):
                    if ending not in reference:
                        file_name += '_' + ending
        if os.path.exists(file_name + '.tiff'):
            os.remove(file_name + ".tiff")
        if os.path.exists(self.luts[index]):
            lut_path = self.luts[index]
        else:
            for folder in ['lut', 'luts', 'cube', 'looks', 'look', 'style']:
                if os.path.exists(folder + "/" + self.luts[index]):
                    lut_path = folder + "/" + self.luts[index]
                    break
        try:
            lut_path
        except NameError:
            print(f"LUT {self.luts[index]} not found")
            return
        ffmpeg.input(src.split('.')[0] + "_log.tiff").filter('lut3d', file=lut_path).output(
            file_name + '.tiff', loglevel="quiet").run()
        return file_name + '.tiff'

    @staticmethod
    def lut_name_ending(name):
        name = name.split('/')[-1]
        name = name.split('\\')[-1]
        name = name.split('.')[0]
        return name.split('_')

    def convert_jpg(self, src):
        """Converts to jpg and removes src tiff file."""
        image = Image.open(src).convert('RGB')
        os.remove(src)
        if self.canvas:
            image = self.add_canvas(np.array(image))
        imageio.imwrite(src.split('.')[0] + '.jpg', image, quality=100)
        return src.split('.')[0] + '.jpg'

    def add_canvas(self, image):
        """Adds background canvas to image."""
        img_ratio = image.shape[1] / image.shape[0]
        if img_ratio > self.output_ratio:
            output_resolution = (
                int(image.shape[1] / self.output_ratio * self.output_scale), int(image.shape[1] * self.output_scale))
        else:
            output_resolution = (
                int(image.shape[0] * self.output_scale), int(image.shape[0] * self.output_ratio * self.output_scale))
        offset = np.subtract(output_resolution, image.shape[:2]) // 2
        canvas = np.tensordot(np.ones(output_resolution), self.output_color, axes=0)
        canvas[offset[0]:offset[0] + image.shape[0], offset[1]:offset[1] + image.shape[1]] = image
        return canvas.astype(dtype='uint8')

    def add_metadata(self, src, metadata, exp_comp, gamma):
        """Adds metadata to image file."""
        metadata = {key: metadata[key] for key in metadata if key.startswith("EXIF") and key[5:] in self.METADATA_KEYS}
        metadata['EXIF:Artist'] = self.artist
        metadata['EXIF:ExposureCompensation'] = exp_comp
        metadata['EXIF:Gamma'] = gamma
        with exiftool.ExifToolHelper() as et:
            et.set_tags([src], metadata, '-overwrite_original')

    def organize_files(self, src, file_list, metadata):
        """Moves files into target folders."""
        # create path
        path = f"{metadata['EXIF:DateTimeOriginal'][:4]}/{metadata['EXIF:DateTimeOriginal'][:10].replace(':', '-')}/"

        # move files
        self.move_file(src, path + '/RAW/')
        for index, file in enumerate(file_list):
            path_extension = ""
            if index:
                if self.luts[0].split('_')[0] == self.luts[index].split('_')[0]:
                    path_extension = self.luts[index].split('_')[-1].split('.')[0] + '/'
                else:
                    path_extension += self.luts[index].split('_')[0].split('.')[0] + '/'
            self.move_file(file, path + path_extension)

    @staticmethod
    def move_file(src, path):
        """Moves src file to path."""
        if not os.path.exists(path):
            os.makedirs(path)
        os.replace(src, path + src)


def init_child(semaphore_):
    global semaphore
    semaphore = semaphore_


def fraction(arg):
    if "/" in str(arg):
        return float(arg.split('/')[0]) / float(arg.split('/')[1])
    else:
        return float(arg)


def hex_color(arg):
    if str(arg) == "white":
        return [255, 255, 255]
    if str(arg) == "black":
        return [0, 0, 0]
    return list(int(arg[i:i + 2], 16) for i in (0, 2, 4))


def main():
    parser = argparse.ArgumentParser(
        description="Develop and organize all raw files in the current directory by running processing.py.")
    parser.add_argument('file', default=None, nargs='*', type=str,
                        help="Name of file from subfolder to edit without extension. Can also be range with '-'")
    parser.add_argument('--formats', action='store_true', help="Print built-in film formats.")
    parser.add_argument('--list_cameras', action='store_true', help="Print all cameras from lensfunpy.")
    parser.add_argument('--list_lenses', action='store_true', help="Print all lenses from lensfunpy.")
    parser.add_argument('--cleanup', action='store_true',
                        help="Delete RAW files if JPEG was deleted. Requires files to be specified")
    parser.add_argument('--format', type=str, choices=Raw2Film.FORMATS.keys(), default=None, help="Select film format")
    parser.add_argument('--no-crop', dest='crop', action='store_false', help="Preserve source aspect ratio.")
    parser.add_argument('--no-resolution', dest='resolution', action='store_false', help="Turn off blur and sharpen.")
    parser.add_argument('--no-halation', dest='halation', action='store_false', help="Turn off halation.")
    parser.add_argument('--no-grain', dest='grain', help="Turn off halation.", action='store_false')
    parser.add_argument('--no-organize', dest='organize', action='store_false', help="Do no organize files.")
    parser.add_argument('--no-correct', dest='correct', action='store_false', help="Turn off lens correction")
    parser.add_argument('--canvas', action='store_true', help="Add canvas to output image.")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="Turn off GPU acceleration.")
    parser.add_argument('--wb', default='standard', choices=['standard', 'auto', 'daylight', 'tungsten', 'camera'],
                        help="Specify white balance mode.")
    parser.add_argument('--tiff', action='store_true',
                        help="Output ARRI LogC3 .tiff files. Used to test and develop LUTs.")
    parser.add_argument('--rename', action='store_true',
                        help="Rename to match Google Photos photo stacking naming scheme")
    parser.add_argument('--keep-exp', dest='keep_exp', action='store_true',
                        help="Keep the exposure and gamma of previously rendered images.")
    parser.add_argument('--bw-grain', dest='bw_grain', action='store_true',
                        help="Monochrome insted of full color grain")
    parser.add_argument('--exp', type=fraction, default=0, help="By how many stops to adjust exposure")
    parser.add_argument('--gamma', type=fraction, default=1.,
                        help="Adjust gamma curve without effecting middle exposure to change contrast")
    parser.add_argument('--width', type=fraction, default=36, help="Simulated film width in mm.")
    parser.add_argument('--height', type=fraction, default=24, help="Simulated film height in mm.")
    parser.add_argument('--ratio', type=fraction, default="4/5", help="Canvas aspect ratio.")
    parser.add_argument('--scale', type=fraction, default=1., help="Canvas border scale.")
    parser.add_argument('--rotation', type=fraction, default=0., help="Angle by which to rotate image.")
    parser.add_argument('--color', type=hex_color, default="000000", help="Color of canvas as hex value.")
    parser.add_argument('--artist', type=str, default="Jan Lohse", help="Artist name in metadata.")
    parser.add_argument('--luts', type=str, default=["Fuji_Natural.cube"], nargs='+',
                        help="Specify list of LUTs separated by comma.")
    parser.add_argument('--nd', type=int, default=1, help="0:No ND adjustment. 1: Automatic 3 stop ND recognition "
                                                          "for Fuji X100 cameras. 2: Force 3 stop ND adjustment.")
    parser.add_argument('--cores', type=int, default=0,
                        help="How many cpu threads to use. Default is maximum available")

    if os.path.isfile('config.txt'):
        parser.add_argument('--config', is_config_file=True, default='config.txt')

    args = parser.parse_args()

    if not args.cores:
        args.cores = os.cpu_count()
    if args.formats:
        return formats_message()
    if args.list_cameras:
        return list_cameras()
    if args.list_lenses:
        return list_lenses()
    if args.cleanup:
        for file in args.file:
            cleanup_files(file)
        return
    if args.format:
        args.width, args.height = Raw2Film.FORMATS[args.format]

    if args.file:
        files = []
        for file in args.file:
            files += copy_from_subfolder(file)
        if not files:
            print("No matching files have been found.")
    else:
        files = [file for file in os.listdir() if file.lower().endswith(Raw2Film.EXTENSION_LIST)]
    if not files:
        return

    if args.cuda and not is_cupy_available:
        args.cuda = False
        warnings.formatwarning = lambda msg, cat, *args, **kwargs: f'{cat.__name__}: {msg}\n'
        warnings.warn("Cupy not working. Turning off gpu acceleration. Supress warning with --no-cuda option.")

    raw2film = Raw2Film(**vars(args))

    start = time.time()
    counter = 1
    result = raw2film.process_runner((0, files.pop(0)))
    if result:
        print(f"{result} processed successfully {counter}/{len(files) + 1}")
    else:
        cleaner(raw2film)
        sys.exit()
    end = time.time()
    raw2film.sleep_time = (end - start) / args.cores

    semaphore = Semaphore(1)
    with Pool(args.cores, initializer=init_child, initargs=(semaphore,)) as p:
        try:
            for result in p.imap_unordered(raw2film.process_runner, enumerate(files)):
                if result:
                    counter += 1
                    print(f"{result} processed successfully {counter}/{len(files) + 1}")
                if not result:
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            p.terminate()
            p.join()
            cleaner(raw2film)


def cleaner(raw2film):
    print("terminating...")
    time.sleep(1)
    for file in os.listdir():
        if (raw2film.organize and file.endswith('.jpg')) or (not raw2film.tiff and file.endswith('.tiff')):
            os.remove(file)


def formats_message():
    """Outputs all built-in formats."""
    key_length = max([len(key) for key in Raw2Film.FORMATS])
    print(f"key {' ' * (key_length - 3)} width mm x height mm")
    for key in Raw2Film.FORMATS:
        print(f"{key} {' ' * (key_length - len(key))} {Raw2Film.FORMATS[key][0]} mm x {Raw2Film.FORMATS[key][1]} mm")


def list_cameras():
    """Output cameras from lensfunpy"""
    # noinspection PyUnresolvedReferences
    db = lensfunpy.Database()
    for camera in db.cameras:
        print(camera.maker, ":", camera.model)
    return


def list_lenses():
    """Output lenses from lensfunpy."""
    # noinspection PyUnresolvedReferences
    db = lensfunpy.Database()
    for lens in db.lenses:
        print(lens.maker, ":", lens.model)
    return


def copy_from_subfolder(file):
    name_start, name_end = prep_file_name(file)

    files = []

    for path in Path().rglob('./*.*'):
        filename = str(path).split('\\')[-1]
        name = filename.split('.')[0]
        if (name_start <= name <= name_end and filename.lower().endswith(Raw2Film.EXTENSION_LIST)
                and filename not in files):
            files.append(filename)
            if not os.path.isfile(filename):
                copy(path, '.', )

    return files


def cleanup_files(file):
    if not file:
        print("Specify the files to clean to avoid errors")
        return

    name_start, name_end = prep_file_name(file)

    for path in Path().rglob('./*/*.*'):
        filename = str(path).split('\\')[-1]
        name = filename.split('.')[0].split('_')[0]
        if name_start <= name <= name_end and (filename.lower().endswith(Raw2Film.EXTENSION_LIST) or
                                               (filename.lower().endswith('jpg') and '_' in filename)):
            if not any(Path().rglob(f'*{name}.jpg')) and not os.path.isfile(filename):
                print("deleted", filename)
                os.remove(path)

    # remove empty subfolders
    for dir_path, dir_names, _ in os.walk('.', topdown=False):
        for dir_name in dir_names:
            full_path = os.path.join(dir_path, dir_name)
            if not os.listdir(full_path) and '20' in full_path:
                print("deleted", full_path)
                os.rmdir(full_path)


def prep_file_name(file):
    name_start = file.split('-')[0]
    if '-' in file:
        end = file.split('-')[1]
        name_end = name_start[:-len(end)] + end
    else:
        name_end = file
    if name_start > name_end:
        name_start, name_end = name_end, name_start

    return name_start, name_end


# runs image processing on all raw files in parallel
if __name__ == '__main__':
    main()
