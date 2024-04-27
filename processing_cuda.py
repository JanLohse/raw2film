import argparse
import math
import os
import sys
import time
from multiprocessing import Pool, Semaphore

import colour
import exiftool
import ffmpeg
import imageio.v2 as imageio
import lensfunpy
import numpy as cp
import numpy as np
import rawpy
from lensfunpy import util as lensfunpy_util
from scipy import ndimage
from scipy import ndimage as cdimage


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
                                   [0.0615, 0.1946, 1.0275]])
    REC2020_TO_REC709 = cp.array([[1.6605, -.1246, -.0182],
                                  [-.5879, 1.1330, -.1006],
                                  [-.0728, -.0084, 1.1187]])
    CAMERA_DB = {"X100S": "Fujifilm : X100S",
                 "DMC-GX80": "Panasonic : DMC-GX80",
                 "DC-FZ10002": "Panasonic : DC-FZ10002"}
    LENS_DB = {"X100S": "Fujifilm : X100 & compatibles (Standard)",
               "LUMIX G 25/F1.7": "Panasonic : Lumix G 25mm f/1.7 Asph.",
               "LUMIX G VARIO 12-32/F3.5-5.6": "Panasonic : Lumix G Vario 12-32mm f/3.5-5.6 Asph. Mega OIS",
               "DC-FZ10002": "Leica : FZ1000 & compatibles"}

    def __init__(self, crop=True, blur=True, sharpen=True, halation=True, grain=True, organize=True, canvas=False, nd=0,
                 width=36, height=24, ratio=4 / 5, scale=1., color=None, artist="Jan Lohse", luts=None, tiff=False,
                 wb='standard', exp=0, zoom=1., correct=True, cores=None, sleep_time=0, rename=False, rotation=0,
                 cuda=False, **args):
        self.crop = crop
        self.blur = blur
        self.sharpen = sharpen
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
        if self.correct:
            rgb = np.asarray(self.lens_correction(rgb, metadata))

        if not run_count:
            rgb = self.film_emulation(src, rgb, metadata)
        else:
            with semaphore:
                rgb = self.film_emulation(src, rgb, metadata)

        self.save_tiff(src, rgb, metadata)
        if self.tiff:
            return

        file_list = [self.apply_lut(src, i, metadata) for i in range(len(self.luts))]
        file_list = [self.convert_jpg(file) for file in file_list]
        os.remove(src.split('.')[0] + "_log.tiff")

        for file in file_list:
            self.add_metadata(file, metadata)
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
        XYZ = colour.RGB_to_XYZ(rgb, "ITU-R BT.2020")
        xy = colour.XYZ_to_xy(XYZ)
        CCT = colour.xy_to_CCT(xy, "Hernandez 1999")
        return CCT

    @staticmethod
    def kelvin_to_BT2020(kelvin):
        xy = colour.CCT_to_xy(kelvin, "Kang 2002")
        XYZ = colour.xy_to_XYZ(xy)
        rgb = colour.XYZ_to_RGB(XYZ, "ITU-R BT.2020")
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

    def film_emulation(self, src, rgb, metadata):
        rgb = cp.asarray(rgb)

        if self.rotation:
            rgb = self.rotate(rgb)

        if self.wb == 'tungsten':
            daylight_rgb = self.kelvin_to_BT2020(5600)
            tungsten_rgb = self.kelvin_to_BT2020(4400)
            rgb = cp.dot(rgb, cp.diag(daylight_rgb / tungsten_rgb))

        lower, upper, max_amount = 2400, 8000, 1200
        if self.wb == 'standard':
            if self.cuda:
                image_kelvin = self.BT2020_to_kelvin(cp.asarray([cp.mean(x) for x in cp.dsplit(rgb, 3)]).get())
            else:
                image_kelvin = self.BT2020_to_kelvin(cp.asarray([cp.mean(x) for x in cp.dsplit(rgb, 3)]))
            value, target = image_kelvin, image_kelvin
            if image_kelvin <= lower:
                value, target = lower, lower + max_amount
            elif lower < image_kelvin < lower + 2 * max_amount:
                target = 0.5 * value + 0.5 * (lower + 2 * max_amount)
            elif upper - max_amount < image_kelvin < upper:
                target = 0.5 * value + 0.5 * (upper - max_amount)
            elif upper <= image_kelvin:
                value, target = upper, upper - max_amount
            rgb = cp.dot(rgb, cp.diag(self.kelvin_to_BT2020(target) / self.kelvin_to_BT2020(value)))

        """Adjusts exposure, aspect ratio and texture, and outputs tiff file in ARRI LogC3 color space."""
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
        exposure = self.calc_exposure(cdimage.gaussian_filter(rgb, sigma=3))
        middle, max_under, max_over, slope, slope_offset = -3, -.75, .66, .9, .5
        lower_bound = -exposure + middle + max_under
        sloped = -slope * exposure + middle + slope_offset
        upper_bound = -exposure + middle + max_over
        adjustment = max(lower_bound, min(sloped, upper_bound))
        rgb *= 2 ** (adjustment + self.exp)

        # texture
        scale = max(rgb.shape) / (80 * self.width)

        if self.blur:
            rgb = self.gaussian_blur(rgb, sigma=.5 * scale)

        if self.sharpen:
            rgb = cp.log2(rgb + 2 ** -16)
            rgb = rgb + cp.clip(rgb - self.gaussian_blur(rgb, sigma=scale), a_min=-2, a_max=2)
            rgb = cp.exp2(rgb) - 2 ** -16

        if self.halation:
            threshold = .2
            r, g, b = cp.dsplit(np.clip(rgb - threshold, a_min=0, a_max=None), 3)
            r = cdimage.gaussian_filter(r, sigma=2.2 * scale)
            g = .8 * cdimage.gaussian_filter(g, sigma=2 * scale)
            b = cdimage.gaussian_filter(b, sigma=0.3 * scale)
            rgb += cp.clip(np.dstack((r, g, b)) - cp.clip(rgb - threshold, a_min=0, a_max=None), a_min=0, a_max=None)

        if self.grain:
            rgb = cp.log2(rgb + 2 ** -16)
            noise = cp.random.rand(*rgb.shape) - .5
            noise = cdimage.gaussian_filter(noise, sigma=.5 * scale)
            rgb += noise * (scale / 2)
            rgb = cp.exp2(rgb) - 2 ** -16

        if self.cuda:
            rgb = rgb.get()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp.get_default_memory_pool().free_all_blocks()
        return rgb

    def rotate(self, rgb):
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
            crop_height = (rgb.shape[0] - h) // 2
            crop_width = (rgb.shape[1] - w) // 2
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

    @staticmethod
    def calc_exposure(rgb, lum_vec=np.array([.2127, .7152, .0722]), crop=.8):
        """Calculates exposure value of the rgb image."""
        lum_mat = cp.dot(np.clip(cp.dot(rgb, cp.asarray(Raw2Film.REC2020_TO_REC709)), a_min=0, a_max=None),
                         cp.asarray(lum_vec))
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
        return cp.average(cp.log2(lum_mat + cp.ones_like(lum_mat) * 2 ** -16))

    @staticmethod
    def gaussian_blur(rgb, sigma=1):
        """Applies gaussian blur per channel of rgb image."""
        r, g, b = cp.dsplit(rgb, 3)
        r = cdimage.gaussian_filter(r, sigma=sigma)
        g = cdimage.gaussian_filter(g, sigma=sigma)
        b = cdimage.gaussian_filter(b, sigma=sigma)
        return cp.dstack((r, g, b))

    def save_tiff(self, src, rgb, metadata):
        rgb = colour.models.log_encoding_ARRILogC3(rgb)
        rgb = np.clip(np.dot(rgb, self.REC2020_TO_ARRIWCG), a_min=0, a_max=1)
        rgb = (rgb * (2 ** 16 - 1)).astype(dtype='uint16')
        imageio.imsave(src.split(".")[0] + "_log.tiff", rgb)

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
                if self.luts[0].split('_')[0] == self.luts[index].split('_')[0]:
                    file_name += '_' + self.luts[index].split('_')[-1].split('.')[0]
                else:
                    file_name += '_' + self.luts[index].split('_')[0].split('.')[0]
        if os.path.exists(file_name + '.tiff'):
            os.remove(file_name + ".tiff")
        ffmpeg.input(src.split('.')[0] + "_log.tiff").filter('lut3d', file=self.luts[index]).output(
            file_name + '.tiff', loglevel="quiet").run()
        return file_name + '.tiff'

    def convert_jpg(self, src):
        """Converts to jpg and removes src tiff file."""
        image = (imageio.imread(src) / 2 ** 8).astype(dtype='uint8')
        os.remove(src)
        if self.canvas:
            image = self.add_canvas(image)
        imageio.imsave(src.split('.')[0] + '.jpg', image, quality=100)
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

    def add_metadata(self, src, metadata):
        """Adds metadata to image file."""
        metadata = {key: metadata[key] for key in metadata if key.startswith("EXIF") and key[5:] in self.METADATA_KEYS}
        metadata['EXIF:Artist'] = self.artist
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
    parser.add_argument('--formats', dest='formats', help="Print built-in film formats.", default=False, const=True,
                        nargs='?')
    parser.add_argument('--list_cameras', dest='list_cameras', help="Print all cameras from lensfunpy.", default=False,
                        const=True, nargs='?')
    parser.add_argument('--list_lenses', dest='list_lenses', help="Print all lenses from lensfunpy.", default=False,
                        const=True, nargs='?')
    parser.add_argument('--no-crop', dest='crop', help="Preserve source aspect ratio.", default=True, const=False,
                        nargs='?')
    parser.add_argument('--no-blur', dest='blur', help="Turn off gaussian blur filter.", default=True, const=False,
                        nargs='?')
    parser.add_argument('--no-sharpen', dest='sharpen', help="Turn off sharpening filter.", default=True, const=False,
                        nargs='?')
    parser.add_argument('--no-halation', dest='halation', help="Turn off halation.", default=True, const=False,
                        nargs='?')
    parser.add_argument('--no-grain', dest='grain', help="Turn off halation.", default=True, const=False, nargs='?')
    parser.add_argument('--no-organize', dest='organize', help="Do no organize files.", default=True, const=False,
                        nargs='?')
    parser.add_argument('--no-correct', dest='correct', help="Turn off lens correction", default=True, const=False,
                        nargs='?')
    parser.add_argument('--canvas', dest='canvas', help="Add canvas to output image.", default=False, const=True,
                        nargs='?')
    parser.add_argument('--no-cuda', dest='cuda', help="Turn off GPU acceleration.", default=True, const=False,
                        nargs='?')
    parser.add_argument('--wb', dest='wb', help="Specify white balance mode.", default='standard',
                        choices=['standard', 'auto', 'daylight', 'tungsten', 'camera'])
    parser.add_argument('--daylight_wb', dest='daylight_wb', help="Forces the use of daylight white balance.",
                        default=False, const=True, nargs='?')
    parser.add_argument('--tiff', dest='tiff', help="Output ARRI LogC3 .tiff files. Used to test and develop LUTs.",
                        default=False, const=True, nargs='?')
    parser.add_argument('--rename', dest='rename', help="Rename to match Google Photos photo stacking naming scheme",
                        default=False, const=True, nargs='?')
    parser.add_argument('--exp', dest='exp', help="By how many stops to adjust exposure", type=fraction, default=0)
    parser.add_argument('--width', dest='width', help="Simulated film width in mm.", type=fraction, default=36)
    parser.add_argument('--height', dest='height', help="Simulated film height in mm.", type=fraction, default=24)
    parser.add_argument('--ratio', dest='ratio', help="Canvas aspect ratio.", type=fraction, default="4/5")
    parser.add_argument('--scale', dest='scale', help="Canvas border scale.", type=fraction, default=1.)
    parser.add_argument('--rotation', dest='rotation', help="Angle by which to rotate image.", type=fraction,
                        default=0.)
    parser.add_argument('--color', dest='color', help="Color of canvas as hex value.", type=hex_color, default="000000")
    parser.add_argument('--artist', dest='artist', help="Artist name in metadata.", type=str, default="Jan Lohse")
    parser.add_argument('--luts', dest='luts', help="Specify list of LUTs separated by comma.", type=str,
                        default=["Fuji_Standard.cube", "BW.cube"], nargs='+')
    parser.add_argument('--nd', dest='nd',
                        help="0:No ND adjustment. 1: Automatic 3 stop ND recognition for Fuji X100 cameras. "
                             "2: Force 3 stop ND adjustment.", type=int, default=1)
    parser.add_argument('--cores', dest='cores', help="How many cpu threads to use. Default is maximum available",
                        type=int, default=0)

    args = parser.parse_args()

    if not args.cores:
        args.cores = os.cpu_count()
    if args.formats:
        return formats_message()
    if args.list_cameras:
        return list_cameras()
    if args.list_lenses:
        return list_lenses()

    if args.cuda:
        try:
            global cp, cdimage
            from cupyx.scipy import ndimage as cdimage
            import cupy as cp
            cp.cuda.set_allocator(None)
            cp.cuda.set_pinned_memory_allocator(None)
        except ImportError:
            args.cuda = False
            raise Warning("Cupy not working. Turning off gpu acceleration. Supress warning with --no-cuda option.")

    raw2film = Raw2Film(**vars(args))
    files = [file for file in os.listdir() if file.lower().endswith(Raw2Film.EXTENSION_LIST)]
    if not files:
        return

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


# runs image processing on all raw files in parallel
if __name__ == '__main__':
    main()
