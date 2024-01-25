import os
import sys
from multiprocessing import Pool

import colour
import exiftool
import ffmpeg
import imageio.v2 as imageio
import lensfunpy
import numpy as np
import rawpy
from lensfunpy import util as lensfunpy_util
from scipy import ndimage


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
    REC2020_TO_REC709 = np.array([[1.6605, -.1246, -.0182],
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
                 auto_wb=False, camera_wb=False, tungsten_wb=False, daylight_wb=False, exp=0, zoom=1., correct=True):
        if luts is None:
            luts = ["Fuji_Vibrant.cube", "BW.cube"]
        if color is None:
            color = [0, 0, 0]
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
        self.auto_wb = auto_wb
        self.camera_wb = camera_wb
        self.tungsten_wb = tungsten_wb
        self.daylight_wb = daylight_wb
        self.tiff = tiff
        self.exp = exp
        self.zoom = zoom
        self.nd = nd
        self.correct = correct

    def process_image(self, src):
        """Manages image processing pipeline."""
        rgb, metadata = self.raw_to_linear(src)

        if self.correct:
            rgb = self.lens_correction(rgb, metadata)

        self.film_emulation(src, rgb, metadata)

        if self.tiff:
            return

        file_list = [self.apply_lut(src, self.luts, i) for i in range(len(self.luts))]
        file_list = [self.convert_jpg(file) for file in file_list]
        os.remove(src.split('.')[0] + "_log.tiff")

        for file in file_list:
            self.add_metadata(file, metadata)
        if self.organize:
            self.organize_files(src, file_list, metadata)

        print(f"{src} processed successfully", flush=True)

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
                                  use_camera_wb=self.camera_wb, use_auto_wb=self.auto_wb,
                                  demosaic_algorithm=rawpy.DemosaicAlgorithm(11), four_color_rgb=True)
        rgb = rgb.astype(dtype='float32')
        rgb /= 2 ** 16 - 1

        if not self.camera_wb and not self.auto_wb and not self.daylight_wb and self.tungsten_wb:
            daylight_rgb = self.kelvin_to_BT2020(5600)
            tungsten_rgb = self.kelvin_to_BT2020(4400)
            rgb = np.dot(rgb, np.diag(daylight_rgb / tungsten_rgb))

        lower, upper, max_amount = 2400, 8000, 1200
        if not self.camera_wb and not self.auto_wb and not self.daylight_wb and not self.tungsten_wb:
            image_kelvin = self.BT2020_to_kelvin([np.mean(x) for x in np.dsplit(rgb, 3)])
            value, target = image_kelvin, image_kelvin
            if image_kelvin <= lower:
                value, target = lower, lower + max_amount
            elif lower < image_kelvin < lower + 2 * max_amount:
                target = 0.5 * value + 0.5 * (lower + 2 * max_amount)
            elif upper - max_amount < image_kelvin < upper:
                target = 0.5 * value + 0.5 * (upper - max_amount)
            elif upper <= image_kelvin:
                value, target = upper, upper - max_amount
            rgb = np.dot(rgb, np.diag(self.kelvin_to_BT2020(target) / self.kelvin_to_BT2020(value)))

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
        return rgb

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
        """Adjusts exposure, aspect ratio and texture, and outputs tiff file in ARRI LogC3 color space."""
        # crop to specified aspect ratio
        if self.crop:
            rgb = self.crop_image(rgb, aspect=self.width / self.height)

        # adjust exposure
        if 'EXIF:FNumber' in metadata:
            rgb *= metadata['EXIF:FNumber'] ** 2 / metadata['EXIF:ISO'] / metadata['EXIF:ExposureTime']
        else:
            rgb *= 4 ** 2 / metadata['EXIF:ISO'] / metadata['EXIF:ExposureTime']
        # adjust exposure if ND filter is used on Fuji X100 camera (sadly imprecise)
        if ('x100' in metadata['EXIF:Model'].lower() and metadata['EXIF:BrightnessValue'] > 3
                and metadata['Composite:LightValue'] - metadata['EXIF:BrightnessValue'] < 1.33):
            rgb *= 8
        exposure = self.calc_exposure(ndimage.gaussian_filter(rgb, sigma=3))
        middle, max_under, max_over, slope, slope_offset = -3, -.75, 1., .9, .5
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
            rgb = np.log2(rgb + 2 ** -16)
            rgb = rgb + np.clip(rgb - self.gaussian_blur(rgb, sigma=scale), a_min=-2, a_max=2)
            rgb = np.exp2(rgb) - 2 ** -16

        if self.halation:
            threshold = .2
            r, g, b = np.dsplit(np.clip(rgb - threshold, a_min=0, a_max=None), 3)
            r = ndimage.gaussian_filter(r, sigma=2.2 * scale)
            g = .8 * ndimage.gaussian_filter(g, sigma=2 * scale)
            b = ndimage.gaussian_filter(b, sigma=0.3 * scale)
            rgb += np.clip(np.dstack((r, g, b)) - np.clip(rgb - threshold, a_min=0, a_max=None), a_min=0, a_max=None)

        if self.grain:
            rgb = np.log2(rgb + 2 ** -16)
            noise = np.random.rand(*rgb.shape) - .5
            noise = ndimage.gaussian_filter(noise, sigma=.5 * scale)
            rgb += noise * (scale / 2)
            rgb = np.exp2(rgb) - 2 ** -16

        # generate logarithmic tiff file
        rgb = colour.models.log_encoding_ARRILogC3(rgb)
        rgb = np.clip(np.dot(rgb, self.REC2020_TO_ARRIWCG), a_min=0, a_max=1)
        rgb = (rgb * (2 ** 16 - 1)).astype(dtype='uint16')
        imageio.imsave(src.split(".")[0] + "_log.tiff", rgb)

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
        lum_mat = np.dot(np.clip(np.dot(rgb, Raw2Film.REC2020_TO_REC709), a_min=0, a_max=None), lum_vec)
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
        return np.average(np.log2(lum_mat + np.ones_like(lum_mat) * 2 ** -16))

    @staticmethod
    def gaussian_blur(rgb, sigma=1):
        """Applies gaussian blur per channel of rgb image."""
        r, g, b = np.dsplit(rgb, 3)
        r = ndimage.gaussian_filter(r, sigma=sigma)
        g = ndimage.gaussian_filter(g, sigma=sigma)
        b = ndimage.gaussian_filter(b, sigma=sigma)
        return np.dstack((r, g, b))

    @staticmethod
    def apply_lut(src, luts, index):
        """Loads tiff file and applies LUT, generates jpg."""
        extension = '.tiff'
        if index:
            if luts[0].split('_')[0] == luts[index].split('_')[0]:
                extension = '_' + luts[index].split('_')[-1].split('.')[0] + extension
            else:
                extension = '_' + luts[index].split('_')[0].split('.')[0] + extension
        if os.path.exists(src.split('.')[0] + extension):
            os.remove(src.split('.')[0] + ".tiff")
        ffmpeg.input(src.split('.')[0] + "_log.tiff").filter('lut3d', file=luts[index]).output(
            src.split('.')[0] + extension, loglevel="quiet").run()
        return src.split('.')[0] + extension

    def convert_jpg(self, src):
        """Converts to jpg and removes src tiff file."""
        image = (imageio.imread(src) / 2 ** 8).astype(dtype='uint8')
        os.remove(src)
        if self.canvas:
            image = self.add_canvas(image)
        imageio.imsave(src.split('.')[0] + ".jpg", image, quality=100)
        return src.split('.')[0] + ".jpg"

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
        if file_list:
            self.move_file(file_list.pop(0), path)
        for file in file_list:
            self.move_file(file, path + '/' + file.split('_')[-1].split('.')[0] + '/')

    @staticmethod
    def move_file(src, path):
        """Moves src file to path."""
        if not os.path.exists(path):
            os.makedirs(path)
        os.replace(src, path + src)


def main(argv):
    bool_params = ['crop', 'blur', 'sharpen', 'halation', 'grain', 'organize', 'canvas', 'camera_wb', 'auto_wb',
                   'tungsten_wb', 'daylight_wb', 'tiff', 'correct']
    float_params = ['width', 'height', 'ratio', 'scale', 'exp', 'zoom', 'nd']

    cores = None
    params = {}
    for arg in argv:
        if arg[0] == '"' and arg[1] == '"':
            arg = arg[1:-1]
        if not arg.startswith('--'):
            fail(arg)
        command = arg[2:]
        if command == 'help':
            return help_message()
        elif command == 'formats':
            return formats_message()
        elif command.startswith('cores='):
            cores = int(command.split('=')[1])
        elif command in bool_params:
            params[command] = True
        elif command == 'list_cameras':
            return list_cameras()
        elif command == 'list_lenses':
            return list_lenses()
        elif command.startswith('no-'):
            command = command[3:]
            if command not in bool_params:
                fail(arg)
            params[command] = False
        elif '=' in command:
            command, parameter = command.split('=')
            if parameter[0] == '"' and parameter[1] == '"':
                parameter = parameter[1:-1]
            if command in float_params:
                if '/' in parameter:
                    parameter = parameter.split('/')
                    parameter = float(parameter[0]) / float(parameter[1])
                else:
                    parameter = float(parameter)
                params[command] = parameter
            elif command == 'color':
                params['color'] = list(int(parameter[i:i + 2], 16) for i in (0, 2, 4))
            elif command == 'lut':
                params['luts'] = parameter.split(',')
            elif command == 'artist':
                params['artist'] = parameter
            else:
                fail(arg)
        else:
            fail(arg)

    raw2film = Raw2Film(**params)
    files = [file for file in os.listdir() if file.lower().endswith(Raw2Film.EXTENSION_LIST)]

    with Pool(cores) as p:
        p.map(raw2film.process_image, files)


def fail(arg):
    print(f"Argument {arg} unknown. Use --help to get more info.")
    sys.exit()


def help_message():
    """Outputs a guide to raw2film."""
    print("""Develop and organize all raw files in the current directory by running processing.py.

Options:
  --help            Print help.
  --formats         Print built-in film formats.
  --list_cameras    Print all cameras from lensfunpy.
  --list_lenses     Print all lenses from lensfunpy.
  --no-crop         Preserve source aspect ratio.
  --no-blur         Turn off gaussian blur filter.
  --no-sharpen      Turn off sharpening filter.
  --no-halation     Turn off halation.
  --no-grain        Turn off grain.
  --no-organize     Do not organize files.
  --canvas          Add canvas to output images.
  --auto_wb         Use automatic white balance adjustment from rawpy.
  --camera_wb       Use as-shot white balance. --auto_wb has priority if both are used.
  --tungsten_wb     Forces the use of tungsten white balance.
  --daylight_wb     Forces the use of daylight white balance.
  --tiff            Output ARRI LogC3 .tiff files. Used to test and develop LUTs.
  --no-correct      Turn off lens correction.
  --exp=<f>         Set how many stops f to increase or decrease the exposure of the output.
  --zoom=<z>        By what factor z to zoom into the original image. Value should be at least 1.
  --width=<w>       Set simulated film width to w mm.
  --height=<h>      Set simulated film height to h mm.
  --ratio=<r>       Set canvas aspect ratio to r.
          <w>/<h>   Set canvas aspect ratio to w/h.
  --scale=<s>       Multiply canvas size by s.
  --color=<hex>     Set canvas color to #hex.
  --artist=<name>   Set artist name in metadata to name.
  --lut=<1,2,...>   Set output LUTs to those listed. Separate LUT names with ',' and leave no space.
                    LUT 1 is the primary LUT. Others are saved to subfolders.
  --nd=<1>          0: Turn of ND adjustment.
                    1: Apply 3 stop nd filter adjustment on X100 cameras if deemed necessary. 
                    2: Force 3 stop nd adjustment. 
  --cores=<c>       How many (virtual) cpu cores c to use. Default is all available.

All boolean options can be used with no- or without, depending on which value is desired.
Above we show the options used to change the default value.
""")


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
    main(sys.argv[1:])
