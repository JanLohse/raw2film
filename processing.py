import os
import sys
from multiprocessing import Pool

import colour
import ffmpeg
import imageio.v2 as imageio
import numpy as np
import rawpy
from exif import Image
from scipy import ndimage


class Raw2Film:
    METADATA_KEYS = ('make', 'model', 'datetime', 'exposure_time', 'f_number', 'exposure_program',
                     'photographic_sensitivity', 'datetime_original',
                     'datetime_digitized', 'exposure_bias_value',
                     'max_aperture_value', 'metering_mode', 'light_source', 'flash', 'focal_length',
                     'subsec_time', 'subsec_time_original', 'subsec_time_digitized', 'color_space',
                     'pixel_x_dimension', 'pixel_y_dimension', 'sensing_method', 'custom_rendered', 'exposure_mode',
                     'white_balance',
                     'focal_length_in_35mm_film', 'scene_capture_type', 'gain_control')
    EXTENSION_LIST = ('.RW2', '.DNG', '.CRW', '.CR2', '.CR3', '.NEF', '.ORF', '.ORI', '.RAF', '.RWL', '.PEF', '.PTX')
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

    def __init__(self, crop=True, blur=True, sharpen=True, halation=True, grain=True, organize=True, canvas=False,
                 width=36, height=24, ratio=4 / 5, scale=1., color=None, artist='Jan Lohse', luts=None, tiff=False,
                 auto_wb=False, camera_wb=False, tungsten_wb=False, daylight_wb=False, exp=0):
        if luts is None:
            luts = ['Filmbox_Vibrant.cube', 'Filmbox_BW.cube']
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
        self.exp=exp

    def process_image(self, src):
        """Manages image processing pipeline."""
        rgb, metadata = self.raw_to_linear(src)
        self.film_emulation(src, rgb, metadata)

        if self.tiff:
            return

        file_list = [self.apply_lut(src, lut, first=(lut == self.luts[0])) for lut in self.luts]
        file_list = [self.convert_jpg(file) for file in file_list]
        os.remove(src.split('.')[0] + '_log.tiff')

        for file in file_list:
            self.add_metadata(file, metadata)
        if self.organize:
            self.organize_files(src, file_list, metadata)

        print(f"{src} processed successfully", flush=True)

    def raw_to_linear(self, src):
        """Takes raw file location and outputs linear rgb data and metadata."""
        # read metadata
        with open(src, 'rb') as img_file:
            metadata = Image(img_file)

        # convert raw file to linear data
        with rawpy.imread(src) as raw:
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
        XYZ = colour.RGB_to_XYZ(rgb, 'ITU-R BT.2020')
        xy = colour.XYZ_to_xy(XYZ)
        CCT = colour.xy_to_CCT(xy, 'Hernandez 1999')
        return CCT

    @staticmethod
    def kelvin_to_BT2020(kelvin):
        xy = colour.CCT_to_xy(kelvin, 'Kang 2002')
        XYZ = colour.xy_to_XYZ(xy)
        rgb = colour.XYZ_to_RGB(XYZ, 'ITU-R BT.2020')
        return rgb

    def film_emulation(self, src, rgb, metadata):
        """Adjusts exposure, aspect ratio and texture, and outputs tiff file in ARRI LogC3 color space."""
        # crop to specified aspect ratio
        if self.crop:
            rgb = self.crop_image(rgb, aspect=self.width / self.height)

        # adjust exposure
        if metadata.f_number:
            rgb *= metadata.f_number ** 2 / metadata.photographic_sensitivity / metadata.exposure_time
        else:
            rgb *= 4 ** 2 / metadata.photographic_sensitivity / metadata.exposure_time
        exposure = self.calc_exposure(ndimage.gaussian_filter(rgb, sigma=3))
        middle, max_under, max_over, slope, slope_offset = -3, -.75, .75, .9, .5
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
        imageio.imsave(src.split(".")[0] + '_log.tiff', rgb)

    @staticmethod
    def crop_image(rgb, aspect=1.5):
        """Crops rgb data to aspect ratio."""
        x, y, c = rgb.shape
        if x > y:
            if x > aspect * y:
                return rgb[round(x / 2 - y * aspect / 2): round(x / 2 + y * aspect / 2), :, :]
            else:
                return rgb[:, round(y / 2 - x / aspect / 2): round(y / 2 + x / aspect / 2), :]
        elif y > aspect * x:
            return rgb[:, round(y / 2 - x * aspect / 2): round(y / 2 + x * aspect / 2), :]
        else:
            return rgb[round(x / 2 - y / aspect / 2): round(x / 2 + y / aspect / 2), :, :]

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
    def apply_lut(src, lut, first=False):
        """Loads tiff file and applies LUT, generates jpg."""
        extension = '.tiff'
        if not first:
            extension = '_' + lut.split('_')[-1].split('.')[0] + extension
        if os.path.exists(src.split('.')[0] + extension):
            os.remove(src.split('.')[0] + '.tiff')
        ffmpeg.input(src.split('.')[0] + '_log.tiff').filter('lut3d', file=lut).output(src.split('.')[0] + extension,
                                                                                       loglevel='quiet').run()
        return src.split('.')[0] + extension

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
        with open(src, 'rb') as img_file:
            temp_metadata = Image(img_file)
        temp_metadata.artist = self.artist
        for key in [key for key in self.METADATA_KEYS if key in metadata.list_all()]:
            temp_metadata[key] = metadata[key]
        with open(src, 'wb') as image_file:
            image_file.write(temp_metadata.get_file())

    def organize_files(self, src, file_list, metadata):
        """Moves files into target folders."""
        # create path
        path = f"{metadata.datetime[:4]}/{metadata.datetime[:10].replace(':', '-')}/"

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
                   'tungsten_wb', 'daylight_wb', 'tiff']
    float_params = ['width', 'height', 'ratio', 'scale', 'color', 'exp']

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
        elif command in bool_params:
            params[command] = True
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
                params['output_color'] = list(int(parameter[i:i + 2], 16) for i in (0, 2, 4))
            elif command == 'lut':
                params['luts'] = parameter.split(',')
            elif command == 'artist':
                params['artist'] = parameter
            else:
                fail(arg)
        else:
            fail(arg)

    raw2film = Raw2Film(**params)
    files = [x for x in os.listdir() if x.endswith(Raw2Film.EXTENSION_LIST)]

    with Pool() as p:
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
  --exp=<f>         Set how many stops f to increase or decrease the exposure of the output.
  --width=<w>       Set simulated film width to w mm.
  --height=<h>      Set simulated film height to h mm.
  --ratio=<r>       Set canvas aspect ratio to r.
          <w>/<h>   Set canvas aspect ratio to w/h.
  --scale=<s>       Multiply canvas size by s.
  --color=<hex>     Set canvas color to #hex.
  --artist=<name>   Set artist name in metadata to name.
  --lut=<1,2,...>   Set output LUTs to those listed. Separate LUT names with ',' and leave no space.
                    LUT 1 is the primary LUT. Others are saved to subfolders.

All boolean options can be used with no- or without, depending on which value is desired.
Above we show the options used to change the default value.
""")


def formats_message():
    """Outputs all built-in formats."""
    key_length = max([len(key) for key in Raw2Film.FORMATS])
    print(f"key {' ' * (key_length - 3)} width mm x height mm")
    for key in Raw2Film.FORMATS:
        print(f"{key} {' ' * (key_length - len(key))} {Raw2Film.FORMATS[key][0]} mm x {Raw2Film.FORMATS[key][1]} mm")


# runs image processing on all raw files in parallel
if __name__ == '__main__':
    main(sys.argv[1:])
