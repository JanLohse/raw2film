import os
import sys
from functools import partial
from multiprocessing import Pool

import ffmpeg
import imageio.v2 as imageio
import numpy as np
import rawpy
from colour.models import log_encoding_ARRILogC3
from exif import Image
from scipy import ndimage

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


def process_image(src, luts=None, organize=True, **kwargs):
    """Manages image processing pipeline."""
    if luts is None:
        luts = ['FilmboxFull_Vibrant.cube', 'FilmboxFull_BW.cube']
    rgb, metadata = raw_to_linear(src)
    film_emulation(src, rgb, metadata, **kwargs)
    file_list = [apply_lut(src, lut, first=(lut == luts[0])) for lut in luts]
    file_list = [convert_jpg(file, **kwargs) for file in file_list]
    os.remove(src.split('.')[0] + '_log.tiff')
    for file in file_list:
        add_metadata(file, metadata, **kwargs)
    if organize:
        organize_files(src, file_list, metadata)
    print(f"{src} processed successfully", flush=True)


def raw_to_linear(src):
    """Takes raw file location and outputs linear rgb data and metadata."""
    # read metadata
    with open(src, 'rb') as img_file:
        metadata = Image(img_file)

    # convert raw file to linear data
    with rawpy.imread(src) as raw:
        rgb = raw.postprocess(output_color=rawpy.ColorSpace(6), gamma=(1, 1),
                              output_bps=16, no_auto_bright=True,
                              demosaic_algorithm=rawpy.DemosaicAlgorithm(11), four_color_rgb=True)
    rgb = rgb.astype(dtype='float32')
    rgb /= 2 ** 16 - 1

    return rgb, metadata


def film_emulation(src, rgb, metadata, width=None, height=None, crop=True, blur=True, sharpen=True, halation=True,
                   grain=True, **kwargs):
    """Adjusts exposure, aspect ratio and texture, and outputs tiff file in ARRI LogC3 color space."""
    if width is None:
        width = FORMATS['135'][0]
    if height is None:
        height = FORMATS['135'][1]

    # crop to specified aspect ratio
    if crop:
        rgb = crop_image(rgb, aspect=width / height)

    # adjust exposure
    if metadata.f_number:
        rgb *= metadata.f_number ** 2 / metadata.photographic_sensitivity / metadata.exposure_time
    else:
        rgb *= 4 ** 2 / metadata.photographic_sensitivity / metadata.exposure_time
    exposure = calc_exposure(ndimage.gaussian_filter(rgb, sigma=3))
    adjustment = -.85 * exposure - 2.35
    rgb *= 2 ** adjustment

    scale = max(rgb.shape) / (80 * width)

    # texture
    if blur:
        rgb = gaussian_blur(rgb, sigma=.5 * scale)

    if sharpen:
        rgb = np.log2(rgb + 2 ** -16)
        rgb = rgb + np.clip(rgb - gaussian_blur(rgb, sigma=scale), a_min=-2, a_max=2)
        rgb = np.exp2(rgb) - 2 ** -16

    if halation:
        threshold = .2
        r, g, b = np.dsplit(np.clip(rgb - threshold, a_min=0, a_max=None), 3)
        r = ndimage.gaussian_filter(r, sigma=2.2 * scale)
        g = .8 * ndimage.gaussian_filter(g, sigma=2 * scale)
        b = ndimage.gaussian_filter(b, sigma=0.3 * scale)
        rgb += np.clip(np.dstack((r, g, b)) - np.clip(rgb - threshold, a_min=0, a_max=None), a_min=0, a_max=None)

    if grain:
        rgb = np.log2(rgb + 2 ** -16)
        noise = np.random.rand(*rgb.shape) - .5
        noise = ndimage.gaussian_filter(noise, sigma=.5 * scale)
        rgb += noise * (scale / 2)
        rgb = np.exp2(rgb) - 2 ** -16

    # generate logarithmic tiff file
    rgb = log_encoding_ARRILogC3(rgb)
    rgb = np.clip(np.dot(rgb, REC2020_TO_ARRIWCG), a_min=0, a_max=1)
    rgb = (rgb * (2 ** 16 - 1)).astype(dtype='uint16')
    imageio.imsave(src.split(".")[0] + '_log.tiff', rgb)


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


def calc_exposure(rgb, lum_vec=np.array([.2127, .7152, .0722]), crop=.8):
    """Calculates exposure value of the rgb image."""
    lum_mat = np.dot(np.clip(np.dot(rgb, REC2020_TO_REC709), a_min=0, a_max=None), lum_vec)
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


def gaussian_blur(rgb, sigma=1):
    """Applies gaussian blur per channel of rgb image."""
    r, g, b = np.dsplit(rgb, 3)
    r = ndimage.gaussian_filter(r, sigma=sigma)
    g = ndimage.gaussian_filter(g, sigma=sigma)
    b = ndimage.gaussian_filter(b, sigma=sigma)
    return np.dstack((r, g, b))


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


def convert_jpg(src, canvas=False, **kwargs):
    """Converts to jpg and removes src tiff file."""
    image = (imageio.imread(src) / 2 ** 8).astype(dtype='uint8')
    os.remove(src)
    if canvas:
        image = add_canvas(image, **kwargs)
    imageio.imsave(src.split('.')[0] + '.jpg', image, quality=100)
    return src.split('.')[0] + '.jpg'


def add_canvas(image, output_ratio=4 / 5, output_scale=1., output_color=(0, 0, 0), **kwargs):
    """Adds background canvas to image."""
    img_ratio = image.shape[1] / image.shape[0]
    if img_ratio > output_ratio:
        output_resolution = (int(image.shape[1] / output_ratio * output_scale), int(image.shape[1] * output_scale))
    else:
        output_resolution = (int(image.shape[0] * output_scale), int(image.shape[0] * output_ratio * output_scale))
    offset = np.subtract(output_resolution, image.shape[:2]) // 2
    canvas = np.tensordot(np.ones(output_resolution), output_color, axes=0)
    canvas[offset[0]:offset[0] + image.shape[0], offset[1]:offset[1] + image.shape[1]] = image
    return canvas.astype(dtype='uint8')


def add_metadata(src, metadata, artist='Jan Lohse', **kwargs):
    """Adds metadata to image file."""
    with open(src, 'rb') as img_file:
        temp_metadata = Image(img_file)
    temp_metadata.artist = artist
    for key in [key for key in METADATA_KEYS if key in metadata.list_all()]:
        temp_metadata[key] = metadata[key]
    with open(src, 'wb') as image_file:
        image_file.write(temp_metadata.get_file())


def organize_files(src, file_list, metadata):
    """Moves files into target folders."""
    # create path
    path = f"{metadata.datetime[:4]}/{metadata.datetime[:10].replace(':', '-')}/"

    # move files
    move_file(src, path + '/RAW/')
    if file_list:
        move_file(file_list.pop(0), path)
    for file in file_list:
        move_file(file, path + '/' + file.split('_')[-1].split('.')[0] + '/')


def move_file(src, path):
    """Moves src file to path."""
    if not os.path.exists(path):
        os.makedirs(path)
    os.replace(src, path + src)


def main(argv):
    params = dict()
    for arg in argv:
        if arg[0] == '"' and arg[1] == '"':
            arg = arg[1:-1]
        if arg == '--help':
            return help_message()
        elif arg == '--formats':
            return formats_message()
        elif arg == '--no-crop':
            params['crop'] = False
        elif arg == '--no-blur':
            params['blur'] = False
        elif arg == '--no-sharpen':
            params['sharpen'] = False
        elif arg == '--no-halation':
            params['halation'] = False
        elif arg == '--no-grain':
            params['grain'] = False
        elif arg == '--no-organize':
            params['organize'] = False
        elif arg == '--canvas':
            params['canvas'] = True
        elif '=' in arg:
            command, parameter = arg.split('=')
            if parameter[0] == '"' and parameter[1] == '"':
                parameter = parameter[1:-1]
            if command == '--format':
                params['width'], params['height'] = FORMATS[parameter]
            if command == '--width':
                params['width'] = float(parameter)
            elif command == '--height':
                params['height'] = float(parameter)
            elif command == '--ratio':
                if '/' in parameter:
                    parameter = parameter.split('/')
                    parameter = float(parameter[0]) / float(parameter[1])
                params['output_ratio'] = float(parameter)
            elif command == '--scale':
                params['output_scale'] = float(parameter)
            elif command == '--color':
                params['output_color'] = list(int(parameter[i:i + 2], 16) for i in (0, 2, 4))
            elif command == '--lut':
                params['luts'] = parameter.split(',')
            elif command == '--artist':
                params['artist'] = parameter
            else:
                print(f"Argument {arg} unknown. Use --help to get more info.")
                return
        else:
            print(f"Argument {arg} unknown. Use --help to get more info.")
            return
    files = [x for x in os.listdir() if x.endswith(EXTENSION_LIST)]

    with Pool() as p:
        p.map(partial(process_image, **params), files)


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
  --width=<w>       Set simulated film width to w mm.
  --height=<h>      Set simulated film height to h mm.
  --ratio=<r>       Set canvas aspect ratio to r.
          <w>/<h>   Set canvas aspect ratio to w/h.
  --scale=<s>       Multiply canvas size by s.
  --color=<hex>     Set canvas color to #hex.
  --artist=<name>   Set artist name in metadata to name.
  --lut=<1,2,...>   Set output LUTs to those listed. Separate LUT names with ',' and leave no space.
                    LUT 1 is the primary LUT. Others are saved to subfolders.
    """)


def formats_message():
    """Outputs all built-in formats."""
    key_length = max([len(key) for key in FORMATS])
    print(f"key {' ' * (key_length - 3)} width mm x height mm")
    for key in FORMATS:
        print(f"{key} {' ' * (key_length - len(key))} {FORMATS[key][0]} mm x {FORMATS[key][1]} mm")


# runs image processing on all raw files in parallel
if __name__ == '__main__':
    main(sys.argv[1:])
