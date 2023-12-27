import math
from multiprocessing import Pool
import rawpy
import imageio.v2 as imageio
import numpy as np
import ffmpeg
import os
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

# LUTs to be applied to the image. First in the list is defaul, others will be saved in subfolders.
LUTS = ['Filmbox100vib.cube', 'BW.cube']

# Artist name added to the output metadata.
ARTIST = 'Jan Lohse'

# What features to activate. 
# Default is all True.
CROP = True
BLUR = True
SHARPEN = True
HALATION = True
GRAIN = True
ORGANIZE = True

# Specify width and height of simulated film frame. Matches resolution and aspect ratio.
# Either select from dictionary or specify manual values. To preserve orientation width should be the larger value.
# Roll film formats: 110, 120, 120-4.5, 120-6, 120-9, 135, 135-half, xpan
# Sheet film formats: 4x5, 5x7, 8x10, 11x14
# Motion picture formats: scope, flat, academy, super16, IMAX, 65mm
WIDTH, HEIGHT = FORMATS['135']

# manages image processing pipeline
def process_image(src):
    rgb, metadata = raw_to_linear(src)
    film_emulation(src, rgb, metadata)
    file_list = [apply_lut(src, lut) for lut in LUTS]
    os.remove(src.split('.')[0] + '_log.tiff')
    for file in file_list:
        add_metadata(file, metadata)
    if ORGANIZE:
        organize_files(src, file_list, metadata)
    print(f"{src} processed successfully", flush=True)


# takes raw file location and outputs linear rgb data and metadata
def raw_to_linear(src):
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


# adjusts exposure, aspect ratio and texture, outputs tiff file
def film_emulation(src, rgb, metadata):
    # crop to specified aspect ratio
    if CROP:
        rgb = crop(rgb, aspect=WIDTH/HEIGHT)

    # adjust exposure
    if metadata.f_number:
        rgb *= metadata.f_number ** 2 / metadata.photographic_sensitivity / metadata.exposure_time        
    else:
        rgb *= 4 ** 2 / metadata.photographic_sensitivity / metadata.exposure_time
    rgb *= 2 ** (-calc_exposure(ndimage.gaussian_filter(rgb, sigma=3)) / 1.15 - 2)

    scale = max(rgb.shape) / (80 * WIDTH)

    # texture
    if BLUR:
        rgb = gaussian_blur(rgb, sigma=.5 * scale)

    if SHARPEN:
        rgb = np.log2(rgb + 2**-16)
        rgb = rgb + np.clip(rgb - gaussian_blur(rgb, sigma=scale), a_min=-2, a_max=2)
        rgb = np.exp2(rgb) - 2**-16

    if HALATION:
        threshold = .2
        r, g, b = np.dsplit(np.clip(rgb - threshold, a_min=0, a_max=None), 3)
        r = ndimage.gaussian_filter(r, sigma=2.2 * scale)
        g = .8 * ndimage.gaussian_filter(g, sigma=2 * scale)
        b = ndimage.gaussian_filter(b, sigma=0.3 * scale)
        rgb += np.clip(np.dstack((r, g, b)) - np.clip(rgb - threshold, a_min=0, a_max=None), a_min=0, a_max=None)

    if GRAIN:
        rgb = np.log2(rgb + 2**-16)
        noise = np.random.rand(*rgb.shape) - .5
        noise = ndimage.gaussian_filter(noise, sigma=.5 * scale)
        rgb += noise * (scale / 2)
        rgb = np.exp2(rgb) - 2**-16
 
    # generate logarithmic tiff file
    rgb = (np.clip((np.log2(rgb + np.ones(rgb.shape)) / 4) * 2 ** 16, a_min=0, a_max=2**16 - 1)).astype(dtype='uint16')
    imageio.imsave(src.split(".")[0] + "_log.tiff", rgb)


# crop image
def crop(rgb, aspect=1.5):
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


# calculate appropriate exposure adjustment
def calc_exposure(rgb, lum_vec=np.array([.2127, .7152, .0722]), crop=True):
    lum_mat = np.dot(rgb, lum_vec)
    if crop:
        ratio = lum_mat.shape[0] / lum_mat.shape[1]
        if ratio > 1:
            width = int((lum_mat.shape[0] - ratio**.5 / ratio * lum_mat.shape[0] * .75)/2)
            height = int((lum_mat.shape[1] - lum_mat.shape[1] * .75)/2)
        else:
            width = int((lum_mat.shape[0] - lum_mat.shape[0] * .75)/2)
            ratio = 1/ratio
            height = int((lum_mat.shape[1] - ratio**.5 / ratio * lum_mat.shape[1] * .75)/2)
        lum_mat = lum_mat[width : -width , height : -height]
    return np.average(np.log2(lum_mat + np.ones_like(lum_mat) * 2**-16))


# applies gaussian blur to each channel individually
def gaussian_blur(rgb, sigma=1):
    r, g, b = np.dsplit(rgb, 3)
    r = ndimage.gaussian_filter(r, sigma=sigma)
    g = ndimage.gaussian_filter(g, sigma=sigma)
    b = ndimage.gaussian_filter(b, sigma=sigma)
    return np.dstack((r, g, b))


# loads tiff file and applies lut, generates jpg
def apply_lut(src, lut):
    if os.path.exists(src.split('.')[0] + '.tiff'):
        os.remove(src.split(".")[0] + ".tiff")
    ffmpeg.input(src.split(".")[0] + '_log.tiff').filter('lut3d', file=lut).output(src.split(".")[0] + '.tiff',
                                                                                   loglevel="quiet").run()
    extension = '.jpg'
    if os.path.exists(src.split(".")[0] + extension):
        extension = "_" +  lut.split('.')[0] + extension
    imageio.imsave(src.split(".")[0] + extension,
                 (imageio.imread(src.split(".")[0] + '.tiff') / 2 ** 8).astype(dtype='uint8'), quality=100)
    os.remove(src.split(".")[0] + ".tiff")
    return src.split('.')[0] + extension


# adds metadata from original image
def add_metadata(src, metadata):
    with open(src, 'rb') as img_file:
        temp_metadata = Image(img_file)
    temp_metadata.artist = ARTIST
    for key in [key for key in METADATA_KEYS if key in metadata.list_all()]:
        temp_metadata[key] = metadata[key]
    with open(src, 'wb') as image_file:
        image_file.write(temp_metadata.get_file())


# moves files into corresponding folders
def organize_files(src, file_list, metadata):
    # create path
    path = f"{metadata.datetime[:4]}/{metadata.datetime[:10].replace(':', '-')}/"

    # move files
    move_file(src, path + "/RAW/")
    if file_list:
        move_file(file_list.pop(0), path)
    for file in file_list:
        move_file(file, path + "/" + file.split("_")[-1].split(".")[0] + "/")


# moves src file to path folder
def move_file(src, path):
    if not os.path.exists(path):
        os.makedirs(path)
    os.replace(src, path + src)


# runs image processing on all raw files in parallel
if __name__ == '__main__':
    files = [x for x in os.listdir() if x.endswith(EXTENSION_LIST)]

    with Pool() as p:
        p.map(process_image, files)