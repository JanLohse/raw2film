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

LUT = 'FilmboxCustom.cube'
LUT_BW = 'BW.cube' # None if no BW is desired
ARTIST = 'Jan Lohse'
HALATION = True
TEXTURE = True
ORGANIZE_RAW = True

def gaussian_blur(rgb, sigma=1):
    r, g, b = np.dsplit(rgb, 3)
    r = ndimage.gaussian_filter(r, sigma=sigma)
    g = ndimage.gaussian_filter(g, sigma=sigma)
    b = ndimage.gaussian_filter(b, sigma=sigma)
    return np.dstack((r, g, b))


# calculate appropriate exposure adjustment
def calc_exposure(rgb, lum_vec=np.array([.2127, .7152, .0722])):
    lum_mat = np.dot(rgb, lum_vec)
    return np.average(np.log2(lum_mat + np.ones_like(lum_mat) * 2**-16))


# adds metadata from original image
def add_metadata(src, metadata):
    with open(src, 'rb') as img_file:
        temp_metadata = Image(img_file)
    temp_metadata.artist = ARTIST
    for key in METADATA_KEYS:
        temp_metadata[key] = metadata[key]
    with open(src, 'wb') as image_file:
        image_file.write(temp_metadata.get_file())


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


# compute processed image for given raw file
def process_image(src):
    # read metadata
    with open(src, 'rb') as img_file:
        metadata = Image(img_file)

    # create path
    path = f"{metadata.datetime[:4]}/{metadata.datetime[:10].replace(':', '-')}"
    if not os.path.exists(path + "/RAW"):
        try:
            os.makedirs(path + "/RAW")
        except:
            pass
    if LUT_BW and not os.path.exists(path + "/BW"):
        try:
            os.makedirs(path + "/BW")
        except:
            pass
    if os.path.exists(f"{path}/{src.split('.')[0]}.jpg"):
        os.replace(src, f"{path}/RAW/{src}")
        return

    # convert raw file to linear data
    with rawpy.imread(src) as raw:
        rgb = raw.postprocess(user_wb=[2.673, 1., 1.502, 1.], output_color=rawpy.ColorSpace(6), gamma=(1, 1),
                              output_bps=16, no_auto_bright=True,
                              demosaic_algorithm=rawpy.DemosaicAlgorithm(11), four_color_rgb=True)
    rgb = rgb.astype(dtype='float32')
    rgb /= 2 ** 16

    # adjust exposure
    rgb *= metadata.f_number ** 2 / metadata.photographic_sensitivity / metadata.exposure_time
    rgb *= 2 ** (-calc_exposure(rgb) / 1.2 - 3)

    # crop to 3:2 ratio
    rgb = crop(rgb)

    # halation
    if HALATION:
        scale = max(rgb.shape) / 4608
        r, g, b = np.dsplit(np.clip(gaussian_blur(rgb, sigma=7 * scale) - rgb, a_min=0, a_max=None), 3)
        rgb += 1.1 * scale * np.dstack((.36 * r, .22 * g, .1 * b))

    # blur and sharpen
    if TEXTURE:
        rgb = gaussian_blur(rgb, sigma=1.25 * scale)
        rgb = 2 * rgb - gaussian_blur(rgb, sigma=1.6 * scale)
 
 
 
    # add lut and generate jpg
    rgb = (np.clip((np.log2(rgb + np.ones(rgb.shape)) / 4) * 2 ** 16, a_min=0, a_max=2**16 - 1)).astype(dtype='uint16')
    imageio.imsave(src.split(".")[0] + "_log.tiff", rgb)
    try:
        os.remove(src.split(".")[0] + ".tiff")
    except OSError:
        pass
    ffmpeg.input(src.split(".")[0] + '_log.tiff').filter('lut3d', file=LUT).output(src.split(".")[0] + '.tiff',
                                                                                   loglevel="quiet").run()
    imageio.imsave(path + "/" + src.split(".")[0] + '.jpg',
                 (imageio.imread(src.split(".")[0] + '.tiff') / 2 ** 8).astype(dtype='uint8'), quality=100)
    # create BW version
    if LUT_BW:
        os.remove(src.split(".")[0] + ".tiff")
        ffmpeg.input(src.split(".")[0] + '_log.tiff').filter('lut3d', file=LUT_BW).output(src.split(".")[0] + '.tiff',
                                                                                       loglevel="quiet").run()
        imageio.imsave(path + "/BW/" + src.split(".")[0] + 'BW.jpg',
                    (imageio.imread(src.split(".")[0] + '.tiff') / 2 ** 8).astype(dtype='uint8'), quality=100)
    os.remove(src.split(".")[0] + "_log.tiff")
    os.remove(src.split(".")[0] + ".tiff")
    
    # add metadata to output file
    add_metadata(path + "/" + src.split(".")[0] + '.jpg', metadata)
    if LUT_BW:
        add_metadata(path + "/BW/" + src.split(".")[0] + 'BW.jpg', metadata)
    
    if ORGANIZE_RAW:
        os.replace(src, f"{path}/RAW/{src}")
    print(f"{src} processed successfully", flush=True)


if __name__ == '__main__':
    files = [x for x in os.listdir() if x.endswith(EXTENSION_LIST)]
    with Pool() as p:
        p.map(process_image, files)
