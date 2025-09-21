import math
import os
import time
from pathlib import Path
from shutil import copy

import exiftool

from raw2film import data
import numpy as np

def find_data(metadata, db):
    """Search for camera and lens name in metadata"""
    cam, lens = None, None

    def check_tags(metadata, tags, endings):
        result = []
        for tag in tags:
            if tag in metadata:
                result.append(metadata[tag])
        for ending in endings:
            for key in metadata:
                if key.endswith(ending):
                    result.append(metadata[key])
        if not result:
            return [None]
        return result

    cam_makes = check_tags(metadata, ['EXIF:Make'], ['Make'])
    cam_models = check_tags(metadata, ['EXIF:Model'], ['Model'])
    lens_makes = check_tags(metadata, ['EXIF:LensMake'], ['LensMake'])
    lens_models = check_tags(metadata, ['EXIF:LensModel', 'EXIF:LensType'], ['LensModel', 'LensType', 'Lens'])

    if cam_makes != [None]:
        for cam_make in cam_makes:
            if cam_make is not None:
                cam_make = str(cam_make)
            for cam_model in cam_models:
                if cam_model is not None:
                    cam_model = str(cam_model)
                cam = db.find_cameras(cam_make, cam_model, loose_search=True)
                if cam:
                    cam = cam[0]
                    break
            else:
                continue
            break
        if cam is not None and cam:
            for lens_make in lens_makes:
                if lens_make is not None:
                    lens_make = str(lens_make)
                for lens_model in lens_models:
                    if lens_model is not None:
                        lens_model = str(lens_model)
                    lens = db.find_lenses(cam, lens_make, lens_model, loose_search=True)
                    if lens:
                        lens = lens[0]
                        break
                else:
                    continue
                break
        elif cam is not None:
            cam = None

    return cam, lens


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


def cleaner(raw2film):
    print("terminating...")
    time.sleep(1)
    for file in os.listdir():
        if (raw2film.organize and file.endswith('.jpg')) or (not raw2film.tiff and file.endswith('.tiff')):
            os.remove(file)


def copy_from_subfolder(file):
    name_start, name_end = prep_file_name(file)

    files = []

    for path in Path().rglob('./*.*'):
        filename = str(path).split('\\')[-1]
        name = filename.split('.')[0]
        if (name_start <= name <= name_end and filename.lower().endswith(
                data.EXTENSION_LIST) and filename not in files):
            files.append(filename)
            if not os.path.isfile(filename):
                copy(path, '../..', )

    return files


def cleanup_files(file):
    if not file:
        print("Specify the files to clean to avoid errors")
        return

    name_start, name_end = prep_file_name(file)

    for path in Path().rglob('./*/*.*'):
        filename = str(path).split('\\')[-1]
        name = filename.split('.')[0].split('_')[0]
        if name_start <= name <= name_end and (filename.lower().endswith(data.EXTENSION_LIST) or (
                filename.lower().endswith('jpg') and '_' in filename)):
            if not any(Path().rglob(f'*{name}.jpg')) and not os.path.isfile(filename):
                print("deleted", filename)
                os.remove(path)

    # remove empty subfolders
    for dir_path, dir_names, _ in os.walk('../..', topdown=False):
        for dir_name in dir_names:
            full_path = os.path.join(dir_path, dir_name)
            if not os.listdir(full_path) and '20' in full_path:
                print("deleted", full_path)
                os.rmdir(full_path)


def organize_files(src, file, metadata):
    """Moves files into target folders."""
    # create path
    path = f"{metadata['EXIF:DateTimeOriginal'][:4]}/{metadata['EXIF:DateTimeOriginal'][:10].replace(':', '-')}/"

    # move files
    move_file(src, path + '/RAW/')
    move_file(file, path)


def move_file(src, path):
    """Moves src file to path."""
    if not os.path.exists(path):
        os.makedirs(path)
    os.replace(src, path + src)


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


def add_metadata(src, metadata, exp_comp):
    metadata = {key: metadata[key] for key in metadata if key.startswith("EXIF") and key[5:] in data.METADATA_KEYS}
    metadata['EXIF:ExposureCompensation'] = exp_comp
    with exiftool.ExifToolHelper() as et:
        et.set_tags([src], metadata, '-overwrite_original')


import numpy as np
from numba import njit


@njit
def generate_histogram(image, height=100):
    """
    Generate an RGB histogram as an image-like numpy array with logarithmic y-scaling.

    Parameters
    ----------
    image : np.ndarray
        Input RGB image (H x W x 3), dtype should be uint8 (0-255).
    height : int
        Height of the histogram image.

    Returns
    -------
    hist_img : np.ndarray
        Histogram as a numpy array of shape (height, 256, 3).
    """
    # Initialize bins
    hist_r = np.zeros(256, dtype=np.float32)
    hist_g = np.zeros(256, dtype=np.float32)
    hist_b = np.zeros(256, dtype=np.float32)

    h, w, _ = image.shape

    # Count frequencies
    for i in range(h):
        for j in range(w):
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            hist_r[r] += 1
            hist_g[g] += 1
            hist_b[b] += 1

    # Normalize histograms to 1
    max_val = max(hist_r.max(), hist_g.max(), hist_b.max())
    if max_val == 0:
        max_val = 1  # avoid division by zero

    hist_r /= max_val
    hist_g /= max_val
    hist_b /= max_val

    # Apply log transform (avoid log(0))
    for i in range(256):
        hist_r[i] = np.log1p(hist_r[i])
        hist_g[i] = np.log1p(hist_g[i])
        hist_b[i] = np.log1p(hist_b[i])

    # Smooth with a 1-pixel kernel (moving average over neighbors)
    smoothed_r = np.empty_like(hist_r)
    smoothed_g = np.empty_like(hist_g)
    smoothed_b = np.empty_like(hist_b)

    for i in range(256):
        left = i - 1 if i > 0 else i
        right = i + 1 if i < 255 else i
        smoothed_r[i] = (hist_r[left] + hist_r[i] + hist_r[right]) / 3
        smoothed_g[i] = (hist_g[left] + hist_g[i] + hist_g[right]) / 3
        smoothed_b[i] = (hist_b[left] + hist_b[i] + hist_b[right]) / 3

    hist_r, hist_g, hist_b = smoothed_r, smoothed_g, smoothed_b

    # normalize to fit in height
    max_val = max(hist_r.max(), hist_g.max(), hist_b.max())
    if max_val == 0:
        max_val = 1  # avoid division by zero

    hist_r = (hist_r * height) / max_val
    hist_g = (hist_g * height) / max_val
    hist_b = (hist_b * height) / max_val

    # Create histogram image
    hist_img = np.zeros((height, 256, 3), dtype=np.uint8)

    for x in range(256):
        r_val = hist_r[x]
        g_val = hist_g[x]
        b_val = hist_b[x]

        for y in range(height - r_val, height):
            hist_img[y, x, 0] = 255
        for y in range(height - g_val, height):
            hist_img[y, x, 1] = 255
        for y in range(height - b_val, height):
            hist_img[y, x, 2] = 255

    return hist_img
