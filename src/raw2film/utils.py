import os
import time
from pathlib import Path
from shutil import copy

from raw2film import data


def find_data(metadata):
    """Search for camera and lens name in metadata"""
    values = list(metadata.values())
    cam, lens = None, None
    for key in data.CAMERA_DB:
        if key in values:
            cam = data.CAMERA_DB[key].split(':')
    for key in data.LENS_DB:
        if key in values:
            lens = data.LENS_DB[key].split(':')
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


def formats_message():
    """Outputs all built-in formats."""
    key_length = max([len(key) for key in data.FORMATS])
    print(f"key {' ' * (key_length - 3)} width mm x height mm")
    for key in data.FORMATS:
        print(f"{key} {' ' * (key_length - len(key))} {data.FORMATS[key][0]} mm x {data.FORMATS[key][1]} mm")


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
        if (name_start <= name <= name_end and filename.lower().endswith(data.EXTENSION_LIST)
                and filename not in files):
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
        if name_start <= name <= name_end and (filename.lower().endswith(data.EXTENSION_LIST) or
                                               (filename.lower().endswith('jpg') and '_' in filename)):
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
