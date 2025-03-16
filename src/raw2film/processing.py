import math
import operator
import os
import sys
import time
from multiprocessing import Pool, Semaphore
from pathlib import Path

import configargparse as argparse
import exiftool
import ffmpeg
import imageio.v3 as imageio
import numpy as np
import rawpy
import torch
from PIL import Image
from spectral_film_lut.negative_film.kodak_portra_400 import KodakPortra400
from spectral_film_lut.print_film.kodak_endura_premier import KodakEnduraPremier
from spectral_film_lut.utils import create_lut
import colour

from raw2film import data, effects, color_processing
from raw2film import utils
from raw2film.utils import hex_color, fraction


class Raw2Film:

    def __init__(self, crop=True, resolution=True, halation=True, grain=True, organize=True, canvas=False, nd=0,
                 width=36, height=24, ratio=4 / 5, scale=1., color=None, artist="Jan Lohse",
                 wb='standard', exp=0, zoom=1., correct=True, cores=None, sleep_time=0, rename=False, rotation=0,
                 keep_exp=False, stock="250D", **kwargs):
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
        self.wb = wb
        self.exp = exp
        self.zoom = zoom
        self.nd = nd
        self.correct = correct
        self.sleep_time = sleep_time
        self.cores = cores
        self.rename = rename
        self.rotation = rotation
        self.keep_exp = keep_exp
        self.stock = data.FILM_DB[stock]
        self.negative = KodakPortra400()
        self.print = KodakEnduraPremier()

    def process_runner(self, starter: tuple[int, str]):
        run_count, src = starter
        try:
            if 0 < run_count < self.cores:
                time.sleep(self.sleep_time * run_count)
            self.process_image(src)
        except KeyboardInterrupt:
            return False
        return src

    def process_image(self, src: str):
        """Manages image processing pipeline."""

        rgb, metadata = self.raw_to_linear(src)

        if self.keep_exp:
            exp_comp = self.find_exp(src, self.exp)
        else:
            exp_comp = self.exp

        if self.correct:
            rgb = np.asarray(effects.lens_correction(rgb, metadata))

        rgb = self.film_emulation(rgb, metadata, exp_comp)

        start = time.time()
        rgb = Raw2Film.save_tiff(src, rgb)
        print(f"save tiff {time.time() - start:.2f} seconds", flush=True)
        start = time.time()

        file = self.apply_lut(rgb, src)

        print(f"apply lut {time.time() - start:.2f} seconds", flush=True)
        start = time.time()

        self.add_metadata(file, metadata, exp_comp)

        if self.organize:
            utils.organize_files(src, file, metadata)

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
            rgb = raw.postprocess(output_color=rawpy.ColorSpace(5), gamma=(1, 1), output_bps=16, no_auto_bright=True,
                                  use_camera_wb=(self.wb == 'camera'), use_auto_wb=(self.wb == 'auto'),
                                  demosaic_algorithm=rawpy.DemosaicAlgorithm(11), four_color_rgb=True)
        rgb = rgb.astype(dtype='float64')
        rgb /= 2 ** 16 - 1

        return rgb, metadata

    def film_emulation(self, rgb, metadata, exp_comp=0.):
        rgb = np.asarray(rgb, dtype=np.float32)

        if self.rotation:
            rgb = effects.rotate(self.rotation, rgb)

        if self.wb == 'tungsten':
            daylight_rgb = color_processing.kelvin_to_XYZ(5600)
            tungsten_rgb = color_processing.kelvin_to_XYZ(4400)
            rgb = np.dot(rgb, np.diag(daylight_rgb / tungsten_rgb))
        elif self.wb == 'standard':
            lower, upper, max_amount = 2400, 8000, 1200
            image_kelvin = color_processing.XYZ_to_kelvin(np.mean(rgb, axis=(0, 1)))
            value, target = image_kelvin, image_kelvin
            if image_kelvin <= lower:
                value, target = lower, lower + max_amount
            elif lower < image_kelvin < lower + 2 * max_amount:
                target = 0.5 * value + 0.5 * (lower + 2 * max_amount)
            elif upper - max_amount < image_kelvin < upper:
                target = 0.5 * value + 0.5 * (upper - max_amount)
            elif upper <= image_kelvin:
                value, target = upper, upper - max_amount
            rgb *= color_processing.kelvin_to_XYZ(target) / color_processing.kelvin_to_XYZ(value)

        # crop to specified aspect ratio
        if self.crop:
            rgb = effects.crop_image(self.zoom, rgb, aspect=self.width / self.height)

        # adjust exposure
        if 'EXIF:FNumber' in metadata and metadata['EXIF:FNumber']:
            rgb *= metadata['EXIF:FNumber'] ** 2 / metadata['EXIF:ISO'] / metadata['EXIF:ExposureTime']
        else:
            rgb *= 4 ** 2 / metadata['EXIF:ISO'] / metadata['EXIF:ExposureTime']
        # adjust exposure if ND filter is used on Fuji X100 camera (sadly imprecise)
        if ('x100' in metadata['EXIF:Model'].lower() and metadata['EXIF:BrightnessValue'] > 3
                and metadata['Composite:LightValue'] - metadata['EXIF:BrightnessValue'] < 1.5):
            rgb *= 8
        exposure = color_processing.calc_exposure(rgb)
        middle, max_under, max_over, slope, slope_offset = -3, -.75, .66, .9, .5
        lower_bound = -exposure + middle + max_under
        sloped = -slope * exposure + middle + slope_offset
        upper_bound = -exposure + middle + max_over
        adjustment = max(lower_bound, min(sloped, upper_bound))
        rgb *= 2 ** (adjustment + exp_comp)

        # texture
        scale = max(rgb.shape) / self.width  # pixels per mm

        if self.halation:
            blured = effects.exponential_blur(rgb, scale / 4)
            color_factors = np.dot(np.array([1.2, 0.5, 0], dtype=np.float32), data.REC709_TO_XYZ)
            rgb += np.multiply(blured, color_factors)
            rgb = np.divide(rgb, color_factors + 1)

        if self.resolution:
            rgb = np.log(rgb + 2 ** -16) / np.log(2)
            rgb = effects.film_sharpness(self.stock, rgb, scale)
            if not self.grain:
                rgb = np.exp(rgb * np.log(2)) - 2 ** -16

        if self.grain:
            if not self.resolution:
                rgb = np.log(rgb + 2 ** -16) / np.log(2)
            # compute scaling factor of exposure rms in regard to measuring device size
            std_factor = math.sqrt((math.pi * 0.024) ** 2 * scale ** 2)
            strength = 1.
            noise = np.dot(torch.empty(rgb.shape, dtype=torch.float32).normal_(), data.REC709_TO_XYZ)
            rough_noise = np.multiply(noise, np.array(self.stock['rough'],
                                                      dtype=np.float32) * std_factor / 1000 * strength)
            clean_noise = np.multiply(noise, np.array(self.stock['clean'],
                                                      dtype=np.float32) * std_factor / 1000 * strength)
            rough_scale, clean_scale = 0.005, 0.002
            if scale * rough_scale * 2 * math.sqrt(math.pi) > 1:
                rough_noise = effects.gaussian_blur(rough_noise, scale * rough_scale) * (
                        scale * rough_scale * 2 + math.sqrt(math.pi))
            if scale * clean_scale * 2 * math.sqrt(math.pi) > 1:
                clean_noise = effects.gaussian_blur(clean_noise, scale * clean_scale) * (
                        scale * clean_scale * 2 + math.sqrt(math.pi))
            noise_blending = np.clip((1 / 11) * (rgb + 1) + 0.5, a_min=0, a_max=1) ** (1 / 3)
            rgb += rough_noise * (1 - noise_blending) + clean_noise * noise_blending
            rgb = np.exp(rgb * np.log(2)) - 2 ** -16

        return rgb

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
            return fallback_exp

        # pick candidate with the shortest path and the shortest name
        path = sorted(options, key=operator.itemgetter(2, 1))[0][0]

        # return exposure compensation value of reference file
        with exiftool.ExifToolHelper() as et:
            meta = et.get_metadata(path)[0]
            exp_comp = meta['EXIF:ExposureCompensation']
        return exp_comp

    @staticmethod
    def save_tiff(src, rgb):
        # convert to arri wcg
        rgb = colour.XYZ_to_RGB(rgb, "ARRI Wide Gamut 3", apply_cctf_encoding=True)

        rgb = (rgb * (2 ** 16 - 1)).astype(dtype='uint16')

        return rgb

    def apply_lut(self, rgb, src):
        """Loads tiff file and applies LUT, generates jpg."""
        file_name = src.split('.')[0] + '.jpg'
        if os.path.exists(file_name):
            os.remove(file_name)
        lut_path = create_lut(self.negative, self.print, input_colourspace="ARRI Wide Gamut 3")
        height, width, _ = rgb.shape
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb48', s='{}x{}'.format(width, height))
            .filter('lut3d', file=lut_path)
            .output(file_name, loglevel="quiet", **{'q:v': 1}).overwrite_output().run_async(pipe_stdin=True)
        )
        process.stdin.write(rgb.tobytes())
        process.stdin.close()
        process.wait()
        return file_name

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

    def add_metadata(self, src, metadata, exp_comp):
        """Adds metadata to image file."""
        metadata = {key: metadata[key] for key in metadata if key.startswith("EXIF") and key[5:] in data.METADATA_KEYS}
        metadata['EXIF:Artist'] = self.artist
        metadata['EXIF:ExposureCompensation'] = exp_comp
        with exiftool.ExifToolHelper() as et:
            et.set_tags([src], metadata, '-overwrite_original')


def init_child(semaphore_):
    global semaphore
    semaphore = semaphore_


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
    parser.add_argument('--format', type=str, choices=data.FORMATS.keys(), default=None, help="Select film format")
    parser.add_argument('--stock', type=str, choices=data.FILM_DB.keys(), default="250D",
                        help="Select film stock for grain and resolution")
    parser.add_argument('--no-crop', dest='crop', action='store_false', help="Preserve source aspect ratio.")
    parser.add_argument('--no-resolution', dest='resolution', action='store_false', help="Turn off blur and sharpen.")
    parser.add_argument('--no-halation', dest='halation', action='store_false', help="Turn off halation.")
    parser.add_argument('--no-grain', dest='grain', help="Turn off halation.", action='store_false')
    parser.add_argument('--no-organize', dest='organize', action='store_false', help="Do no organize files.")
    parser.add_argument('--no-correct', dest='correct', action='store_false', help="Turn off lens correction")
    parser.add_argument('--canvas', action='store_true', help="Add canvas to output image.")
    parser.add_argument('--wb', default='standard', choices=['standard', 'auto', 'daylight', 'tungsten', 'camera'],
                        help="Specify white balance mode.")
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
        return utils.formats_message()
    if args.list_cameras:
        return utils.list_cameras()
    if args.list_lenses:
        return utils.list_lenses()
    if args.cleanup:
        for file in args.file:
            utils.cleanup_files(file)
        return
    if args.format:
        args.width, args.height = data.FORMATS[args.format]

    if args.file:
        files = []
        for file in args.file:
            files += utils.copy_from_subfolder(file)
        if not files:
            print("No matching files have been found.")
    else:
        files = [file for file in os.listdir() if file.lower().endswith(data.EXTENSION_LIST)]
    if not files:
        return

    raw2film = Raw2Film(**vars(args))

    start = time.time()
    counter = 1
    result = raw2film.process_runner((0, files.pop(0)))
    if result:
        print(f"{result} processed successfully {counter}/{len(files) + 1}")
    else:
        utils.cleaner(raw2film)
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
            utils.cleaner(raw2film)


# runs image processing on all raw files in parallel
if __name__ == '__main__':
    main()
