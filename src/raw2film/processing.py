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
from raw2film import data, effects, color_processing
from raw2film import utils
from raw2film.utils import hex_color, fraction
from spectral_film_lut.film_spectral import FilmSpectral
from spectral_film_lut.negative_film.kodak_5207 import Kodak5207
from spectral_film_lut.negative_film.kodak_portra_400 import KodakPortra400
from spectral_film_lut.print_film.kodak_endura_premier import KodakEnduraPremier
from spectral_film_lut.utils import create_lut
from spectral_film_lut.film_spectral import default_dtype


class Raw2Film:

    def __init__(self, crop=True, resolution=True, halation=True, grain=True, organize=True, canvas=False, nd=0,
                 width=36, height=24, ratio=4 / 5, scale=1., color=None, artist="Jan Lohse", exp=0, zoom=1.,
                 correct=True, cores=None, sleep_time=0, rename=False, rotation=0, keep_exp=False, **kwargs):
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
        self.exp = exp
        self.zoom = zoom
        self.nd = nd
        self.correct = correct
        self.sleep_time = sleep_time
        self.cores = cores
        self.rename = rename
        self.rotation = rotation
        self.keep_exp = keep_exp
        self.negative = KodakPortra400()
        self.print = KodakEnduraPremier()
        self.grain_stock = Kodak5207()

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

        file = self.apply_lut(rgb, src)

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
                                  use_camera_wb=False, use_auto_wb=False,
                                  demosaic_algorithm=rawpy.DemosaicAlgorithm(11), four_color_rgb=True)
        rgb = rgb.astype(dtype='float64')
        rgb /= 2 ** 16 - 1

        return rgb, metadata

    def film_emulation(self, rgb, metadata, exp_comp=0.):
        rgb = np.asarray(rgb, dtype=np.float32)

        if self.rotation:
            rgb = effects.rotate(self.rotation, rgb)

        # crop to specified aspect ratio
        if self.crop:
            rgb = effects.crop_image(self.zoom, rgb, aspect=self.width / self.height)

        # TODO: white balance
        # TODO: resolution scaling

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
        exp_comp += adjustment

        # texture
        scale = max(rgb.shape) / self.width  # pixels per mm

        if self.halation:
            blured = effects.exponential_blur(rgb, scale / 4)
            color_factors = np.dot(np.array([1.2, 0.5, 0], dtype=np.float32), data.REC709_TO_XYZ)
            rgb += np.multiply(blured, color_factors)
            rgb = np.divide(rgb, color_factors + 1)

        transform = FilmSpectral.generate_conversion(self.negative, mode='negative', input_colourspace=None,
                                                     exp_comp=exp_comp, )
        rgb = transform(rgb)

        if self.resolution:
            rgb = effects.film_sharpness(self.negative, rgb, scale)

        if self.grain:
            # compute scaling factor of exposure rms in regard to measuring device size
            std_factor = math.sqrt(math.pi) * 0.024 * scale / 6
            noise = np.dot(torch.empty(rgb.shape, dtype=torch.float32).normal_(), data.REC709_TO_XYZ)

            red_rms = np.interp(rgb[..., 0], self.grain_stock.red_rms_density, self.grain_stock.red_rms * std_factor)
            green_rms = np.interp(rgb[..., 1], self.grain_stock.green_rms_density, self.grain_stock.green_rms * std_factor)
            blue_rms = np.interp(rgb[..., 2], self.grain_stock.blue_rms_density, self.grain_stock.blue_rms * std_factor)
            rms = np.stack([red_rms, green_rms, blue_rms], axis=-1, dtype=default_dtype)

            noise = np.multiply(noise, rms)
            grain_size = 0.002
            if scale * grain_size * 2 * math.sqrt(math.pi) > 1:
                noise = effects.gaussian_blur(noise, scale * grain_size) * (scale * grain_size * 2 * math.sqrt(math.pi))
            rgb += noise

        rgb = np.clip(rgb, 0, 1)

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

    def apply_lut(self, rgb, src):
        """Loads tiff file and applies LUT, generates jpg."""
        file_name = src.split('.')[0] + '.jpg'
        if os.path.exists(file_name):
            os.remove(file_name)
        lut_path = create_lut(self.negative, self.print, input_colourspace="ARRI Wide Gamut 3", mode='print')
        height, width, _ = rgb.shape
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb48', s='{}x{}'.format(width, height))
            .filter('lut3d', file=lut_path)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1, loglevel="quiet")
            .run_async(pipe_stdin=True, pipe_stdout=True)
        )
        rgb *= 2 ** 16 - 1
        process.stdin.write(rgb.astype('uint16').tobytes())
        process.stdin.close()
        rgb = process.stdout.read(width * height * 3)
        process.wait()
        rgb = np.frombuffer(rgb, np.uint8).reshape([height, width, 3])
        imageio.imwrite(file_name, rgb, quality=100)
        return file_name

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
