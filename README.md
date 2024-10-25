# raw2film
Automatic raw image development with built-in film emulation.

## Installation
Install required packages as follows:
- [`ffmpeg-python`](https://pypi.org/project/ffmpeg-python/)
- [`imageio`](https://pypi.org/project/imageio/)
- [`numpy`](https://pypi.org/project/numpy/)
- [`rawpy`](https://pypi.org/project/rawpy/)
- [`PyExifTool`](https://pypi.org/project/PyExifTool/)
- [`SciPy`](https://pypi.org/project/SciPy/)
- [`lensfunpy`](https://pypi.org/project/lensfunpy/)
- [`opencv-python`](https://pypi.org/project/opencv-python/)
- [`pillow`](https://pypi.org/project/pillow/)

Optional:
- [`cupy`](https://pypi.org/project/cupy/)

## Usage
Put `processing.py` and the `.cube` files in the same folder as the raw images and run `processing.py`.
With a working `cupy` installation `processing_cuda.py` can be used instead to make use of gpu acceleration.

Get a list of parameters with the argument `--help`.

### Lens correction
For cameras which can easily be matched from their metadata it is applied automatically.
Otherwise they can be added to `CAMERA_DB` and `LENS_DB` manually to ensure they are found by `lensfunpy`.
For this find an identifying tag in the metadata of the file for each (e.g. using [ExifTool](https://exiftool.org/)) and find their tag for `lensfunpy` by running the options `--list_cameras` and `--list_lenses`.
Then add the identifier and `lensfunpy` tag to the databases in `processing.py`.

## Features
- Develop raw images into `.jpg` files and organize the output by date.
- Apply basic film emulation including _halation_, _grain_, and _resolution_.
- Automatic _exposure adjustment_ based solely on scene brightness regardless of exposure.
- White balance tries to adhere to daylight balance, but will adjust slightly.
- Support for ARRI LogC3 LUTs, including output under multiple LUTs to generate alternative versions.
- Emulate various _film formats_ by matching resolution and aspect ratio.
- Optionally add a _canvas_ with a fixed output aspect-ratio, e.g. to get a uniform output for Instagram posts.
- Lens correction using `lensfunpy`.
- Film emulation LUTs based on [Filmbox](https://videovillage.com/filmbox/):
  - Natural/Standard/Vibrant: Variants with different saturation and black levels.
  - Kodak: Kodak Vision3 250D with Kodak 2383 print film.
  - Fuji: Fuji Eterna Vivid 160T 8543 with Fuji 3513 print film.
  - BW: Kodak Double-X black and white film.
