# raw2film
Automatic raw image development with built-in film emulation.

## Installation
Install required packages as follows:
- [`colour-science`](https://pypi.org/project/colour-science/)
- [`ffmpeg-python`](https://pypi.org/project/ffmpeg-python/)
- [`imageio`](https://pypi.org/project/imageio/)
- [`numpy`](https://pypi.org/project/numpy/)
- [`rawpy`](https://pypi.org/project/rawpy/)
- [`PyExifTool`](https://pypi.org/project/PyExifTool/)
- [`SciPy`](https://pypi.org/project/SciPy/)
- [`lensfunpy`](https://pypi.org/project/lensfunpy/)

Optional (speeds up distortion correction):
- [`opencv-python`](https://pypi.org/project/opencv-python/)

## Usage
Put `processing.py` and the `.cube` files in the same folder as the raw images and run `processing.py`.

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
  - Fuji: Kodak Vision3 250D with Fuji 3513 print film.
  - Eterna: Fuji Eterna 250T with Fuji 3513 print film.
  - BW: Kodak Double-X black and white film.
