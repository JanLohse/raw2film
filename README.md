# raw2film
Automatic raw image development with built-in film emulation.

## Installation
Install required packages as follows:
- [`colour-science`](https://pypi.org/project/colour-science/)
- [`ffmpeg-python`](https://pypi.org/project/ffmpeg-python/)
- [`imageio`](https://pypi.org/project/imageio/)
- [`numpy`](https://pypi.org/project/numpy/)
- [`rawpy`](https://pypi.org/project/rawpy/)
- [`exif`](https://pypi.org/project/exif/)
- [`SciPy`](https://pypi.org/project/SciPy/)

## Usage
Put `processing.py` and the `.cube` files in the same folder as the raw images and the run `processing.py`.

Get a list of parameters with the argument `--helpt`.

## Features
- Develop raw images into `.jpg` files and organize the output by date.
- Apply basic film emulation including _halation_, _grain_, and _resolution_.
- Automatic _exposure adjustment_ based solely on scene brightness regardless of exposure.
- White balance tries to adhere to daylight balance, but will adjust slightly.
- Support for ARRI LogC3 LUTs, including output under multiple LUTs to generate alternative versions.
- Emulate various _film formats_ by matching resolution and aspect ratio.
- Optionally add a _canvas_ with a fixed output aspect-ratio, e.g. to get a uniform output for Instagram posts.
