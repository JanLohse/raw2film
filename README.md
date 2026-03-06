# raw2film

[![Docs](https://img.shields.io/badge/docs-online-blue)](https://janlohse.github.io/raw2film/)
[![CI](https://github.com/JanLohse/raw2film/actions/workflows/python-app.yml/badge.svg)](https://github.com/JanLohse/raw2film/actions/workflows/python-app.yml)
[![Version](https://img.shields.io/github/v/release/JanLohse/raw2film)](https://github.com/JanLohse/raw2film/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/JanLohse/raw2film?tab=MIT-1-ov-file#readme)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13%20|%203.14-blue)](https://www.python.org/)

raw2film is full raw image editor with a focus on realistic film emulation.

The looks are based on published film datasheets and use the image processing pipeline
from [Spectral Film LUT](https://github.com/JanLohse/spectral_film_lut).

The film emulation includes:

- Both negative and print material emulation for a huge variety of emulsions.
- Grain with varying intensity based on brightness and hue.
- Halation to add natural glow to highlights (no data available, so intensity should be
  adjusted to taste).
- Resolution and micro-contrast matches mtf chart for each film stock.
- Set the simulated frame size to match resolution, grain intensity, and aspect ratio.

<img src="https://github.com/user-attachments/assets/13915205-b34f-4184-9c08-2d6a68b90706#gh-light-mode-only" alt="main gui" width="100%">
<img src="https://github.com/user-attachments/assets/f5aa60a0-c4df-4d1f-b298-0b8dfd271af4#gh-dark-mode-only" alt="main gui" style="display:none;" width="100%">

## Installation

### Requirements

To run raw2film it is required to have installed [exiftool](https://exiftool.org/) on
your system.
On Linux this can be done easily with

```bash
sudo apt install exiftool
```

### Windows

The easiest way to run raw2film is to download the latest `.exe` from
the [releases](../../releases) section.
(There might be issues with Windows Defender, in which case it is recommended to use the
python package.)

### Linux

Download the `.AppImage` from the [releases](../../releases) section or install the
python package.

### Python Package

You can also install the program using pip:

```bash
pip install git+https://github.com/JanLohse/raw2film
```

Then run with `raw2film`.

### GPU acceleration

To make use of your GPUs CUDA cores you need to install the python package CuPy and run
the program from the python package.
CUDA is used to display the preview, which results in a more responsive UI.
It can result in slower exports though, as there might be a lack of VRAM.

## Usage

The interface is designed to be familiar for anyone who has used a raw editor before.

- The image bar on the bottom lets you select one or multiple images to edit at once. (
  Select multiple with Shift or Ctrl.)
- Copy settings from one image to the selected ones by clicking on the thumbnail with
  the middle mouse button.
- Double click on a settings label to reset to the default value.
- Many shortcuts are available. Hover over a setting to see its description and
  shortcut.
- By default a simplified render is activated for preview to make the software more
  responsive. Activate the full preview under view to see the full film characterisitcs.

### Filmstock Selector

When clicking on the magnifying glass a window opens to search and browse through the
available film stocks.

<img src="https://github.com/user-attachments/assets/54b82d41-9386-49e6-9d3e-b389e8f1585c#gh-light-mode-only" alt="main gui" width="100%">
<img src="https://github.com/user-attachments/assets/3886eea0-f21f-43de-a319-f84105daae50#gh-dark-mode-only" alt="main gui" style="display:none;" width="100%">
