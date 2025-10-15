# raw2film

raw2film is full raw image editor with a focus on realisitic film emulation.

The looks are based on published film datasheets and use the image processing pipeline from [Spectral Film LUT](https://github.com/JanLohse/spectral_film_lut).

The film emulation includes:
- Both negative and print material emulation for a huge variety of emulsions.
- Grain with varying intensity based on brightness and hue.
- Halation to add natural glow to highlights (no data available, so intensity should be adjusted to taste).
- Resolution and micro-contrast matches mtf chart for each filmstock.
- Set the simulated frame size to match resolution, grain intensity, and aspect ratio.

## Installation

### Requirements
To run raw2film it is required to have installed [exiftool](https://exiftool.org/) on your system.
On Linux this can be done easily with
```bash
sudo apt install exiftool
```

### Windows
The easiest way to run raw2film is to download the latest `.exe` from the [releases](../../releases) section.  
(There might be issues with Windows Defender, in which case it is recommended to use the python package.)

### Python Package
You can also install the program using pip:  

```bash
pip install git+https://github.com/JanLohse/raw2film
```
Then run with `raw2film`.

### GPU acceleration
To make use of your GPUs CUDA cores you need to install the python package CuPy and run the program from the python package.
CUDA is used to display the preview, which results in a more responsive UI.
It can result in slower exports though, as there might be a lack of VRAM.

## Usage

The interface is designed to be familiar for anyone who has used a raw editor before.

- The image bar on the bottom lets you select one or multiple images to edit at once. (Select multiple with Shift or Ctrl.)
- Copy settings from one image to the selected ones by clicking on the thumbnail with the middle mouse button.
- Double click on a settings label to reset to the default value.
- Many shortcuts are available. Hover over a setting to see its description and shortcut.
- By default a simplified render is activated for preview to make the software more responsive. Activate the full preview under view to see the full film characterisitcs.

### Main GUI
<img width="1121" height="817" alt="main gui" src="https://github.com/user-attachments/assets/d78bcb17-7f96-402e-aee7-36c04b9fac03" />

### Filmstock Selector
When clicking on the magnifying glass a window opens to search and browse through the available film stocks.
<img width="854" height="604" alt="filmstock selector" src="https://github.com/user-attachments/assets/c10af392-a531-40e4-8f4c-8a564807a9d3" />
