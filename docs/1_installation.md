---
icon: lucide/download
tags:
  - Guide
  - Setup
---

# Installation

To run Raw2Film it is required to have installed [exiftool](https://exiftool.org/) on
your system.
On Linux this can be done easily with

```bash
sudo apt install exiftool
```

=== "Windows"
    Download the latest `.exe` from
    the [releases](https://github.com/JanLohse/raw2film/releases) page and run it.

    Alternatively, install via Python (see [below](#python-package)).

    !!! bug "ExifTool not found"
        If ExifTool is not
        found despite being installed and put on `PATH`, try placing the `.exe` of Raw2Film and
        ExifTool in the same folder.

=== "Linux"
    Download the `.AppImage` from
    the [releases](https://github.com/JanLohse/raw2film/releases) page and make it
    executable:

    ```bash
    chmod +x raw2film-{version}.AppImage
    ./raw2film-{version}.AppImage
    ```

    Alternatively, install via Python (see [below](#python-package)).

=== "macOS"
    There is currently no native binary available for macOS.
    Install and run the application using a Python-based method.
    See the [Python Package](#python-package) section below.

## Python Package

Install the application using your preferred Python package manager.

=== "pip"
    Installs the package into the current Python environment:

    ```bash
    pip install raw2film
    ```

=== "pipx"
    Recommended for installing standalone applications globally in an isolated
    environment:

    ```bash
    pipx install raw2film
    ```

=== "uv"
    Install the application as an isolated tool:

    ```bash
    uv tool install raw2film
    ```

    Alternatively, run directly from a cloned repository without installing:

    ```bash
    uvx raw2film
    ```

After installation, run the application:

```bash
raw2film
```

## Legacy CUDA support

CUDA support has been removed in the current versions. There are plans to add more
universal GPU support, not reliant on the proprietary CUDA drivers. Recently there
have also been improvements added that make the live preview much faster than before
on CPU.

A legacy CUDA branch has been added. Importantly it is also far out of date in many
regards. When using it additionally installing [CuPy package](https://cupy.dev) is
necessary to activate the GPU functionality. To disable CUDA on an installation with
CUDA capabilities, use the argument `--no-cuda`.
