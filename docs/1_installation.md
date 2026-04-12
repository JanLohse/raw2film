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
    pip install git+https://github.com/JanLohse/raw2film
    ```

=== "pipx"
    Recommended for installing standalone applications globally in an isolated
    environment:

    ```bash
    pipx install git+https://github.com/JanLohse/raw2film
    ```

=== "uv"
    Install the application as an isolated tool:

    ```bash
    uv tool install git+https://github.com/JanLohse/raw2film
    ```

    Alternatively, run directly from a cloned repository without installing:

    ```bash
    git clone https://github.com/JanLohse/raw2film
    cd raw2film
    uv run raw2film
    ```

After installation, run the application:

```bash
raw2film
```

## CUDA support

For hardware acceleration we make use of CuPy. It might be removed in future releases
though.
There is not a relevant speed-up for generating LUTs, that justifies the added
complexity in code.

Once CUDA support is removed, a legacy CUDA branch will be added. To use CUDA pull that
branch
and install using pip, and additionally install the https://cupy.dev/ package.
To disable CUDA on an installation with CUDA capabilities, use the argument `--no-cuda`.
