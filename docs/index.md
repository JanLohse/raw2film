# raw2film

[![GitHub](https://img.shields.io/badge/GitHub-repo-blue?logo=github)](https://github.com/JanLohse/raw2film)
[![CI](https://github.com/JanLohse/raw2film/actions/workflows/python-app.yml/badge.svg)](https://github.com/JanLohse/raw2film/actions/workflows/python-app.yml)
[![Version](https://img.shields.io/github/v/release/JanLohse/raw2film)](https://github.com/JanLohse/raw2film/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/JanLohse/raw2film?tab=MIT-1-ov-file#readme)
[![Python](https://img.shields.io/badge/python-3.11%20|%203.12%20|%203.13%20|%203.14-blue)](https://www.python.org/)

Raw2Film is full raw image editor with a focus on realistic film emulation.

The looks are based on published film datasheets and use the image processing pipeline
from [Spectral Film LUT](https://github.com/JanLohse/spectral_film_lut).

The film emulation includes:

- Both negative and print material emulation for a huge variety of emulsions.
- Grain with varying intensity based on brightness and hue.
- Halation to add natural glow to highlights (no data available, so intensity should be
  adjusted to taste).
- Resolution and micro-contrast matches mtf chart for each film stock.
- Set the simulated frame size to match resolution, grain intensity, and aspect ratio.

<img width="100%" alt="Spectral Film LUT main ui" src="https://github.com/user-attachments/assets/b7e31c74-d522-42dd-97e8-9374c9efa8ee" />
