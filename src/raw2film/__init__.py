"""
[![GitHub](https://img.shields.io/badge/GitHub-repo-blue?logo=github)](https://github.com/JanLohse/raw2film)
.. include:: ../../README.md
   :start-line: 3
"""

from pathlib import Path

try:
    from ._version import __version__
except ImportError:
    __version__ = ""

R2F_BASE_DIR = Path(__file__).resolve().parent
R2F_BASE_DIR = str(R2F_BASE_DIR).replace("\\", "/")
