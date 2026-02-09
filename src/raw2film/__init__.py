from pathlib import Path

try:
    from ._version import __version__
except ImportError:
    __version__ = ""

R2F_BASE_DIR = Path(__file__).resolve().parent
R2F_BASE_DIR = str(R2F_BASE_DIR).replace("\\", "/")