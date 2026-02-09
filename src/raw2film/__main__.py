import ctypes
import sys

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon
from spectral_film_lut.splash_screen import launch_splash_screen
from raw2film import __version__, R2F_BASE_DIR


def run():
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("jan_lohse.raw2film")
    app, splash_screen = launch_splash_screen("Raw2Film", __version__)

    icon = QIcon()
    for size in [256, 128, 64, 48, 32, 16]:
        path = f"{R2F_BASE_DIR}/resources/raw2film_{size}.png"
        icon.addFile(path, QSize(size, size))


    from raw2film.gui import MainWindow
    from spectral_film_lut.film_loader import load_ui

    load_ui(MainWindow, splash_screen, app, 0.16)


if __name__ == "__main__":
    run()
