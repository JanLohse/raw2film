import ctypes
import os
import sys

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QIcon
from spectral_film_lut.splash_screen import launch_splash_screen

from raw2film import R2F_BASE_DIR, __version__

os.environ["NUMBA_THREADING_LAYER"] = "workqueue"


def run(exit_immediately: bool = False):
    if sys.platform == "win32":
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "jan_lohse.raw2film"
        )
    app, splash_screen = launch_splash_screen("Raw2Film", __version__)

    icon = QIcon()
    for size in [256, 128, 64, 48, 32, 16]:
        path = f"{R2F_BASE_DIR}/resources/raw2film_{size}.png"
        icon.addFile(path, QSize(size, size))

    from spectral_film_lut.film_loader import load_ui

    from raw2film.gui import MainWindow

    load_ui(MainWindow, splash_screen, app, exit_immediately=exit_immediately)


if __name__ == "__main__":
    run()
