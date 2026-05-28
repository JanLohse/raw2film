import math
import sys
import time
import traceback

import numpy as np
from PyQt6.QtCore import (
    QObject,
    QRunnable,
    QSize,
    Qt,
    QThreadPool,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(float)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        # Add the callback to our kwargs
        self.kwargs["progress_callback"] = self.signals.progress

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class CpuGenerator:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.running = False

    def generate_new(self, width, height, sleep_time=0.5):
        assert not self.running, "Multiple generations triggered at once."
        self.running = True
        time.sleep(sleep_time)
        image = self.rng.integers(0, 255, (width, height, 3), dtype=np.uint8)
        self.running = False
        return image


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image = QLabel("show some image")
        self.image.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )
        self.image.setMinimumSize(QSize(256, 256))

        refresh_button = QPushButton("press me")
        refresh_button.setFixedHeight(25)
        refresh_button.pressed.connect(self.trigger_update)

        refresh_slider = QSlider(Qt.Orientation.Horizontal)
        refresh_slider.setFixedHeight(25)
        refresh_slider.valueChanged.connect(self.trigger_update)

        side_bar = QVBoxLayout()
        side_bar.addWidget(refresh_button)
        side_bar.addWidget(refresh_slider)
        side_bar.addStretch()

        side_bar_widget = QWidget()
        side_bar_widget.setLayout(side_bar)

        self.cpu_generator = CpuGenerator()

        page_splitter = QSplitter()
        page_splitter.addWidget(self.image)
        page_splitter.addWidget(side_bar_widget)
        page_splitter.setStretchFactor(0, 1)
        page_splitter.setStretchFactor(1, 0)

        self.setCentralWidget(page_splitter)

        self.running = False
        self.waiting = False

        self.threadpool = QThreadPool()

    def start_worker(self, function, semaphore=True, *args, **kwargs):
        if semaphore:
            if self.running:
                self.waiting = True
                return
            else:
                self.running = True
        worker = Worker(function, *args, **kwargs)
        if semaphore:
            worker.signals.finished.connect(self.update_finished)
        self.threadpool.start(worker)

    def update_finished(self):
        self.running = False
        if self.waiting:
            self.waiting = False
            self.start_worker(self.update_image)

    def trigger_update(self):
        self.start_worker(self.update_image)

    def resizeEvent(self, a0):
        self.trigger_update()
        super().resizeEvent(a0)

    def update_image(self, ratio=1.5, **_):
        pixel_ratio = self.devicePixelRatioF()
        width = math.floor(self.image.width() * pixel_ratio)
        height = math.floor(self.image.height() * pixel_ratio)
        width = min(width, math.floor(height * ratio))
        height = min(height, math.floor(width // ratio))

        image = self.cpu_generator.generate_new(width, height)

        image = QImage(image, width, height, 3 * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        pixmap.setDevicePixelRatio(pixel_ratio)
        self.image.setPixmap(pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()

    window.show()

    sys.exit(app.exec())
