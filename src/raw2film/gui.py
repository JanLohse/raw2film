import imageio
import numpy as np
from PyQt6.QtCore import QSize, QThreadPool
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QGridLayout, QSizePolicy, QCheckBox
from raw2film.raw_conversion import raw_to_linear, process_image, crop_rotate_zoom
from spectral_film_lut import FILMSTOCKS
from spectral_film_lut.utils import *


class MainWindow(QMainWindow):
    def __init__(self, filmstocks):
        super().__init__()

        self.setWindowTitle("Raw2Film")

        self.filmstocks = filmstocks

        pagelayout = QHBoxLayout()
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed))
        sidelayout = QGridLayout()
        widget.setLayout(sidelayout)
        sidelayout.setAlignment(Qt.AlignmentFlag.AlignBottom)

        self.image = QLabel("Select a reference image for the preview")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(QSize(256, 256))
        self.pixmap = QPixmap()

        pagelayout.addWidget(self.image)
        pagelayout.addWidget(widget, alignment=Qt.AlignmentFlag.AlignBottom)

        self.side_counter = -1

        def add_option(widget, name=None, default=None, setter=None):
            self.side_counter += 1
            sidelayout.addWidget(widget, self.side_counter, 1)
            label = QLabel(name, alignment=(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
            sidelayout.addWidget(label, self.side_counter, 0)
            if default is not None and setter is not None:
                label.mouseDoubleClickEvent = lambda *args: setter(default)
                setter(default)

        self.image_selector = FileSelector()
        add_option(self.image_selector, "Reference image:")

        self.lens_correction = QCheckBox()
        add_option(self.lens_correction, "Lens correction:", False, self.lens_correction.setChecked)

        self.full_preview = QCheckBox()
        add_option(self.full_preview, "Full preview:", False, self.full_preview.setChecked)

        self.halation = QCheckBox()
        add_option(self.halation, "Halation:", True, self.halation.setChecked)
        self.sharpness = QCheckBox()
        add_option(self.sharpness, "Sharpness:", True, self.sharpness.setChecked)
        self.grain = QCheckBox()
        add_option(self.grain, "Grain:", True, self.grain.setChecked)

        self.exp_comp = Slider()
        self.exp_comp.setMinMaxTicks(-2, 2, 1, 6)
        add_option(self.exp_comp, "Exposure:", 0, self.exp_comp.setValue)

        self.wb_modes = {"Native": None, "Daylight": 5500, "Cloudy": 6500, "Shade": 7500, "Tungsten": 2850, "Fluorescent": 3800, "Custom": None}
        self.wb_mode = QComboBox()
        self.wb_mode.addItems(list(self.wb_modes.keys()))
        add_option(self.wb_mode, "WB:", "Daylight", self.wb_mode.setCurrentText)

        self.exp_wb = Slider()
        self.exp_wb.setMinMaxTicks(2000, 12000, 50)
        add_option(self.exp_wb, "Kelvin:", 6500, self.exp_wb.setValue)

        # TODO: add rotate and flip buttons
        self.rotation = Slider()
        self.rotation.setMinMaxTicks(-90, 90, 1)
        add_option(self.rotation, "Rotation angle:", 0, self.rotation.setValue)

        self.zoom = Slider()
        self.zoom.setMinMaxTicks(1, 2, 1, 100)
        add_option(self.zoom, "Zoom:", 0, self.zoom.setValue)

        self.negative_selector = QComboBox()
        self.negative_selector.addItems(list(filmstocks.keys()))
        add_option(self.negative_selector, "Negativ stock:", "Kodak5207", self.negative_selector.setCurrentText)

        self.red_light = Slider()
        self.red_light.setMinMaxTicks(-2, 2, 1, 6)
        add_option(self.red_light, "Red printer light:", 0, self.red_light.setValue)
        self.green_light = Slider()
        self.green_light.setMinMaxTicks(-2, 2, 1, 6)
        add_option(self.green_light, "Green printer light:", 0, self.green_light.setValue)
        self.blue_light = Slider()
        self.blue_light.setMinMaxTicks(-2, 2, 1, 6)
        add_option(self.blue_light, "Blue printer light:", 0, self.blue_light.setValue)

        self.link_lights = QCheckBox()
        self.link_lights.setChecked(True)
        self.link_lights.setText("link lights")
        add_option(self.link_lights)

        filmstocks["None"] = None
        self.print_selector = QComboBox()
        self.print_selector.addItems(["None"] + list(filmstocks.keys()))
        add_option(self.print_selector, "Print stock:", "Kodak2383", self.print_selector.setCurrentText)

        self.projector_kelvin = Slider()
        self.projector_kelvin.setMinMaxTicks(2700, 10000, 100)
        add_option(self.projector_kelvin, "Projector wb:", 6500, self.projector_kelvin.setValue)

        self.white_point = Slider()
        self.white_point.setMinMaxTicks(.5, 2., 1, 20)
        add_option(self.white_point, "White point:", 1., self.white_point.setValue)

        self.save_lut_button = QPushButton("Save image")
        self.save_lut_button.released.connect(self.save_image_dialog)
        add_option(self.save_lut_button)

        self.negative_selector.currentTextChanged.connect(self.parameter_changed)
        self.print_selector.currentTextChanged.connect(self.print_light_changed)
        self.image_selector.textChanged.connect(self.load_image)
        self.projector_kelvin.valueChanged.connect(self.parameter_changed)
        self.exp_comp.valueChanged.connect(self.parameter_changed)
        self.wb_mode.currentTextChanged.connect(self.changed_wb_mode)
        self.exp_wb.valueChanged.connect(self.changed_exp_wb)
        self.red_light.valueChanged.connect(self.lights_changed)
        self.green_light.valueChanged.connect(self.lights_changed)
        self.blue_light.valueChanged.connect(self.lights_changed)
        self.white_point.valueChanged.connect(self.parameter_changed)
        self.lens_correction.checkStateChanged.connect(self.load_image)
        self.full_preview.checkStateChanged.connect(self.parameter_changed)
        self.halation.checkStateChanged.connect(self.parameter_changed)
        self.sharpness.checkStateChanged.connect(self.parameter_changed)
        self.grain.checkStateChanged.connect(self.parameter_changed)
        self.rotation.valueChanged.connect(self.crop_zoom_changed)
        self.zoom.valueChanged.connect(self.crop_zoom_changed)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

        self.resize(QSize(1024, 512))

        self.waiting = False
        self.running = False

        self.threadpool = QThreadPool()

        self.XYZ_image = None
        self.preview_image = None
        self.value_changed = False

    def scale_pixmap(self):
        if not self.pixmap.isNull():
            scaled_pixmap = self.pixmap.scaled(self.image.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation)
            self.image.setPixmap(scaled_pixmap)

    def load_image(self):
        self.XYZ_image = raw_to_linear(self.image_selector.currentText(),
                                       lens_correction=self.lens_correction.isChecked())
        self.parameter_changed()

    def resizeEvent(self, event):
        self.scale_pixmap()
        super().resizeEvent(event)

    def changed_wb_mode(self):
        mode = self.wb_mode.currentText()
        if mode == "Native":
            self.exp_wb.setValue(self.filmstocks[self.negative_selector.currentText()].exposure_kelvin)
        elif mode != "Custom":
            self.exp_wb.setValue(self.wb_modes[mode])
            self.parameter_changed()

    def changed_exp_wb(self):
        kelvin = self.exp_wb.getValue()
        curr_mode = self.wb_mode.currentText()
        if curr_mode == "Custom":
            self.parameter_changed()
        elif self.wb_modes[curr_mode] != kelvin:
            self.wb_mode.setCurrentText("Custom")
            self.parameter_changed()

    def setup_params(self):
        kwargs = {"negative_film": self.filmstocks[self.negative_selector.currentText()],
                  "print_film": self.filmstocks[self.print_selector.currentText()],
                  "projector_kelvin": self.projector_kelvin.getValue(), "exp_comp": self.exp_comp.getValue(),
                  "printer_light_comp": np.array(
                      [self.red_light.getValue(), self.green_light.getValue(), self.blue_light.getValue()]),
                  "white_point": self.white_point.getValue(), "zoom": self.zoom.getValue(),
                  "rotation": self.rotation.getValue(),
                  "exposure_kelvin": self.exp_wb.getValue(),
                  "halation": self.halation.isChecked(),
                  "sharpness": self.sharpness.isChecked(),
                  "grain": self.grain.isChecked(),}
        return kwargs

    def lights_changed(self, value):
        if self.link_lights.isChecked():
            if value == self.red_light.getPosition() == self.green_light.getPosition() == self.blue_light.getPosition():
                self.parameter_changed()
            else:
                self.red_light.setPosition(value)
                self.green_light.setPosition(value)
                self.blue_light.setPosition(value)
        else:
            self.parameter_changed()

    def print_light_changed(self):
        if self.print_selector.currentText() == "None":
            self.red_light.setDisabled(True)
            self.green_light.setDisabled(True)
            self.blue_light.setDisabled(True)
            self.link_lights.setDisabled(True)
        else:
            self.red_light.setDisabled(False)
            self.green_light.setDisabled(False)
            self.blue_light.setDisabled(False)
            self.link_lights.setDisabled(False)
        self.parameter_changed()

    def print_output(self, s):
        return

    def update_finished(self):
        self.running = False
        if self.waiting:
            self.waiting = False
            self.start_worker(self.update_preview)

    def progress_fn(self, n):
        return

    def parameter_changed(self):
        self.value_changed = True

        self.start_worker(self.update_preview)

    def crop_zoom_changed(self):
        self.value_changed = False

        self.start_worker(self.update_preview)

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
        worker.signals.progress.connect(self.progress_fn)

        self.threadpool.start(worker)

    def update_preview(self, verbose=False, *args, **kwargs):
        if self.XYZ_image is None:
            return
        kwargs = self.setup_params()
        if self.value_changed or self.full_preview.isChecked():
            self.preview_image = process_image(self.XYZ_image, fast_mode=not self.full_preview.isChecked(), **kwargs)
        image = self.preview_image
        if not self.full_preview.isChecked():
            image = crop_rotate_zoom(image, **kwargs)
        height, width, _ = image.shape
        image = QImage(np.require(image, np.uint8, 'C'), width, height, 3 * width, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(image)
        self.image.setPixmap(self.pixmap)
        self.scale_pixmap()

    def save_image(self, filename, **kwargs):
        kwargs = self.setup_params()
        image = raw_to_linear(self.image_selector.currentText(), lens_correction=self.lens_correction.isChecked(),
                              half_size=False)
        image = process_image(image, fast_mode=False, **kwargs)
        if '.' not in filename:
            filename += '.jpg'
        imageio.imwrite(filename, image, quality=100, format='.jpg')

    def save_image_dialog(self):
        filename, ok = QFileDialog.getSaveFileName(self)

        if ok:
            self.start_worker(self.save_image, filename=filename, semaphore=False)


def gui_main():
    filmstocks = {k: v() for k, v in FILMSTOCKS.items() if v is not None}

    app = QApplication(sys.argv)
    w = MainWindow(filmstocks)
    w.show()
    app.exec()


if __name__ == '__main__':
    gui_main()
