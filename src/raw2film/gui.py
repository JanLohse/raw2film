import imageio
import lensfunpy
import numpy as np
from PyQt6.QtCore import QSize, QThreadPool
from PyQt6.QtGui import QPixmap, QImage, QIntValidator, QDoubleValidator
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QGridLayout, QSizePolicy, QCheckBox
from spectral_film_lut import NEGATIVE_FILM, REVERSAL_FILM, PRINT_FILM
from spectral_film_lut.utils import *

from raw2film import data, utils, effects
from raw2film.raw_conversion import raw_to_linear, process_image, crop_rotate_zoom
from raw2film.utils import add_metadata


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Raw2Film")

        negative_stocks = dict(NEGATIVE_FILM, **REVERSAL_FILM)
        self.negative_stocks = {k: v() for k, v in negative_stocks.items() if v is not None}
        self.print_stocks = {k: v() for k, v in PRINT_FILM.items() if v is not None}

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

        self.hideable_widgets = []

        def add_option(widget, name=None, default=None, setter=None, hideable=False):
            self.side_counter += 1
            sidelayout.addWidget(widget, self.side_counter, 1)
            label = QLabel(name, alignment=(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
            sidelayout.addWidget(label, self.side_counter, 0)
            if hideable:
                self.hideable_widgets.append(widget)
                self.hideable_widgets.append(label)
            if default is not None and setter is not None:
                label.mouseDoubleClickEvent = lambda *args: setter(default)
                setter(default)

        self.image_selector = FileSelector()
        self.image_selector.filetype=f"Raw (*{' *'.join(data.EXTENSION_LIST)})"
        add_option(self.image_selector, "Image:")

        self.advanced_controls = QCheckBox()
        add_option(self.advanced_controls, "Advanced controls:", False, self.advanced_controls.setChecked)

        self.lensfunpy_db = lensfunpy.Database()
        self.cameras = {camera.maker + " " + camera.model: camera for camera in self.lensfunpy_db.cameras}
        self.lenses = {lens.model: lens for lens in self.lensfunpy_db.lenses}
        self.cameras["None"] = None
        self.lenses["None"] = None

        self.lens_correction = QCheckBox()
        add_option(self.lens_correction, "Lens correction:", False, self.lens_correction.setChecked)
        self.camera_selector = QComboBox()
        self.camera_selector.setMaximumWidth(180)
        self.camera_selector.addItems(self.cameras.keys())
        add_option(self.camera_selector, "Camera model:", "None", self.camera_selector.setCurrentText, hideable=True)
        self.lens_selector = QComboBox()
        self.lens_selector.setMaximumWidth(180)
        self.lens_selector.addItems(self.lenses.keys())
        add_option(self.lens_selector, "Lens model:", "None", self.lens_selector.setCurrentText, hideable=True)

        self.full_preview = QCheckBox()
        add_option(self.full_preview, "Full preview:", False, self.full_preview.setChecked, hideable=True)

        self.halation = QCheckBox()
        add_option(self.halation, "Halation:", True, self.halation.setChecked, hideable=True)
        self.sharpness = QCheckBox()
        add_option(self.sharpness, "Sharpness:", True, self.sharpness.setChecked, hideable=True)
        self.grain = QCheckBox()
        add_option(self.grain, "Grain:", True, self.grain.setChecked, hideable=True)

        self.exp_comp = Slider()
        self.exp_comp.setMinMaxTicks(-2, 2, 1, 6)
        add_option(self.exp_comp, "Exposure:", 0, self.exp_comp.setValue)

        self.wb_modes = {"Native": None, "Daylight": 5500, "Cloudy": 6500, "Shade": 7500, "Tungsten": 2850,
                         "Fluorescent": 3800, "Custom": None}
        self.wb_mode = QComboBox()
        self.wb_mode.addItems(list(self.wb_modes.keys()))
        add_option(self.wb_mode, "WB:", "Daylight", self.wb_mode.setCurrentText)

        self.exp_wb = Slider()
        self.exp_wb.setMinMaxTicks(2000, 12000, 50)
        add_option(self.exp_wb, "Kelvin:", 6500, self.exp_wb.setValue)

        self.rotate = QWidget()
        rotate_layout = QHBoxLayout()
        self.rotate_left = QPushButton("Left")
        self.rotate_right = QPushButton("Right")
        rotate_layout.addWidget(self.rotate_left)
        rotate_layout.addWidget(self.rotate_right)
        rotate_layout.setContentsMargins(0, 0, 0, 0)
        self.rotate.setLayout(rotate_layout)
        add_option(self.rotate, "Rotate:")

        self.rotation = Slider()
        self.rotation.setMinMaxTicks(-90, 90, 1)
        add_option(self.rotation, "Rotation angle:", 0, self.rotation.setValue)

        self.zoom = Slider()
        self.zoom.setMinMaxTicks(1, 2, 1, 100)
        add_option(self.zoom, "Zoom:", 0, self.zoom.setValue)

        self.format_selector = QComboBox()
        self.format_selector.addItems(list(data.FORMATS.keys()))
        add_option(self.format_selector, "Format:", "135", self.format_selector.setCurrentText)
        self.width = QLineEdit()
        self.width.setValidator(QDoubleValidator())
        add_option(self.width, "Width:", "36", self.width.setText, hideable=True)
        self.height = QLineEdit()
        self.height.setValidator(QDoubleValidator())
        add_option(self.height, "Height:", "24", self.height.setText, hideable=True)

        self.grain_size = Slider()
        self.grain_size.setMinMaxTicks(0.2, 2, 1, 10)
        add_option(self.grain_size, "Grain size (microns):", 1, self.grain_size.setValue, hideable=True)

        self.negative_selector = QComboBox()
        self.negative_selector.addItems(list(negative_stocks.keys()))
        add_option(self.negative_selector, "Negativ stock:", "KodakPortra400", self.negative_selector.setCurrentText)

        self.red_light = Slider()
        self.red_light.setMinMaxTicks(-1, 1, 1, 20)
        add_option(self.red_light, "Red printer light:", 0, self.red_light.setValue, hideable=True)
        self.green_light = Slider()
        self.green_light.setMinMaxTicks(-1, 1, 1, 20)
        add_option(self.green_light, "Green printer light:", 0, self.green_light.setValue, hideable=True)
        self.blue_light = Slider()
        self.blue_light.setMinMaxTicks(-1, 1, 1, 20)
        add_option(self.blue_light, "Blue printer light:", 0, self.blue_light.setValue, hideable=True)

        self.link_lights = QCheckBox()
        self.link_lights.setChecked(True)
        self.link_lights.setText("link lights")
        add_option(self.link_lights, hideable=True)

        self.print_selector = QComboBox()
        self.print_selector.addItems(["None"] + list(self.print_stocks.keys()))
        add_option(self.print_selector, "Print stock:", "KodakEnduraPremier", self.print_selector.setCurrentText)

        self.projector_kelvin = Slider()
        self.projector_kelvin.setMinMaxTicks(2700, 10000, 100)
        add_option(self.projector_kelvin, "Projector wb:", 6500, self.projector_kelvin.setValue, hideable=True)

        self.white_point = Slider()
        self.white_point.setMinMaxTicks(.5, 2., 1, 20)
        add_option(self.white_point, "White point:", 1., self.white_point.setValue, hideable=True)

        self.output_resolution = QLineEdit()
        self.output_resolution.setValidator(QIntValidator())
        add_option(self.output_resolution, "Output resolution:", "", hideable=True)

        self.save_lut_button = QPushButton("Save image")
        self.save_lut_button.released.connect(self.save_image_dialog)
        add_option(self.save_lut_button)

        self.negative_selector.currentTextChanged.connect(self.changed_negative)
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
        self.lens_correction.checkStateChanged.connect(self.apply_lens_correction)
        self.full_preview.checkStateChanged.connect(self.parameter_changed)
        self.halation.checkStateChanged.connect(self.parameter_changed)
        self.sharpness.checkStateChanged.connect(self.parameter_changed)
        self.grain.checkStateChanged.connect(self.parameter_changed)
        self.rotation.valueChanged.connect(self.crop_zoom_changed)
        self.zoom.valueChanged.connect(self.crop_zoom_changed)
        self.format_selector.currentTextChanged.connect(self.format_changed)
        self.advanced_controls.checkStateChanged.connect(self.hide_controls)
        self.grain_size.valueChanged.connect(self.parameter_changed)
        self.rotate_right.released.connect(self.rotate_right_pressed)
        self.rotate_left.released.connect(self.rotate_left_pressed)
        self.lens_selector.currentTextChanged.connect(self.apply_lens_correction)
        self.camera_selector.currentTextChanged.connect(self.apply_lens_correction)
        self.width.textChanged.connect(self.crop_zoom_changed)
        self.height.textChanged.connect(self.crop_zoom_changed)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)

        self.resize(QSize(1024, 512))

        self.waiting = False
        self.running = False

        self.threadpool = QThreadPool()

        self.XYZ_image = None
        self.corrected_image = None
        self.preview_image = None
        self.value_changed = False
        self.rotation_state = 0
        self.metadata = None

        self.hide_controls()
        self.changed_wb_mode()

    def format_changed(self):
        width, height = data.FORMATS[self.format_selector.currentText()]
        self.width.setText(str(width))
        self.height.setText(str(height))

    def rotate_left_pressed(self):
        self.rotation_state = (self.rotation_state + 1) % 4
        self.update_preview(rotate_only=True)

    def rotate_right_pressed(self):
        self.rotation_state = (self.rotation_state - 1) % 4
        self.update_preview(rotate_only=True)

    def hide_controls(self):
        if self.advanced_controls.isChecked():
            for widget in self.hideable_widgets:
                widget.show()
        else:
            for widget in self.hideable_widgets:
                widget.hide()

    def scale_pixmap(self):
        if not self.pixmap.isNull():
            scaled_pixmap = self.pixmap.scaled(self.image.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation)
            self.image.setPixmap(scaled_pixmap)

    def load_image(self):
        self.XYZ_image, self.metadata = raw_to_linear(self.image_selector.currentText())
        cam, lens = utils.find_data(self.metadata, self.lensfunpy_db)
        if cam is not None:
            self.camera_selector.setCurrentText(cam.maker + " " + cam.model)
        if lens is not None:
            self.lens_selector.setCurrentText(lens.model)
        if self.lens_correction.isChecked():
            self.apply_lens_correction()
        else:
            self.parameter_changed()

    def apply_lens_correction(self):
        if self.lens_correction.isChecked():
            cam = self.cameras[self.camera_selector.currentText()]
            lens = self.lenses[self.lens_selector.currentText()]
            if cam is not None and lens is not None:
                self.corrected_image = effects.lens_correction(self.XYZ_image.astype(np.float64), self.metadata, cam,
                                                               lens)
            else:
                self.corrected_image = self.XYZ_image
        self.parameter_changed()

    def resizeEvent(self, event):
        self.scale_pixmap()
        super().resizeEvent(event)

    def changed_wb_mode(self):
        mode = self.wb_mode.currentText()
        if mode == "Native":
            self.exp_wb.setValue(self.negative_stocks[self.negative_selector.currentText()].exposure_kelvin)
        elif mode != "Custom":
            self.exp_wb.setValue(self.wb_modes[mode])
            self.parameter_changed()

    def changed_exp_wb(self):
        kelvin = self.exp_wb.getValue()
        curr_mode = self.wb_mode.currentText()
        if curr_mode == "Custom":
            self.parameter_changed()
        elif curr_mode == "Native" and kelvin == self.negative_stocks[
            self.negative_selector.currentText()].exposure_kelvin:
            self.parameter_changed()
        elif self.wb_modes[curr_mode] != kelvin:
            self.wb_mode.setCurrentText("Custom")
            self.parameter_changed()

    def changed_negative(self):
        negative = self.negative_selector.currentText()
        self.print_selector.setEnabled(negative not in REVERSAL_FILM)
        curr_kelvin = self.exp_wb.getValue()
        if self.wb_mode.currentText() == "Native":
            self.changed_wb_mode()
            if curr_kelvin == self.exp_wb.getValue():
                self.parameter_changed()
        else:
            self.parameter_changed()

    def setup_params(self):
        kwargs = {"negative_film": self.negative_stocks[self.negative_selector.currentText()],
                  "print_film": self.print_stocks[
                      self.print_selector.currentText()] if self.print_selector.isEnabled() else None,
                  "projector_kelvin": self.projector_kelvin.getValue(), "exp_comp": self.exp_comp.getValue(),
                  "printer_light_comp": np.array(
                      [self.red_light.getValue(), self.green_light.getValue(), self.blue_light.getValue()]),
                  "white_point": self.white_point.getValue(), "zoom": self.zoom.getValue(),
                  "rotation": self.rotation.getValue(), "exposure_kelvin": self.exp_wb.getValue(),
                  "halation": self.halation.isChecked(), "sharpness": self.sharpness.isChecked(),
                  "grain": self.grain.isChecked(), "frame_width": float(self.width.text()),
                  "frame_height": float(self.height.text()), "grain_size": self.grain_size.getValue() / 1000,
                  "rotate_times": self.rotation_state, "src": self.image_selector.currentText()}
        resolution = self.output_resolution.text()
        if resolution != "":
            kwargs["resolution"] = int(resolution)
        if self.lens_correction.isChecked():
            cam = self.cameras[self.camera_selector.currentText()]
            lens = self.lenses[self.lens_selector.currentText()]
            if cam is not None and lens is not None:
                kwargs["cam"] = cam
                kwargs["lens"] = lens
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

    def update_preview(self, rotate_only=False, *args, **kwargs):
        if self.XYZ_image is None:
            return
        kwargs = self.setup_params()
        if (self.value_changed or self.full_preview.isChecked()) and not rotate_only:
            if self.lens_correction.isChecked():
                image = self.corrected_image
            else:
                image = self.XYZ_image
            self.preview_image = process_image(image, fast_mode=not self.full_preview.isChecked(),
                                               metadata=self.metadata, **kwargs)
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
        image, metadata = raw_to_linear(kwargs["src"], half_size=False, )
        if "cam" in kwargs and "lens" in kwargs:
            image = effects.lens_correction(image, metadata, kwargs["cam"], kwargs["lens"])
        image = process_image(image, fast_mode=False, **kwargs)
        if '.' not in filename:
            filename += '.jpg'
        imageio.imwrite(filename, image, quality=100, format='.jpg')
        add_metadata(filename, metadata, exp_comp=kwargs['exp_comp'])

    def save_image_dialog(self):
        filename, ok = QFileDialog.getSaveFileName(self)

        if ok:
            self.start_worker(self.save_image, filename=filename, semaphore=False)


def gui_main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == '__main__':
    gui_main()
