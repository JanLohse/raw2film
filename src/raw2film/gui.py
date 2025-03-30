import json
from functools import cache
from functools import lru_cache

import exiftool
import imageio
import lensfunpy
from PyQt6.QtCore import QSize, QThreadPool
from PyQt6.QtGui import QPixmap, QImage, QIntValidator, QDoubleValidator, QAction
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QGridLayout, QSizePolicy, QCheckBox, QVBoxLayout
from spectral_film_lut import NEGATIVE_FILM, REVERSAL_FILM, PRINT_FILM
from spectral_film_lut.utils import *

from raw2film import data, utils
from raw2film.raw_conversion import *
from raw2film.utils import add_metadata
from raw2film.image_bar import ImageBar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Raw2Film")

        negative_stocks = {**NEGATIVE_FILM, **REVERSAL_FILM}
        self.negative_stocks = {k: v() for k, v in negative_stocks.items() if v is not None}
        self.print_stocks = {k: v() for k, v in PRINT_FILM.items() if v is not None}
        self.print_stocks["None"] = None


        page_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        top_widget = QWidget()
        top_widget.setLayout(top_layout)
        sidebar = QWidget()
        sidebar.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed))
        side_layout = QGridLayout()
        sidebar.setLayout(side_layout)
        side_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        top_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.image = QLabel("Select a reference image for the preview")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        self.image.setMinimumSize(QSize(256, 256))
        self.pixmap = QPixmap()

        self.image_bar = ImageBar()
        page_layout.addWidget(top_widget)
        page_layout.addWidget(self.image_bar)

        top_layout.addWidget(self.image)
        top_layout.addWidget(sidebar, alignment=Qt.AlignmentFlag.AlignBottom)

        self.side_counter = -1

        self.hideable_widgets = []

        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        view_menu = menu.addMenu("View")

        def add_option(widget, name=None, default=None, setter=None, hideable=False):
            self.side_counter += 1
            side_layout.addWidget(widget, self.side_counter, 1)
            label = QLabel(name, alignment=(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter))
            side_layout.addWidget(label, self.side_counter, 0)
            if hideable:
                self.hideable_widgets.append(widget)
                self.hideable_widgets.append(label)
            if default is not None and setter is not None:
                label.mouseDoubleClickEvent = lambda *args: setter(default)
                setter(default)

        self.image_selector = QAction("Open images", self)
        file_menu.addAction(self.image_selector)
        self.folder_selector = QAction("Open folder", self)
        file_menu.addAction(self.folder_selector)
        self.save_image_button = QAction("Save current image")
        self.save_image_button.triggered.connect(self.save_image_dialog)
        file_menu.addAction(self.save_image_button)
        self.save_all_button = QAction("Save all images")
        self.save_all_button.triggered.connect(self.save_all_images)
        file_menu.addAction(self.save_all_button)
        self.save_settings_button = QAction("Save settings")
        file_menu.addAction(self.save_settings_button)
        self.load_settings_button = QAction("Load settings")
        file_menu.addAction(self.load_settings_button)

        self.advanced_controls = QAction("Advanced Controls", self)
        self.advanced_controls.setCheckable(True)
        view_menu.addAction(self.advanced_controls)
        self.p3_preview = QAction("Display P3 preview", self)
        self.p3_preview.setCheckable(True)
        view_menu.addAction(self.p3_preview)
        self.full_preview = QAction("Full preview")
        self.full_preview.setCheckable(True)
        view_menu.addAction(self.full_preview)

        self.profiles = QComboBox()
        self.profiles.addItems(["Custom", "Profile 1", "Profile 2", "Profile 3"])
        add_option(self.profiles, "Profile:", "Profile 1", self.profiles.setCurrentText)

        self.lensfunpy_db = lensfunpy.Database()
        self.cameras = {camera.maker + " " + camera.model: camera for camera in self.lensfunpy_db.cameras}
        self.lenses = {lens.model: lens for lens in self.lensfunpy_db.lenses}
        self.cameras["None"] = None
        self.lenses["None"] = None

        self.lens_correction = QCheckBox()
        add_option(self.lens_correction, "Lens correction:", True, self.lens_correction.setChecked)
        self.camera_selector = QComboBox()
        self.camera_selector.setMaximumWidth(180)
        self.camera_selector.addItems(self.cameras.keys())
        add_option(self.camera_selector, "Camera model:", "None", self.camera_selector.setCurrentText, hideable=True)
        self.lens_selector = QComboBox()
        self.lens_selector.setMaximumWidth(180)
        self.lens_selector.addItems(self.lenses.keys())
        add_option(self.lens_selector, "Lens model:", "None", self.lens_selector.setCurrentText, hideable=True)

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
        self.grain_size.setMinMaxTicks(1, 6, 1, 2)
        add_option(self.grain_size, "Grain size (microns):", 4, self.grain_size.setValue, hideable=True)

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

        self.negative_selector.currentTextChanged.connect(self.changed_negative)
        self.print_selector.currentTextChanged.connect(self.print_light_changed)
        self.image_selector.triggered.connect(self.load_images)
        self.folder_selector.triggered.connect(self.load_folder)
        self.projector_kelvin.valueChanged.connect(self.parameter_changed)
        self.exp_comp.valueChanged.connect(self.parameter_changed)
        self.wb_mode.currentTextChanged.connect(self.changed_wb_mode)
        self.exp_wb.valueChanged.connect(self.changed_exp_wb)
        self.red_light.valueChanged.connect(self.lights_changed)
        self.green_light.valueChanged.connect(self.lights_changed)
        self.blue_light.valueChanged.connect(self.lights_changed)
        self.white_point.valueChanged.connect(self.parameter_changed)
        self.lens_correction.checkStateChanged.connect(self.parameter_changed)
        self.full_preview.triggered.connect(self.parameter_changed)
        self.halation.checkStateChanged.connect(self.parameter_changed)
        self.sharpness.checkStateChanged.connect(self.parameter_changed)
        self.grain.checkStateChanged.connect(self.parameter_changed)
        self.rotation.valueChanged.connect(self.crop_zoom_changed)
        self.zoom.valueChanged.connect(self.crop_zoom_changed)
        self.format_selector.currentTextChanged.connect(self.format_changed)
        self.advanced_controls.triggered.connect(self.hide_controls)
        self.grain_size.valueChanged.connect(self.parameter_changed)
        self.rotate_right.released.connect(self.rotate_right_pressed)
        self.rotate_left.released.connect(self.rotate_left_pressed)
        self.lens_selector.currentTextChanged.connect(self.parameter_changed)
        self.camera_selector.currentTextChanged.connect(self.parameter_changed)
        self.width.textChanged.connect(self.crop_zoom_changed)
        self.height.textChanged.connect(self.crop_zoom_changed)
        self.profiles.currentTextChanged.connect(self.load_profile_params)
        self.p3_preview.triggered.connect(self.parameter_changed)
        self.save_settings_button.triggered.connect(self.save_settings)
        self.load_settings_button.triggered.connect(self.load_settings)
        self.image_bar.image_changed.connect(self.load_image)

        widget = QWidget()
        widget.setLayout(page_layout)
        self.setCentralWidget(widget)

        self.resize(QSize(1024, 512))

        self.waiting = False
        self.running = False

        self.threadpool = QThreadPool()

        self.corrected_image = None
        self.preview_image = None
        self.value_changed = False
        self.rotation_state = 0
        self.metadata = None
        self.active = True

        self.hide_controls()
        self.changed_wb_mode()

        self.filenames = None

        self.image_params = {}
        self.profiles_params = {}

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

    def load_images(self):
        filenames, ok = QFileDialog.getOpenFileNames(self, 'Open raw images', '',
                                                     filter=f"Raw (*{' *'.join(data.EXTENSION_LIST)})")

        if ok:
            self.filenames = {filename.split("/")[-1]: filename for filename in filenames}
            self.image_bar.clear_images()
            self.image_bar.load_images(filenames)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select image folder', '')
        self.filenames = {filename.split("/")[-1]: folder + "/" + filename for filename in os.listdir(folder) if
                          filename.lower().endswith(data.EXTENSION_LIST)}
        self.image_bar.clear_images()
        self.image_bar.load_images(self.filenames.values())

    def load_image_process(self, src, **kwargs):
        self.metadata = self.load_metadata(src)
        if src in self.image_params:
            kwargs = self.image_params[src]
            self.load_image_params(src)
        else:
            self.reset_image_params()
        if src not in self.image_params or "cam" not in kwargs or "lens" not in kwargs:
            cam, lens = utils.find_data(self.metadata, self.lensfunpy_db)
            if cam is not None:
                self.camera_selector.setCurrentText(cam.maker + " " + cam.model)
            if lens is not None:
                self.lens_selector.setCurrentText(lens.model)
        self.active = True
        self.parameter_changed()

    def load_image(self, src):
        self.start_worker(self.load_image_process, src=src, semaphore=False)

    @cache
    def load_metadata(self, src):
        with exiftool.ExifToolHelper() as et:
            metadata = et.get_metadata(src)[0]
        return metadata

    @lru_cache
    def load_raw_image(self, src, cam=None, lens=None):
        if cam is not None and lens is not None:
            cam = self.cameras[cam]
            lens = self.lenses[lens]

            return effects.lens_correction(raw_to_linear(src), self.metadata, cam, lens)
        else:
            return raw_to_linear(src)

    def xyz_image(self):
        src = self.image_bar.current_image()
        if self.lens_correction.isChecked():
            cam = self.camera_selector.currentText()
            lens = self.lens_selector.currentText()
            return self.load_raw_image(src, cam, lens)
        else:
            return self.load_raw_image(src)

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

    def setup_image_params(self):
        kwargs = {"exp_comp": self.exp_comp.getValue(), "zoom": self.zoom.getValue(),
                  "rotation": self.rotation.getValue(), "exposure_kelvin": self.exp_wb.getValue(),
                  "rotate_times": self.rotation_state, "src": self.image_bar.current_image().split("/")[-1],
                  "format": self.format_selector.currentText(), "lens_correction": self.lens_correction.isChecked(),
                  "profile": self.profiles.currentText(), "wb_mode": self.wb_mode.currentText(),
                  "cam": self.camera_selector.currentText(), "lens": self.lens_selector.currentText()}
        return kwargs

    def setup_profile_params(self):
        kwargs = {"negative_film": self.negative_selector.currentText(),
                  "print_film": self.print_selector.currentText() if self.print_selector.isEnabled() else None,
                  "projector_kelvin": self.projector_kelvin.getValue(), "printer_light_comp": (
                self.red_light.getValue(), self.green_light.getValue(), self.blue_light.getValue()),
                  "white_point": self.white_point.getValue(), "halation": self.halation.isChecked(),
                  "sharpness": self.sharpness.isChecked(), "grain": self.grain.isChecked(),
                  "frame_width": float(self.width.text()), "frame_height": float(self.height.text()),
                  "grain_size": self.grain_size.getValue() / 1000, "link_lights": self.link_lights.isChecked(),
                  "format": self.format_selector.currentText(), }
        resolution = self.output_resolution.text()
        if resolution != "":
            kwargs["resolution"] = int(resolution)
        return kwargs

    def load_image_params(self, src):
        kwargs = self.image_params[src]
        self.exp_comp.setValue(kwargs["exp_comp"])
        self.zoom.setValue(kwargs["zoom"])
        self.rotation_state = kwargs["rotate_times"]
        self.rotation.setValue(kwargs["rotation"])
        self.exp_wb.setValue(kwargs["exposure_kelvin"])
        self.wb_mode.setCurrentText(kwargs["wb_mode"])
        self.lens_correction.setChecked(kwargs["lens_correction"])
        self.profiles.setCurrentText(kwargs["profile"])
        self.camera_selector.setCurrentText(kwargs["cam"])
        self.lens_selector.setCurrentText(kwargs["lens"])
        self.load_profile_params()

    def reset_image_params(self):
        self.exp_comp.setValue(0)
        self.zoom.setValue(0)
        self.rotation_state = 0
        self.rotation.setValue(0)
        self.wb_mode.setCurrentText("Native")

    def load_profile_params(self):
        profile = self.profiles.currentText()
        if profile == "Custom":
            profile = self.image_bar.current_image().split("/")[-1]
        if profile in self.profiles_params:
            kwargs = self.profiles_params[profile]
            self.projector_kelvin.setValue(kwargs["projector_kelvin"])
            self.red_light.setValue(kwargs["printer_light_comp"][0])
            self.green_light.setValue(kwargs["printer_light_comp"][1])
            self.blue_light.setValue(kwargs["printer_light_comp"][2])
            self.white_point.setValue(kwargs["white_point"])
            self.halation.setChecked(kwargs["halation"])
            self.sharpness.setChecked(kwargs["sharpness"])
            self.grain.setChecked(kwargs["grain"])
            self.format_selector.setCurrentText(kwargs["format"])
            self.width.setText(str(kwargs["frame_width"]))
            self.height.setText(str(kwargs["frame_height"]))
            self.grain_size.setValue(kwargs["grain_size"] * 1000)
            self.negative_selector.setCurrentText(kwargs["negative_film"])
            self.print_selector.setCurrentText(kwargs["print_film"])
            self.link_lights.setChecked(kwargs["link_lights"])
        else:
            self.profiles_params[profile] = self.setup_profile_params()

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
        if self.active:
            self.value_changed = True

            self.start_worker(self.update_preview)

    def crop_zoom_changed(self):
        if self.active:
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
        if self.image_bar.current_image() is None:
            return
        image_args = self.setup_image_params()
        profile_args = self.setup_profile_params()
        if "resolution" in profile_args:
            profile_args.pop("resolution")
        self.image_params[image_args["src"]] = image_args
        image_args["src"] = self.filenames[image_args["src"]]
        if image_args["profile"] != "Custom":
            self.profiles_params[image_args["profile"]] = profile_args
        else:
            self.profiles_params[image_args["src"]] = profile_args
        processing_args = {**image_args, **profile_args}
        processing_args["negative_film"] = self.negative_stocks[processing_args["negative_film"]]
        if "print_film" in processing_args and processing_args["print_film"] is not None:
            processing_args["print_film"] = self.print_stocks[processing_args["print_film"]]
        if self.p3_preview.isChecked():
            processing_args["output_colourspace"] = "Display P3"
        if (self.value_changed or self.full_preview.isChecked()) and not rotate_only:
            image = self.xyz_image()
            self.preview_image = process_image(image, fast_mode=not self.full_preview.isChecked(),
                                               metadata=self.metadata, **processing_args)
        image = self.preview_image
        if not self.full_preview.isChecked():
            image = crop_rotate_zoom(image, **processing_args)
        height, width, _ = image.shape
        image = QImage(np.require(image, np.uint8, 'C'), width, height, 3 * width, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(image)
        self.image.setPixmap(self.pixmap)
        self.scale_pixmap()

    def save_image(self, src, filename, **kwargs):
        short = src.split("/")[-1]
        if short not in self.image_params:
            image_args = self.setup_image_params()
            image_args["src"] = short
            image_args["exp_comp"] = 0
            image_args["zoom"] = 0
            image_args["rotation"] = 0
            image_args["rotation_state"] = 0
            image_args["wb_mode"] = "Native"
            self.image_params[short] = image_args
        else:
            image_args = self.image_params[short]
        if image_args["profile"] != "Custom":
            profile_args = self.profiles_params[image_args["profile"]]
        else:
            profile_args = self.profiles_params[image_args["src"]]
        image_args["src"] = self.filenames[short]
        processing_args = {**image_args, **profile_args}
        processing_args["negative_film"] = self.negative_stocks[processing_args["negative_film"]]
        if "print_film" in processing_args and processing_args["print_film"] is not None:
            processing_args["print_film"] = self.print_stocks[processing_args["print_film"]]
        image = raw_to_linear(src, half_size=False)
        metadata = self.load_metadata(src)

        if processing_args["lens_correction"]:
            image = effects.lens_correction(image, metadata, self.cameras[processing_args["cam"]],
                                            self.lenses[processing_args["lens"]])
        image = process_image(image, fast_mode=False, **processing_args)
        if '.' not in filename:
            filename += '.jpg'
        imageio.imwrite(filename, image, quality=100, format='.jpg')
        add_metadata(filename, metadata, exp_comp=processing_args['exp_comp'])
        print(f"exported {filename}")

    def save_image_dialog(self):
        filename, ok = QFileDialog.getSaveFileName(self)
        src = self.image_bar.current_image()
        if ok:
            self.start_worker(self.save_image, src=src, filename=filename, semaphore=False)

    def save_all_process(self, folder, filenames, **kwargs):
        for filename in filenames:
            self.save_image(src=filename, filename=folder + "/" + filename.split("/")[-1].split(".")[0])

    def save_all_images(self):
        folder = QFileDialog.getExistingDirectory(self)
        self.start_worker(self.save_all_process, folder=folder, filenames=self.filenames.values(), semaphore=False)

    def save_settings(self):
        filename, ok = QFileDialog.getSaveFileName(self, "Select file name", "raw2film_settings.json", "*.json")
        if ok:
            complete_dict = {"image_params": self.image_params, "profile_params": self.profiles_params}
            with open(filename, "w") as f:
                json.dump(complete_dict, f)

    def load_settings(self):
        filename, ok = QFileDialog.getOpenFileName(self)
        if ok:
            with open(filename, "r") as f:
                complete_dict = json.load(f)
            self.image_params = complete_dict["image_params"]
            self.profiles_params = complete_dict["profile_params"]


def gui_main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == '__main__':
    gui_main()
