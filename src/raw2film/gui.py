import json
from functools import cache
from functools import lru_cache

import exiftool
import imageio
import lensfunpy
from PyQt6.QtCore import QSize, QThreadPool
from PyQt6.QtGui import QPixmap, QImage, QIntValidator, QDoubleValidator, QAction, QShortcut, QKeySequence
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QGridLayout, QSizePolicy, QCheckBox, QVBoxLayout, \
    QInputDialog, QMessageBox
from raw2film import data, utils
from raw2film.image_bar import ImageBar
from raw2film.raw_conversion import *
from raw2film.utils import add_metadata
from spectral_film_lut import NEGATIVE_FILM, REVERSAL_FILM, PRINT_FILM
from spectral_film_lut.utils import *


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
        profile_menu = menu.addMenu("Profile")

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
        self.image_selector.setShortcut(QKeySequence("Ctrl+O"))
        file_menu.addAction(self.image_selector)
        self.folder_selector = QAction("Open folder", self)
        self.folder_selector.setShortcut(QKeySequence("Ctrl+Shift+O"))
        file_menu.addAction(self.folder_selector)
        self.save_image_button = QAction("Export current image")
        self.save_image_button.triggered.connect(self.save_image_dialog)
        file_menu.addAction(self.save_image_button)
        self.save_all_button = QAction("Export all images")
        self.save_all_button.triggered.connect(self.save_all_images)
        self.save_selected_button = QAction("Export selected images")
        self.save_selected_button.triggered.connect(self.save_selected_images)
        file_menu.addAction(self.save_selected_button)
        file_menu.addAction(self.save_all_button)
        self.save_settings_button = QAction("Save settings")
        file_menu.addAction(self.save_settings_button)
        self.load_settings_button = QAction("Load settings")
        file_menu.addAction(self.load_settings_button)
        self.delete_profile_button = QAction("Delete profile")
        profile_menu.addAction(self.delete_profile_button)

        self.advanced_controls = QAction("Advanced Controls", self)
        self.advanced_controls.setShortcut(QKeySequence("Ctrl+Shift+A"))
        self.advanced_controls.setCheckable(True)
        view_menu.addAction(self.advanced_controls)
        self.p3_preview = QAction("Display P3 preview", self)
        self.p3_preview.setShortcut(QKeySequence("Ctrl+Shift+P"))
        self.p3_preview.setCheckable(True)
        view_menu.addAction(self.p3_preview)
        self.full_preview = QAction("Full preview")
        self.full_preview.setShortcut(QKeySequence("Ctrl+Shift+F"))
        self.full_preview.setCheckable(True)
        view_menu.addAction(self.full_preview)

        self.default_profile_params = {"negative_film": "KodakPortra400", "print_film": "KodakEnduraPremier",
                                       "red_light": 0, "green_light": 0, "blue_light": 0, "white_point": 1,
                                       "halation": True, "sharpness": True, "grain": True, "format": "135",
                                       "grain_size": 0.002, "link_lights": True}
        self.default_image_params = {"exp_comp": 0, "zoom": 1, "rotate_times": 0, "rotation": 0, "wb_mode": "Daylight",
                                     "profile": "Default", "lens_correction": True}

        self.profile_selector = QComboBox()
        self.profile_selector.addItem("Default")
        self.add_profile = QPushButton("+")
        self.add_profile.setFixedWidth(25)
        profile_widget = QWidget()
        profile_widget_layout = QHBoxLayout()
        profile_widget.setLayout(profile_widget_layout)
        profile_widget_layout.addWidget(self.profile_selector)
        profile_widget_layout.addWidget(self.add_profile)
        profile_widget_layout.setContentsMargins(0, 0, 0, 0)
        add_option(profile_widget, "Profile:", "Default", self.profile_selector.setCurrentText)

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

        self.wb_modes = {"Daylight": 5500, "Cloudy": 6500, "Shade": 7500, "Tungsten": 2850, "Fluorescent": 3800,
                         "Custom": None}
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
        self.rotation.setMinMaxTicks(-90, 90, 1, 2)
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
        self.grain_size.setMinMaxTicks(0.1, 6, 1, 10)
        add_option(self.grain_size, "Grain size (microns):", 2, self.grain_size.setValue, hideable=True)

        self.halation_size = Slider()
        self.halation_size.setMinMaxTicks(0.5, 2, 1, 2)
        add_option(self.halation_size, "Halation size:", 1, self.halation_size.setValue, hideable=True)
        self.halation_green = Slider()
        self.halation_green.setMinMaxTicks(0, 1, 1, 10)
        add_option(self.halation_green, "Halation color:", .4, self.halation_green.setValue, hideable=True)

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

        QShortcut(QKeySequence('Up'), self).activated.connect(self.exp_comp.increase)
        QShortcut(QKeySequence('Down'), self).activated.connect(self.exp_comp.decrease)
        QShortcut(QKeySequence("Ctrl+Right"), self).activated.connect(self.rotation.increase)
        QShortcut(QKeySequence("Ctrl+Left"), self).activated.connect(self.rotation.decrease)
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(self.rotate_image)
        QShortcut(QKeySequence("Ctrl++"), self).activated.connect(lambda: self.zoom.increase(5))
        QShortcut(QKeySequence("Ctrl+-"), self).activated.connect(lambda: self.zoom.decrease(5))
        QShortcut(QKeySequence("Shift+Ctrl++"), self).activated.connect(self.zoom.increase)
        QShortcut(QKeySequence("Shift+Ctrl+-"), self).activated.connect(self.zoom.decrease)
        QShortcut(QKeySequence("1"), self).activated.connect(
            lambda: self.profile_selector.setCurrentIndex(min(0, self.profile_selector.count() - 1)))
        QShortcut(QKeySequence("2"), self).activated.connect(
            lambda: self.profile_selector.setCurrentIndex(min(1, self.profile_selector.count() - 1)))
        QShortcut(QKeySequence("3"), self).activated.connect(
            lambda: self.profile_selector.setCurrentIndex(min(2, self.profile_selector.count() - 1)))
        QShortcut(QKeySequence("4"), self).activated.connect(
            lambda: self.profile_selector.setCurrentIndex(min(3, self.profile_selector.count() - 1)))
        QShortcut(QKeySequence("5"), self).activated.connect(
            lambda: self.profile_selector.setCurrentIndex(min(4, self.profile_selector.count() - 1)))
        QShortcut(QKeySequence("6"), self).activated.connect(
            lambda: self.profile_selector.setCurrentIndex(min(5, self.profile_selector.count() - 1)))
        QShortcut(QKeySequence("7"), self).activated.connect(
            lambda: self.profile_selector.setCurrentIndex(min(6, self.profile_selector.count() - 1)))
        QShortcut(QKeySequence("8"), self).activated.connect(
            lambda: self.profile_selector.setCurrentIndex(min(7, self.profile_selector.count() - 1)))
        QShortcut(QKeySequence("9"), self).activated.connect(
            lambda: self.profile_selector.setCurrentIndex(min(8, self.profile_selector.count() - 1)))
        QShortcut(QKeySequence("Shift+D"), self).activated.connect(lambda: self.wb_mode.setCurrentText("Daylight"))
        QShortcut(QKeySequence("Shift+C"), self).activated.connect(lambda: self.wb_mode.setCurrentText("Cloudy"))
        QShortcut(QKeySequence("Shift+S"), self).activated.connect(lambda: self.wb_mode.setCurrentText("Shade"))
        QShortcut(QKeySequence("Shift+T"), self).activated.connect(lambda: self.wb_mode.setCurrentText("Tungsten"))
        QShortcut(QKeySequence("Shift+F"), self).activated.connect(lambda: self.wb_mode.setCurrentText("Fluorescent"))

        self.negative_selector.currentTextChanged.connect(self.changed_negative)
        self.print_selector.currentTextChanged.connect(lambda x: self.profile_changed(x, "print_film"))
        self.image_selector.triggered.connect(self.load_images)
        self.folder_selector.triggered.connect(self.load_folder)
        self.projector_kelvin.valueChanged.connect(lambda x: self.profile_changed(x, "projector_kelvin"))
        self.exp_comp.valueChanged.connect(lambda x: self.setting_changed(x, "exp_comp"))
        self.wb_mode.currentTextChanged.connect(self.changed_wb_mode)
        self.exp_wb.valueChanged.connect(lambda x: self.setting_changed(x, "exposure_kelvin"))
        self.red_light.valueChanged.connect(lambda x: self.light_changed(x, "red_light"))
        self.green_light.valueChanged.connect(lambda x: self.light_changed(x, "green_light"))
        self.blue_light.valueChanged.connect(lambda x: self.light_changed(x, "blue_light"))
        self.white_point.valueChanged.connect(lambda x: self.profile_changed(x, "white_point"))
        self.lens_correction.stateChanged.connect(lambda x: self.setting_changed(x, "lens_correction"))
        self.full_preview.triggered.connect(lambda: self.parameter_changed())
        self.halation.stateChanged.connect(lambda x: self.profile_changed(x, "halation"))
        self.sharpness.stateChanged.connect(lambda x: self.profile_changed(x, "sharpness"))
        self.grain.stateChanged.connect(lambda x: self.profile_changed(x, "grain"))
        self.rotation.valueChanged.connect(lambda x: self.setting_changed(x, "rotation", crop_zoom=True))
        self.zoom.valueChanged.connect(lambda x: self.setting_changed(x, "zoom", crop_zoom=True))
        self.format_selector.currentTextChanged.connect(self.format_changed)
        self.advanced_controls.triggered.connect(self.hide_controls)
        self.grain_size.valueChanged.connect(lambda x: self.profile_changed(x / 1000, "grain_size"))
        self.halation_size.valueChanged.connect(lambda x: self.profile_changed(x, "halation_size"))
        self.halation_green.valueChanged.connect(lambda x: self.profile_changed(x, "halation_green_factor"))
        self.rotate_right.released.connect(self.rotate_image)
        self.rotate_left.released.connect(lambda: self.rotate_image(-1))
        self.lens_selector.currentTextChanged.connect(lambda x: self.setting_changed(x, "lens"))
        self.camera_selector.currentTextChanged.connect(lambda x: self.setting_changed(x, "cam"))
        self.width.textChanged.connect(lambda x: self.profile_changed(float(x), "frame_width", crop_zoom=True))
        self.height.textChanged.connect(lambda x: self.profile_changed(float(x), "frame_height", crop_zoom=True))
        self.profile_selector.currentTextChanged.connect(self.load_profile_params)
        self.p3_preview.triggered.connect(lambda: self.parameter_changed())
        self.save_settings_button.triggered.connect(self.save_settings)
        self.load_settings_button.triggered.connect(self.load_settings)
        self.image_bar.image_changed.connect(self.load_image)
        self.add_profile.released.connect(self.add_profile_prompt)
        self.delete_profile_button.triggered.connect(self.delete_profile)

        widget = QWidget()
        widget.setLayout(page_layout)
        self.setCentralWidget(widget)

        self.resize(QSize(1024, 512))

        self.waiting = False
        self.running = False

        self.threadpool = QThreadPool()

        self.corrected_image = None
        self.preview_image = None
        self.rotate_times = 0
        self.active = True  # prevent from running update_preview by setting inactive
        self.loading = False  # prevent from storing settings while setting widget values

        self.hide_controls()

        self.filenames = None

        self.image_params = {}
        self.profile_params = {}

    def delete_profile(self):
        current_profile = self.profile_selector.currentText()
        reply = QMessageBox()
        reply.setText(f"Delete profile {current_profile}?")
        reply.setStandardButtons(QMessageBox.StandardButton.Yes |
                                 QMessageBox.StandardButton.No)
        x = reply.exec()
        if x == QMessageBox.StandardButton.Yes:
            if current_profile == "Default":
                self.profile_params[current_profile] = {}
            else:
                self.profile_selector.setCurrentText("Default")
                self.profile_selector.removeItem(self.profile_selector.findText(current_profile))
                if current_profile in self.profile_params:
                    self.profile_params.pop(current_profile)
                for image in self.image_params:
                    if 'profile' in self.image_params[image]:
                        self.image_params[image]['profile'] = "Default"
            self.load_profile_params()

    def add_profile_prompt(self):
        text, ok = QInputDialog.getText(self, "Add profile", "Profile name:")
        if ok:
            self.profile_selector.addItem(text)

    def format_changed(self, format):
        width, height = data.FORMATS[format]
        self.width.setText(str(width))
        self.height.setText(str(height))

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
        if folder:
            self.filenames = {filename.split("/")[-1]: folder + "/" + filename for filename in os.listdir(folder) if
                              filename.lower().endswith(data.EXTENSION_LIST)}
            self.image_bar.clear_images()
            self.image_bar.load_images(self.filenames.values())

    def load_image(self, src, **kwargs):
        self.start_worker(self.load_image_process, src=src)

    def load_image_process(self, src, **kwargs):
        src_short = src.split("/")[-1]
        if src_short not in self.image_params:
            self.image_params[src_short] = {}
            metadata = self.load_metadata(src)
            cam, lens = utils.find_data(metadata, self.lensfunpy_db)
            if cam is not None:
                self.image_params[src_short]["cam"] = cam.maker + " " + cam.model
            if lens is not None:
                self.image_params[src_short]["lens"] = lens.model
        if "profile" not in self.image_params[src_short]:
            self.image_params[src_short]["profile"] = self.profile_selector.currentText()
        self.load_image_params(src_short)
        self.update_preview(src)

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

            return effects.lens_correction(raw_to_linear(src), self.load_metadata(src), cam, lens)
        else:
            return raw_to_linear(src)

    def xyz_image(self, src=None):
        if src is None:
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

    def changed_wb_mode(self, mode):
        if mode != "Custom":
            self.exp_wb.setValue(self.wb_modes[mode])
            self.parameter_changed()

    def changed_negative(self, negative):
        self.active = False
        self.print_selector.setEnabled(negative not in REVERSAL_FILM)
        if self.print_selector.isEnabled():
            self.profile_changed(self.print_selector.currentText(), "print_film")
        else:
            self.profile_changed(None, "print_film")
        if negative in REVERSAL_FILM and self.negative_stocks[negative].projection_kelvin is not None:
            self.projector_kelvin.setValue(self.negative_stocks[negative].projection_kelvin)
        elif self.print_selector.currentText() != "None" and self.print_stocks[
            self.print_selector.currentText()].projection_kelvin is not None:
            self.projector_kelvin.setValue(self.print_stocks[self.print_selector.currentText()].projection_kelvin)
        self.active = True
        self.profile_changed(negative, "negative_film")

    def profile_changed(self, value, key, crop_zoom=False):
        if self.loading:
            return
        profile = self.profile_selector.currentText()
        if profile not in self.profile_params:
            self.profile_params[profile] = {}
        self.profile_params[profile][key] = value
        if crop_zoom:
            self.crop_zoom_changed()
        else:
            self.parameter_changed()

    def setting_changed(self, value, key, crop_zoom=False):
        if self.loading:
            return
        for src in self.image_bar.get_highlighted():
            src_short = src.split("/")[-1]
            if src_short not in self.image_params:
                self.image_params[src_short] = {}
            self.image_params[src_short][key] = value
            if key == "exposure_kelvin":
                image_params = self.setup_image_params(src_short)
                wb_mode = image_params["wb_mode"]
                if wb_mode != "Custom" and value != self.wb_modes[wb_mode]:
                    self.image_params[src_short]["wb_mode"] = "Custom"
                    self.wb_mode.setCurrentText("Custom")
        if crop_zoom:
            self.crop_zoom_changed()
        else:
            self.parameter_changed()

    def setup_profile_params(self, profile, src=None):
        if profile in self.profile_params:
            return {**self.default_profile_params, **self.profile_params[profile]}
        else:
            return self.default_profile_params

    def load_image_params(self, src):
        image_params = self.setup_image_params(src)
        self.loading = True
        self.exp_comp.setValue(image_params["exp_comp"])
        self.zoom.setValue(image_params["zoom"])
        if "rotate_times" in image_params:
            self.rotate_times = image_params["rotate_times"]
        if "rotation" in image_params:
            self.rotation.setValue(image_params["rotation"])
        if "exposure_kelvin" in image_params:
            self.exp_wb.setValue(image_params["exposure_kelvin"])
        self.wb_mode.setCurrentText(image_params["wb_mode"])
        if "lens_correction" in image_params:
            self.lens_correction.setChecked(image_params["lens_correction"])
        if "profile" in image_params:
            self.profile_selector.setCurrentText(image_params["profile"])
        if "cam" in image_params:
            self.camera_selector.setCurrentText(image_params["cam"])
        if "lens" in image_params:
            self.lens_selector.setCurrentText(image_params["lens"])
        self.loading = False

    def load_profile_params(self, profile=None):
        if profile is None:
            profile = self.profile_selector.currentText()
        self.active = False
        self.setting_changed(profile, "profile")
        self.loading = True
        profile_params = self.setup_profile_params(profile)
        if "projector_kelvin" in profile_params:
            self.projector_kelvin.setValue(profile_params["projector_kelvin"])
        self.red_light.setValue(profile_params["red_light"])
        self.green_light.setValue(profile_params["green_light"])
        self.blue_light.setValue(profile_params["blue_light"])
        self.white_point.setValue(profile_params["white_point"])
        self.halation.setChecked(profile_params["halation"])
        self.sharpness.setChecked(profile_params["sharpness"])
        self.grain.setChecked(profile_params["grain"])
        if "frame_width" in profile_params:
            self.width.setText(str(profile_params["frame_width"]))
        if "frame_height" in profile_params:
            self.height.setText(str(profile_params["frame_height"]))
        self.grain_size.setValue(profile_params["grain_size"] * 1000)
        self.negative_selector.setCurrentText(profile_params["negative_film"])
        self.print_selector.setCurrentText(profile_params["print_film"])
        self.link_lights.setChecked(profile_params["link_lights"])
        self.loading = False
        self.format_selector.setCurrentText(profile_params["format"])
        self.active = True
        self.parameter_changed()

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

    def print_output(self, s):
        return

    def update_finished(self):
        self.running = False
        if self.waiting:
            self.waiting = False
            self.start_worker(self.update_preview)

    def progress_fn(self, n):
        return

    def parameter_changed(self, src=None):
        if self.active:
            self.start_worker(self.update_preview, src=src)

    def crop_zoom_changed(self):
        if self.active:
            self.start_worker(self.update_preview, value_changed=False)

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

    def update_preview(self, src=None, value_changed=True, *args, **kwargs):
        if src is None:
            if self.image_bar.current_image() is None:
                return
            else:
                src = self.image_bar.current_image()
        src_short = src.split("/")[-1]
        if src_short not in self.image_params:
            self.load_image_process(src)
            return
        else:
            self.load_image_params(src_short)
        image_args = self.setup_image_params(src_short)
        profile_args = self.setup_profile_params(image_args["profile"], src)
        processing_args = {**self.default_profile_params, **image_args, **profile_args}
        if "resolution" in processing_args:
            processing_args.pop("resolution")
        processing_args["negative_film"] = self.negative_stocks[processing_args["negative_film"]]
        if "print_film" in processing_args and processing_args["print_film"] is not None:
            processing_args["print_film"] = self.print_stocks[processing_args["print_film"]]
        if self.p3_preview.isChecked():
            processing_args["output_colourspace"] = "Display P3"
        if value_changed or self.full_preview.isChecked():
            image = self.xyz_image(src)
            self.preview_image = process_image(image, fast_mode=not self.full_preview.isChecked(),
                                               metadata=self.load_metadata(src), **processing_args)
        image = self.preview_image
        if not self.full_preview.isChecked():
            image = crop_rotate_zoom(image, **processing_args)
        height, width, _ = image.shape
        image = QImage(np.require(image, np.uint8, 'C'), width, height, 3 * width, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(image)
        self.image.setPixmap(self.pixmap)
        self.scale_pixmap()

    def setup_image_params(self, src):
        image_params = {**self.default_image_params, **self.image_params[src]}

        if image_params["wb_mode"] != "Custom":
            image_params["exposure_kelvin"] = self.wb_modes[image_params["wb_mode"]]

        return image_params

    def save_image(self, src, filename, **kwargs):
        src_short = src.split("/")[-1]
        if src_short in self.image_params:
            image_args = self.setup_image_params(src_short)
        else:
            image_args = self.default_image_params
        profile_args = self.setup_profile_params(image_args["profile"], src)
        processing_args = {**self.default_profile_params, **image_args, **profile_args}
        processing_args["negative_film"] = self.negative_stocks[processing_args["negative_film"]]
        if self.output_resolution.text() != "":
            processing_args["resolution"] = int(self.output_resolution.text())
        if "print_film" in processing_args and processing_args["print_film"] is not None:
            processing_args["print_film"] = self.print_stocks[processing_args["print_film"]]
        image = raw_to_linear(src, half_size=False)
        metadata = self.load_metadata(src)
        if processing_args["lens_correction"]:
            if "cam" not in processing_args or "lens" not in processing_args:
                cam, lens = utils.find_data(metadata, self.lensfunpy_db)
                if cam is not None:
                    processing_args["cam"] = cam.maker + " " + cam.model
                else:
                    processing_args["cam"] = None
                if lens is not None:
                    processing_args["lens"] = lens.model
                else:
                    processing_args["lens"] = None
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
        if folder:
            self.start_worker(self.save_all_process, folder=folder, filenames=self.filenames.values(), semaphore=False)

    def save_selected_images(self):
        folder = QFileDialog.getExistingDirectory(self)
        if folder:
            self.start_worker(self.save_all_process, folder=folder, filenames=self.image_bar.get_highlighted(),
                              semaphore=False)

    def save_settings(self):
        filename, ok = QFileDialog.getSaveFileName(self, "Select file name", "raw2film_settings.json", "*.json")
        if ok:
            complete_dict = {"image_params": self.image_params, "profile_params": self.profile_params}
            with open(filename, "w") as f:
                json.dump(complete_dict, f)

    def load_settings(self):
        filename, ok = QFileDialog.getOpenFileName(self)
        if ok:
            with open(filename, "r") as f:
                complete_dict = json.load(f)
            self.image_params = {**complete_dict["image_params"], **self.image_params}
            self.profile_params = {**complete_dict["profile_params"], **self.profile_params}
            for profile in self.profile_params:
                if self.profile_selector.findText(profile) == -1:
                    self.profile_selector.addItem(profile)

    def light_changed(self, value, light_name):
        if self.loading:
            return
        if self.link_lights.isChecked():
            self.loading = True
            self.red_light.setValue(value)
            self.green_light.setValue(value)
            self.blue_light.setValue(value)
            self.loading = False
            self.active = False
            self.profile_changed(value, "red_light")
            self.profile_changed(value, "green_light")
            self.profile_changed(value, "blue_light")
            self.active = True
            self.parameter_changed()
        else:
            self.profile_changed(value, light_name)

    def rotate_image(self, direction=1):
        self.rotate_times = (self.rotate_times - direction) % 4
        self.setting_changed(self.rotate_times, "rotate_times", crop_zoom=True)


def gui_main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == '__main__':
    gui_main()
