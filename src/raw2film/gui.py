import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cache, partial, lru_cache

import exiftool
import imageio
import lensfunpy
from PyQt6.QtCore import QSize, QThreadPool, QThread, QRegularExpression, QSettings, QTimer
from PyQt6.QtGui import QPixmap, QImage, QAction, QShortcut, QKeySequence, QRegularExpressionValidator, QIntValidator
from PyQt6.QtWidgets import QApplication, QMainWindow, QComboBox, QGridLayout, QSizePolicy, QCheckBox, QVBoxLayout, \
    QInputDialog, QMessageBox, QDialog, QProgressDialog, QScrollArea, QSplitter
from raw2film.color_processing import rec709_to_displayP3
from spectral_film_lut import REVERSAL_FILM
from spectral_film_lut.film_loader import load_ui
from spectral_film_lut.filmstock_selector import FilmStockSelector

from raw2film import data, utils
from raw2film.image_bar import ImageBar
from raw2film.raw_conversion import *
from raw2film.utils import add_metadata, generate_histogram


class MultiWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.cancel_event = threading.Event()

    def run_tasks(self, func, tasks, max_workers=None, **kwargs):
        semaphore = threading.Semaphore(1)
        kwargs['semaphore'] = semaphore
        kwargs['cancel_event'] = self.cancel_event  # Pass cancel flag to each task

        def func_wrapper(i, *args, **kwargs):
            if self.cancel_event.is_set():
                return "Cancelled"

            if i < max_workers:
                time.sleep(i)
            return func(*args, **kwargs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(func_wrapper, i, *task, **kwargs) for i, task in enumerate(tasks) if
                       not self.cancel_event.is_set()]

            for future in as_completed(futures):
                if self.cancel_event.is_set():
                    break
                self.progress.emit(future.result())

        self.finished.emit()

    def cancel(self):
        self.cancel_event.set()
        self.finished.emit()


class MainWindow(QMainWindow):
    ui_update = pyqtSignal(dict)

    def __init__(self, filmstocks):
        super().__init__()

        self.setWindowTitle("Raw2Film")

        self.filmstocks = filmstocks
        filmstock_info = {x: {'Year': filmstocks[x].year, 'Manufacturer': filmstocks[x].manufacturer, 'Type':
            {"camerapositive": "Slide", "cameranegative": "Negative", "printnegative": "Print",
             "printpositive": "SlidePrint"}[filmstocks[x].stage + filmstocks[x].type], 'Medium': filmstocks[x].medium,
                              'Sensitivity': f"ISO {filmstocks[x].iso}" if filmstocks[x].iso is not None else None,
                              'sensitivity': filmstocks[x].iso if filmstocks[x].iso is not None else None,
                              'resolution': f"{filmstocks[x].resolution} lines/mm" if filmstocks[
                                                                                          x].resolution is not None else None,
                              'Resolution': filmstocks[x].resolution if filmstocks[x].resolution is not None else None,
                              'Granularity': f"{filmstocks[x].rms} rms" if filmstocks[x].rms is not None else None,
                              'Decade': f"{filmstocks[x].year // 10 * 10}s" if filmstocks[x].year is not None else None,
                              'stage': filmstocks[x].stage,
                              'Chromaticity': 'BW' if filmstocks[x].density_measure == 'bw' else 'Color',
                              'image': QImage(np.require(filmstocks[x].color_checker, np.uint8, 'C'), 6, 4, 18,
                                              QImage.Format.Format_RGB888), 'Gamma': round(filmstocks[x].gamma, 3),
                              'Alias': filmstocks[x].alias, 'Comment': filmstocks[x].comment} for x in filmstocks}

        self.settings = QSettings("JanLohse", "Raw2Film")

        page_splitter = QSplitter(Qt.Orientation.Vertical)
        top_splitter = QSplitter()
        sidebar = QWidget()
        side_layout = QGridLayout()
        sidebar.setLayout(side_layout)
        side_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setWidget(sidebar)
        scroll_area.setMinimumWidth(280)

        self.image = QLabel("Select a reference image for the preview")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        self.image.setMinimumSize(QSize(256, 256))
        self.pixmap = QPixmap()

        self.image_bar = ImageBar()
        page_splitter.addWidget(top_splitter)
        page_splitter.addWidget(self.image_bar)
        page_splitter.setContentsMargins(8, 8, 8, 8)
        top_splitter.setContentsMargins(0, 0, 0, 0)

        page_splitter.setStretchFactor(0, 1)
        page_splitter.setStretchFactor(1, 0)

        top_splitter.addWidget(self.image)
        top_splitter.addWidget(scroll_area)
        top_splitter.setStretchFactor(0, 1)
        top_splitter.setStretchFactor(1, 0)

        self.side_counter = -1

        self.hideable_widgets = []

        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        view_menu = menu.addMenu("View")
        edit_menu = menu.addMenu("Edit")

        def add_option(widget, name=None, default=None, setter=None, hideable=False, tool_tip=None):
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
            if tool_tip is not None:
                label.setToolTip(tool_tip)
                widget.setToolTip(tool_tip)

        self.image_selector = QAction("Open images", self)
        self.image_selector.setShortcut(QKeySequence("Ctrl+O"))
        file_menu.addAction(self.image_selector)
        self.folder_selector = QAction("Open folder", self)
        self.folder_selector.setShortcut(QKeySequence("Ctrl+Shift+O"))
        file_menu.addAction(self.folder_selector)
        self.quick_save_button = QAction("Quick save settings")
        self.quick_save_button.setShortcut("Ctrl+S")
        file_menu.addAction(self.quick_save_button)
        self.save_image_button = QAction("Quick export jpg")
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
        self.close_highlighted_button = QAction("Close selected images")
        self.close_highlighted_button.setShortcut("Del")
        file_menu.addAction(self.close_highlighted_button)
        self.delete_highlighted_button = QAction("Delete selected images")
        self.delete_highlighted_button.setShortcut("Shift+Del")
        file_menu.addAction(self.delete_highlighted_button)
        self.deselect_all_button = QAction("Deselect all")
        self.deselect_all_button.setShortcut("Ctrl+D")
        edit_menu.addAction(self.deselect_all_button)
        self.reset_image_button = QAction("Reset image")
        edit_menu.addAction(self.reset_image_button)
        self.reset_all_images_button = QAction("Reset all images")
        edit_menu.addAction(self.reset_all_images_button)
        self.reset_profile_button = QAction("Reset profile")
        edit_menu.addAction(self.reset_profile_button)
        self.delete_profile_button = QAction("Delete profile")
        edit_menu.addAction(self.delete_profile_button)
        self.delete_all_profiles_button = QAction("Delete all profiles")
        edit_menu.addAction(self.delete_all_profiles_button)

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

        self.dflt_prf_params = {"negative_film": "KodakPortra400", "print_film": "FujiCrystalArchiveMaxima",
                                "red_light": 0, "green_light": 0, "blue_light": 0, "halation": True, "sharpness": True,
                                "grain": 2, "format": "135", "frame_width": 36, "frame_height": 24,
                                "grain_size": 0.0025, "halation_size": 1, "halation_green_factor": 0.4,
                                "projector_kelvin": 6500, "halation_intensity": 1, "black_offset": 0,
                                "pre_flash_neg": -4, "pre_flash_print": -4, "sat_adjust": 1}
        self.dflt_img_params = {"exp_comp": 0, "zoom": 1, "rotate_times": 0, "rotation": 0, "exposure_kelvin": 6000,
                                "profile": "Default", "lens_correction": True, "canvas_mode": "No", "canvas_scale": 1,
                                "canvas_ratio": 0.8, "highlight_burn": 0, "burn_scale": 50, "flip": False, "tint": 0}

        self.histogram = QLabel()
        self.histogram.setScaledContents(True)
        self.histogram.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.histogram.setMinimumSize(0, 0)
        side_layout.addWidget(self.histogram, 0, 0, 1, 2)
        self.side_counter += 1

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
        add_option(profile_widget, "Profile:", self.dflt_img_params["profile"], self.profile_selector.setCurrentText,
                   tool_tip="""Select the profile specifying film and format parameters.
(1-9: select profile)""")

        self.lensfunpy_db = lensfunpy.Database()
        self.cameras = {camera.maker + " " + camera.model: camera for camera in self.lensfunpy_db.cameras}
        self.lenses = {lens.model: lens for lens in self.lensfunpy_db.lenses}
        self.cameras["None"] = None
        self.lenses["None"] = None

        self.lens_correction = QCheckBox()
        add_option(self.lens_correction, "Lens correction:", self.dflt_img_params["lens_correction"],
                   self.lens_correction.setChecked, hideable=True, tool_tip="Correct lens distortion and vignetting.")
        self.camera_selector = QComboBox()
        self.camera_selector.setMinimumWidth(100)
        self.camera_selector.addItems(self.cameras.keys())
        add_option(self.camera_selector, "Camera model:", "None", self.camera_selector.setCurrentText, hideable=True,
                   tool_tip="Select camera model for lens correction.")
        self.lens_selector = QComboBox()
        self.lens_selector.setMinimumWidth(100)
        self.lens_selector.addItems(self.lenses.keys())
        add_option(self.lens_selector, "Lens model:", "None", self.lens_selector.setCurrentText, hideable=True,
                   tool_tip="Select lens model for lens correction.")

        self.halation = QCheckBox()
        add_option(self.halation, "Halation:", self.dflt_prf_params["halation"], self.halation.setChecked,
                   hideable=True,
                   tool_tip="Activate halation, a warm glow around highlights,\nresulting from reflections on the back of the film.")
        self.sharpness = QCheckBox()
        add_option(self.sharpness, "Sharpness:", self.dflt_prf_params["sharpness"], self.sharpness.setChecked,
                   hideable=True, tool_tip="Emulate the resolution and micro-contrast of film.")
        self.grain = QCheckBox()
        self.grain.setTristate(True)
        add_option(self.grain, "Grain:", self.dflt_prf_params["grain"], self.grain.setChecked, hideable=True, tool_tip="""
<table>
  <tr><td><u>Unchecked: </u></td><td>No film grain.</td></tr>
  <tr><td><u>Partially Checked: </u></td><td>Monochromatic film grain.</td></tr>
  <tr><td><u>Checked: </u></td><td>RGB film grain.</td></tr>
</table>
""")

        self.exp_comp = Slider()
        self.exp_comp.setMinMaxTicks(-3, 3, 1, 10)
        add_option(self.exp_comp, "Exposure:", self.dflt_img_params["exp_comp"], self.exp_comp.setValue, tool_tip="""Adjust exposure in stops.
(Up: increase exposure)
(Down: decrease exposure)""")

        self.wb_modes = {"Default": 6000, "Daylight": 5500, "Cloudy": 6500, "Shade": 7500, "Tungsten": 2800,
                         "Fluorescent": 3800, "Custom": None}
        self.wb_mode = QComboBox()
        self.wb_mode.addItems(list(self.wb_modes.keys()))
        add_option(self.wb_mode, "WB:", "Daylight", self.wb_mode.setCurrentText, tool_tip="""Select preset white balance.
(Shift+D: daylight)
(Shift+C: cloudy)
(Shift+S: shade)
(Shift+F: fluorescent)
(Shift+T: Tungsten)""")

        self.exp_wb = Slider()
        self.exp_wb.setMinMaxTicks(2000, 12000, 100)
        add_option(self.exp_wb, "Kelvin:", self.dflt_img_params["exposure_kelvin"], self.exp_wb.setValue,
                   tool_tip="Adjust white balance in kelvin.")

        self.tint = Slider()
        self.tint.setMinMaxTicks(-1, 1, 1, 100)
        add_option(self.tint, "Tint:", self.dflt_img_params["tint"], self.tint.setValue,
                   tool_tip="Change tint on green-magenta axis.")

        self.pre_flash_neg = Slider()
        self.pre_flash_neg.setMinMaxTicks(-4, -1, 1, 10)
        add_option(self.pre_flash_neg, "Pre-flash neg.:", self.dflt_prf_params["pre_flash_neg"],
                   self.pre_flash_neg.setValue, tool_tip="""Simulate the effect of exposing the negative with uniform light.
Helps to reduce contrast by lifting the shadows.
Set in how many stops below middle gray the uniform exposure is.
-4 will turn off the effect completely.""")
        self.pre_flash_print = Slider()
        self.pre_flash_print.setMinMaxTicks(-4, -1, 1, 10)
        add_option(self.pre_flash_print, "Pre-flash print:", self.dflt_prf_params["pre_flash_print"],
                   self.pre_flash_print.setValue, tool_tip="""Simulate the effect of exposing the print film with uniform light.
Helps to reduce contrast by lowering the highlights.
Set in how many stops below middle gray the uniform exposure is.
-4 will turn off the effect completely.
(Ctrl+Up: increase)
(Ctrl+Down: decrease)""")
        self.highlight_burn = Slider()
        self.highlight_burn.setMinMaxTicks(0, 1, 1, 20)
        add_option(self.highlight_burn, "Highlight burn:", self.dflt_img_params["highlight_burn"],
                   self.highlight_burn.setValue, tool_tip="""Lower the brightness of bright areas on the print film.
Only visible in preview if GPU acceleration is enabled
or full-preview is enabled.
(Shift+Up: increase)
(Shift+Down: decrease)""")
        self.burn_scale = Slider()
        self.burn_scale.setMinMaxTicks(1, 200)
        add_option(self.burn_scale, "Burn scale:", self.dflt_img_params["burn_scale"], self.burn_scale.setValue,
                   hideable=True, tool_tip="How much blur is applied to the highlight burn.")

        self.rotate = QWidget()
        rotate_layout = QHBoxLayout()
        self.rotate_left = QPushButton("Left")
        self.rotate_right = QPushButton("Right")
        self.flip_button = QPushButton("Flip")
        for btn in (self.rotate_left, self.rotate_right, self.flip_button):
            btn.setMinimumWidth(10)
            btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            rotate_layout.addWidget(btn, stretch=1)
        rotate_layout.setContentsMargins(0, 0, 0, 0)
        rotate_layout.setSpacing(0)
        self.rotate.setLayout(rotate_layout)
        add_option(self.rotate, "Rotate:", tool_tip="""Rotate the image by 90 degrees or change the orientation.
(Ctrl+R: rotate right)""")

        self.rotation = Slider()
        self.rotation.setMinMaxTicks(-90, 90, 1, 4)
        add_option(self.rotation, "Rotation angle:", self.dflt_img_params["rotation"], self.rotation.setValue, tool_tip="""Rotate by an angle in degrees.
(Ctrl+Right: rotate right)
(Ctrl+Left: rotate left)""")

        self.zoom = Slider()
        self.zoom.setMinMaxTicks(1, 2, 1, 100)
        add_option(self.zoom, "Zoom:", self.dflt_img_params["zoom"], self.zoom.setValue, tool_tip="""Crop into the image.
(Ctrl+Plus: zoom in)
(Ctrl+Minus: zoom out)""")

        self.format_selector = QComboBox()
        self.format_selector.addItems(list(data.FORMATS.keys()) + ["Custom"])
        add_option(self.format_selector, "Format:", self.dflt_prf_params["format"], self.format_selector.setCurrentText,
                   tool_tip="""Select a preset film format.
Adjusts scale of film characteristics (halation, resolution, grain)
and changes aspect ratio.""")

        self.frame_size = QWidget()
        frame_layout = QHBoxLayout()
        self.frame_width = QLineEdit()
        regex = QRegularExpression(r"[0-9]*|[0-9]+\.[0-9]*")
        self.frame_width.setValidator(QRegularExpressionValidator(regex))
        self.frame_height = QLineEdit()
        self.frame_height.setValidator(QRegularExpressionValidator(regex))
        frame_layout.addWidget(self.frame_width)
        frame_layout.addWidget(self.frame_height)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_size.setLayout(frame_layout)
        add_option(self.frame_size, "Width/Height:", self.dflt_prf_params["format"],
                   self.format_selector.setCurrentText, hideable=True, tool_tip="""Specify simulated film frame size.
Adjusts scale of film characteristics (halation, resolution, grain)
and changes aspect ratio.""")

        self.grain_size = Slider()
        self.grain_size.setMinMaxTicks(1, 5, 1, 10)
        add_option(self.grain_size, "Grain size (microns):", self.dflt_prf_params["grain_size"] * 1000,
                   self.grain_size.setValue, hideable=True, tool_tip="Size of simulated film grains.")

        self.halation_size = Slider()
        self.halation_size.setMinMaxTicks(0.5, 8, 1, 4)
        add_option(self.halation_size, "Halation size:", self.dflt_prf_params["halation_size"],
                   self.halation_size.setValue, hideable=True, tool_tip="""How far the halation spreads.
Halation is a warm glow around highlights,\nresulting from reflections on the film backing.""")
        self.halation_green = Slider()
        self.halation_green.setMinMaxTicks(0, 1, 1, 20)
        add_option(self.halation_green, "Halation color:", self.dflt_prf_params["halation_green_factor"],
                   self.halation_green.setValue, hideable=True, tool_tip="""How red or yellow the halation is.
Specifies how strongly the halation reaches into the green sensitive layer.""")
        self.halation_intensity = Slider()
        self.halation_intensity.setMinMaxTicks(0.5, 4, 1, 4)
        add_option(self.halation_intensity, "Halation intensity:", self.dflt_prf_params["halation_intensity"],
                   self.halation_intensity.setValue, hideable=True, tool_tip="""How intense the halation is.
Halation is a warm glow around highlights,
resulting from reflections on the film backing.""")

        negative_info = {x: y for x, y in filmstock_info.items() if y['stage'] == 'camera'}
        sort_keys_negative = ["Name", "Year", "Resolution", "Granularity", "sensitivity", "Gamma"]
        group_keys_negative = ["Manufacturer", "Type", "Decade", "Medium"]
        list_keys_negative = ["Manufacturer", "Type", "Year", "Sensitivity", "Chromaticity"]
        sidebar_keys_negative = ["Alias", "Manufacturer", "Type", "Year", "Sensitivity", "resolution", "Granularity",
                                 "Medium", "Chromaticity", "Gamma", "Comment"]
        self.filmstocks["None"] = None
        self.negative_selector = FilmStockSelector(negative_info, sort_keys=sort_keys_negative, image_key="image",
                                                   group_keys=group_keys_negative, list_keys=list_keys_negative,
                                                   sidebar_keys=sidebar_keys_negative, default_group="Manufacturer")
        self.negative_selector.setMinimumWidth(100)
        add_option(self.negative_selector, "Negativ stock:", self.dflt_prf_params["negative_film"],
                   self.negative_selector.setCurrentText, tool_tip="""Which negative film stock to emulate.
Affects colors, resolution, and graininess""")

        self.red_light = Slider()
        self.red_light.setMinMaxTicks(-0.75, 0.75, 1, 50)
        add_option(self.red_light, "Red printer light:", self.dflt_prf_params["red_light"], self.red_light.setValue,
                   hideable=True, tool_tip="""How strong the simulated red light is during printing.
Decreases how red the print is.""")
        self.green_light = Slider()
        self.green_light.setMinMaxTicks(-0.75, 0.75, 1, 50)
        add_option(self.green_light, "Green printer light:", self.dflt_prf_params["green_light"],
                   self.green_light.setValue, hideable=True, tool_tip="""How strong the simulated green light is during printing.
Decreases how green the print is.""")
        self.blue_light = Slider()
        self.blue_light.setMinMaxTicks(-0.75, 0.75, 1, 50)
        add_option(self.blue_light, "Blue printer light:", self.dflt_prf_params["blue_light"], self.blue_light.setValue,
                   hideable=True, tool_tip="""How strong the simulated blue light is during printing.
Decreases how blue the print is.""")

        self.link_lights = QCheckBox()
        self.link_lights.setChecked(True)
        self.link_lights.setText("link lights")
        add_option(self.link_lights, hideable=True,
                   tool_tip="Whether to adjust the printer lights individually or not.")

        print_info = {x: y for x, y in filmstock_info.items() if y['stage'] == 'print'}
        print_info["None"] = {}
        sort_keys_print = ["Name", "Year", "Gamma"]
        group_keys_print = ["Manufacturer", "Type", "Decade", "Medium"]
        list_keys_print = ["Manufacturer", "Type", "Year", "Chromaticity"]
        sidebar_keys_print = ["Alias", "Manufacturer", "Type", "Year", "Medium", "Chromaticity", "Gamma", "Comment"]
        self.print_selector = FilmStockSelector(print_info, sort_keys=sort_keys_print, group_keys=group_keys_print,
                                                list_keys=list_keys_print, sidebar_keys=sidebar_keys_print,
                                                default_group="Manufacturer", image_key="image")
        self.print_selector.setMinimumWidth(100)
        add_option(self.print_selector, "Print stock:", self.dflt_prf_params["print_film"],
                   self.print_selector.setCurrentText, tool_tip="""Which print material to emulate.
Affects only colors.""")

        self.projector_kelvin = Slider()
        self.projector_kelvin.setMinMaxTicks(2700, 10000, 100)
        add_option(self.projector_kelvin, "Projector wb:", self.dflt_prf_params["projector_kelvin"],
                   self.projector_kelvin.setValue, hideable=True,
                   tool_tip="Under what light temperature to view the print or slide.")

        self.saturation_slider = Slider()
        self.saturation_slider.setMinMaxTicks(0, 2, 1, 100)
        add_option(self.saturation_slider, "Saturation:", self.dflt_prf_params["sat_adjust"],
                   self.saturation_slider.setValue, hideable=True,
                   tool_tip="Adjust the saturation in the display color space.")

        self.canvas_mode = QComboBox()
        self.canvas_mode.addItems(
            ["No", "Proportional white", "Proportional black", "Uniform white", "Uniform black", "Fixed white",
             "Fixed black"])
        add_option(self.canvas_mode, "Canvas:", self.dflt_img_params["canvas_mode"], self.canvas_mode.setCurrentText,
                   hideable=True, tool_tip="What type of border to add to the image.")

        self.canvas_scale = Slider()
        self.canvas_scale.setMinMaxTicks(1, 2, 1, 40)
        add_option(self.canvas_scale, "Canvas scale:", self.dflt_img_params["canvas_scale"], self.canvas_scale.setValue,
                   hideable=True, tool_tip="How big the canvas is.")

        self.canvas_size = QWidget()
        canvas_layout = QHBoxLayout()
        self.canvas_width = QLineEdit()
        self.canvas_width.setValidator(QRegularExpressionValidator(regex))
        self.canvas_height = QLineEdit()
        self.canvas_height.setValidator(QRegularExpressionValidator(regex))
        canvas_layout.addWidget(self.canvas_width)
        canvas_layout.addWidget(self.canvas_height)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_size.setLayout(canvas_layout)
        add_option(self.canvas_size, "Width/Height:", 1,
                   lambda x: (self.canvas_width.setText("5"), self.canvas_height.setText("4")), hideable=True,
                   tool_tip="""Aspect ratio of added canvas.""")

        self.black_offset = Slider()
        self.black_offset.setMinMaxTicks(-2, 2, 1, 10)
        add_option(self.black_offset, "Black offset:", self.dflt_prf_params["black_offset"], self.black_offset.setValue,
                   hideable=True, tool_tip="Change the black value without affecting other areas.")

        QShortcut(QKeySequence('Up'), self).activated.connect(self.exp_comp.increase)
        QShortcut(QKeySequence('Down'), self).activated.connect(self.exp_comp.decrease)
        QShortcut(QKeySequence("Ctrl+Right"), self).activated.connect(self.rotation.increase)
        QShortcut(QKeySequence("Ctrl+Left"), self).activated.connect(self.rotation.decrease)
        QShortcut(QKeySequence("Shift+Up"), self).activated.connect(self.highlight_burn.increase)
        QShortcut(QKeySequence("Shift+Down"), self).activated.connect(self.highlight_burn.decrease)
        QShortcut(QKeySequence("Ctrl+Up"), self).activated.connect(self.pre_flash_print.increase)
        QShortcut(QKeySequence("Ctrl+Down"), self).activated.connect(self.pre_flash_print.decrease)
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
        self.tint.valueChanged.connect(lambda x: self.setting_changed(x, "tint"))
        self.red_light.valueChanged.connect(lambda x: self.light_changed(x, "red_light"))
        self.green_light.valueChanged.connect(lambda x: self.light_changed(x, "green_light"))
        self.blue_light.valueChanged.connect(lambda x: self.light_changed(x, "blue_light"))
        self.lens_correction.stateChanged.connect(lambda x: self.setting_changed(x, "lens_correction"))
        self.full_preview.triggered.connect(lambda: self.parameter_changed())
        self.halation.stateChanged.connect(lambda x: self.profile_changed(x, "halation"))
        self.sharpness.stateChanged.connect(lambda x: self.profile_changed(x, "sharpness"))
        self.grain.stateChanged.connect(lambda x: self.profile_changed(x, "grain"))
        self.rotation.valueChanged.connect(lambda x: self.setting_changed(x, "rotation"))
        self.zoom.valueChanged.connect(lambda x: self.setting_changed(x, "zoom"))
        self.format_selector.currentTextChanged.connect(self.format_changed)
        self.advanced_controls.triggered.connect(self.hide_controls)
        self.grain_size.valueChanged.connect(lambda x: self.profile_changed(x / 1000, "grain_size"))
        self.halation_size.valueChanged.connect(lambda x: self.profile_changed(x, "halation_size"))
        self.halation_intensity.valueChanged.connect(lambda x: self.profile_changed(x, "halation_intensity"))
        self.halation_green.valueChanged.connect(lambda x: self.profile_changed(x, "halation_green_factor"))
        self.rotate_right.released.connect(self.rotate_image)
        self.rotate_left.released.connect(lambda: self.rotate_image(-1))
        self.flip_button.released.connect(self.flip_image)
        self.lens_selector.currentTextChanged.connect(lambda x: self.setting_changed(x, "lens"))
        self.camera_selector.currentTextChanged.connect(lambda x: self.setting_changed(x, "cam"))
        self.frame_width.textChanged.connect(lambda x: self.profile_changed(x, "frame_width"))
        self.frame_height.textChanged.connect(lambda x: self.profile_changed(x, "frame_height"))
        self.profile_selector.currentTextChanged.connect(self.load_profile_params)
        self.p3_preview.triggered.connect(lambda: self.parameter_changed())
        self.save_settings_button.triggered.connect(self.save_settings_dialogue)
        self.load_settings_button.triggered.connect(self.load_settings_dialogue)
        self.image_bar.image_changed.connect(self.load_image)
        self.add_profile.released.connect(self.add_profile_prompt)
        self.delete_profile_button.triggered.connect(self.delete_profile)
        self.delete_all_profiles_button.triggered.connect(self.delete_all_profiles)
        self.pre_flash_neg.valueChanged.connect(lambda x: self.profile_changed(x, "pre_flash_neg"))
        self.pre_flash_print.valueChanged.connect(lambda x: self.profile_changed(x, "pre_flash_print"))
        self.highlight_burn.valueChanged.connect(lambda x: self.setting_changed(x, "highlight_burn"))
        self.burn_scale.valueChanged.connect(lambda x: self.setting_changed(x, "burn_scale"))
        self.saturation_slider.valueChanged.connect(lambda x: self.profile_changed(x, "sat_adjust"))
        self.quick_save_button.triggered.connect(self.quick_save)
        self.close_highlighted_button.triggered.connect(self.image_bar.close_highlighted)
        self.delete_highlighted_button.triggered.connect(self.delete_highlighted)
        self.image_bar.copy_settings.connect(self.copy_settings)
        self.deselect_all_button.triggered.connect(self.image_bar.deselect_all)
        self.reset_image_button.triggered.connect(self.reset_image)
        self.reset_all_images_button.triggered.connect(self.reset_all_images)
        self.reset_profile_button.triggered.connect(self.reset_profile)
        self.canvas_mode.currentTextChanged.connect(lambda x: self.setting_changed(x, "canvas_mode"))
        self.canvas_scale.valueChanged.connect(lambda x: self.setting_changed(x, "canvas_scale"))
        self.canvas_width.textChanged.connect(lambda: self.setting_changed(float(self.canvas_height.text()) / float(
            self.canvas_width.text()) if self.canvas_width.text() and self.canvas_height.text() else 0.8,
                                                                           "canvas_ratio"))
        self.canvas_height.textChanged.connect(lambda: self.setting_changed(float(self.canvas_height.text()) / float(
            self.canvas_width.text()) if self.canvas_width.text() and self.canvas_height.text() else 0.8,
                                                                            "canvas_ratio"))
        self.black_offset.valueChanged.connect(lambda x: self.profile_changed(x, "black_offset"))
        self.ui_update.connect(self.load_image_params_to_ui)

        self.setCentralWidget(page_splitter)

        self.resize(QSize(1080, 720))

        self.waiting = False
        self.running = False

        self.threadpool = QThreadPool()

        self.corrected_image = None
        self.preview_image = None
        self.rotate_times = 0
        self.flip_image = False
        self.active = True  # prevent from running update_preview by setting inactive
        self.loading = False  # prevent from storing settings while setting widget values

        self.hide_controls()

        self.image_params = {}
        self.profile_params = {}

        self.load_settings_system()

        self.save_timer = time.time()

        self.load_profile_params()

        QTimer.singleShot(0, self.test_exiftool)

    def test_exiftool(self):
        try:
            exiftool.ExifToolHelper()
        except FileNotFoundError:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Error")
            msg.setText("ExifTool not found.\nPlease install ExifTool first.")
            msg.setStandardButtons(QMessageBox.StandardButton.Close)

            # When Close is clicked, quit the app
            if msg.exec() == QMessageBox.StandardButton.Close:
                QApplication.quit()

    def reset_image(self):
        for src in self.image_bar.get_highlighted():
            src_short = src.split("/")[-1]
            if src_short in self.image_params:
                self.image_params.pop(src_short)
        self.load_image(self.image_bar.selected_label.image_path)

    def reset_all_images(self):
        self.image_params = {}
        if self.image_bar.selected_label is not None:
            self.load_image(self.image_bar.selected_label.image_path)

    def reset_profile(self):
        profile = self.profile_selector.currentText()
        if profile in self.profile_params:
            self.profile_params[profile] = {}
        self.load_profile_params()

    def copy_settings(self, src):
        src_short = src.split("/")[-1]
        if src_short in self.image_params:
            for image in self.image_bar.get_highlighted():
                image_short = image.split("/")[-1]
                lens, cam = None, None
                if image_short in self.image_params and "lens" in self.image_params[image_short]:
                    lens = self.image_params[image_short]["lens"]
                if image_short in self.image_params and "cam" in self.image_params[image_short]:
                    cam = self.image_params[image_short]["cam"]
                self.image_params[image_short] = self.image_params[src_short].copy()
                if lens is not None:
                    self.image_params[image_short]["lens"] = lens
                if cam is not None:
                    self.image_params[image_short]["cam"] = cam
        else:
            for image in self.image_bar.get_highlighted():
                if image in self.image_params:
                    self.image_params.pop(image.split("/")[-1])
        self.load_image(self.image_bar.selected_label.image_path)

    def delete_highlighted(self):
        reply = QMessageBox()
        number_images = len(self.image_bar.get_highlighted())
        if number_images == 1:
            reply.setText(f"Delete 1 image permanently?")
        elif number_images > 1:
            reply.setText(f"Delete {number_images} images permanently?")
        else:
            return
        reply.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        x = reply.exec()
        if x == QMessageBox.StandardButton.Yes:
            for image in self.image_bar.get_highlighted():
                os.remove(image)
            self.image_bar.close_highlighted()

    def delete_profile(self):
        current_profile = self.profile_selector.currentText()
        reply = QMessageBox()
        reply.setText(f"Delete profile {current_profile}?")
        reply.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
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
                    if 'profile' in self.image_params[image] and self.image_params[image]['profile'] == current_profile:
                        self.image_params[image]['profile'] = "Default"
            self.load_profile_params()

    def delete_all_profiles(self):
        reply = QMessageBox()
        profile_count = len(self.profile_params)
        if not profile_count:
            return
        if profile_count == 1:
            message = f"Delete 1 profile permanently?"
        elif profile_count > 1:
            message = f"Delete {profile_count} profiles permanently?"
        reply.setText(message)
        reply.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        x = reply.exec()
        if x == QMessageBox.StandardButton.Yes:
            profiles = tuple(self.profile_params.keys())
            for profile in profiles:
                if profile == "Default":
                    self.profile_params[profile] = {}
                else:
                    self.profile_selector.setCurrentText("Default")
                    self.profile_selector.removeItem(self.profile_selector.findText(profile))
                    if profile in self.profile_params:
                        self.profile_params.pop(profile)
                    for image in self.image_params:
                        if 'profile' in self.image_params[image] and self.image_params[image]['profile'] == profile:
                            self.image_params[image]['profile'] = "Default"
            self.load_profile_params()

    def add_profile_prompt(self):
        text, ok = QInputDialog.getText(self, "Add profile", "Profile name:")
        if ok:
            self.profile_selector.addItem(text)

    def format_changed(self, format):
        if format != "Custom" and not self.loading:
            width, height = data.FORMATS[format]
            self.frame_width.setText(str(width))
            self.frame_height.setText(str(height))

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
                                                     filter=f"RAW (*{' *'.join(data.EXTENSION_LIST)})")

        if ok:
            for folder in set(["/".join(filename.split("/")[:-1]) for filename in filenames]):
                self.load_settings_directory(folder)
            self.image_bar.load_images(filenames)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select image folder', '')
        if folder:
            self.load_settings_directory(folder)
            filenames = [folder + "/" + filename for filename in os.listdir(folder) if
                         filename.lower().endswith(data.EXTENSION_LIST)]
            self.image_bar.load_images(filenames)

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
            else:
                self.image_params[src_short]["cam"] = "None"
            if lens is not None:
                self.image_params[src_short]["lens"] = lens.model
            else:
                self.image_params[src_short]["lens"] = "None"
        if "profile" not in self.image_params[src_short]:
            self.image_params[src_short]["profile"] = self.profile_selector.currentText()
        if "exposure_kelvin" not in self.image_params[src_short]:
            self.image_params[src_short]["exposure_kelvin"] = self.exp_wb.getValue()
        self.load_image_params(src_short)
        if self.active:
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
        self.active = True
        self.profile_changed(negative, "negative_film")

    def profile_changed(self, value, key):
        if self.loading:
            return
        if key in ["frame_width", "frame_height"]:
            if not value:
                return
            else:
                value = float(value)
            width, height = self.frame_width.text(), self.frame_height.text()
            if width and height:
                dimensions = (float(width), float(height))
                if dimensions in data.FORMATS.values():
                    format_name = list(data.FORMATS.keys())[list(data.FORMATS.values()).index(dimensions)]
                    self.format_selector.setCurrentText(format_name)
                else:
                    self.format_selector.setCurrentText("Custom")
        profile = self.profile_selector.currentText()
        if profile not in self.profile_params:
            self.profile_params[profile] = {}
        self.profile_params[profile][key] = value
        self.parameter_changed()

    def quick_save(self):
        self.save_settings_system()
        self.save_timer = time.time()

    def setting_changed(self, value, key):
        if self.loading:
            return
        current_image = self.image_bar.current_image()
        if current_image is not None:
            current_image = current_image.split("/")[-1]
            if current_image in self.image_params:
                if key in self.image_params[current_image] and value == self.image_params[current_image][key]:
                    return  # no change
            elif (key in self.dflt_img_params and self.dflt_img_params[key] == value) or (
                    key in self.dflt_prf_params and self.dflt_prf_params[key] == value):
                return  # default value and nothing to overwrite
        if time.time() - self.save_timer > 10:
            self.quick_save()
        for src in self.image_bar.get_highlighted():
            src_short = src.split("/")[-1]
            if src_short not in self.image_params:
                self.image_params[src_short] = {}
                if "profile" not in self.image_params[src_short]:
                    self.image_params[src_short]["profile"] = self.profile_selector.currentText()
                if "exposure_kelvin" not in self.image_params[src_short]:
                    self.image_params[src_short]["exposure_kelvin"] = self.exp_wb.getValue()
            self.image_params[src_short][key] = value
        if key == "exposure_kelvin":
            self.update_wb_mode(value)
        self.parameter_changed()

    def update_wb_mode(self, value):
        if value in self.wb_modes.values():
            self.wb_mode.setCurrentText(list(self.wb_modes.keys())[list(self.wb_modes.values()).index(value)])
        else:
            self.wb_mode.setCurrentText("Custom")

    def setup_profile_params(self, profile, src=None):
        if profile in self.profile_params:
            return {**self.dflt_prf_params, **self.profile_params[profile]}
        else:
            return self.dflt_prf_params

    def load_image_params(self, src):
        image_params = self.setup_image_params(src)
        self.ui_update.emit(image_params)

    def load_image_params_to_ui(self, image_params):
        def set_safely(widget, method_name, key):
            if key in image_params:
                widget.blockSignals(True)
                getattr(widget, method_name)(image_params[key])
                widget.blockSignals(False)

        # Use the helper to set values safely
        set_safely(self.exp_comp, "setValue", "exp_comp")
        set_safely(self.zoom, "setValue", "zoom")

        if "rotate_times" in image_params:
            self.rotate_times = image_params["rotate_times"]

        if "flip" in image_params:
            self.flip = image_params["flip"]

        set_safely(self.rotation, "setValue", "rotation")
        set_safely(self.exp_wb, "setValue", "exposure_kelvin")
        set_safely(self.tint, "setValue", "tint")

        set_safely(self, "update_wb_mode", "exposure_kelvin")

        set_safely(self.lens_correction, "setChecked", "lens_correction")
        set_safely(self.canvas_mode, "setCurrentText", "canvas_mode")
        set_safely(self.canvas_scale, "setValue", "canvas_scale")

        set_safely(self.camera_selector, "setCurrentText", "cam")
        set_safely(self.lens_selector, "setCurrentText", "lens")

        set_safely(self.highlight_burn, "setValue", "highlight_burn")
        set_safely(self.burn_scale, "setValue", "burn_scale")

        if "profile" in image_params:
            self.profile_selector.setCurrentText(image_params["profile"])

    def load_profile_params(self, profile=None):
        if profile is None:
            profile = self.profile_selector.currentText()
        self.active = False
        self.setting_changed(profile, "profile")
        self.loading = True
        profile_params = self.setup_profile_params(profile)
        self.red_light.setValue(profile_params["red_light"])
        self.green_light.setValue(profile_params["green_light"])
        self.blue_light.setValue(profile_params["blue_light"])
        self.halation.setChecked(profile_params["halation"])
        self.halation_size.setValue(profile_params["halation_size"])
        self.halation_green.setValue(profile_params["halation_green_factor"])
        self.halation_intensity.setValue(profile_params["halation_intensity"])
        self.sharpness.setChecked(profile_params["sharpness"])
        self.pre_flash_neg.setValue(profile_params["pre_flash_neg"])
        self.pre_flash_print.setValue(profile_params["pre_flash_print"])
        self.grain.setCheckState(Qt.CheckState(profile_params["grain"]))
        if "frame_width" in profile_params:
            self.frame_width.setText(str(profile_params["frame_width"]))
        if "frame_height" in profile_params:
            self.frame_height.setText(str(profile_params["frame_height"]))
        if "frame_width" in profile_params and "frame_height" in profile_params:
            dimensions = (profile_params["frame_width"], profile_params["frame_height"])
            if dimensions in data.FORMATS.values():
                format_name = list(data.FORMATS.keys())[list(data.FORMATS.values()).index(dimensions)]
                self.format_selector.setCurrentText(format_name)
        self.grain_size.setValue(profile_params["grain_size"] * 1000)
        self.negative_selector.setCurrentText(profile_params["negative_film"])
        self.print_selector.setCurrentText(profile_params["print_film"])
        self.black_offset.setValue(profile_params["black_offset"])
        self.saturation_slider.setValue(profile_params["sat_adjust"])

        if "projector_kelvin" in profile_params:
            self.projector_kelvin.setValue(profile_params["projector_kelvin"])
        self.loading = False
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
        processing_args = {**self.dflt_prf_params, **image_args, **profile_args}
        processing_args["negative_film"] = self.filmstocks[processing_args["negative_film"]]
        if "print_film" in processing_args and processing_args["print_film"] is not None:
            processing_args["print_film"] = self.filmstocks[processing_args["print_film"]]
        if self.p3_preview.isChecked():
            processing_args["output_colourspace"] = "Display P3"
        if value_changed or self.full_preview.isChecked():
            image = self.xyz_image(src)
            processing_args["resolution"] = max(self.image.height(), self.image.width())
            if not self.full_preview.isChecked():
                processing_args["resolution"] = min(processing_args["resolution"], 1080)
                if processing_args["highlight_burn"]:
                    processing_args["sharpness"] = False
                    processing_args["grain"] = False
                    processing_args["halation"] = False
                else:
                    processing_args["fast_mode"] = True

            self.preview_image = process_image(image, metadata=self.load_metadata(src), **processing_args)
        image = self.preview_image
        height, width, _ = image.shape
        histogram = generate_histogram(image, 80)
        if self.p3_preview.isChecked():
            histogram = rec709_to_displayP3(histogram)
        histogram = QPixmap.fromImage(QImage(histogram, histogram.shape[1], histogram.shape[0], 3 * histogram.shape[1],
                                             QImage.Format.Format_RGB888))
        self.histogram.setPixmap(histogram)
        image = QImage(image, width, height, 3 * width, QImage.Format.Format_RGB888)
        self.pixmap = QPixmap.fromImage(image)
        self.image.setPixmap(self.pixmap)
        self.image.setToolTip(src)
        self.scale_pixmap()

    def setup_image_params(self, src):
        image_params = {**self.dflt_img_params, **self.image_params[src]}

        return image_params

    def save_image(self, src, filename, add_year=False, add_date=False, move_raw=0, quality=100, close=False,
                   resolution=None, semaphore=None, **kwargs):
        src_short = src.split("/")[-1]
        if src_short in self.image_params:
            image_args = self.setup_image_params(src_short)
        else:
            image_args = self.dflt_img_params
        profile_args = self.setup_profile_params(image_args["profile"], src)
        processing_args = {**self.dflt_prf_params, **image_args, **profile_args}
        processing_args["negative_film"] = self.filmstocks[processing_args["negative_film"]]
        if semaphore is not None:
            processing_args["semaphore"] = semaphore
        if resolution is not None:
            processing_args["resolution"] = resolution
        if "print_film" in processing_args and processing_args["print_film"] is not None:
            processing_args["print_film"] = self.filmstocks[processing_args["print_film"]]
        image = raw_to_linear(src, half_size=False)
        metadata = self.load_metadata(src)
        if processing_args["lens_correction"]:
            if "cam" not in processing_args or "lens" not in processing_args:
                cam, lens = utils.find_data(metadata, self.lensfunpy_db)
                if cam is not None or lens is not None:
                    processing_args["cam"] = cam.maker + " " + cam.model
                    processing_args["lens"] = lens.model
            if "cam" in processing_args and "lens" in processing_args:
                image = effects.lens_correction(image, metadata, self.cameras[processing_args["cam"]],
                                                self.lenses[processing_args["lens"]])
        image = process_image(image, fast_mode=False, **processing_args)
        path = "/".join(filename.split("/")[:-1])
        if path:
            path += "/"
        if add_year:
            path += metadata['EXIF:DateTimeOriginal'][:4] + "/"
        if add_date:
            path += metadata['EXIF:DateTimeOriginal'][:10].replace(':', '-') + "/"
        filename = filename.split("/")[-1]
        if '.' not in filename:
            filename += '.jpg'
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except FileExistsError:
            pass
        if move_raw:
            if not os.path.exists(path + 'RAW'):
                os.makedirs(path + 'RAW')
            # self.save_settings_directory(path + 'RAW', src=src_short)
            if move_raw == 2:
                os.replace(src, path + 'RAW/' + src.split('/')[-1])
            elif move_raw == 1 and not os.path.isfile(path + 'RAW/' + src.split('/')[-1]):
                shutil.copy2(src, path + 'RAW/' + src.split('/')[-1])
        imageio.imwrite(path + filename, image, quality=quality, format='.jpg')
        add_metadata(path + filename, metadata, exp_comp=processing_args['exp_comp'])
        if close:
            self.image_bar.close_single_image(src)
        return f"exported {filename}"

    def save_image_dialog(self):
        src = self.image_bar.current_image()
        if src is None:
            return
        filename = src.split("/")[-1].split(".")[0]
        filename, ok = QFileDialog.getSaveFileName(self, "Choose output file", filename, "*.jpg")
        if ok:
            self.start_worker(self.save_image, src=src, filename=filename, semaphore=False)

    def save_multiple_process(self, folder, filenames, **kwargs):
        self.active = False
        if cuda_available:
            xp.get_default_memory_pool().free_all_blocks()
            xp.get_default_pinned_memory_pool().free_all_blocks()
        self.task_count = len(filenames)
        self.progress_dialog = QProgressDialog("Starting export...", "Cancel", 0, self.task_count, self)
        self.progress_dialog.setWindowTitle("Export")
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        QApplication.processEvents()

        self.thread = QThread()
        self.worker = MultiWorker()
        self.progress_dialog.canceled.connect(self.worker.cancel)
        self.worker.moveToThread(self.thread)

        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.tasks_finished)
        tasks = [(filename, folder + "/" + filename.split("/")[-1].split(".")[0]) for filename in filenames]
        self.thread.started.connect(partial(self.worker.run_tasks, self.save_image, tasks, **kwargs))
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        self.tasks_done = 0

    def update_progress(self, value):
        self.tasks_done += 1
        self.progress_dialog.setValue(self.tasks_done)
        self.progress_dialog.setLabelText(f"{value} ({self.tasks_done}/{self.task_count})")

    def tasks_finished(self):
        self.active = True
        self.progress_dialog.close()
        if cuda_available:
            xp.get_default_memory_pool().free_all_blocks()
            xp.get_default_pinned_memory_pool().free_all_blocks()
        if self.image_bar.current_image() is not None:
            self.load_image(self.image_bar.current_image())

    def save_image_setting_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Export settings")

        layout = QVBoxLayout()

        quality_slider = Slider()
        quality_slider.setMinMaxTicks(0, 100)
        quality_slider.setValue(100)
        layout.addWidget(QLabel("JPEG quality:"))
        layout.addWidget(quality_slider)

        sort_by_year = QCheckBox("Sort by year")
        sort_by_year.setChecked(True)
        layout.addWidget(sort_by_year)

        sort_by_date = QCheckBox("Sort by date")
        sort_by_date.setChecked(True)
        layout.addWidget(sort_by_date)

        move_raw = QCheckBox("Move raw file to subfolder")
        move_raw.setTristate(True)
        move_raw.setToolTip("Checked: move file \nPartially checked: copy file\nUnchecked: do nothing to raw file")
        layout.addWidget(move_raw)

        close_checkbox = QCheckBox("Close images after export")
        move_raw.stateChanged.connect(lambda x: close_checkbox.setEnabled(x != 2))
        move_raw.setChecked(True)
        close_checkbox.setChecked(True)
        layout.addWidget(close_checkbox)

        resolution_field = QLineEdit()
        resolution_field.setValidator(QIntValidator())
        layout.addWidget(QLabel("Resolution:"))
        layout.addWidget(resolution_field)

        thread_slider = Slider()
        cpu_count = os.cpu_count()
        thread_slider.setMinMaxTicks(1, os.cpu_count())
        thread_slider.setValue(max(cpu_count // 2, 1))
        layout.addWidget(QLabel("Threads:"))
        layout.addWidget(thread_slider)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)
        dialog.setLayout(layout)

        # Connect buttons
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        if dialog.exec():
            if resolution_field.text():
                resolution = int(resolution_field.text())
            else:
                resolution = None
            kwargs = {"move_raw": move_raw.checkState().value, "add_year": sort_by_year.isChecked(),
                      "close": close_checkbox.isChecked() or move_raw.checkState().value == 2,
                      "quality": int(quality_slider.getValue()), "add_date": sort_by_date.isChecked(),
                      "resolution": resolution, "max_workers": int(thread_slider.getValue())}
            return True, kwargs
        else:
            return False, {}

    def save_all_images(self):
        if not self.image_bar.get_all():
            return
        ok, kwargs = self.save_image_setting_dialog()

        if ok:
            folder = QFileDialog.getExistingDirectory(self)
            if folder:
                self.save_multiple_process(folder=folder, filenames=self.image_bar.get_all(), **kwargs)

    def save_selected_images(self):
        if not self.image_bar.get_highlighted():
            return
        ok, kwargs = self.save_image_setting_dialog()

        if ok:
            folder = QFileDialog.getExistingDirectory(self)
            if folder:
                self.save_multiple_process(folder=folder, filenames=self.image_bar.get_highlighted(), **kwargs)

    def save_settings_dialogue(self):
        filename, ok = QFileDialog.getSaveFileName(self, "Select file name", "raw2film_settings.json", "*.json")
        if ok:
            self.save_settings(filename)

    def save_settings_directory(self, root='', src=None, **kwargs):
        if root:
            filename = root + "/raw2film_settings.json"
        else:
            filename = "raw2film_settings.json"
        self.save_settings(filename, src)

    def save_settings(self, filename, src=None):
        if src is None:
            complete_dict = {"image_params": self.image_params, "profile_params": self.profile_params}
        else:
            if "profile" in self.image_params[src] and self.image_params[src]["profile"] in self.profile_params:
                profile = self.image_params[src]["profile"]
                complete_dict = {"image_params": {src: self.image_params[src]},
                                 "profile_params": {profile: self.profile_params[profile]}}
            else:
                complete_dict = {"image_params": {src: self.image_params[src]}, "profile_params": {}}
        if Path(filename).is_file():
            with open(filename, "r") as f:
                old_dict = json.load(f)
            complete_dict["image_params"] = {**old_dict["image_params"], **complete_dict["image_params"]}
            complete_dict["profile_params"] = {**old_dict["profile_params"], **complete_dict["profile_params"]}
        with open(filename, "w") as f:
            json.dump(complete_dict, f)

    def save_settings_system(self):
        self.settings.setValue("profile_params", json.dumps(self.profile_params))
        self.settings.setValue("image_params", json.dumps(self.image_params))

    def load_settings_system(self):
        self.profile_params = json.loads(self.settings.value("profile_params", "{}"))
        self.image_params = json.loads(self.settings.value("image_params", "{}"))

        for profile in self.profile_params:
            if self.profile_selector.findText(profile) == -1:
                self.profile_selector.addItem(profile)

    def load_settings_dialogue(self):
        filename, ok = QFileDialog.getOpenFileName(self)
        if ok:
            self.load_settings(filename)

    def load_settings_directory(self, root=''):
        if root:
            filename = root + "/raw2film_settings.json"
        else:
            filename = "raw2film_settings.json"
        if Path(filename).is_file():
            self.load_settings(filename)

    def load_settings(self, filename):
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
        self.setting_changed(self.rotate_times, "rotate_times")

    def flip_image(self):
        self.flip = not self.flip
        self.setting_changed(self.flip, "flip")


def gui_main():
    load_ui(MainWindow, "Raw2Film")


if __name__ == '__main__':
    gui_main()
