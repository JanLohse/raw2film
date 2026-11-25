"""The main GUI implementation."""

import time

import exiftool
import numpy as np
import wgpu
from PIL import ImageCms
from PyQt6.QtCore import (
    QEvent,
    QSettings,
    QSize,
    QThreadPool,
    QTimer,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QIcon,
    QImage,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QWidget,
)
from rendercanvas.qt import QRenderWidget
from spectral_film_lut.ui.theme import BORDER_RADIUS, THEME
from spectral_film_lut.ui.widgets.controls import (
    AboutDialog,
)

from raw2film import R2F_BASE_DIR, __version__
from raw2film.cpu_processor import CpuProcessor
from raw2film.gpu_processor import GpuProcessor
from raw2film.gui_objects import (
    AutoShortcutsDialog,
)
from raw2film.ui.export import ExportMixin
from raw2film.ui.image_management import ImageManagementMixin
from raw2film.ui.layout import WindowLayoutMixin
from raw2film.ui.panel_setup import PanelSetupMixin
from raw2film.ui.preview import PreviewMixin
from raw2film.ui.settings import SettingsMixin
from raw2film.ui.signals import SignalSetupMixin


class MainWindow(
    ImageManagementMixin,
    PreviewMixin,
    ExportMixin,
    SettingsMixin,
    PanelSetupMixin,
    WindowLayoutMixin,
    SignalSetupMixin,
    QMainWindow,
):
    """The main window of raw2film."""

    ui_update = pyqtSignal(dict)

    info_update = pyqtSignal(dict, str, str)

    """The UI has updated."""

    def __init__(self, filmstocks):  # noqa: D107
        super().__init__()

        self.flip = False
        self.setWindowTitle(f"Raw2Film {__version__}")
        self.setStyleSheet(THEME)

        icon = QIcon()
        for size in [256, 128, 64, 48, 32, 16]:
            path = f"{R2F_BASE_DIR}/resources/raw2film_{size}.png"
            icon.addFile(path, QSize(size, size))

        self.setWindowIcon(icon)

        self.filmstocks = filmstocks
        self._raw_image_cache = {}
        filmstock_info = {
            x: {
                "Year": filmstocks[x].year,
                "Manufacturer": filmstocks[x].manufacturer,
                "Type": {
                    "camerapositive": "Slide",
                    "cameranegative": "Negative",
                    "printnegative": "Print",
                    "printpositive": "SlidePrint",
                }[filmstocks[x].stage + filmstocks[x].film_type],
                "Medium": filmstocks[x].medium,
                "Sensitivity": f"ISO {filmstocks[x].iso}"
                if filmstocks[x].iso is not None
                else None,
                "sensitivity": filmstocks[x].iso if filmstocks[x].iso is not None else None,
                "resolution": f"{filmstocks[x].resolution} lines/mm"
                if filmstocks[x].resolution is not None
                else None,
                "Resolution": filmstocks[x].resolution
                if filmstocks[x].resolution is not None
                else None,
                "Granularity": f"{filmstocks[x].rms} rms"
                if filmstocks[x].rms is not None
                else None,
                "Decade": f"{filmstocks[x].year // 10 * 10}s"
                if filmstocks[x].year is not None
                else None,
                "stage": filmstocks[x].stage,
                "Chromaticity": "BW" if filmstocks[x].density_measure == "bw" else "Color",
                "image": QImage(
                    np.require(filmstocks[x].color_checker, np.uint8, "C"),
                    6,
                    4,
                    18,
                    QImage.Format.Format_RGB888,
                ),
                "Gamma": round(filmstocks[x].gamma, 3),
                "Alias": filmstocks[x].alias,
                "Comment": filmstocks[x].comment,
            }
            for x in filmstocks
        }
        self.reversal_stocks = {
            name
            for name, stock in filmstocks.items()
            if stock.stage == "camera" and stock.film_type == "positive"
        }

        self.settings = QSettings("JanLohse", "Raw2Film")

        (
            page_splitter,
            scroll_area,
            basic_settings_group,
            profile_settings_group,
            film_effects_group,
            image_correction_group,
            advanced_printing_group,
            canvas_group,
        ) = self._create_window_shell(filmstock_info)
        self._create_menus()
        self.dflt_prf_params = {
            "negative_film": "Kodak Portra 400",
            "print_film": "Fuji Crystal Archive Maxima",
            "red_light": 0,
            "green_light": 0,
            "blue_light": 0,
            "halation": True,
            "sharpness": True,
            "grain": 1,
            "film_format": "135",
            "frame_width": 36,
            "frame_height": 24,
            "grain_size": 6,
            "grain_intensity": 1,
            "halation_size": 1.0,
            "halation_green_factor": 0.3,
            "projector_kelvin": 6500,
            "inversion_gamma": 3.0,
            "idealized_curve": False,
            "halation_intensity": 1,
            "shadow_comp": 0,
            "white_clip": False,
            "white_balance": False,
            "sat_adjust": 1,
            "grain_sigma": 0.4,
            "gamma_func": "sRGB",
            "push_pull": 0.0,
            "sharpening_strength": 0.0,
            "sharpening_sigma": 1.0,
            "color_masking": 0.5,
            "dye_inhibition": 0.5,
        }
        self.dflt_img_params = {
            "exp_comp": 0,
            "zoom": 1,
            "rotate_times": 0,
            "rotation": 0,
            "exp_kelvin": 6000,
            "profile": "Default",
            "canvas_mode": "No",
            "canvas_scale": 1.0,
            "canvas_ratio": 0.8,
            "highlight_burn": 0,
            "burn_scale": 50,
            "flip": False,
            "tint": 0,
            "chroma_nr": 0,
        }

        self._create_controls(
            filmstock_info,
            basic_settings_group,
            profile_settings_group,
            film_effects_group,
            image_correction_group,
            advanced_printing_group,
            canvas_group,
        )
        self._connect_controls()
        self.setCentralWidget(page_splitter)

        self.resize(QSize(1440, 960))
        scroll_area.resize(QSize(500, 500))
        basic_settings_group.setChecked()
        profile_settings_group.setChecked()

        self.waiting = False
        self.running = False

        self.threadpool = QThreadPool()

        self.corrected_image = None
        self.rotate_times = 0
        self.flip_image = False
        self.active = True  # prevent from running update_preview by setting inactive
        self.loading = False  # prevent from storing settings while setting widget values
        self.display_icc_path = None
        self.softproof_icc_path = None
        self.icc_transform = None
        self.display_intent = ImageCms.Intent.RELATIVE_COLORIMETRIC
        self.softproof_intent = ImageCms.Intent.ABSOLUTE_COLORIMETRIC
        self.srgb_profile = ImageCms.createProfile("sRGB")
        self.icc_bpc = False

        self.gpu_processor = GpuProcessor(self.cameras, self.lenses)
        self.cpu_processor = CpuProcessor(self.cameras, self.lenses)

        self.image_params = {}
        self.profile_params = {}

        self.load_settings_system()
        self.load_view_settings()

        self.save_timer = time.time()

        self.load_profile_params()

        self.et = None

        QTimer.singleShot(0, self.setup_exiftool)

    def eventFilter(self, watched, event):
        """Handle Qt events forwarded to this window."""
        if watched == self.image and event.type() == QEvent.Type.Resize:
            self.parameter_changed()

        return super().eventFilter(watched, event)

    def create_context(self):
        """Replace the render contexts after switching between GPU and bitmap mode."""
        target_mode = "wgpu" if self.gpu_processing.isChecked() else "bitmap"
        if self.context_mode == target_mode:
            return
        self.context_mode = target_mode

        sizes = self.top_splitter.sizes()
        sidebar_width = sizes[1] if len(sizes) > 1 else 320

        old_image = self.image
        old_histogram = self.histogram

        old_image.setParent(None)
        old_image.deleteLater()

        self.image = QRenderWidget(update_mode="ondemand")
        self.image.installEventFilter(self)

        self.image_stack.insertWidget(1, self.image)

        old_histogram.setParent(None)
        old_histogram.deleteLater()

        self.histogram = QRenderWidget(update_mode="ondemand")
        self.histogram.setMinimumSize(0, 80)
        self.histogram.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        histogram_container = QWidget()
        container_layout = QHBoxLayout(histogram_container)
        container_layout.setContentsMargins(BORDER_RADIUS - 1, 0, BORDER_RADIUS - 1, 0)
        container_layout.setSpacing(0)
        container_layout.addWidget(self.histogram)

        self.sidebar_layout.insertWidget(0, histogram_container)

        self.image.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )
        self.image.setMinimumSize(QSize(256, 256))

        if self.context_mode == "wgpu":
            self.image_context = self.image.get_wgpu_context()
            self.image_context.configure(
                device=self.gpu_processor.device,
                format=wgpu.TextureFormat.rgba8unorm,
                usage=(
                    wgpu.TextureUsage.COPY_DST
                    | wgpu.TextureUsage.RENDER_ATTACHMENT
                    | wgpu.TextureUsage.STORAGE_BINDING
                    | wgpu.TextureUsage.TEXTURE_BINDING
                ),
            )
            self.histogram_context = self.histogram.get_wgpu_context()
            self.histogram_context.configure(
                device=self.gpu_processor.device,
                format=wgpu.TextureFormat.rgba8unorm,
                usage=(
                    wgpu.TextureUsage.COPY_DST
                    | wgpu.TextureUsage.RENDER_ATTACHMENT
                    | wgpu.TextureUsage.STORAGE_BINDING
                ),
            )
        elif self.context_mode == "bitmap":
            self.image_context = self.image.get_bitmap_context()
            self.histogram_context = self.histogram.get_bitmap_context()

        self.top_splitter.setStretchFactor(0, 1)
        self.top_splitter.setStretchFactor(1, 0)
        self.top_splitter.setSizes([10000, sidebar_width])

        if self.image_bar.current_image() is not None:
            self.image_stack.setCurrentIndex(1)
        else:
            self.image_stack.setCurrentIndex(0)

    def setup_exiftool(self):
        """Initialize ExifTool and show an error when it is unavailable."""
        try:
            self.et = exiftool.ExifToolHelper()
        except FileNotFoundError:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("Error")
            msg.setText("ExifTool not found.\nPlease install ExifTool first.")
            msg.setStandardButtons(QMessageBox.StandardButton.Close)

            # When Close is clicked, quit the app
            if msg.exec() == QMessageBox.StandardButton.Close:
                QApplication.quit()

    def show_about(self):
        """Open the application about dialog."""
        app_links = {
            "GitHub Page": "https://github.com/JanLohse/raw2film",
            "PyPI Page": "https://pypi.org/project/raw2film/",
            "Documentation": "https://janlohse.github.io/raw2film",
        }

        about_box = AboutDialog(
            parent=self,
            app_name="Raw2Film",
            version=__version__,
            author="Jan Lohse",
            year="2026",
            license_type="MIT License",
            links=app_links,
        )
        about_box.exec()

    def show_shortcuts_dialog(self):
        """Show the shortcut dialog."""
        shortcuts_map = {}

        # Grab shortcuts from all QActions
        for action in self.findChildren(QAction):
            if not action.shortcut().isEmpty():
                name = action.text().replace("&", "")
                shortcuts_map[name] = action.shortcut().toString()

        # Grab shortcuts from all standalone QShortcut objects
        for shortcut in self.findChildren(QShortcut):
            if not shortcut.key().isEmpty():
                desc = f"Shortcut ({shortcut.key().toString()})"

                if shortcut.objectName():
                    desc = shortcut.objectName()

                shortcuts_map[desc] = shortcut.key().toString()

        # Open the updated dialog
        dialog = AutoShortcutsDialog(shortcuts_map, self)
        dialog.exec()

    def closeEvent(self, event):
        """Save settings and terminate ExifTool before closing."""
        self.quick_save()
        if self.et is not None:
            self.et.terminate()
        super().closeEvent(event)

    def show_empty_placeholder(self):
        """Switch the preview stack to the empty placeholder."""
        self.image_stack.setCurrentWidget(self.empty_placeholder)

    def resizeEvent(self, event):  # noqa: D102
        super().resizeEvent(event)
        if hasattr(self, "top_splitter"):
            self._update_splitter_sizes()

    def _update_splitter_sizes(self):
        total_width = self.top_splitter.width()
        if total_width <= 0:
            return

        sidebar_width = 370
        info_visible = self.info_container.isVisible()
        info_width = 280 if info_visible else 0

        handle_width = self.top_splitter.handleWidth()
        num_handles = 2 if info_visible else 1

        # Calculate remaining width strictly for the image container
        image_width = total_width - info_width - sidebar_width - (handle_width * num_handles)
        image_width = max(200, image_width)

        if info_visible:
            self.top_splitter.setSizes([info_width, image_width, sidebar_width])
        else:
            self.top_splitter.setSizes([0, image_width, sidebar_width])

    def toggle_image_info(self, checked):
        """Show or hide the image information sidebar."""
        self.info_container.setVisible(checked)
        self.settings.setValue("image_info_visible", "1" if checked else "0")
        self._update_splitter_sizes()
