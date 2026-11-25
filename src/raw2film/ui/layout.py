"""Build the main window layout and menus."""

from PyQt6.QtCore import (
    QSize,
    Qt,
)
from PyQt6.QtGui import (
    QAction,
    QKeySequence,
)
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)
from rendercanvas.qt import QRenderWidget
from spectral_film_lut.ui.theme import BORDER_RADIUS, OUTLINE_COLOR, THEME
from spectral_film_lut.ui.widgets.controls import (
    AnimatedButton,
)

from raw2film.gui_objects import (
    ImageInfoWidget,
    SidebarGroup,
)
from raw2film.image_bar import ImageBar


class WindowLayoutMixin:
    """Build the main window layout and menus."""

    def _create_window_shell(self, filmstock_info):
        self.histogram = QRenderWidget(update_mode="ondemand")
        self.histogram.setMinimumSize(QSize(256, 256))
        self.histogram.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.histogram.setMinimumSize(0, 80)
        self.histogram_context = None

        page_splitter = QSplitter(Qt.Orientation.Vertical)
        sidebar_widget = QWidget()

        sidebar_widget.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.sidebar_layout = QVBoxLayout(sidebar_widget)
        self.sidebar_layout.addWidget(self.histogram)
        self.sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.histogram.setContentsMargins(BORDER_RADIUS - 1, 0, BORDER_RADIUS - 1, 0)
        self.sidebar_layout.setSpacing(0)
        sidebar_settings = QWidget()
        side_layout = QVBoxLayout()
        sidebar_settings.setLayout(side_layout)
        side_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        def create_line():
            line = QFrame(self)
            line.setFrameShape(QFrame.Shape.HLine)
            line.setStyleSheet(f"background-color: {OUTLINE_COLOR};")
            line.setMaximumHeight(1)
            return line

        basic_settings_group = SidebarGroup("Basic editing", self)
        side_layout.addWidget(basic_settings_group)
        side_layout.addWidget(create_line())
        profile_settings_group = SidebarGroup("Profile settings", self)
        side_layout.addWidget(profile_settings_group)
        side_layout.addWidget(create_line())
        film_effects_group = SidebarGroup("Film effects", self)
        side_layout.addWidget(film_effects_group)
        side_layout.addWidget(create_line())
        image_correction_group = SidebarGroup("Image correction", self)
        side_layout.addWidget(image_correction_group)
        side_layout.addWidget(create_line())
        advanced_printing_group = SidebarGroup("Advanced printing techniques", self)
        side_layout.addWidget(advanced_printing_group)
        side_layout.addWidget(create_line())
        canvas_group = SidebarGroup("Canvas", self)
        side_layout.addWidget(canvas_group)

        sidebar_container = QFrame(self)
        sidebar_container.setObjectName("scroll")
        sidebar_container_layout = QVBoxLayout(sidebar_container)
        sidebar_container_layout.setContentsMargins(4, 4, 4, 4)
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setWidget(sidebar_settings)
        scroll_area.setMinimumWidth(280)
        sidebar_container_layout.addWidget(scroll_area)

        self.image = QRenderWidget(update_mode="ondemand")
        self.image.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )
        self.image.setMinimumSize(QSize(256, 256))
        self.image_context = None
        self.context_mode = None

        # Create placeholder widget with two buttons
        self.empty_placeholder = QWidget()
        placeholder_layout = QHBoxLayout(self.empty_placeholder)
        placeholder_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_open_images = AnimatedButton("Open images", parent=self)
        btn_open_folder = AnimatedButton("Open folder", parent=self)

        btn_open_images.clicked.connect(self.load_images)
        btn_open_folder.clicked.connect(self.load_folder)

        placeholder_layout.addWidget(btn_open_images)
        placeholder_layout.addWidget(btn_open_folder)

        # Create Image Info container widget (left of image preview)
        self.info_container = QFrame(self)
        self.info_container.setObjectName("scroll")
        info_container_layout = QVBoxLayout(self.info_container)
        info_container_layout.setContentsMargins(4, 4, 4, 4)
        info_scroll = QScrollArea(self)
        info_scroll.setWidgetResizable(True)
        info_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.image_info_widget = ImageInfoWidget(self)
        info_scroll.setWidget(self.image_info_widget)
        info_scroll.setMinimumWidth(260)
        info_container_layout.addWidget(info_scroll)

        self.sidebar_layout.addWidget(sidebar_container)

        # Create container widget with QStackedLayout
        self.image_container = QWidget()
        self.image_stack = QStackedLayout(self.image_container)
        self.image_stack.addWidget(self.empty_placeholder)  # Index 0: Placeholder
        self.image_stack.addWidget(self.image)  # Index 1: Canvas image

        self.image_bar = ImageBar()
        image_bar_container = QFrame(self)
        image_bar_container.setObjectName("scroll")
        image_bar_container_layout = QVBoxLayout(image_bar_container)
        image_bar_container_layout.setContentsMargins(4, 4, 4, 4)
        image_bar_container_layout.addWidget(self.image_bar)

        self.top_splitter = QSplitter()
        page_splitter.addWidget(self.top_splitter)
        page_splitter.addWidget(image_bar_container)
        page_splitter.setContentsMargins(8, 8, 8, 8)
        self.top_splitter.setContentsMargins(0, 0, 0, 0)

        page_splitter.setStretchFactor(0, 1)
        page_splitter.setStretchFactor(1, 0)

        self.sidebar_layout.addWidget(sidebar_container)

        self.top_splitter.addWidget(self.info_container)
        self.top_splitter.addWidget(self.image_container)
        self.top_splitter.addWidget(sidebar_widget)

        self.top_splitter.setCollapsible(0, False)
        self.top_splitter.setCollapsible(1, False)
        self.top_splitter.setCollapsible(2, False)

        self.top_splitter.setStretchFactor(0, 0)
        self.top_splitter.setStretchFactor(1, 1)
        self.top_splitter.setStretchFactor(2, 0)

        self.top_splitter.setSizes([0, 10000, 370])
        self.info_container.hide()
        return (
            page_splitter,
            scroll_area,
            basic_settings_group,
            profile_settings_group,
            film_effects_group,
            image_correction_group,
            advanced_printing_group,
            canvas_group,
        )

    def _create_menus(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.setStyleSheet(THEME)
        view_menu = menu.addMenu("View")
        edit_menu = menu.addMenu("Edit")
        view_menu.setToolTipsVisible(True)
        help_menu = menu.addMenu("Help")

        self.image_selector = QAction("Open images", self)
        self.image_selector.setShortcut(QKeySequence("Ctrl+O"))
        file_menu.addAction(self.image_selector)
        self.folder_selector = QAction("Open folder", self)
        self.folder_selector.setShortcut(QKeySequence("Ctrl+Shift+O"))
        file_menu.addAction(self.folder_selector)
        self.quick_save_button = QAction("Quick save settings", self)
        self.quick_save_button.setShortcut("Ctrl+S")
        file_menu.addAction(self.quick_save_button)
        self.save_image_button = QAction("Quick export jpg", self)
        self.save_image_button.triggered.connect(self.save_image_dialog)
        file_menu.addAction(self.save_image_button)
        self.save_all_button = QAction("Export all images", self)
        self.save_all_button.triggered.connect(self.save_all_images)
        self.save_selected_button = QAction("Export selected images", self)
        self.save_selected_button.triggered.connect(self.save_selected_images)
        file_menu.addAction(self.save_selected_button)
        file_menu.addAction(self.save_all_button)
        self.save_settings_button = QAction("Save settings", self)
        file_menu.addAction(self.save_settings_button)
        self.load_settings_button = QAction("Load settings", self)
        file_menu.addAction(self.load_settings_button)
        self.close_highlighted_button = QAction("Close selected images", self)
        self.close_highlighted_button.setShortcut("Del")
        file_menu.addAction(self.close_highlighted_button)
        self.delete_highlighted_button = QAction("Delete selected images", self)
        self.delete_highlighted_button.setShortcut("Shift+Del")
        file_menu.addAction(self.delete_highlighted_button)
        self.deselect_all_button = QAction("Deselect all", self)
        self.deselect_all_button.setShortcut("Ctrl+D")
        edit_menu.addAction(self.deselect_all_button)
        self.auto_lens_correct = QAction("Auto lens correct", self)
        """Whether to toggle lens correction automatically when loading a new image."""
        self.auto_lens_correct.setCheckable(True)
        self.auto_lens_correct.setChecked(True)
        edit_menu.addAction(self.auto_lens_correct)
        self.reset_image_button = QAction("Reset image", self)
        edit_menu.addAction(self.reset_image_button)
        self.reset_all_images_button = QAction("Reset all images", self)
        edit_menu.addAction(self.reset_all_images_button)
        self.reset_profile_button = QAction("Reset profile", self)
        edit_menu.addAction(self.reset_profile_button)
        self.delete_profile_button = QAction("Delete profile", self)
        edit_menu.addAction(self.delete_profile_button)
        self.delete_all_profiles_button = QAction("Delete all profiles", self)
        edit_menu.addAction(self.delete_all_profiles_button)
        self.image_info_action = QAction("Image info", self)
        self.image_info_action.setShortcut(QKeySequence("Ctrl+I"))
        self.image_info_action.setCheckable(True)
        self.image_info_action.setChecked(False)
        self.image_info_action.triggered.connect(self.toggle_image_info)
        view_menu.addAction(self.image_info_action)

        self.full_preview = QAction("Full preview", self)
        self.full_preview.setShortcut(QKeySequence("Ctrl+Shift+F"))
        self.full_preview.setCheckable(True)
        self.full_preview.setChecked(True)
        view_menu.addAction(self.full_preview)
        self.gpu_processing = QAction("GPU rendering", self)
        self.gpu_processing.setCheckable(True)
        self.gpu_processing.setChecked(True)
        view_menu.addAction(self.gpu_processing)
        self.half_res_preview = QAction("Half res. preview", self)
        self.half_res_preview.setCheckable(True)
        self.half_res_preview.setShortcut(QKeySequence("Ctrl+Shift+H"))
        view_menu.addAction(self.half_res_preview)
        self.load_display_icc_button = QAction("Load display ICC profile", self)
        self.load_display_icc_button.setCheckable(True)
        view_menu.addAction(self.load_display_icc_button)
        self.reset_display_icc_button = QAction("Reset display ICC profile", self)
        self.reset_display_icc_button.setVisible(False)
        view_menu.addAction(self.reset_display_icc_button)

        display_intent_menu = view_menu.addMenu("Display rendering intent")
        self.display_absolute_intent = QAction("Absolute colorimetric", self)
        self.display_absolute_intent.setCheckable(True)
        display_intent_menu.addAction(self.display_absolute_intent)
        self.display_relative_intent = QAction("Relative colorimetric", self)
        self.display_relative_intent.setCheckable(True)
        self.display_relative_intent.setChecked(True)
        display_intent_menu.addAction(self.display_relative_intent)
        self.display_relative_bpc_intent = QAction("Relative w/ black point compensation", self)
        self.display_relative_bpc_intent.setCheckable(True)
        self.display_relative_bpc_intent.setChecked(False)
        display_intent_menu.addAction(self.display_relative_bpc_intent)
        self.display_perceptual_intent = QAction("Perceptual", self)
        self.display_perceptual_intent.setCheckable(True)
        display_intent_menu.addAction(self.display_perceptual_intent)
        self.display_saturation_intent = QAction("Saturation preserving", self)
        self.display_saturation_intent.setCheckable(True)
        display_intent_menu.addAction(self.display_saturation_intent)

        self.load_softproof_icc_button = QAction("Load soft proofing ICC profile", self)
        self.load_softproof_icc_button.setCheckable(True)
        view_menu.addAction(self.load_softproof_icc_button)
        self.reset_softproof_icc_button = QAction("Reset soft proofing ICC profile", self)
        self.reset_softproof_icc_button.setVisible(False)
        view_menu.addAction(self.reset_softproof_icc_button)

        softproof_intent_menu = view_menu.addMenu("Soft proofing rendering intent")
        self.softproof_absolute_intent = QAction("Absolute colorimetric", self)
        self.softproof_absolute_intent.setCheckable(True)
        softproof_intent_menu.addAction(self.softproof_absolute_intent)
        self.softproof_relative_intent = QAction("Relative colorimetric", self)
        self.softproof_relative_intent.setCheckable(True)
        self.softproof_relative_intent.setChecked(True)
        softproof_intent_menu.addAction(self.softproof_relative_intent)
        self.softproof_perceptual_intent = QAction("Perceptual", self)
        self.softproof_perceptual_intent.setCheckable(True)
        softproof_intent_menu.addAction(self.softproof_perceptual_intent)
        self.softproof_saturation_intent = QAction("Saturation preserving", self)
        self.softproof_saturation_intent.setCheckable(True)
        softproof_intent_menu.addAction(self.softproof_saturation_intent)

        about_button = help_menu.addAction("About")
        about_button.triggered.connect(self.show_about)
        shortcut_button = help_menu.addAction("Shortcuts")
        shortcut_button.triggered.connect(self.show_shortcuts_dialog)
