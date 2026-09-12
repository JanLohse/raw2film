"""
The image bar on the bottom of the application.
"""

import gc
import io

import rawpy
from PIL import Image, ImageOps
from PyQt6.QtCore import QRect, QRectF, QSize, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPixmap,
    QShortcut,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from spectral_film_lut.css_theme import BUTTON_RADIUS


class RoundedLabel(QLabel):
    """A label with rounded corners."""

    def __init__(self, radius=12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius = radius
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        rect = QRectF(self.rect())
        path = QPainterPath()
        path.addRoundedRect(rect, self.radius, self.radius)
        painter.setClipPath(path)

        # draw the pixmap (if present)
        pix = self.pixmap()
        if pix is not None and not pix.isNull():
            scaled = pix.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            painter.drawPixmap(self.rect(), scaled)
        else:
            # fallback: draw normal background & text
            super().paintEvent(event)


_thumbnail_color = {
    "default": "transparent",
    "highlighted": "#808080",
    "selected": "#dedede",
}


class Thumbnail(QFrame):
    """A thumbnail frame for the image bar."""

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setToolTip(image_path)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # QLabel for image
        self.label = RoundedLabel(BUTTON_RADIUS)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.settings_dot = QFrame(self)
        self.settings_dot.setFixedSize(8, 8)
        self.settings_dot.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.settings_dot.setStyleSheet(
            """
            QFrame {
                background-color: #dedede;
                border: 1px solid rgba(0, 0, 0, 70);
                border-radius: 4px;
            }
            """
        )
        shadow = QGraphicsDropShadowEffect(self.settings_dot)
        shadow.setBlurRadius(8)
        shadow.setOffset(0, 0)
        shadow.setColor(Qt.GlobalColor.black)
        self.settings_dot.setGraphicsEffect(shadow)
        self.settings_dot.hide()

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

        self.setObjectName("Thumbnail")
        self.set_state()

        self._pixmap = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        self.loaded = True

        with rawpy.imread(self.image_path) as raw:
            thumb = raw.extract_thumb()
            img = Image.open(io.BytesIO(thumb.data))
            img = ImageOps.exif_transpose(img)

        img = img.convert("RGBA")
        data = img.tobytes("raw", "RGBA")
        qimage = QImage(data, img.width, img.height, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)

        pixmap = pixmap.scaledToHeight(256, Qt.TransformationMode.FastTransformation)
        self.setPixmap(pixmap)

    def resizeEvent(self, event):
        """Ensure the label is always a square and the image scales."""
        height = (event.size() * self.devicePixelRatioF()).height()
        if self._pixmap:
            scaled_pixmap = self._pixmap.scaledToHeight(
                height, Qt.TransformationMode.FastTransformation
            )
            scaled_pixmap.setDevicePixelRatio(self.devicePixelRatioF())
            self.label.setPixmap(scaled_pixmap)
        self.settings_dot.move(self.width() - self.settings_dot.width() - 6, 6)
        self.settings_dot.raise_()

    def setPixmap(self, pixmap: QPixmap):
        if pixmap:
            self._pixmap = pixmap
            scaled_pixmap = self._pixmap.scaledToHeight(
                round(self.height() * self.devicePixelRatioF()),
                Qt.TransformationMode.FastTransformation,
            )
            scaled_pixmap.setDevicePixelRatio(self.devicePixelRatioF())
            self.label.setPixmap(scaled_pixmap)

    def set_state(self, state="default"):
        bq_color = _thumbnail_color.get(state, _thumbnail_color["default"])
        outline_thickness = 3
        self.setStyleSheet(f"""
#Thumbnail {{
    border-radius: {BUTTON_RADIUS + outline_thickness}px;
    border: {outline_thickness}px solid {bq_color};
    background-color: {bq_color};
}}
""")

    def set_has_settings(self, has_settings: bool):
        self.settings_dot.setVisible(has_settings)

    def sizeHint(self):
        if self.loaded:
            return self.label.sizeHint()
        else:
            return QSize(self.width(), self.width())

    def minimumSizeHint(self):
        if self.loaded:
            return self.label.minimumSizeHint()
        else:
            return QSize(self.width(), self.width())

    def size(self):
        if self.loaded:
            return self.label.minimumSizeHint()
        else:
            return QSize(self.width(), self.width())


class ImageBar(QScrollArea):
    """A vertical scrollable image selector bar."""

    image_changed = pyqtSignal(str)
    """The selected image has changed."""
    copy_settings = pyqtSignal(str)
    """Settings are to be copied between images."""
    images_empty = pyqtSignal()
    """Emitted when the image list becomes empty after closing images."""

    def __init__(self):
        super().__init__()
        self.selected_label = None
        self.highlighted_labels = set()
        self.image_labels = []
        self.settings_images = set()

        self.threadpool = QThreadPool()

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.container = QWidget()
        self.container.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred
        )
        self.setMinimumHeight(100)
        self.setMaximumHeight(250)
        self.height_hint = 130

        self.image_layout = QHBoxLayout(self.container)
        self.image_layout.setSpacing(3)
        self.image_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.container.setLayout(self.image_layout)

        self.setWidget(self.container)

        def create_shortcut(key_sequence: str, func, name: str | None = None):
            shortcut = QShortcut(QKeySequence(key_sequence), self)
            if name is not None:
                shortcut.setObjectName(name)
            shortcut.activated.connect(func)

        create_shortcut("Ctrl+A", self.highlight_all, "Highlight all images")
        create_shortcut("Right", lambda: self.arrow_pressed("right"), "Next image")
        create_shortcut(
            "Shift+Right",
            lambda: self.arrow_pressed("right", shift=True),
            "Extend selection right",
        )
        create_shortcut("Left", lambda: self.arrow_pressed("left"), "Previous image")
        create_shortcut(
            "Shift+Left",
            lambda: self.arrow_pressed("left", shift=True),
            "Extend selection left",
        )

        self.horizontalScrollBar().valueChanged.connect(self.check_visible)
        self.horizontalScrollBar().rangeChanged.connect(self.check_visible)

    def resizeEvent(self, event):
        self.container.resize(event.size())
        super().resizeEvent(event)

    def sizeHint(self):
        return QSize(super().sizeHint().width(), self.height_hint)

    def clear_images(self):
        for i in reversed(range(self.image_layout.count())):
            widget = self.image_layout.takeAt(i).widget()
            widget.deleteLater()
        self.image_labels = []
        self.highlighted_labels = set()
        self.selected_label = None
        gc.collect()

    def _state_for_label(self, label):
        if label == self.selected_label:
            return "selected"
        if label in self.highlighted_labels:
            return "highlighted"
        return "default"

    def refresh_thumbnail_states(self):
        for label in self.image_labels:
            label.set_state(self._state_for_label(label))
            label.set_has_settings(
                label.image_path.split("/")[-1] in self.settings_images
            )

    def set_settings_images(self, image_paths):
        settings_images = {image_path.split("/")[-1] for image_path in image_paths}
        if settings_images == self.settings_images:
            return
        self.settings_images = settings_images
        self.refresh_thumbnail_states()

    def load_images(self, image_paths):
        self.clear_images()
        for img_path in sorted(image_paths, key=lambda x: x.split("/")[-1]):
            label = Thumbnail(img_path, self)
            label.mousePressEvent = lambda event, lbl=label: self.label_mouse_event(
                event, lbl
            )
            self.image_layout.addWidget(label)
            self.image_labels.append(label)
        self.refresh_thumbnail_states()

        if self.image_labels:
            self.select_image(self.image_labels[0])

        QApplication.processEvents()
        QTimer.singleShot(0, self.check_visible)

    def label_mouse_event(self, event, label):
        if event.button() == Qt.MouseButton.LeftButton:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.highlight_image(label)
            elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self.highlight_shift(label)
            else:
                self.select_image(label)
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.copy_settings.emit(label.image_path)

    def highlight_shift(self, label):
        if self.selected_label is None or self.selected_label == label:
            return
        selected_index = self.image_labels.index(self.selected_label)
        clicked_index = self.image_labels.index(label)
        self.highlighted_labels = {
            self.image_labels[index]
            for index in range(
                min(selected_index, clicked_index),
                max(selected_index, clicked_index) + 1,
            )
        }
        self.refresh_thumbnail_states()

    def highlight_image(self, label, friendly=False):
        if not self.selected_label == label:
            if label in self.highlighted_labels and not friendly:
                self.highlighted_labels.remove(label)
            else:
                self.highlighted_labels.add(label)
            self.refresh_thumbnail_states()

    def highlight_all(self):
        if len(self.highlighted_labels) == len(self.image_labels):
            self.highlighted_labels = set()
        else:
            self.highlighted_labels = set(self.image_labels)
        self.refresh_thumbnail_states()

    def select_image(self, label):
        if label == self.selected_label:
            return
        if self.selected_label:
            if label in self.highlighted_labels:
                self.highlighted_labels.add(label)
            else:
                self.highlighted_labels = {label}
        elif label not in self.highlighted_labels:
            self.highlighted_labels = {label}
        self.selected_label = label
        self.refresh_thumbnail_states()
        self.image_changed.emit(label.image_path)
        self.ensure_visible(label)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta)

    def current_image(self):
        if self.selected_label is None:
            return None
        else:
            return self.selected_label.image_path

    def arrow_pressed(self, key, shift=False):
        if not self.image_labels:
            return

        if self.selected_label in self.image_labels:
            current_index = self.image_labels.index(self.selected_label)
        else:
            current_index = -1

        if key == "right":
            if current_index < len(self.image_labels) - 1:
                target_index = current_index + 1
            else:
                target_index = 0
        elif key == "left":
            if current_index > 0:
                target_index = current_index - 1
            else:
                target_index = -1
        else:
            return
        if shift:
            self.highlight_image(self.image_labels[target_index], friendly=True)
        self.select_image(self.image_labels[target_index])

    def ensure_visible(self, label):
        """Scrolls the view to make sure the selected image is visible"""
        x = label.pos().x()
        label_width = label.width()
        area_width = self.width()
        scrollbar_x = self.horizontalScrollBar().value()
        if x < scrollbar_x:
            self.horizontalScrollBar().setValue(x - label_width)
        elif x + label_width > scrollbar_x + area_width:
            self.horizontalScrollBar().setValue(x - area_width + 2 * label_width)

    def get_highlighted(self):
        return sorted([label.image_path for label in self.highlighted_labels])

    def get_all(self):
        return [label.image_path for label in self.image_labels]

    def close_labels(self, labels):
        # Copy labels to avoid modifying the iterable while iterating
        labels_to_close = list(labels)

        if self.selected_label is not None:
            new_selected = self.image_labels.index(self.selected_label)
        else:
            new_selected = None

        for image_label in labels_to_close:
            # skip labels that are no longer present
            if image_label not in self.image_labels:
                continue
            index = self.image_labels.index(image_label)
            # Remove widget from layout and list
            widget = self.image_layout.itemAt(index).widget()
            if widget is not None:
                widget.setParent(None)
            self.image_labels.pop(index)
            if image_label == self.selected_label:
                self.selected_label = None
            self.highlighted_labels.discard(image_label)
            if (
                new_selected is not None
                and index <= new_selected
                and (index or new_selected)
            ):
                if index < new_selected:
                    new_selected -= 1
                if new_selected >= len(self.image_labels) - 1:
                    new_selected = len(self.image_labels) - 1
        self.refresh_thumbnail_states()
        QTimer.singleShot(0, self.check_visible)
        # Notify listeners if the bar is now empty
        if not self.image_labels:
            self.images_empty.emit()
        return new_selected

    def close_highlighted(self):
        new_selected = self.close_labels(self.highlighted_labels)
        self.highlighted_labels = set()
        if self.image_labels:
            self.select_image(self.image_labels[new_selected])

    def deselect_all(self):
        if self.selected_label is not None:
            self.highlighted_labels = {self.selected_label}
        else:
            self.highlighted_labels = set()
        self.refresh_thumbnail_states()

    def close_single_image(self, src):
        for label in self.image_labels:
            if label.image_path == src:
                new_selected = self.close_labels(
                    [
                        label,
                    ]
                )
                if new_selected is not None and self.image_labels:
                    self.select_image(self.image_labels[new_selected])
                return

    def check_visible(self):
        viewport = self.viewport().rect()
        for lbl in self.image_labels:
            # Map label's geometry to viewport coordinates
            rect = self.viewport().mapFromGlobal(lbl.mapToGlobal(lbl.rect().topLeft()))
            label_rect = QRect(rect, lbl.size())
            if viewport.intersects(label_rect):
                lbl.load()
