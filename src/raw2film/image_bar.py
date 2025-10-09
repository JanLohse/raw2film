import gc
import io

import rawpy
from PIL import Image, ImageOps
from PyQt6.QtCore import QThreadPool, QSize, QRect, QTimer
from PyQt6.QtGui import QPixmap, QShortcut, QKeySequence, QImage, QPainter, QPen
from PyQt6.QtWidgets import QScrollArea, QSizePolicy, QApplication
from spectral_film_lut.utils import *


from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QSizePolicy
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from PIL import Image, ImageOps
import rawpy, io


class Thumbnail(QFrame):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setToolTip(image_path)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Use QFrame for border handling
        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(0)

        # Inner QLabel for image
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        # Layout to hold the QLabel
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        self.setStyleSheet("")

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

        pixmap = pixmap.scaledToHeight(256, Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(pixmap)

    def resizeEvent(self, event):
        """Ensure the label is always a square and the image scales."""
        height = event.size().height()
        if self._pixmap:
            self.label.setPixmap(
                self._pixmap.scaledToHeight(
                    height, Qt.TransformationMode.SmoothTransformation
                )
            )

    def setPixmap(self, pixmap: QPixmap):
        if pixmap:
            self._pixmap = pixmap
            self.label.setPixmap(
                self._pixmap.scaledToHeight(
                    self.height(), Qt.TransformationMode.SmoothTransformation
                )
            )

    def set_state(self, state):
        match state:
            case "selected":
                self.setLineWidth(3)
            case "highlighted":
                self.setLineWidth(2)
            case "default":
                self.setLineWidth(0)

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
    image_changed = pyqtSignal(str)
    copy_settings = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.selected_label = None
        self.highlighted_labels = set()
        self.image_labels = []

        self.threadpool = QThreadPool()

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.container = QWidget()
        self.container.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
        self.setMinimumHeight(100)
        self.setMaximumHeight(250)
        self.height_hint = 130

        self.image_layout = QHBoxLayout(self.container)
        self.image_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.container.setLayout(self.image_layout)

        self.setWidget(self.container)

        QShortcut(QKeySequence('Ctrl+A'), self).activated.connect(self.highlight_all)
        QShortcut(QKeySequence('Right'), self).activated.connect(lambda: self.arrow_pressed('right'))
        QShortcut(QKeySequence('Shift+Right'), self).activated.connect(lambda: self.arrow_pressed('right', shift=True))
        QShortcut(QKeySequence('Left'), self).activated.connect(lambda: self.arrow_pressed('left'))
        QShortcut(QKeySequence('Shift+Left'), self).activated.connect(lambda: self.arrow_pressed('left', shift=True))

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
        self.selected_label = None
        gc.collect()

    def load_images(self, image_paths):
        self.clear_images()
        for img_path in sorted(image_paths, key=lambda x: x.split("/")[-1]):
            label = Thumbnail(img_path, self)
            label.mousePressEvent = lambda event, lbl=label: self.label_mouse_event(event, lbl)
            self.image_layout.addWidget(label)
            self.image_labels.append(label)
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
        for highlighted_label in self.highlighted_labels:
            highlighted_label.set_state("default")
        self.highlighted_labels = {self.image_labels[index] for index in
                                   range(min(selected_index, clicked_index), max(selected_index, clicked_index) + 1)}
        for highlighted_label in self.highlighted_labels:
            highlighted_label.set_state("highlighted")
        self.selected_label.set_state("selected")

    def highlight_image(self, label, friendly=False):
        if not self.selected_label == label:
            if label in self.highlighted_labels and not friendly:
                self.highlighted_labels.remove(label)
                label.set_state("default")
            else:
                self.highlighted_labels.add(label)
                label.set_state("highlighted")

    def highlight_all(self):
        if len(self.highlighted_labels) == len(self.image_labels):
            for label in self.image_labels:
                label.set_state("default")
            self.highlighted_labels = set()
            if self.selected_label is not None:
                self.highlighted_labels.add(self.selected_label)
                self.selected_label.set_state("selected")
        else:
            for label in self.image_labels:
                label.set_state("highlighted")
            self.highlighted_labels = set(self.image_labels)
            if self.selected_label is not None:
                self.selected_label.set_state("selected")

    def select_image(self, label):
        if label == self.selected_label:
            return
        if self.selected_label:
            if label in self.highlighted_labels:
                self.selected_label.set_state("highlighted")
                self.highlighted_labels.add(label)
            else:
                self.selected_label.set_state("default")
                for highlighted_label in self.highlighted_labels:
                    highlighted_label.set_state("default")
                self.highlighted_labels = {label}
        elif not label in self.highlighted_labels:
            for highlighted_label in self.highlighted_labels:
                highlighted_label.set_state("default")
            self.highlighted_labels = {label}
        self.selected_label = label
        label.set_state("selected")
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
        """ Scrolls the view to make sure the selected image is visible """
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
        if self.selected_label is not None:
            new_selected = self.image_labels.index(self.selected_label)
        else:
            new_selected = None
        for image_label in labels:
            index = self.image_labels.index(image_label)
            self.image_layout.itemAt(index).widget().setParent(None)
            self.image_labels.pop(index)
            if image_label == self.selected_label:
                self.selected_label = None
            if new_selected is not None and index <= new_selected and (index or new_selected):
                if index < new_selected:
                    new_selected -= 1
                if new_selected >= len(self.image_labels) - 1:
                    new_selected = len(self.image_labels) - 1
        QTimer.singleShot(0, self.check_visible)
        return new_selected

    def close_highlighted(self):
        new_selected = self.close_labels(self.highlighted_labels)
        self.highlighted_labels = set()
        if self.image_labels:
            self.select_image(self.image_labels[new_selected])

    def deselect_all(self):
        for label in self.highlighted_labels:
            if label != self.selected_label:
                label.set_state("default")
        if self.selected_label is not None:
            self.highlighted_labels = {self.selected_label}
        else:
            self.highlighted_labels = set()

    def close_single_image(self, src):
        for label in self.image_labels:
            if label.image_path == src:
                new_selected = self.close_labels([label,])
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
