import rawpy
from PyQt6.QtCore import QThreadPool, QSize
from PyQt6.QtGui import QPixmap, QShortcut, QKeySequence
from PyQt6.QtWidgets import QScrollArea, QSizePolicy
from spectral_film_lut.utils import *
import gc


class Thumbnail(QLabel):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setToolTip(image_path)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 1px solid gray; border-radius: 10px;")
        self._pixmap = None
        self.loaded = False

    def load(self):
        with rawpy.imread(self.image_path) as raw:
            thumb = raw.extract_thumb()
        pixmap = QPixmap()
        pixmap.loadFromData(thumb.data, "JPEG")
        self.setPixmap(pixmap)
        self.loaded = True

    def resizeEvent(self, event):
        """Ensure the label is always a square and the image scales."""
        size = min(self.width(), self.height())
        self.resize(size, size)
        if self._pixmap:
            super().setPixmap(
                self._pixmap.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio
                )
            )

    def setPixmap(self, pixmap: QPixmap):
        if pixmap:
            self._pixmap = pixmap
            super().setPixmap(
                self._pixmap.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio
                )
            )

    def set_state(self, state):
        match state:
            case "selected":
                self.setStyleSheet("border: 1px solid darkgray; background-color: gray; border-radius: 10px;")
            case "highlighted":
                self.setStyleSheet("border: 1px solid darkgray; background-color: lightgray; border-radius: 10px;")
            case "default":
                self.setStyleSheet("border: 1px solid gray; background-color: transparent; border-radius: 10px;")


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

    def sizeHint(self):
        return QSize(super().sizeHint().width(), self.height_hint)

    def load_images(self, image_paths):
        for img_path in sorted(image_paths, key=lambda x: x.split("/")[-1]):
            label = Thumbnail(img_path, self)
            label.mousePressEvent = lambda event, lbl=label: self.label_mouse_event(event, lbl)
            self.image_layout.addWidget(label)
            self.image_labels.append(label)

        self.start_lazy_loading()

    def clear_images(self):
        for i in reversed(range(self.image_layout.count())):
            widget = self.image_layout.takeAt(i).widget()
            widget.clear()
            widget.deleteLater()
        gc.collect()
        self.image_labels = []
        self.selected_label = None

    def start_lazy_loading(self):
        worker = Worker(self.lazy_load_images)
        self.threadpool.start(worker)

    def lazy_load_images(self, **kwargs):
        for label in self.image_labels:
            label.load()

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
