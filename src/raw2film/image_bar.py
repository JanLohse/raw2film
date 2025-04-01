import rawpy
from PyQt6.QtCore import QThreadPool
from PyQt6.QtGui import QPixmap, QWheelEvent
from PyQt6.QtWidgets import QScrollArea, QSizePolicy
from spectral_film_lut.utils import *


class Thumbnail(QLabel):
    def __init__(self, image_path, thumbnail_size, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.thumbnail_size = thumbnail_size
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 1px solid gray; border-radius: 7px;")
        self.setFixedSize(self.thumbnail_size, self.thumbnail_size)
        self.loaded = False

    def load(self):
        with rawpy.imread(self.image_path) as raw:
            thumb = raw.extract_thumb()
        pixmap = QPixmap()
        pixmap.loadFromData(thumb.data, "JPEG", )
        pixmap = pixmap.scaled(self.thumbnail_size, self.thumbnail_size, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        self.setPixmap(pixmap)
        self.loaded = True

    def set_selected(self, selected):
        if selected:
            self.setStyleSheet("border: 3px solid darkgray; background-color: gray; border-radius: 10px;")
        else:
            self.setStyleSheet("border: 1px solid gray; background-color: transparent; border-radius: 10px;")


class ImageBar(QScrollArea):
    image_changed = pyqtSignal(str)

    def __init__(self, thumbnail_size=100, padding=15):
        super().__init__()
        self.thumbnail_size = thumbnail_size
        self.bar_height = self.thumbnail_size + padding * 2
        self.selected_label = None
        self.image_labels = []

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(self.bar_height)

        self.container = QWidget()
        self.container.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)

        self.image_layout = QHBoxLayout(self.container)
        self.image_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.container.setLayout(self.image_layout)

        self.setWidget(self.container)

    def load_images(self, image_paths):
        for img_path in image_paths:
            label = Thumbnail(img_path, self.thumbnail_size, self)
            label.mousePressEvent = lambda event, lbl=label: self.select_image(lbl)
            self.image_layout.addWidget(label)
            self.image_labels.append(label)

        threadpool = QThreadPool()
        worker = Worker(self.lazy_load_images)
        threadpool.start(worker)

    def clear_images(self):
        for i in reversed(range(self.image_layout.count())):
            self.image_layout.itemAt(i).widget().setParent(None)
        self.image_labels = []
        self.selected_label = None

    def start_lazy_loading(self):
        worker = Worker(self.lazy_load_images)
        self.threadpool.start(worker)

    def lazy_load_images(self, **kwargs):
        for label in self.image_labels:
            label.load()

    def select_image(self, label):
        if self.selected_label:
            self.selected_label.set_selected(False)
        self.selected_label = label
        self.selected_label.set_selected(True)
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

    def keyPressEvent(self, event):
        if not self.image_labels:
            return

        if self.selected_label in self.image_labels:
            current_index = self.image_labels.index(self.selected_label)
        else:
            current_index = -1

        if event.key() == Qt.Key.Key_Right:
            if current_index < len(self.image_labels) - 1:
                self.select_image(self.image_labels[current_index + 1])
            else:
                self.select_image(self.image_labels[0])
        elif event.key() == Qt.Key.Key_Left:
            if current_index > 0:
                self.select_image(self.image_labels[current_index - 1])
            else:
                self.select_image(self.image_labels[-1])

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
