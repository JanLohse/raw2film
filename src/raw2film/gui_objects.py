"""Additional GUI objects used by Raw2Film."""

import queue
import threading
import time

from PyQt6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QObject,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QSize,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import (
    QColor,
    QIcon,
)
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from spectral_film_lut import BASE_DIR
from spectral_film_lut.css_theme import BASE_COLOR
from spectral_film_lut.gui_objects import (
    AnimatedButton,
    AnimatedToolButton,
)


class CpuWorker(QObject):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._is_canceled = False

    @pyqtSlot()
    def cancel(self):
        self._is_canceled = True

    def run_tasks(self, func_execute_cpu, tasks, **kwargs):
        self._is_canceled = False
        total = len(tasks)

        start = time.time()

        for idx, task in enumerate(tasks, 1):
            if self._is_canceled:
                break
            status_msg = func_execute_cpu(
                task, current_idx=idx, total_count=total, **kwargs
            )
            self.progress.emit(status_msg, idx)

        print(f"total {time.time() - start:.2f}s")

        self.finished.emit()


class GpuWorker(QObject):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._is_canceled = False
        self.payload_queue = None

    @pyqtSlot()
    def cancel(self):
        self._is_canceled = True
        if self.payload_queue is not None:
            try:
                self.payload_queue.put((None, None, None), block=False)
            except queue.Full:
                pass

    def run_tasks(self, func_prepare_cpu, func_execute_gpu, tasks, **kwargs):
        self._is_canceled = False
        total = len(tasks)
        self.payload_queue = queue.Queue(maxsize=1)
        start = time.time()

        # Background Producer (CPU raw decoding)
        def cpu_producer():
            for idx, task in enumerate(tasks, 1):
                if self._is_canceled:
                    break
                try:
                    pipeline_payload = func_prepare_cpu(task[0], **kwargs)
                    while not self._is_canceled:
                        try:
                            self.payload_queue.put(
                                (idx, task, pipeline_payload), timeout=0.1
                            )
                            break
                        except queue.Full:
                            continue
                except Exception:
                    self.payload_queue.put((idx, task, None))
            self.payload_queue.put((None, None, None))

        producer_thread = threading.Thread(target=cpu_producer, daemon=True)
        producer_thread.start()

        # Consumer (GPU loop running on the QThread)
        while True:
            if self._is_canceled:
                break

            idx, task, pipeline_payload = self.payload_queue.get()
            if idx is None:
                break
            if pipeline_payload is None:
                self.payload_queue.task_done()
                continue

            status_msg = func_execute_gpu(
                task, pipeline_payload, current_idx=idx, total_count=total, **kwargs
            )
            self.progress.emit(status_msg, idx)
            self.payload_queue.task_done()

        producer_thread.join(timeout=1.0)

        print(f"total {time.time() - start:.2f}s")

        self.finished.emit()


class AutoShortcutsDialog(QDialog):
    """
    A dialog that automatically builds a list of shortcuts from both QActions and
    QShortcuts.
    """

    def __init__(self, shortcuts_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Keyboard Shortcuts")
        self.setMinimumSize(400, 350)

        layout = QVBoxLayout(self)

        text_area = QTextEdit(self)
        text_area.setReadOnly(True)

        html_content = """
        <h3>Active Keyboard Shortcuts</h3>
        <table border="0" cellpadding="6" cellspacing="0" width="100%">
        """

        for description, shortcut_str in sorted(shortcuts_data.items()):
            html_content += (
                f"<tr><td>{description}</td><td><b>{shortcut_str}</b></td></tr>"
            )

        html_content += "</table>"
        text_area.setHtml(html_content)
        layout.addWidget(text_area)

        close_btn = AnimatedButton("Close", parent=self)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


DOWN_ARROW_ICON = QIcon(f"{BASE_DIR}/resources/down_arrow.svg")
RIGHT_ARROW_ICON = QIcon(f"{BASE_DIR}/resources/right_arrow.svg")


class SidebarGroup(QWidget):
    """A group wrapper for a sidebar that is collapsible."""

    def __init__(self, title="", parent=None):
        super().__init__(parent)

        self.toggle_button = AnimatedToolButton(parent=self)
        self.toggle_button._checked_color = QColor(BASE_COLOR)
        self.toggle_button.setText("  " + title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.toggle_button.setIcon(RIGHT_ARROW_ICON)
        self.toggle_button.pressed.connect(self.on_pressed)
        self.toggle_button.setStyleSheet("background: transparent;")
        self.toggle_button.setIconSize(QSize(12, 12))

        self.toggle_animation = QParallelAnimationGroup(self)

        self.content_area = QScrollArea(maximumHeight=0, minimumHeight=0)
        self.content_area.setContentsMargins(0, 0, 0, 0)
        self.content_area.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.content_area.setFrameShape(QFrame.Shape.NoFrame)
        self.content_layout = QGridLayout()
        self.content_area.setLayout(self.content_layout)
        self.content_counter = -1

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)

        self.toggle_animation.addAnimation(QPropertyAnimation(self, b"minimumHeight"))
        self.toggle_animation.addAnimation(QPropertyAnimation(self, b"maximumHeight"))
        self.toggle_animation.addAnimation(
            QPropertyAnimation(self.content_area, b"maximumHeight")
        )

    def setChecked(self):
        self.toggle_button.setChecked(True)
        self.toggle_button.setIcon(DOWN_ARROW_ICON)
        collapsed_height = self.sizeHint().height() - self.content_area.maximumHeight()
        content_height = self.content_layout.sizeHint().height()
        self.setMinimumHeight(collapsed_height + content_height)
        self.setMaximumHeight(collapsed_height + content_height)
        self.content_area.setMaximumHeight(content_height)

    @pyqtSlot()
    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setIcon(RIGHT_ARROW_ICON if checked else DOWN_ARROW_ICON)
        self.toggle_animation.setDirection(
            QAbstractAnimation.Direction.Backward
            if checked
            else QAbstractAnimation.Direction.Forward
        )
        self.toggle_animation.start()

    def add_option(self, widget, name=None, default=None, setter=None, tool_tip=None):
        self.content_counter += 1
        label = QLabel(
            name,
            alignment=(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
        )
        self.content_layout.addWidget(label, self.content_counter, 0)
        self.content_layout.addWidget(widget, self.content_counter, 1)
        if default is not None and setter is not None:
            label.mouseDoubleClickEvent = lambda *args: setter(default)
            setter(default)
        if tool_tip is not None:
            label.setToolTip(tool_tip)
        self.update_animation()

    def update_animation(self):
        collapsed_height = self.sizeHint().height() - self.content_area.maximumHeight()
        content_height = self.content_layout.sizeHint().height()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(300)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)
            animation.setEasingCurve(QEasingCurve.Type.InOutCubic)

        content_animation = self.toggle_animation.animationAt(
            self.toggle_animation.animationCount() - 1
        )
        content_animation.setDuration(300)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)
        content_animation.setEasingCurve(QEasingCurve.Type.InOutCubic)


class ImageInfoWidget(QWidget):
    """A widget showing image metadata details."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QGridLayout(self)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.setContentsMargins(8, 8, 8, 8)

        self.layout.setColumnStretch(0, 0)
        self.layout.setColumnStretch(1, 1)

        self.labels = {}
        fields = [
            ("Filename", "filename"),
            ("Location", "location"),
            ("Date", "date"),
            ("Camera model", "camera"),
            ("Lens model", "lens"),
            ("Focal length", "focal_length"),
            ("Resolution", "resolution"),
            ("File size", "file_size"),
            ("Aperture", "f_number"),
            ("ISO", "iso"),
            ("Shutter speed", "shutter_speed"),
        ]

        for i, (label_text, key) in enumerate(fields):
            title_label = QLabel(
                label_text + ":",
                alignment=(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            )
            title_label.setStyleSheet("font-weight: bold; color: gray;")

            val_label = QLabel(
                "-",
                alignment=(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            )
            val_label.setWordWrap(True)
            val_label.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
            )
            val_label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )

            self.layout.addWidget(title_label, i, 0)
            self.layout.addWidget(val_label, i, 1)
            self.labels[key] = val_label

    @pyqtSlot(dict, str, str)
    def update_info(self, metadata, cam_str, lens_str):
        if not metadata:
            for lbl in self.labels.values():
                lbl.setText("-")
            return

        self.labels["filename"].setText(str(metadata.get("File:FileName", "-")))
        self.labels["location"].setText(str(metadata.get("File:Directory", "-")))
        self.labels["date"].setText(
            str(
                metadata.get(
                    "EXIF:DateTimeOriginal",
                    metadata.get("File:FileModifyDate", "-"),
                )
            )
        )

        # Robust camera model resolution (handling spoofed/generic MODEL-NAME tags)
        resolved_cam = cam_str
        model = metadata.get("EXIF:Model", "").strip()
        make = metadata.get("EXIF:Make", "").strip()
        sony_model_id = metadata.get("MakerNotes:SonyModelID")

        sony_model_map = {
            388: "Sony a7 IV (ILCE-7M4)",
            389: "Sony ZV-1F",
            390: "Sony a7R V (ILCE-7RM5)",
            391: "Sony FX30 (ILME-FX30)",
            392: "Sony a9 III (ILCE-9M3)",
        }

        if not resolved_cam or resolved_cam == "None" or model == "MODEL-NAME":
            if sony_model_id in sony_model_map:
                resolved_cam = sony_model_map[sony_model_id]
            elif make and model and model != "MODEL-NAME":
                resolved_cam = f"{make} {model}"
            elif model and model != "MODEL-NAME":
                resolved_cam = model
            elif make:
                resolved_cam = make
            else:
                resolved_cam = "-"
        self.labels["camera"].setText(resolved_cam)

        # Robust lens model resolution
        resolved_lens = lens_str
        if not resolved_lens or resolved_lens == "None":
            lens_model = metadata.get("EXIF:LensModel", "").strip()
            lens_make = metadata.get("EXIF:LensMake", "").strip()
            lens_info = metadata.get("EXIF:LensInfo", "").strip()

            if lens_model:
                resolved_lens = (
                    f"{lens_make} {lens_model}".strip()
                    if lens_make and not lens_model.startswith(lens_make)
                    else lens_model
                )
            elif lens_info:
                resolved_lens = lens_info
            else:
                resolved_lens = "-"
        self.labels["lens"].setText(resolved_lens)

        fl = metadata.get("EXIF:FocalLength")
        self.labels["focal_length"].setText(f"{fl} mm" if fl else "-")

        res = (
            metadata.get("Composite:ImageSize")
            or f"{metadata.get('EXIF:ExifImageWidth', '')} "
            f"{metadata.get('EXIF:ExifImageHeight', '')}".strip()
        )
        self.labels["resolution"].setText(str(res) if res and res != " " else "-")

        fs = metadata.get("File:FileSize")
        if fs:
            try:
                fs_mb = int(fs) / (1024 * 1024)
                fs_str = f"{fs_mb:.1f} MB"
            except Exception:
                fs_str = str(fs)
        else:
            fs_str = "-"
        self.labels["file_size"].setText(fs_str)

        fnum = metadata.get("EXIF:FNumber")
        self.labels["f_number"].setText(f"f/{fnum}" if fnum else "-")

        iso = metadata.get("EXIF:ISO")
        self.labels["iso"].setText(str(iso) if iso else "-")

        ss = metadata.get("EXIF:ExposureTime")
        if ss:
            try:
                ss_val = float(ss)
                if ss_val < 1:
                    denom = round(1.0 / ss_val)
                    ss_str = f"1/{denom}s"
                else:
                    ss_str = f"{ss_val}s"
            except Exception:
                ss_str = str(ss)
        else:
            ss_str = "-"
        self.labels["shutter_speed"].setText(ss_str)
