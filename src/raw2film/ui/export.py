"""Main-window methods for export."""

import logging
import os
import shutil
import time
from functools import partial

from PIL import Image
from PyQt6.QtCore import (
    Qt,
    QThread,
    QTimer,
)
from PyQt6.QtGui import (
    QIntValidator,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QProgressDialog,
    QVBoxLayout,
)
from spectral_film_lut.gui_objects import (
    AnimatedButton,
    HoverLineEdit,
    Slider,
)

from raw2film.gui_objects import (
    CpuWorker,
    GpuWorker,
)
from raw2film.utils import add_metadata

logger = logging.getLogger(__name__)


class ExportMixin:
    """Single-image and batch export actions for the main window."""

    def _export_processed_image(
        self,
        image,
        task: tuple,
        metadata: dict,
        processing_args: dict,
        start_time: float,
        current_idx: int | None = None,
        total_count: int | None = None,
        **kwargs,
    ) -> str:
        """Centralize file system saving and metadata restoration."""
        src, filename = task
        src_short = src.split("/")[-1]

        add_year = kwargs.get("add_year", False)
        add_date = kwargs.get("add_date", False)
        move_raw = kwargs.get("move_raw", 0)
        quality = kwargs.get("quality", 100)
        close = kwargs.get("close", False)

        # 1. Determine relative folder directory paths
        path = "/".join(filename.split("/")[:-1])
        if path:
            path += "/"
        if add_year:
            path += metadata["EXIF:DateTimeOriginal"][:4] + "/"
        if add_date:
            path += metadata["EXIF:DateTimeOriginal"][:10].replace(":", "-") + "/"

        out_filename = filename.split("/")[-1]
        if "." not in out_filename:
            out_filename += ".jpg"

        # 2. Handle Directory generation
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except FileExistsError:
            pass

        if move_raw:
            raw_dir = os.path.join(path, "RAW")
            if not os.path.exists(raw_dir):
                os.makedirs(raw_dir)
            target_raw = os.path.join(raw_dir, src_short)
            if move_raw == 2:
                os.replace(src, target_raw)
            elif move_raw == 1 and not os.path.isfile(target_raw):
                shutil.copy2(src, target_raw)

        full_output_path = os.path.join(path, out_filename)
        img = Image.fromarray(image)
        img.save(full_output_path, "JPEG", quality=quality)
        add_metadata(self.et, full_output_path, metadata, exp_comp=processing_args["exp_comp"])

        if close:
            QTimer.singleShot(0, lambda: self.image_bar.close_single_image(src))

        logger.info(
            "exported %s in %.2fs",
            out_filename,
            time.time() - start_time,
        )
        if current_idx and total_count:
            return f"exported {out_filename} ({current_idx}/{total_count})"
        return f"exported {out_filename}"

    def save_image(self, src: str, filename: str, **kwargs):
        """Process and export one image to a JPEG file."""
        start = time.time()
        # Fallback to widget check if direct thread state snapshot isn't passed
        auto_lens_correct = (
            kwargs.get("_auto_lens_correct_state") or self.auto_lens_correct.isChecked()
        )

        processing_args, metadata = self._build_processing_params(
            src, auto_lens_correct, kwargs.get("resolution")
        )

        processor = self.gpu_processor if self.gpu_processing.isChecked() else self.cpu_processor

        image = processor.process(src, half_size=False, cache=False, **processing_args)

        return self._export_processed_image(
            image, (src, filename), metadata, processing_args, start, **kwargs
        )

    def save_image_dialog(self):
        """Prompt for an output name and export the selected image."""
        src = self.image_bar.current_image()
        if src is None:
            return
        filename = src.split("/")[-1].split(".")[0]
        filename, ok = QFileDialog.getSaveFileName(self, "Choose output file", filename, "*.jpg")
        if ok:
            self.start_worker(self.save_image, src=src, filename=filename, semaphore=False)

    def save_multiple_process(self, folder, filenames, **kwargs):
        """Configure and run a batch image export."""
        self.active = False
        tasks = [(f, folder + "/" + f.split("/")[-1].split(".")[0]) for f in filenames]

        # 1. Create the dialog
        self.progress_dialog = QProgressDialog(
            "Starting export...", "Cancel", 0, len(tasks), parent=self
        )
        self.progress_dialog.setWindowTitle("Export")
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
        QApplication.processEvents()

        use_gpu = self.gpu_processing.isChecked()
        kwargs["_auto_lens_correct_state"] = self.auto_lens_correct.isChecked()

        self.thread = QThread()

        if use_gpu:
            self.worker = GpuWorker()
            self.thread.started.connect(
                partial(
                    self.worker.run_tasks,
                    self.prepare_cpu_stage,
                    self.save_image_from_preloaded,
                    tasks,
                    **kwargs,
                )
            )
        else:
            self.worker = CpuWorker()
            self.thread.started.connect(
                partial(self.worker.run_tasks, self.execute_cpu_single_step, tasks, **kwargs)
            )

        self.worker.moveToThread(self.thread)
        self.progress_dialog.canceled.connect(
            self.worker.cancel, Qt.ConnectionType.DirectConnection
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.tasks_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def execute_cpu_single_step(self, task: tuple, current_idx: int, total_count: int, **kwargs):
        """Export one image using the CPU processing pipeline."""
        start = time.time()
        src, _filename = task
        auto_lens_correct = kwargs.get("_auto_lens_correct_state", False)

        processing_args, metadata = self._build_processing_params(
            src, auto_lens_correct, kwargs.get("resolution")
        )

        image = self.cpu_processor.process(src, half_size=False, cache=False, **processing_args)

        return self._export_processed_image(
            image,
            task,
            metadata,
            processing_args,
            start,
            current_idx,
            total_count,
            **kwargs,
        )

    def prepare_cpu_stage(self, src: str, **kwargs):
        """Prepare CPU-side data for a GPU batch export."""
        auto_lens_correct = kwargs.get("_auto_lens_correct_state", False)

        processing_args, metadata = self._build_processing_params(
            src, auto_lens_correct, kwargs.get("resolution")
        )
        cpu_payload = self.gpu_processor.extract_image_data_cpu(
            src, half_size=False, cache=False, **processing_args
        )

        return {
            "cpu_payload": cpu_payload,
            "processing_args": processing_args,
            "metadata": metadata,
        }

    def save_image_from_preloaded(
        self,
        task: tuple,
        pipeline_payload: dict,
        current_idx: int | None = None,
        total_count: int | None = None,
        **kwargs,
    ):
        """Export an image using preloaded GPU pipeline data."""
        start = time.time()

        # Unpack preloaded variables
        cpu_payload = pipeline_payload["cpu_payload"]
        processing_args = pipeline_payload["processing_args"]
        metadata = pipeline_payload["metadata"]

        image = self.gpu_processor.process_preloaded(cpu_payload, **processing_args)

        return self._export_processed_image(
            image,
            task,
            metadata,
            processing_args,
            start,
            current_idx,
            total_count,
            **kwargs,
        )

    def update_progress(self, message, value):
        """Update the batch export progress dialog."""
        self.progress_dialog.setLabelText(message)
        self.progress_dialog.setValue(value)

    def tasks_finished(self):
        """Restore UI state after all batch tasks finish."""
        self.active = True
        self.progress_dialog.close()
        if self.image_bar.current_image() is not None:
            self.load_image(self.image_bar.current_image())

    def save_image_setting_dialog(self):
        """Prompt for batch export settings."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Export settings")

        layout = QVBoxLayout()

        quality_slider = Slider(continuous=False)
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
        move_raw.setToolTip(
            "Checked: move file \nPartially checked: copy file\nUnchecked: do nothing to raw file"
        )
        layout.addWidget(move_raw)

        close_checkbox = QCheckBox("Close images after export")
        move_raw.stateChanged.connect(lambda x: close_checkbox.setEnabled(x != 2))
        move_raw.setChecked(True)
        close_checkbox.setChecked(True)
        layout.addWidget(close_checkbox)

        resolution_field = HoverLineEdit(parent=self)
        resolution_field.setValidator(QIntValidator())
        layout.addWidget(QLabel("Resolution:"))
        layout.addWidget(resolution_field)

        # Buttons
        button_layout = QHBoxLayout()
        ok_button = AnimatedButton("OK", parent=self)
        cancel_button = AnimatedButton("Cancel", parent=self)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)
        dialog.setLayout(layout)

        # Connect buttons
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        if dialog.exec():
            resolution = int(resolution_field.text()) if resolution_field.text() else None
            kwargs = {
                "move_raw": move_raw.checkState().value,
                "add_year": sort_by_year.isChecked(),
                "close": close_checkbox.isChecked() or move_raw.checkState().value == 2,
                "quality": int(quality_slider.getValue()),
                "add_date": sort_by_date.isChecked(),
                "resolution": resolution,
            }
            return True, kwargs
        else:
            return False, {}

    def save_all_images(self):
        """Prompt for an export folder and export every loaded image."""
        if not self.image_bar.get_all():
            return
        ok, kwargs = self.save_image_setting_dialog()

        if ok:
            folder = QFileDialog.getExistingDirectory(self)
            if folder:
                self.save_multiple_process(
                    folder=folder, filenames=self.image_bar.get_all(), **kwargs
                )

    def save_selected_images(self):
        """Prompt for an export folder and export every highlighted image."""
        if not self.image_bar.get_highlighted():
            return
        ok, kwargs = self.save_image_setting_dialog()

        if ok:
            folder = QFileDialog.getExistingDirectory(self)
            if folder:
                self.save_multiple_process(
                    folder=folder, filenames=self.image_bar.get_highlighted(), **kwargs
                )
