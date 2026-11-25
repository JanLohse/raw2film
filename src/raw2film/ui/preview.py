"""Main-window methods for preview."""

import logging

import numpy as np
from PyQt6.QtCore import (
    QTimer,
)
from spectral_film_lut.gui_objects import (
    Worker,
)

from raw2film import utils
from raw2film.utils import generate_histogram, load_metadata

logger = logging.getLogger(__name__)


class PreviewMixin:
    """Preview rendering and worker coordination for the main window."""

    def lights_changed(self, value):
        """Update linked or independent color-light controls."""
        if self.link_lights.isChecked():
            if (
                value
                == self.red_light.getPosition()
                == self.green_light.getPosition()
                == self.blue_light.getPosition()
            ):
                self.parameter_changed()
            else:
                self.red_light.setPosition(value)
                self.green_light.setPosition(value)
                self.blue_light.setPosition(value)
        else:
            self.parameter_changed()

    def print_output(self, s):
        """Handle an output status callback."""
        return

    def update_finished(self):
        """Restart queued preview work after a worker finishes."""
        self.running = False
        if self.waiting:
            self.waiting = False
            self.create_context()
            self.start_worker(self.update_preview)

    def progress_fn(self, n):
        """Accept a worker progress update."""
        return

    def parameter_changed(self, src=None):
        """Schedule preview reprocessing after a parameter change."""
        if self.active:
            self.create_context()
            self.start_worker(self.update_preview, src=src)

    def start_worker(self, function, semaphore=True, *args, **kwargs):
        """Start a QRunnable while preventing duplicate processing."""
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

    def numpy_to_canvas(
        self,
        image: np.ndarray,
        full_height: int,
        full_width: int,
    ):
        """Render an RGB image into a bitmap-sized preview canvas."""
        image = np.ascontiguousarray(image)

        image_height, image_width = image.shape[:2]

        canvas_ratio = full_width / full_height
        image_ratio = image_width / image_height

        if canvas_ratio > image_ratio:
            canvas_height = image_height
            canvas_width = round(image_height * canvas_ratio)
        else:
            canvas_width = image_width
            canvas_height = round(image_width / canvas_ratio)

        y_offset = (canvas_height - image_height) // 2
        x_offset = (canvas_width - image_width) // 2

        canvas_rgba = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

        canvas_rgba[y_offset : y_offset + image_height, x_offset : x_offset + image_width, 0:3] = (
            image
        )

        canvas_rgba[y_offset : y_offset + image_height, x_offset : x_offset + image_width, 3] = 255

        self.image_context.set_bitmap(canvas_rgba)

    def update_preview(self, src=None, *args, **kwargs):
        """Process and draw the selected or explicitly supplied image."""
        if src is None:
            if self.image_bar.current_image() is None:
                return
            else:
                src = self.image_bar.current_image()
        src_short = src.split("/")[-1]
        if src_short not in self.image_params:
            self.load_image_process(src)
            return
        self._ensure_image_metadata(src_short, src)

        full_width, full_height = self.image_context.physical_size

        image_args = self.setup_image_params(src_short, src)
        profile_args = self.setup_profile_params(image_args["profile"], src)
        processing_args = {**self.dflt_prf_params, **image_args, **profile_args}

        processing_args["negative_film"] = self.filmstocks[processing_args["negative_film"]]
        if "print_film" in processing_args and processing_args["print_film"] is not None:
            if processing_args["print_film"].startswith("Inversion"):
                processing_args["inversion"] = True
                processing_args["custom_inversion"] = "ACES" not in processing_args["print_film"]
            processing_args["print_film"] = self.filmstocks[processing_args["print_film"]]

        processing_args["resolution"] = (
            full_height,
            full_width,
        )

        if self.half_res_preview.isChecked():
            processing_args["resolution"] = tuple(x // 2 for x in processing_args["resolution"])
        if not self.full_preview.isChecked():
            processing_args["sharpness"] = False
            processing_args["grain"] = False
            processing_args["halation"] = False
        try:
            if self.context_mode == "wgpu":
                self.gpu_processor.process(
                    src,
                    icc_transform=self.icc_transform,
                    dst_texture=self.image_context.get_current_texture(),
                    histogram_texture=self.histogram_context.get_current_texture(),
                    **processing_args,
                )
            else:
                image = self.cpu_processor.process(
                    src,
                    icc_transform=self.icc_transform,
                    **processing_args,
                )
                histogram = generate_histogram(image, height=80)
                self.histogram_context.set_bitmap(histogram)
                _img_height, _img_width, _ = image.shape
                self.numpy_to_canvas(image, full_height, full_width)
        except AttributeError:
            QTimer.singleShot(100, lambda: self.update_preview(src))

        self.histogram.request_draw()
        self.image.request_draw()
        self.image.setToolTip(src)

    def setup_image_params(self, src, full_src=None):
        """Merge default and saved parameters for an image."""
        src_short = src.split("/")[-1]
        image_params = {**self.dflt_img_params, **self.image_params[src_short]}

        return image_params

    def _build_processing_params(
        self, src: str, auto_lens_correct: bool, resolution: int | None
    ) -> tuple[dict, dict]:
        """Centralized parameter setup and validation logic."""
        src_short = src.split("/")[-1]

        # Parameter extraction
        image_args = (
            self.setup_image_params(src_short, src)
            if src_short in self.image_params
            else self.dflt_img_params
        )
        profile_args = self.setup_profile_params(image_args["profile"], src)
        processing_args = {**self.dflt_prf_params, **image_args, **profile_args}

        # 2. Film stock resolution mappings
        processing_args["negative_film"] = self.filmstocks[processing_args["negative_film"]]
        if processing_args.get("print_film") is not None:
            if processing_args["print_film"].startswith("Inversion"):
                processing_args["inversion"] = True
                processing_args["custom_inversion"] = "ACES" not in processing_args["print_film"]
            processing_args["print_film"] = self.filmstocks[processing_args["print_film"]]

        if resolution is not None:
            processing_args["resolution"] = (resolution, resolution)

        metadata = load_metadata(src)

        # 3. Lensfun DB Correction mapping
        if (
            processing_args.get("lens_correction")
            or ("lens_correction" not in processing_args and auto_lens_correct)
        ) and ("cam" not in processing_args or "lens" not in processing_args):
            cam, lens = utils.find_data(metadata, self.lensfunpy_db)
            if cam or lens:
                processing_args["cam"] = f"{cam.maker} {cam.model}"
                processing_args["lens"] = lens.model

        # 4. Global parameter adjustments (Safe check configuration)
        processing_args["chroma_nr"] = processing_args.get("chroma_nr", 0) * 2

        return processing_args, metadata
