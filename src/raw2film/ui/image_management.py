"""Main-window methods for image management."""

import logging
import os
import time

from PyQt6.QtCore import (
    Qt,
)
from PyQt6.QtWidgets import (
    QFileDialog,
    QInputDialog,
    QMessageBox,
)

from raw2film import __version__, data, effects, utils
from raw2film.raw_conversion import raw_to_linear
from raw2film.utils import load_metadata

logger = logging.getLogger(__name__)


class ImageManagementMixin:
    """Image and film-profile interactions for the main window."""

    def reset_image(self):
        """Reset processing settings for every highlighted image."""
        for src in self.image_bar.get_highlighted():
            src_short = src.split("/")[-1]
            if src_short in self.image_params:
                self.image_params.pop(src_short)
        self.sync_thumbnail_settings()
        self.load_image(self.image_bar.selected_label.image_path)

    def reset_all_images(self):
        """Clear processing settings for all images."""
        self.image_params = {}
        self.sync_thumbnail_settings()
        if self.image_bar.selected_label is not None:
            self.load_image(self.image_bar.selected_label.image_path)

    def reset_profile(self):
        """Reset the currently selected processing profile."""
        profile = self.profile_selector.currentText()
        if profile in self.profile_params:
            self.profile_params[profile] = {}
        self.load_profile_params()

    def copy_settings(self, src):
        """Copy the source image settings to every highlighted image."""
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
        self.sync_thumbnail_settings()
        self.load_image(self.image_bar.selected_label.image_path)

    def delete_highlighted(self):
        """Delete every highlighted image after confirmation."""
        reply = QMessageBox()
        number_images = len(self.image_bar.get_highlighted())
        if number_images == 1:
            reply.setText("Delete 1 image permanently?")
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
        """Delete the selected profile after confirmation."""
        current_profile = self.profile_selector.currentText()
        reply = QMessageBox(self)
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
                    if (
                        "profile" in self.image_params[image]
                        and self.image_params[image]["profile"] == current_profile
                    ):
                        self.image_params[image]["profile"] = "Default"
            self.load_profile_params()

    def delete_all_profiles(self):
        """Delete all profiles except the default profile after confirmation."""
        reply = QMessageBox(self)
        profile_count = len(self.profile_params)
        if not profile_count:
            return
        if profile_count == 1:
            message = "Delete 1 profile permanently?"
        else:
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
                        if (
                            "profile" in self.image_params[image]
                            and self.image_params[image]["profile"] == profile
                        ):
                            self.image_params[image]["profile"] = "Default"
            self.load_profile_params()

    def add_profile_prompt(self):
        """Prompt for and add a new processing profile."""
        text, ok = QInputDialog.getText(self, "Add profile", "Profile name:")
        if ok:
            self.profile_params[text] = self.profile_params[
                self.profile_selector.currentText()
            ].copy()
            self.profile_selector.addItem(text)
            self.profile_selector.setCurrentText(text)

    def format_changed(self, film_format: str):
        """Update frame dimensions when a preset format is selected."""
        if film_format != "Custom" and not self.loading:
            width, height = data.FORMATS[film_format]
            self.frame_width.setText(str(width))
            self.frame_height.setText(str(height))

    def _trigger_initial_image_info(self):
        """Load metadata and emit image information for the selected image."""
        current_img = self.image_bar.current_image()
        if current_img:
            src_short = current_img.split("/")[-1]
            self._ensure_image_metadata(src_short, current_img)
            metadata = load_metadata(current_img)
            image_params = self.image_params.get(src_short, {})
            self.info_update.emit(metadata, image_params.get("cam"), image_params.get("lens"))

    def load_images(self):
        """Prompt for and load one or more raw image files."""
        filenames, ok = QFileDialog.getOpenFileNames(
            self,
            "Open raw images",
            "",
            filter=f"RAW (*{' *'.join(data.EXTENSION_LIST)})",
        )

        if ok:
            for folder in {"/".join(filename.split("/")[:-1]) for filename in filenames}:
                self.load_settings_directory(folder)
            self.image_bar.load_images(filenames)
            self.sync_thumbnail_settings()
            self._trigger_initial_image_info()

    def load_folder(self):
        """Prompt for and load every supported image in a folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select image folder", "")
        if folder:
            self.load_settings_directory(folder)
            filenames = [
                folder + "/" + filename
                for filename in os.listdir(folder)
                if filename.lower().endswith(data.EXTENSION_LIST)
            ]
            self.image_bar.load_images(filenames)
            self.sync_thumbnail_settings()
            self._trigger_initial_image_info()

    def load_image(self, src, **kwargs):
        """Load an image and switch the preview stack to it."""
        self.image_stack.setCurrentWidget(self.image)
        self.start_worker(self.load_image_process, src=src)

    def update_window_title(self, image_path: str | None = None):
        """Update the window title to show the currently selected image filename."""
        if image_path is None:
            self.setWindowTitle(f"Raw2Film {__version__}")
        else:
            filename = image_path.split("/")[-1]
            self.setWindowTitle(f"{filename} - Raw2Film {__version__}")

    def _ensure_image_metadata(self, src_short, src=None, force=False):
        image_params = self.image_params.setdefault(src_short, {})

        if image_params.get("_metadata_checked") and not force:
            return image_params

        needs_cam = force or image_params.get("cam") in (None, "", "None")
        needs_lens = force or image_params.get("lens") in (None, "", "None")

        if not (needs_cam or needs_lens):
            if "lens_correction" not in image_params:
                image_params["lens_correction"] = self.auto_lens_correct.isChecked()
            return image_params

        if src is None and hasattr(self, "image_bar"):
            current = self.image_bar.current_image()
            if current is not None and current.split("/")[-1] == src_short:
                src = current

        if src is not None:
            metadata = load_metadata(src)
            cam, lens = utils.find_data(metadata, self.lensfunpy_db)

            if needs_cam:
                image_params["cam"] = f"{cam.maker} {cam.model}" if cam else "None"
            if needs_lens:
                image_params["lens"] = lens.model if lens else "None"

            # Check for Fuji digital teleconverter zoom metadata
            if metadata.get("RAF:RawZoomActive"):
                zoom_size = metadata.get("RAF:RawZoomSize")
                cropped_size = metadata.get("RAF:RawImageCroppedSize") or metadata.get(
                    "RAF:RawImageFullSize"
                )
                if zoom_size and cropped_size:
                    try:
                        zw, _ = map(int, zoom_size.split())
                        bw, _ = map(int, cropped_size.split())
                        if zw > 0:
                            image_params["zoom"] = bw / zw
                    except Exception:
                        pass

        if "lens_correction" not in image_params:
            image_params["lens_correction"] = self.auto_lens_correct.isChecked()

        image_params["_metadata_checked"] = True

        return image_params

    def load_image_process(self, src, **kwargs):
        """Prepare and process a single image source."""
        src_short = src.split("/")[-1]
        self._ensure_image_metadata(src_short, src)
        if "profile" not in self.image_params[src_short]:
            self.image_params[src_short]["profile"] = self.profile_selector.currentText()
        if "exp_kelvin" not in self.image_params[src_short]:
            self.image_params[src_short]["exp_kelvin"] = self.exp_wb.getValue()
        self.load_image_params(src_short, src)

        # Emit info update for the image info widget
        metadata = load_metadata(src)
        image_params = self.image_params.get(src_short, {})
        self.info_update.emit(metadata, image_params.get("cam"), image_params.get("lens"))

        if self.active:
            self.update_preview(src)

    def _ensure_metadata_for_images(self, srcs, progress_callback=None, **kwargs):
        """Worker helper: ensure metadata is available for a list of full-path srcs.

        This runs in a background worker to avoid blocking the UI when
        extracting metadata with exiftool for multiple images at once.
        """
        for src in srcs:
            try:
                src_short = src.split("/")[-1]
                # Force=False: _ensure_image_metadata will skip work if already done
                self._ensure_image_metadata(src_short, src)
            except Exception:
                # Swallow exceptions to avoid crashing the worker thread; nothing
                # special to do on failure here.
                pass

    def load_raw_image(self, src, cam=None, lens=None):
        """Load and optionally lens-correct one raw image."""
        cache_key = (src, cam, lens)
        if cache_key not in self._raw_image_cache:
            image = raw_to_linear(src)

            if cam is not None and lens is not None:
                cam = self.cameras[cam]
                lens = self.lenses[lens]

                image = effects.lens_correction(image, load_metadata(src), cam, lens)

            self._raw_image_cache[cache_key] = image
            if len(self._raw_image_cache) > 32:
                del self._raw_image_cache[next(iter(self._raw_image_cache))]
        return self._raw_image_cache[cache_key]

    def changed_wb_mode(self, mode):
        """Apply a white-balance preset or enable custom adjustment."""
        if mode != "Custom":
            self.exp_wb.setValue(self.wb_modes[mode])
            self.parameter_changed()

    def changed_negative(self, negative):
        """Update UI and profile values when the negative stock changes."""
        self.active = False
        self.print_selector.setEnabled(negative not in self.reversal_stocks)
        if self.print_selector.isEnabled():
            self.profile_changed(self.print_selector.currentText(), "print_film")
        else:
            self.profile_changed(None, "print_film")
        self.active = True
        self.profile_changed(negative, "negative_film")

    def profile_changed(self, value, key):
        """Store a profile value and trigger preview reprocessing."""
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
                    format_name = list(data.FORMATS.keys())[
                        list(data.FORMATS.values()).index(dimensions)
                    ]
                    self.format_selector.setCurrentText(format_name)
                else:
                    self.format_selector.setCurrentText("Custom")
        profile = self.profile_selector.currentText()
        if profile not in self.profile_params:
            self.profile_params[profile] = {}
        self.profile_params[profile][key] = value
        self.parameter_changed()

    def quick_save(self):
        """Persist settings and reset the save timer."""
        self.save_settings_system()
        self.save_timer = time.time()

    def sync_thumbnail_settings(self):
        """Update thumbnails to indicate which images have custom settings."""
        if hasattr(self, "image_bar"):
            current_settings = frozenset(self.image_params.keys())
            if getattr(self, "_thumbnail_settings_images", None) == current_settings:
                return
            self._thumbnail_settings_images = current_settings
            self.image_bar.set_settings_images(current_settings)

    def setting_changed(self, value, key):
        """Store a changed setting for the current profile or image."""
        if self.loading:
            return
        current_image = self.image_bar.current_image()
        if current_image is not None:
            current_image = current_image.split("/")[-1]
            if current_image in self.image_params:
                if (
                    key in self.image_params[current_image]
                    and value == self.image_params[current_image][key]
                ):
                    return  # no change
            elif (key in self.dflt_img_params and self.dflt_img_params[key] == value) or (
                key in self.dflt_prf_params and self.dflt_prf_params[key] == value
            ):
                return  # default value and nothing to overwrite
        if time.time() - self.save_timer > 10:
            self.quick_save()

        # Prepare list of images that need metadata extraction. Extracting
        # metadata (via exiftool) can be slow on the first call, so do that in
        # a background worker instead of blocking the UI here.
        to_metadata = []
        for src in self.image_bar.get_highlighted():
            src_short = src.split("/")[-1]
            if src_short not in self.image_params:
                self.image_params[src_short] = {}
            ip = self.image_params[src_short]

            # Determine whether metadata (cam/lens) likely needs fetching.
            needs_meta = (
                not ip.get("_metadata_checked") and ip.get("cam") in (None, "", "None")
            ) or (not ip.get("_metadata_checked") and ip.get("lens") in (None, "", "None"))
            if needs_meta:
                to_metadata.append(src)

            if "profile" not in ip:
                ip["profile"] = self.profile_selector.currentText()
            if "exp_kelvin" not in ip:
                ip["exp_kelvin"] = self.exp_wb.getValue()
            ip[key] = value

        if to_metadata:
            # Run metadata extraction asynchronously so the UI remains
            # responsive. Do not semaphore this worker.
            self.start_worker(self._ensure_metadata_for_images, semaphore=False, srcs=to_metadata)
        if key == "exp_kelvin":
            self.update_wb_mode(value)
        self.sync_thumbnail_settings()
        self.parameter_changed()

    def update_wb_mode(self, value):
        """Select a preset or custom mode for a Kelvin white-balance value."""
        value = round(value, -2)
        if value in self.wb_modes.values():
            self.wb_mode.setCurrentText(
                list(self.wb_modes.keys())[list(self.wb_modes.values()).index(value)]
            )
        else:
            self.wb_mode.setCurrentText("Custom")

    def setup_profile_params(self, profile, src=None):
        """Merge profile values with the default processing parameters."""
        if profile in self.profile_params:
            return {**self.dflt_prf_params, **self.profile_params[profile]}
        else:
            return self.dflt_prf_params

    def load_image_params(self, src, full_src=None):
        """Load resolved image parameters and emit them to the UI."""
        image_params = self.setup_image_params(src, full_src)
        self.ui_update.emit(image_params)

    def load_image_params_to_ui(self, image_params):
        """Populate widgets from a resolved image parameter dictionary."""

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
        set_safely(self.exp_wb, "setValue", "exp_kelvin")
        set_safely(self.tint, "setValue", "tint")

        set_safely(self, "update_wb_mode", "exp_kelvin")

        set_safely(self.lens_correction, "setChecked", "lens_correction")
        set_safely(self.canvas_mode, "setCurrentText", "canvas_mode")
        set_safely(self.canvas_scale, "setValue", "canvas_scale")

        set_safely(self.camera_selector, "setCurrentText", "cam")
        set_safely(self.lens_selector, "setCurrentText", "lens")

        set_safely(self.highlight_burn, "setValue", "highlight_burn")
        set_safely(self.burn_scale, "setValue", "burn_scale")

        set_safely(self.chroma_nr, "setValue", "chroma_nr")

        if "profile" in image_params:
            self.profile_selector.setCurrentText(image_params["profile"])

        self.sync_thumbnail_settings()

    def load_profile_params(self, profile=None):
        """Load profile values into the processing controls."""
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
        self.sharpening_strength.setValue(profile_params["sharpening_strength"])
        self.sharpening_sigma.setValue(profile_params["sharpening_sigma"])
        self.grain.setCheckState(Qt.CheckState(profile_params["grain"]))
        if "frame_width" in profile_params:
            self.frame_width.setText(str(profile_params["frame_width"]))
        if "frame_height" in profile_params:
            self.frame_height.setText(str(profile_params["frame_height"]))
        if "frame_width" in profile_params and "frame_height" in profile_params:
            dimensions = (profile_params["frame_width"], profile_params["frame_height"])
            if dimensions in data.FORMATS.values():
                format_name = list(data.FORMATS.keys())[
                    list(data.FORMATS.values()).index(dimensions)
                ]
                self.format_selector.setCurrentText(format_name)
        self.grain_size.setValue(profile_params["grain_size"])
        self.grain_intensity.setValue(profile_params.get("grain_intensity", 1))
        self.grain_sigma.setValue(profile_params["grain_sigma"])
        self.negative_selector.setCurrentText(profile_params["negative_film"])
        self.print_selector.setCurrentText(profile_params["print_film"])
        self.shadow_comp.setValue(profile_params["shadow_comp"])
        self.saturation_slider.setValue(profile_params["sat_adjust"])
        self.color_masking.setValue(profile_params["color_masking"])
        self.dye_inhibition.setValue(profile_params["dye_inhibition"])
        self.inversion_gamma.setValue(profile_params["inversion_gamma"])
        self.idealized_curve.setChecked(profile_params["idealized_curve"])
        self.white_clip.setChecked(profile_params["white_clip"])
        self.white_balance.setCheckState(Qt.CheckState(profile_params["white_balance"]))
        self.push_pull.setValue(profile_params["push_pull"])

        if "projector_kelvin" in profile_params:
            self.projector_kelvin.setValue(profile_params["projector_kelvin"])
        self.loading = False
        self.active = True
        self.parameter_changed()
