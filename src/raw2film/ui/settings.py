"""Main-window methods for settings."""

import contextlib
import json
import logging
from pathlib import Path

import PIL.ImageCms
from PIL import ImageCms
from PyQt6.QtCore import (
    QTimer,
)
from PyQt6.QtWidgets import (
    QFileDialog,
    QMessageBox,
)

from raw2film import utils
from raw2film.utils import load_metadata

logger = logging.getLogger(__name__)


class SettingsMixin:
    """Application, image, profile, and display settings for the main window."""

    def save_settings_dialogue(self):
        """Prompt for a settings file name and save it."""
        filename, ok = QFileDialog.getSaveFileName(
            self, "Select file name", "raw2film_settings.json", "*.json"
        )
        if ok:
            self.save_settings(filename)

    def save_settings_directory(self, root="", src=None, **kwargs):
        """Save settings into the requested directory."""
        filename = root + "/raw2film_settings.json" if root else "raw2film_settings.json"
        self.save_settings(filename, src)

    def save_settings(self, filename, src=None):
        """Write complete or image-specific settings to JSON."""
        if src is None:
            complete_dict = {
                "image_params": self.image_params,
                "profile_params": self.profile_params,
            }
        else:
            if (
                "profile" in self.image_params[src]
                and self.image_params[src]["profile"] in self.profile_params
            ):
                profile = self.image_params[src]["profile"]
                complete_dict = {
                    "image_params": {src: self.image_params[src]},
                    "profile_params": {profile: self.profile_params[profile]},
                }
            else:
                complete_dict = {
                    "image_params": {src: self.image_params[src]},
                    "profile_params": {},
                }
        if Path(filename).is_file():
            with open(filename) as f:
                old_dict = json.load(f)
            complete_dict["image_params"] = {
                **old_dict["image_params"],
                **complete_dict["image_params"],
            }
            complete_dict["profile_params"] = {
                **old_dict["profile_params"],
                **complete_dict["profile_params"],
            }
        with open(filename, "w") as f:
            json.dump(complete_dict, f)

    def save_settings_system(self):
        """Persist profile and image settings in application QSettings."""
        self.settings.setValue("profile_params", json.dumps(self.profile_params))
        self.settings.setValue(
            "image_params",
            json.dumps(self._strip_internal_image_params(self.image_params)),
        )

    def load_settings_system(self):
        """Restore profile and image settings from persistent QSettings."""
        self.profile_params = json.loads(self.settings.value("profile_params", "{}"))
        self.image_params = json.loads(self.settings.value("image_params", "{}"))
        for params in self.image_params.values():
            params.pop("_metadata_checked", None)

        for profile in self.profile_params:
            if self.profile_selector.findText(profile) == -1:
                self.profile_selector.addItem(profile)
        self.sync_thumbnail_settings()

    def load_settings_dialogue(self):
        """Prompt for a settings file and load it."""
        filename, ok = QFileDialog.getOpenFileName(self)
        if ok:
            self.load_settings(filename)

    def load_settings_directory(self, root=""):
        """Load settings from raw2film_settings.json in a directory."""
        filename = root + "/raw2film_settings.json" if root else "raw2film_settings.json"
        if Path(filename).is_file():
            self.load_settings(filename)

    def load_settings(self, filename):
        """Merge saved settings from a JSON file into the current session."""
        with open(filename) as f:
            complete_dict = json.load(f)
        self.image_params = {**complete_dict["image_params"], **self.image_params}
        self.profile_params = {**complete_dict["profile_params"], **self.profile_params}
        for params in self.image_params.values():
            params.pop("_metadata_checked", None)
        for profile in self.profile_params:
            if self.profile_selector.findText(profile) == -1:
                self.profile_selector.addItem(profile)
        self.sync_thumbnail_settings()

    def _strip_internal_image_params(self, image_params):
        """Remove runtime-only image metadata before saving settings."""
        return {
            key: {k: v for k, v in params.items() if k != "_metadata_checked"}
            for key, params in image_params.items()
        }

    def light_changed(self, value, light_name):
        """Apply a changed print-light value to linked or individual lights."""
        if self.loading:
            return
        if self.link_lights.isChecked():
            self.loading = True
            self.red_light.setValue(value)
            self.green_light.setValue(value)
            self.blue_light.setValue(value)
            self.loading = False
            self.active = False
            self.profile_changed(value, "red_light")
            self.profile_changed(value, "green_light")
            self.profile_changed(value, "blue_light")
            self.active = True
            self.parameter_changed()
        else:
            self.profile_changed(value, light_name)

    def rotate_image(self, direction=1):
        """Rotate the image clockwise or counterclockwise."""
        self.rotate_times = (self.rotate_times - direction) % 4
        self.setting_changed(self.rotate_times, "rotate_times")

    def flip_image(self):
        """Toggle horizontal image flipping."""
        self.flip = not self.flip
        self.setting_changed(self.flip, "flip")

    def load_view_settings(self):
        """Restore persisted preview, ICC, and lens settings."""
        display_icc_path = self.settings.value("display_icc", None)
        softproof_icc_path = self.settings.value("softproof_icc", None)
        display_intent = self.settings.value("display_rendering_intent", "relative")
        softproof_intent = self.settings.value("softproof_rendering_intent", "absolute")
        info_val = self.settings.value("image_info_visible", None)

        self.set_display_intent(display_intent, False)
        self.set_softproof_intent(softproof_intent, False)
        if display_icc_path is not None:
            self.load_display_icc(display_icc_path)
        if softproof_icc_path is not None:
            self.load_softproof_icc(softproof_icc_path)

        def _to_bool(v):
            if v is None:
                return False
            return str(v).lower() in ("1", "true", "yes")

        gpu_val = self.settings.value("gpu_processing", None)
        half_val = self.settings.value("half_res_preview", None)
        full_val = self.settings.value("full_preview", None)

        if info_val is not None:
            is_visible = _to_bool(info_val)
            self.image_info_action.setChecked(is_visible)
            self.info_container.setVisible(is_visible)
        else:
            self.image_info_action.setChecked(False)
            self.info_container.hide()

        if gpu_val is not None:
            self.gpu_processing.setChecked(_to_bool(gpu_val))
        if half_val is not None:
            self.half_res_preview.setChecked(_to_bool(half_val))
        if full_val is not None:
            self.full_preview.setChecked(_to_bool(full_val))
        auto_lens_val = self.settings.value("auto_lens_correct", None)
        if auto_lens_val is not None:
            self.auto_lens_correct.setChecked(_to_bool(auto_lens_val))

        QTimer.singleShot(0, self._update_splitter_sizes)

        with contextlib.suppress(Exception):
            self.create_context()

    def load_display_icc_dialog(self):
        """Prompt for and load a display ICC profile."""
        icc_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select display ICC profile",
            "",
            "ICC Profile (*.icc *.icm)",
        )
        if icc_path:
            self.load_display_icc(icc_path)
            self.parameter_changed()
        else:
            self.load_display_icc_button.setChecked(not self.load_display_icc_button.isChecked())

    def load_display_icc(self, icc_path):
        """Configure a display ICC profile and rebuild the transform."""
        self.display_icc_path = icc_path

        self.load_display_icc_button.setToolTip(icc_path)
        self.reset_display_icc_button.setVisible(True)
        self.load_display_icc_button.setChecked(True)
        self.settings.setValue("display_icc", self.display_icc_path)

        self.build_icc_transform()

    def reset_display_icc(self):
        """Clear the display ICC profile and rebuild the transform."""
        self.display_icc_path = None

        self.reset_display_icc_button.setVisible(False)
        self.load_display_icc_button.setChecked(False)
        self.load_display_icc_button.setToolTip("")
        self.settings.remove("display_icc")

        self.build_icc_transform()

        self.parameter_changed()

    def load_softproof_icc_dialog(self):
        """Prompt for and load a soft-proof ICC profile."""
        icc_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ICC profile for soft proofing",
            "",
            "ICC Profile (*.icc *.icm)",
        )
        if icc_path:
            self.load_softproof_icc(icc_path)
            self.parameter_changed()
        else:
            self.load_softproof_icc_button.setChecked(
                not self.load_softproof_icc_button.isChecked()
            )

    def load_softproof_icc(self, icc_path):
        """Configure a soft-proof ICC profile and rebuild the transform."""
        self.softproof_icc_path = icc_path

        self.load_softproof_icc_button.setToolTip(icc_path)
        self.load_softproof_icc_button.setChecked(True)
        self.reset_softproof_icc_button.setVisible(True)
        self.settings.setValue("softproof_icc", self.display_icc_path)

        self.build_icc_transform()

    def reset_softproof_icc(self):
        """Clear the soft-proof ICC profile and rebuild the transform."""
        self.softproof_icc_path = None

        self.load_softproof_icc_button.setChecked(False)
        self.reset_softproof_icc_button.setVisible(False)
        self.load_softproof_icc_button.setToolTip("")
        self.settings.remove("softproof_icc")

        self.build_icc_transform()

        self.parameter_changed()

    def build_icc_transform(self):
        """Build or clear the display and soft-proof ICC transform."""
        flags = ImageCms.Flags.BLACKPOINTCOMPENSATION if self.icc_bpc else 0
        try:
            if self.display_icc_path:
                if self.softproof_icc_path:
                    flags |= ImageCms.Flags.SOFTPROOFING
                    self.icc_transform = ImageCms.buildProofTransform(
                        self.srgb_profile,
                        self.display_icc_path,
                        self.softproof_icc_path,
                        "RGB",
                        "RGB",
                        renderingIntent=self.display_intent,
                        proofRenderingIntent=self.softproof_intent,
                        flags=flags,
                    )
                else:
                    self.icc_transform = ImageCms.buildTransform(
                        self.srgb_profile,
                        self.display_icc_path,
                        "RGB",
                        "RGB",
                        renderingIntent=self.display_intent,
                        flags=flags,
                    )
            elif self.softproof_icc_path:
                flags |= ImageCms.Flags.SOFTPROOFING
                self.icc_transform = ImageCms.buildProofTransform(
                    self.srgb_profile,
                    self.srgb_profile,
                    self.softproof_icc_path,
                    "RGB",
                    "RGB",
                    renderingIntent=self.display_intent,
                    proofRenderingIntent=self.softproof_intent,
                    flags=flags,
                )
            else:
                self.icc_transform = None
        except PIL.ImageCms.PyCMSError:
            self.reset_display_icc()
            self.reset_softproof_icc()
            QTimer.singleShot(0, self.icc_loading_warning)

    def set_display_intent(self, intent, build_transform=True):
        """Apply a display rendering intent."""
        self.display_absolute_intent.setChecked(intent == "absolute")
        self.display_relative_intent.setChecked(intent == "relative")
        self.display_relative_bpc_intent.setChecked(intent == "relative_bpc")
        self.display_perceptual_intent.setChecked(intent == "perceptual")
        self.display_saturation_intent.setChecked(intent == "saturation")

        self.display_intent = {
            "absolute": ImageCms.Intent.ABSOLUTE_COLORIMETRIC,
            "relative": ImageCms.Intent.RELATIVE_COLORIMETRIC,
            "relative_bpc": ImageCms.Intent.RELATIVE_COLORIMETRIC,
            "perceptual": ImageCms.Intent.PERCEPTUAL,
            "saturation": ImageCms.Intent.SATURATION,
        }[intent]

        self.icc_bpc = intent == "relative_bpc"

        self.settings.setValue("display_rendering_intent", intent)

        if build_transform:
            self.build_icc_transform()

            self.parameter_changed()

    def set_softproof_intent(self, intent, build_transform=True):
        """Apply a soft-proof rendering intent."""
        self.softproof_absolute_intent.setChecked(intent == "absolute")
        self.softproof_relative_intent.setChecked(intent == "relative")
        self.softproof_perceptual_intent.setChecked(intent == "perceptual")
        self.softproof_saturation_intent.setChecked(intent == "saturation")

        self.softproof_intent = {
            "absolute": ImageCms.Intent.ABSOLUTE_COLORIMETRIC,
            "relative": ImageCms.Intent.RELATIVE_COLORIMETRIC,
            "perceptual": ImageCms.Intent.PERCEPTUAL,
            "saturation": ImageCms.Intent.SATURATION,
        }[intent]

        self.settings.setValue("softproof_rendering_intent", intent)

        if build_transform:
            self.build_icc_transform()

            self.parameter_changed()

    def toggle_gpu_processing(self, checked):
        """Persist GPU processing and recreate the rendering context.

        Recreate the context so the GPU setting takes effect immediately.
        """
        self.settings.setValue("gpu_processing", "1" if checked else "0")
        with contextlib.suppress(Exception):
            self.create_context()
        self.parameter_changed()

    def toggle_half_res_preview(self, checked):
        """Persist half-resolution preview choice and update the preview."""
        self.settings.setValue("half_res_preview", "1" if checked else "0")
        self.parameter_changed()

    def toggle_full_preview(self, checked):
        """Persist full-preview choice and update the preview.

        The full preview action previously only triggered a parameter update
        but did not persist the user choice across sessions. Store the
        state in QSettings so it can be restored on startup.
        """
        self.settings.setValue("full_preview", "1" if checked else "0")
        self.parameter_changed()

    def toggle_auto_lens_correction(self, checked):
        """Persist whether to auto-toggle lens correction for newly loaded images."""
        with contextlib.suppress(Exception):
            self.settings.setValue("auto_lens_correct", "1" if checked else "0")

    def reload_lens_cam_data(self):
        """Enable lens correction and reload camera and lens metadata.

        Update image parameters and trigger a re-render.
        """
        current = self.image_bar.current_image()
        if current is None:
            return
        src = current
        src_short = src.split("/")[-1]

        # Ensure an entry exists for this image
        if src_short not in self.image_params:
            self.image_params[src_short] = {}

        # Load metadata and try to find camera/lens
        metadata = load_metadata(src)
        cam, lens = utils.find_data(metadata, self.lensfunpy_db)

        if cam is not None:
            self.image_params[src_short]["cam"] = cam.maker + " " + cam.model
        else:
            self.image_params[src_short]["cam"] = "None"

        if lens is not None:
            self.image_params[src_short]["lens"] = lens.model
        else:
            self.image_params[src_short]["lens"] = "None"

        # Activate lens correction for this image and update UI
        self.image_params[src_short]["lens_correction"] = True
        self.lens_correction.setChecked(True)

        # Clear any cached raw load that might depend on previous lens/cam
        self._raw_image_cache.clear()

        # Refresh UI and reprocess
        self.load_image_params(src_short, src)
        self.parameter_changed(src)

    def icc_loading_warning(self):
        """Warn that saved ICC profiles could not be restored."""
        QMessageBox.information(
            self,
            "ICC loading failed",
            "The ICC profiles could not be restored from last session. They have been reset.",
        )
