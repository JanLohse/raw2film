"""Connect the main window controls to their handlers."""

from PyQt6.QtGui import (
    QKeySequence,
    QShortcut,
)


class SignalSetupMixin:
    """Connect the main window controls to their handlers."""

    def _connect_controls(self):
        # Helper Method
        def create_shortcut(key_sequence: str, func, name: str | None = None):
            shortcut = QShortcut(QKeySequence(key_sequence), self)
            if name is not None:
                shortcut.setObjectName(name)
            shortcut.activated.connect(func)

        # Exposure & Image Manipulation
        create_shortcut("Up", self.exp_comp.increase, "Increase exposure compensation")
        create_shortcut("Down", self.exp_comp.decrease, "Decrease exposure compensation")
        create_shortcut("Ctrl+Right", self.rotation.increase, "Rotate right")
        create_shortcut("Ctrl+Left", self.rotation.decrease, "Rotate left")
        create_shortcut("Shift+Up", self.highlight_burn.increase, "Increase highlight-burn")
        create_shortcut("Shift+Down", self.highlight_burn.decrease, "Decrease highlight-burn")
        create_shortcut("Ctrl+R", self.rotate_image, "Rotate image 90°")

        #  Zoom Controls
        create_shortcut("Ctrl++", lambda: self.zoom.increase(5), "Zoom in")
        create_shortcut("Ctrl+-", lambda: self.zoom.decrease(5), "Zoom out")
        create_shortcut("Shift+Ctrl++", self.zoom.increase, "Zoom in (fine)")
        create_shortcut("Shift+Ctrl+-", self.zoom.decrease, "Zoom out (fine)")

        #  Profile Selection (Dynamic Loop for 1-9)
        for i in range(1, 10):
            create_shortcut(
                str(i),
                lambda idx=i - 1: self.profile_selector.setCurrentIndex(
                    min(idx, self.profile_selector.count() - 1)
                ),
                f"Select profile slot {i}",
            )

        create_shortcut(
            "Shift+D",
            lambda: self.wb_mode.setCurrentText("Daylight"),
            "Set white balance: Daylight",
        )
        create_shortcut(
            "Shift+C",
            lambda: self.wb_mode.setCurrentText("Cloudy"),
            "Set white balance: Cloudy",
        )
        create_shortcut(
            "Shift+S",
            lambda: self.wb_mode.setCurrentText("Shade"),
            "Set white balance: Shade",
        )
        create_shortcut(
            "Shift+T",
            lambda: self.wb_mode.setCurrentText("Tungsten"),
            "Set white balance: Tungsten",
        )
        create_shortcut(
            "Shift+F",
            lambda: self.wb_mode.setCurrentText("Fluorescent"),
            "Set white balance: Fluorescent",
        )

        self.negative_selector.currentTextChanged.connect(self.changed_negative)
        self.print_selector.currentTextChanged.connect(
            lambda x: self.profile_changed(x, "print_film")
        )
        self.image_selector.triggered.connect(self.load_images)
        self.folder_selector.triggered.connect(self.load_folder)
        self.push_pull.valueChanged.connect(lambda x: self.profile_changed(x, "push_pull"))
        self.projector_kelvin.valueChanged.connect(
            lambda x: self.profile_changed(x, "projector_kelvin")
        )
        self.inversion_gamma.valueChanged.connect(
            lambda x: self.profile_changed(x, "inversion_gamma")
        )
        self.idealized_curve.stateChanged.connect(
            lambda x: self.profile_changed(x, "idealized_curve")
        )
        self.white_clip.stateChanged.connect(lambda x: self.profile_changed(x, "white_clip"))
        self.white_balance.stateChanged.connect(lambda x: self.profile_changed(x, "white_balance"))
        self.exp_comp.valueChanged.connect(lambda x: self.setting_changed(x, "exp_comp"))
        self.wb_mode.currentTextChanged.connect(self.changed_wb_mode)
        self.exp_wb.valueChanged.connect(lambda x: self.setting_changed(x, "exp_kelvin"))
        self.tint.valueChanged.connect(lambda x: self.setting_changed(x, "tint"))
        self.red_light.valueChanged.connect(lambda x: self.light_changed(x, "red_light"))
        self.green_light.valueChanged.connect(lambda x: self.light_changed(x, "green_light"))
        self.blue_light.valueChanged.connect(lambda x: self.light_changed(x, "blue_light"))
        self.lens_correction.stateChanged.connect(
            lambda x: self.setting_changed(x, "lens_correction")
        )
        self.auto_lens_correct.triggered.connect(self.toggle_auto_lens_correction)
        self.full_preview.triggered.connect(self.toggle_full_preview)
        self.gpu_processing.triggered.connect(self.toggle_gpu_processing)
        self.half_res_preview.triggered.connect(self.toggle_half_res_preview)
        self.halation.stateChanged.connect(lambda x: self.profile_changed(x, "halation"))
        self.sharpness.stateChanged.connect(lambda x: self.profile_changed(x, "sharpness"))
        # Profile-bind sharpening parameters
        self.sharpening_strength.valueChanged.connect(
            lambda x: self.profile_changed(x, "sharpening_strength")
        )
        self.sharpening_sigma.valueChanged.connect(
            lambda x: self.profile_changed(x, "sharpening_sigma")
        )
        self.grain.stateChanged.connect(lambda x: self.profile_changed(x, "grain"))
        self.rotation.valueChanged.connect(lambda x: self.setting_changed(x, "rotation"))
        self.zoom.valueChanged.connect(lambda x: self.setting_changed(x, "zoom"))
        self.format_selector.currentTextChanged.connect(self.format_changed)
        self.grain_size.valueChanged.connect(lambda x: self.profile_changed(x, "grain_size"))
        self.grain_intensity.valueChanged.connect(
            lambda x: self.profile_changed(x, "grain_intensity")
        )
        self.grain_sigma.valueChanged.connect(lambda x: self.profile_changed(x, "grain_sigma"))
        self.halation_size.valueChanged.connect(lambda x: self.profile_changed(x, "halation_size"))
        self.halation_intensity.valueChanged.connect(
            lambda x: self.profile_changed(x, "halation_intensity")
        )
        self.halation_green.valueChanged.connect(
            lambda x: self.profile_changed(x, "halation_green_factor")
        )
        self.rotate_right.released.connect(self.rotate_image)
        self.rotate_left.released.connect(lambda: self.rotate_image(-1))
        self.flip_button.released.connect(self.flip_image)
        self.lens_selector.currentTextChanged.connect(lambda x: self.setting_changed(x, "lens"))
        self.camera_selector.currentTextChanged.connect(lambda x: self.setting_changed(x, "cam"))
        self.frame_width.textChanged.connect(lambda x: self.profile_changed(x, "frame_width"))
        self.frame_height.textChanged.connect(lambda x: self.profile_changed(x, "frame_height"))
        self.profile_selector.currentTextChanged.connect(self.load_profile_params)
        self.save_settings_button.triggered.connect(self.save_settings_dialogue)
        self.load_settings_button.triggered.connect(self.load_settings_dialogue)
        self.image_bar.image_changed.connect(self.load_image)
        self.image_bar.image_changed.connect(self.update_window_title)
        self.add_profile.released.connect(self.add_profile_prompt)
        self.delete_profile_button.triggered.connect(self.delete_profile)
        self.delete_all_profiles_button.triggered.connect(self.delete_all_profiles)
        self.highlight_burn.valueChanged.connect(
            lambda x: self.setting_changed(x, "highlight_burn")
        )
        self.burn_scale.valueChanged.connect(lambda x: self.setting_changed(x, "burn_scale"))
        self.saturation_slider.valueChanged.connect(lambda x: self.profile_changed(x, "sat_adjust"))
        self.color_masking.valueChanged.connect(lambda x: self.profile_changed(x, "color_masking"))
        self.quick_save_button.triggered.connect(self.quick_save)
        self.close_highlighted_button.triggered.connect(self.image_bar.close_highlighted)
        self.delete_highlighted_button.triggered.connect(self.delete_highlighted)
        self.image_bar.copy_settings.connect(self.copy_settings)
        self.deselect_all_button.triggered.connect(self.image_bar.deselect_all)
        self.reset_image_button.triggered.connect(self.reset_image)
        self.reset_all_images_button.triggered.connect(self.reset_all_images)
        self.reset_profile_button.triggered.connect(self.reset_profile)
        self.canvas_mode.currentTextChanged.connect(
            lambda x: self.setting_changed(x, "canvas_mode")
        )
        self.canvas_scale.valueChanged.connect(lambda x: self.setting_changed(x, "canvas_scale"))
        self.canvas_width.textChanged.connect(
            lambda: self.setting_changed(
                float(self.canvas_height.text()) / float(self.canvas_width.text())
                if self.canvas_width.text() and self.canvas_height.text()
                else 0.8,
                "canvas_ratio",
            )
        )
        self.canvas_height.textChanged.connect(
            lambda: self.setting_changed(
                float(self.canvas_height.text()) / float(self.canvas_width.text())
                if self.canvas_width.text() and self.canvas_height.text()
                else 0.8,
                "canvas_ratio",
            )
        )
        self.shadow_comp.valueChanged.connect(lambda x: self.profile_changed(x, "shadow_comp"))
        self.chroma_nr.valueChanged.connect(lambda x: self.setting_changed(x, "chroma_nr"))
        self.ui_update.connect(self.load_image_params_to_ui)
        self.load_display_icc_button.triggered.connect(self.load_display_icc_dialog)
        self.reset_display_icc_button.triggered.connect(self.reset_display_icc)
        self.load_softproof_icc_button.triggered.connect(self.load_softproof_icc_dialog)
        self.reset_softproof_icc_button.triggered.connect(self.reset_softproof_icc)
        self.display_saturation_intent.triggered.connect(
            lambda x: self.set_display_intent("saturation")
        )
        self.display_relative_intent.triggered.connect(
            lambda x: self.set_display_intent("relative")
        )
        self.display_relative_bpc_intent.triggered.connect(
            lambda x: self.set_display_intent("relative_bpc")
        )
        self.display_absolute_intent.triggered.connect(
            lambda x: self.set_display_intent("absolute")
        )
        self.display_perceptual_intent.triggered.connect(
            lambda x: self.set_display_intent("perceptual")
        )

        self.softproof_saturation_intent.triggered.connect(
            lambda x: self.set_softproof_intent("saturation")
        )
        self.softproof_relative_intent.triggered.connect(
            lambda x: self.set_softproof_intent("relative")
        )
        self.softproof_absolute_intent.triggered.connect(
            lambda x: self.set_softproof_intent("absolute")
        )
        self.softproof_perceptual_intent.triggered.connect(
            lambda x: self.set_softproof_intent("perceptual")
        )
        self.image_bar.images_empty.connect(self.show_empty_placeholder)
        self.image_bar.images_empty.connect(self.update_window_title)
        self.info_update.connect(self.image_info_widget.update_info)
        self.image_bar.images_empty.connect(lambda: self.info_update.emit({}, "", ""))
