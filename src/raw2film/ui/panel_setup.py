"""Build the main window controls."""

import lensfunpy
import numpy as np
from PyQt6.QtCore import (
    QRegularExpression,
)
from PyQt6.QtGui import (
    QRegularExpressionValidator,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QSizePolicy,
    QWidget,
)
from spectral_film_lut.filmstock_selector import FilmStockSelector
from spectral_film_lut.gui_objects import (
    AnimatedButton,
    HoverLineEdit,
    Slider,
    SliderLog,
    WideComboBox,
)

from raw2film import data


class PanelSetupMixin:
    """Build the main window controls."""

    def _create_controls(
        self,
        filmstock_info,
        basic_settings_group,
        profile_settings_group,
        film_effects_group,
        image_correction_group,
        advanced_printing_group,
        canvas_group,
    ):
        self.profile_selector = WideComboBox(parent=self)
        """Select the profile specifying film and format parameters.
        (1-9: select profile)"""

        self.profile_selector.addItem("Default")
        self.add_profile = AnimatedButton(parent=self)
        self.add_profile.setObjectName("plus")
        self.add_profile.setFixedWidth(25)
        profile_widget = QWidget()
        profile_widget_layout = QHBoxLayout()
        profile_widget.setLayout(profile_widget_layout)
        profile_widget_layout.addWidget(self.profile_selector)
        profile_widget_layout.addWidget(self.add_profile)
        profile_widget_layout.setContentsMargins(0, 0, 0, 0)
        basic_settings_group.add_option(
            profile_widget,
            "Profile",
            self.dflt_img_params["profile"],
            self.profile_selector.setCurrentText,
            tool_tip="Select the profile specifying film and format parameters.\n"
            "(1-9: select profile)",
        )

        self.lensfunpy_db = lensfunpy.Database()
        self.cameras = {
            camera.maker + " " + camera.model: camera for camera in self.lensfunpy_db.cameras
        }
        self.lenses = {lens.model: lens for lens in self.lensfunpy_db.lenses}
        self.cameras["None"] = None
        self.lenses["None"] = None

        # Lens correction checkbox with a reload-data button next to it
        lens_widget = QWidget()
        lens_layout = QHBoxLayout()
        lens_layout.setContentsMargins(0, 0, 0, 0)
        self.lens_correction = QCheckBox()
        """Correct lens distortion and vignetting."""
        # Button to reload camera/lens data for the current image
        self.reload_lens_data_button = AnimatedButton("Reload data", parent=self)
        self.reload_lens_data_button.setObjectName("reload")
        self.reload_lens_data_button.setFixedWidth(100)
        lens_layout.addWidget(self.lens_correction)
        lens_layout.addWidget(self.reload_lens_data_button)
        lens_widget.setLayout(lens_layout)

        image_correction_group.add_option(
            lens_widget,
            "Lens correction",
            tool_tip="Correct lens distortion and vignetting.",
        )

        # Connect the reload button to the handler that enables lens correction
        # for the current image and tries to reload camera/lens data.
        self.reload_lens_data_button.released.connect(lambda: self.reload_lens_cam_data())

        self.camera_selector = WideComboBox(self)
        """Select camera model for lens correction."""
        self.camera_selector.setMinimumWidth(100)
        self.camera_selector.addItems(self.cameras.keys())
        image_correction_group.add_option(
            self.camera_selector,
            "Camera model",
            "None",
            self.camera_selector.setCurrentText,
            tool_tip="Select camera model for lens correction.",
        )

        self.lens_selector = WideComboBox(self)
        """Select lens model for lens correction."""
        self.lens_selector.setMinimumWidth(100)
        self.lens_selector.addItems(self.lenses.keys())
        image_correction_group.add_option(
            self.lens_selector,
            "Lens model",
            "None",
            self.lens_selector.setCurrentText,
            tool_tip="Select lens model for lens correction.",
        )

        self.grain = QCheckBox()
        """
        *Unchecked:* No film grain.
        *Partially Checked:* Monochromatic film grain.
        *Checked:* Monochromatic film grain.
        """
        self.grain.setTristate(True)
        film_effects_group.add_option(
            self.grain,
            "Grain",
            self.dflt_prf_params["grain"],
            self.grain.setChecked,
            tool_tip="""<table>
  <tr><td><u>Unchecked: </u></td><td>No film grain.</td></tr>
  <tr><td><u>Partially Checked: </u></td><td>Monochromatic film grain.</td></tr>
  <tr><td><u>Checked: </u></td><td>RGB film grain.</td></tr>
</table>""",
        )

        self.grain_intensity = Slider(continuous=False)
        """Multiplier for grain intensity (0..2)."""
        self.grain_intensity.setMinMaxTicks(0, 2, 1, 10, self.dflt_prf_params["grain_intensity"])
        film_effects_group.add_option(
            self.grain_intensity,
            "Grain intensity",
            self.dflt_prf_params["grain_intensity"],
            self.grain_intensity.setValue,
            tool_tip="Multiplier for grain intensity (0 = none, 1 = default, 2 = double).",
        )

        self.grain_size = Slider(continuous=False)
        """Size of simulated film grains."""
        self.grain_size.setMinMaxTicks(3, 9, 1, 4, self.dflt_prf_params["grain_size"])
        film_effects_group.add_option(
            self.grain_size,
            "Grain size (microns)",
            self.dflt_prf_params["grain_size"],
            self.grain_size.setValue,
            tool_tip="Size of simulated film grains.",
        )

        self.grain_sigma = Slider(continuous=False)
        """Variance of simulated film grains. Effects perceived uniformity."""
        self.grain_sigma.setMinMaxTicks(0.0, 1.0, 1, 50, self.dflt_prf_params["grain_sigma"])
        film_effects_group.add_option(
            self.grain_sigma,
            "Grain variance",
            self.dflt_prf_params["grain_sigma"],
            self.grain_sigma.setValue,
            tool_tip="Variance of simulated film grains. Effects perceived uniformity.",
        )

        self.sharpness = QCheckBox()
        """Emulate the resolution and micro-contrast of film."""
        film_effects_group.add_option(
            self.sharpness,
            "Sharpness",
            self.dflt_prf_params["sharpness"],
            self.sharpness.setChecked,
            tool_tip="Emulate the resolution and micro-contrast of film.",
        )

        self.sharpening_strength = Slider(continuous=False)
        """Amount of sharpening to apply (0 - no sharpening, 1 - full)."""
        self.sharpening_strength.setMinMaxTicks(
            0.0, 1.0, 1, 100, default=self.dflt_prf_params["sharpening_strength"]
        )
        film_effects_group.add_option(
            self.sharpening_strength,
            "Sharpening strength",
            self.dflt_prf_params["sharpening_strength"],
            self.sharpening_strength.setValue,
            tool_tip="Amount of sharpening to apply (0 - no sharpening, 1 - full).",
        )

        self.sharpening_sigma = Slider(continuous=False)
        """Sigma used for the sharpening kernel (controls radius)."""
        self.sharpening_sigma.setMinMaxTicks(
            0.1, 3.0, 1, 50, default=self.dflt_prf_params["sharpening_sigma"]
        )
        film_effects_group.add_option(
            self.sharpening_sigma,
            "Sharpening sigma",
            self.dflt_prf_params["sharpening_sigma"],
            self.sharpening_sigma.setValue,
            tool_tip="Sigma used for the sharpening kernel (controls radius).",
        )

        self.halation = QCheckBox()
        """
        Activate halation, a warm glow around highlights, resulting from reflections on
        the back of the film.
        """
        film_effects_group.add_option(
            self.halation,
            "Halation",
            self.dflt_prf_params["halation"],
            self.halation.setChecked,
            tool_tip="Activate halation, a warm glow around highlights, resulting\n"
            "from reflections on the back of the film.",
        )

        self.halation_size = SliderLog(continuous=False)
        """
        How far the halation spreads. Halation is a warm glow around highlights,
        resulting from reflections on the film backing.
        """
        self.halation_size.setMinMaxSteps(0.5, 2, 50, self.dflt_prf_params["halation_size"])
        film_effects_group.add_option(
            self.halation_size,
            "Halation size",
            self.dflt_prf_params["halation_size"],
            self.halation_size.setValue,
            tool_tip="How far the halation spreads. Halation is a warm glow around\n"
            "highlights, resulting from reflections on the film backing.",
        )

        self.halation_green = Slider(continuous=False)
        """
        How red or yellow the halation is. Specifies how strongly the halation reaches
        into the green sensitive layer.
        """
        self.halation_green.set_color_gradient(
            np.array([0.6, 0.21, 29.23 / 360]), np.array([0.9, 0.18, 109.77 / 360])
        )
        self.halation_green.setMinMaxTicks(
            0, 1, 1, 20, self.dflt_prf_params["halation_green_factor"]
        )
        film_effects_group.add_option(
            self.halation_green,
            "Halation color",
            self.dflt_prf_params["halation_green_factor"],
            self.halation_green.setValue,
            tool_tip="How red or yellow the halation is. Specifies how strongly the\n"
            "halation reaches into the green sensitive layer.",
        )

        self.halation_intensity = SliderLog(continuous=False)
        self.halation_intensity.setMinMaxSteps(
            0.5, 4, 50, self.dflt_prf_params["halation_intensity"], 1
        )
        film_effects_group.add_option(
            self.halation_intensity,
            "Halation intensity",
            self.dflt_prf_params["halation_intensity"],
            self.halation_intensity.setValue,
            tool_tip="How intense the halation is. Halation is a warm glow around\n"
            "highlights, resulting from reflections on the film backing.",
        )

        self.exp_comp = Slider(continuous=False)
        self.exp_comp.setMinMaxTicks(-3, 3, 1, 20, self.dflt_img_params["exp_comp"])
        basic_settings_group.add_option(
            self.exp_comp,
            "Exposure",
            self.dflt_img_params["exp_comp"],
            self.exp_comp.setValue,
            tool_tip="Adjust exposure in stops.\n"
            "(Up: increase exposure)\n"
            "(Down: decrease exposure)",
        )

        self.wb_modes = {
            "Default": 6000,
            "Daylight": 5500,
            "Cloudy": 6500,
            "Shade": 7500,
            "Tungsten": 2800,
            "Fluorescent": 3800,
            "Custom": None,
        }
        self.wb_mode = WideComboBox(self)
        self.wb_mode.addItems(list(self.wb_modes.keys()))
        basic_settings_group.add_option(
            self.wb_mode,
            "WB",
            "Daylight",
            self.wb_mode.setCurrentText,
            tool_tip="Select preset white balance.\n"
            "(Shift+D: daylight)\n"
            "(Shift+C: cloudy)\n"
            "(Shift+S: shade)\n"
            "(Shift+F: fluorescent)\n"
            "(Shift+T: Tungsten)",
        )

        self.exp_wb = SliderLog(continuous=False)
        self.exp_wb.setMinMaxSteps(2700, 16000, 120, self.dflt_img_params["exp_kelvin"], -2)
        self.exp_wb.set_color_gradient(
            np.array([2 / 3, 0.14, 0.65277]), np.array([2 / 3, 0.14, 0.15277])
        )
        basic_settings_group.add_option(
            self.exp_wb,
            "Kelvin",
            self.dflt_img_params["exp_kelvin"],
            self.exp_wb.setValue,
            tool_tip="Adjust white balance in kelvin.",
        )

        self.tint = Slider(continuous=False)
        self.tint.setMinMaxTicks(-1, 1, 1, 100, default=self.dflt_img_params["tint"])
        self.tint.set_color_gradient(
            np.array([2 / 3, 0.14, 0.90277]), np.array([2 / 3, 0.14, 0.40277])
        )
        basic_settings_group.add_option(
            self.tint,
            "Tint",
            self.dflt_img_params["tint"],
            self.tint.setValue,
            tool_tip="Change tint on green-magenta axis.",
        )

        self.chroma_nr = Slider(continuous=False)
        self.chroma_nr.setMinMaxTicks(0, 10)
        image_correction_group.add_option(
            self.chroma_nr,
            "Chroma NR",
            self.dflt_img_params["chroma_nr"],
            self.chroma_nr.setValue,
            tool_tip="Strength of chroma noise reduction.",
        )

        self.highlight_burn = Slider(continuous=False)
        self.highlight_burn.setMinMaxTicks(
            0, 4, 1, 20, default=self.dflt_img_params["highlight_burn"]
        )
        basic_settings_group.add_option(
            self.highlight_burn,
            "Highlight burn",
            self.dflt_img_params["highlight_burn"],
            self.highlight_burn.setValue,
            tool_tip="Lower the brightness of bright areas. Reach can be configured\n"
            "under 'Burn scale' under 'Advanced printing techniques.\n"
            "Inspired by the burn method in darkroom printing. Is applied in stops.\n"
            "(Shift+Up: increase)\n"
            "(Shift+Down: decrease)",
        )
        self.burn_scale = Slider(continuous=False)
        self.burn_scale.setMinMaxTicks(1, 200, default=self.dflt_img_params["burn_scale"])
        advanced_printing_group.add_option(
            self.burn_scale,
            "Burn scale",
            self.dflt_img_params["burn_scale"],
            self.burn_scale.setValue,
            tool_tip="How much blur is applied to the highlight burn.",
        )

        self.rotate = QWidget()
        rotate_layout = QHBoxLayout()
        self.rotate_left = AnimatedButton(parent=self)
        self.rotate_left.setObjectName("left")
        self.rotate_right = AnimatedButton(parent=self)
        self.rotate_right.setObjectName("right")
        self.flip_button = AnimatedButton(parent=self)
        """Switch between horizontal and vertical orientation."""
        self.flip_button.setObjectName("flip")
        for btn in (self.rotate_left, self.rotate_right, self.flip_button):
            btn.setMinimumWidth(10)
            btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            rotate_layout.addWidget(btn, stretch=1)
        rotate_layout.setContentsMargins(0, 0, 0, 0)
        rotate_layout.setSpacing(0)
        self.rotate.setLayout(rotate_layout)
        basic_settings_group.add_option(
            self.rotate,
            "Rotate",
            tool_tip="Rotate the image by 90 degrees or change the orientation.\n"
            "(Ctrl+R: rotate right)",
        )

        self.rotation = Slider(continuous=False)
        self.rotation.setMinMaxTicks(-90, 90, 1, 4, default=self.dflt_img_params["rotation"])
        basic_settings_group.add_option(
            self.rotation,
            "Rotation angle",
            self.dflt_img_params["rotation"],
            self.rotation.setValue,
            tool_tip="Rotate by an angle in degrees.\n"
            "(Ctrl+Right: rotate right)\n"
            "(Ctrl+Left: rotate left)",
        )

        self.zoom = Slider(continuous=False)
        self.zoom.setMinMaxTicks(1, 2, 1, 100, default=self.dflt_img_params["zoom"])
        basic_settings_group.add_option(
            self.zoom,
            "Zoom",
            self.dflt_img_params["zoom"],
            self.zoom.setValue,
            tool_tip="Crop into the image.\n(Ctrl+Plus: zoom in)\n(Ctrl+Minus: zoom out)",
        )

        self.format_selector = WideComboBox(self)
        self.format_selector.addItems([*list(data.FORMATS.keys()), "Custom"])
        profile_settings_group.add_option(
            self.format_selector,
            "Format",
            self.dflt_prf_params["film_format"],
            self.format_selector.setCurrentText,
            tool_tip="Select a preset film format. Adjusts scale of film\n"
            "characteristics (halation, resolution, grain) and changes aspect ratio.",
        )

        self.frame_size = QWidget()
        frame_layout = QHBoxLayout()
        self.frame_width = HoverLineEdit(parent=self)
        regex = QRegularExpression(r"[0-9]*|[0-9]+\.[0-9]*")
        self.frame_width.setValidator(QRegularExpressionValidator(regex))
        self.frame_height = HoverLineEdit(parent=self)
        self.frame_height.setValidator(QRegularExpressionValidator(regex))
        frame_layout.addWidget(self.frame_width)
        frame_layout.addWidget(self.frame_height)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        self.frame_size.setLayout(frame_layout)
        profile_settings_group.add_option(
            self.frame_size,
            "Width/Height",
            self.dflt_prf_params["film_format"],
            self.format_selector.setCurrentText,
            tool_tip="Specify simulated film frame size. Adjusts scale of film\n"
            "characteristics (halation, resolution, grain) and changes aspect ratio.",
        )

        negative_info = {x: y for x, y in filmstock_info.items() if y["stage"] == "camera"}
        sort_keys_negative = [
            "Name",
            "Year",
            "Resolution",
            "Granularity",
            "sensitivity",
            "Gamma",
        ]
        group_keys_negative = ["Manufacturer", "Type", "Decade", "Medium"]
        list_keys_negative = [
            "Manufacturer",
            "Type",
            "Year",
            "Sensitivity",
            "Chromaticity",
        ]
        sidebar_keys_negative = [
            "Alias",
            "Manufacturer",
            "Type",
            "Year",
            "Sensitivity",
            "resolution",
            "Granularity",
            "Medium",
            "Chromaticity",
            "Gamma",
            "Comment",
        ]
        self.filmstocks["None"] = None
        self.filmstocks["Inversion (ACES like)"] = None
        # self.filmstocks["Inversion (Optimized)"] = None
        self.negative_selector = FilmStockSelector(
            negative_info,
            self,
            self,
            sort_keys=sort_keys_negative,
            image_key="image",
            group_keys=group_keys_negative,
            list_keys=list_keys_negative,
            sidebar_keys=sidebar_keys_negative,
            default_group="Manufacturer",
        )
        self.negative_selector.setMinimumWidth(100)
        profile_settings_group.add_option(
            self.negative_selector,
            "Negative stock",
            self.dflt_prf_params["negative_film"],
            self.negative_selector.setCurrentText,
            tool_tip="Which negative film stock to emulate. Affects colors,\n"
            "resolution, and graininess",
        )

        luma_bright = 0.8
        luma_dark = 0.4
        chroma = 0.2
        hue_offset = 0.06111111
        self.red_light = Slider(continuous=False)
        self.red_light.setMinMaxTicks(-0.75, 0.75, 1, 50)
        self.red_light.set_color_gradient(
            np.array([luma_bright, chroma, hue_offset + 0 / 6]),
            np.array([luma_dark, chroma, hue_offset + 3 / 6]),
        )
        advanced_printing_group.add_option(
            self.red_light,
            "Red printer light",
            self.dflt_prf_params["red_light"],
            self.red_light.setValue,
            tool_tip="How strong the simulated red light is during printing.\n"
            "Decreases how red the print is.",
        )
        self.green_light = Slider(continuous=False)
        self.green_light.setMinMaxTicks(-0.75, 0.75, 1, 50)
        self.green_light.set_color_gradient(
            np.array([luma_bright, chroma, hue_offset + 2 / 6]),
            np.array([luma_dark, chroma, hue_offset + 5 / 6]),
        )
        advanced_printing_group.add_option(
            self.green_light,
            "Green printer light",
            self.dflt_prf_params["green_light"],
            self.green_light.setValue,
            tool_tip="How strong the simulated green light is during printing.\n"
            "Decreases how green the print is.",
        )
        self.blue_light = Slider(continuous=False)
        self.blue_light.setMinMaxTicks(-0.75, 0.75, 1, 50)
        self.blue_light.set_color_gradient(
            np.array([luma_bright, chroma, hue_offset + 4 / 6]),
            np.array([luma_dark, chroma, hue_offset + 1 / 6]),
        )
        advanced_printing_group.add_option(
            self.blue_light,
            "Blue printer light",
            self.dflt_prf_params["blue_light"],
            self.blue_light.setValue,
            tool_tip="How strong the simulated blue light is during printing.\n"
            "Decreases how blue the print is.",
        )

        self.link_lights = QCheckBox()
        self.link_lights.setChecked(True)
        self.link_lights.setText("link lights")
        advanced_printing_group.add_option(
            self.link_lights,
            tool_tip="Whether to adjust the printer lights individually or not.",
        )

        print_info = {x: y for x, y in filmstock_info.items() if y["stage"] == "print"}
        print_info["Inversion (ACES like)"] = {}
        # print_info["Inversion (Optimized)"] = {}
        print_info["None"] = {}
        sort_keys_print = ["Name", "Year", "Gamma"]
        group_keys_print = ["Manufacturer", "Type", "Decade", "Medium"]
        list_keys_print = ["Manufacturer", "Type", "Year", "Chromaticity"]
        sidebar_keys_print = [
            "Alias",
            "Manufacturer",
            "Type",
            "Year",
            "Medium",
            "Chromaticity",
            "Gamma",
            "Comment",
        ]
        self.print_selector = FilmStockSelector(
            print_info,
            self,
            self,
            sort_keys=sort_keys_print,
            group_keys=group_keys_print,
            list_keys=list_keys_print,
            sidebar_keys=sidebar_keys_print,
            default_group="Manufacturer",
            image_key="image",
        )
        self.print_selector.setMinimumWidth(100)
        profile_settings_group.add_option(
            self.print_selector,
            "Print stock",
            self.dflt_prf_params["print_film"],
            self.print_selector.setCurrentText,
            tool_tip="Which print material to emulate. Affects only colors.",
        )

        self.push_pull = Slider(continuous=False)
        """
        How much to push or pull the film, adjusting contrast. Works linearly scaling
        the characteristic curve of the film. Not based on measured data, but a rough
        approximation, useful for controlling contrast.

        Not recommended for use with slide film.
        """
        self.push_pull.setMinMaxTicks(-1.5, 1.5, 1, 20)
        profile_settings_group.add_option(
            self.push_pull,
            "Push/pull",
            self.dflt_prf_params["push_pull"],
            self.push_pull.setValue,
            tool_tip="How much to push or pull the film, adjusting contrast.\n"
            "Works linearly scaling the characteristic curve of the film.\n"
            "Not based on measured data, but a rough approximation useful\n"
            "for controlling contrast.\n"
            "Not recommended for use with slide film.",
        )

        self.projector_kelvin = SliderLog(continuous=False)
        """Under what light temperature to view the print or slide."""
        self.projector_kelvin.setMinMaxSteps(
            2700, 16000, 120, self.dflt_prf_params["projector_kelvin"], -2
        )
        self.projector_kelvin.set_color_gradient(
            np.array([2 / 3, 0.14, 0.15277]), np.array([2 / 3, 0.14, 0.65277])
        )
        profile_settings_group.add_option(
            self.projector_kelvin,
            "Projector WB",
            self.dflt_prf_params["projector_kelvin"],
            self.projector_kelvin.setValue,
            tool_tip="Under what light temperature to view the print or slide.",
        )

        self.inversion_gamma = Slider(continuous=False)
        """
        The gamma applied using the inversion if 'Inversion' is selected or when
        'idealized curve' is checked."""
        self.inversion_gamma.setMinMaxTicks(1, 5, 1, 10)
        profile_settings_group.add_option(
            self.inversion_gamma,
            "Gamma",
            self.dflt_prf_params["inversion_gamma"],
            self.inversion_gamma.setValue,
            tool_tip="The gamma applied using the inversion if 'Inversion' is\n"
            "selected or when 'idealized curve' is checked.",
        )

        self.idealized_curve = QCheckBox("Pure curve")
        """
        Replace the characteristic curve of the print film with an ideal gamma curve.
        Preserves the sensitivity and dye densities of the print film.
        When activated, the gamma is controlled by the inversion gamma.
        """
        self.idealized_curve.setToolTip(
            "Replace the characteristic curve of the print film with an ideal gamma\n"
            "curve. Preserves the sensitivity and dye densities of the print film.\n"
            "When activated, the gamma is controlled by the inversion gamma."
        )

        self.white_clip = QCheckBox("Clip")
        """
        When viewing print film brightness will be increased to clip at exactly 1.0.
        When viewing slide film white balancing is applied, so that a gray patch will
        actually produce the color temperature specified by the  projector kelvin.
        """
        self.white_clip.setToolTip(
            "When viewing print film brightness will be increased to clip at\n"
            "exactly 1.0. When viewing slide film white balancing is\n"
            "applied, so that a gray patch will actually produce the color\n"
            "temperature specified by the  projector kelvin."
        )

        self.white_balance = QCheckBox("WB")
        """Whether to white balance slide film."""
        self.white_balance.setTristate(True)
        self.white_balance.setToolTip(
            "Slide film white balance mode:\n"
            "Unchecked = off\n"
            "Partially checked = balance by peak white\n"
            "Checked = balance by mid gray"
        )

        checker_widget = QWidget()
        checker_widget_layout = QHBoxLayout(checker_widget)
        checker_widget_layout.addWidget(self.idealized_curve)
        checker_widget_layout.addWidget(self.white_clip)
        checker_widget_layout.addWidget(self.white_balance)
        profile_settings_group.add_option(checker_widget)

        self.saturation_slider = Slider(continuous=False)
        self.saturation_slider.setMinMaxTicks(
            0, 2, 1, 100, default=self.dflt_prf_params["sat_adjust"]
        )
        self.saturation_slider.set_color_gradient(
            np.array(
                [
                    0.666,
                    0.0,
                    0.0,
                ]
            ),
            np.array([0.666, 0.25, 2.0]),
            20,
            False,
        )
        profile_settings_group.add_option(
            self.saturation_slider,
            "Saturation",
            self.dflt_prf_params["sat_adjust"],
            self.saturation_slider.setValue,
            tool_tip="Adjust the saturation in the display color space.",
        )

        self.color_masking = Slider(continuous=False)
        """
        How effective the orange color mask of the film is. Value of 1 perfectly
        compensates for color layer cross contamination. An increased value leads to
        higher saturation. There is no documented data on this, so you can play around
        with this to your liking.

        For film without a color mask like slide film this can be used to simulate other
        inter-layer effects. Should probably set lower, but should be experimented with.
        """
        self.color_masking.setMinMaxTicks(
            0, 2, 1, 100, default=self.dflt_prf_params["color_masking"]
        )
        self.color_masking.set_color_gradient(
            np.array(
                [
                    0.666,
                    0.0,
                    0.0,
                ]
            ),
            np.array([0.666, 0.25, 2.0]),
            20,
            False,
        )
        profile_settings_group.add_option(
            self.color_masking,
            "Color masking",
            self.dflt_prf_params["color_masking"],
            self.color_masking.setValue,
            tool_tip="How effective the orange color mask of the film is. Value of 1\n"
            "perfectly compensates for color layer cross contamination. An\n"
            "increased value leads to higher saturation. There is no\n"
            "documented data on this, so you can play around with this to\n"
            "your liking.\n"
            "For film without a color mask like slide film this\n"
            "can be used to simulate other inter-layer effects. Should\n"
            "probably set lower, but should be experimented with.",
        )

        self.canvas_mode = WideComboBox(self)
        self.canvas_mode.addItems(
            [
                "No",
                "Proportional white",
                "Proportional black",
                "Uniform white",
                "Uniform black",
                "Fixed white",
                "Fixed black",
            ]
        )
        canvas_group.add_option(
            self.canvas_mode,
            "Canvas",
            self.dflt_img_params["canvas_mode"],
            self.canvas_mode.setCurrentText,
            tool_tip="What type of border to add to the image.",
        )

        self.canvas_scale = Slider(continuous=False)
        self.canvas_scale.setMinMaxTicks(1, 2, 1, 40, default=self.dflt_img_params["canvas_scale"])
        canvas_group.add_option(
            self.canvas_scale,
            "Canvas scale",
            self.dflt_img_params["canvas_scale"],
            self.canvas_scale.setValue,
            tool_tip="How big the canvas is.",
        )

        self.canvas_size = QWidget()
        canvas_layout = QHBoxLayout()
        self.canvas_width = HoverLineEdit(parent=self)
        self.canvas_width.setValidator(QRegularExpressionValidator(regex))
        self.canvas_height = HoverLineEdit(parent=self)
        self.canvas_height.setValidator(QRegularExpressionValidator(regex))
        canvas_layout.addWidget(self.canvas_width)
        canvas_layout.addWidget(self.canvas_height)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_size.setLayout(canvas_layout)
        canvas_group.add_option(
            self.canvas_size,
            "Width/Height",
            1,
            lambda x: (self.canvas_width.setText("5"), self.canvas_height.setText("4")),
            tool_tip="Aspect ratio of added canvas.",
        )

        self.shadow_comp = Slider(continuous=False)
        """
        Specify black compensation. For positive values blacks are lifted with a film
        like curves. For negative values shadows are darkened without crushing any
        details.
        """
        self.shadow_comp.setMinMaxTicks(-1, 1, 1, 50)
        profile_settings_group.add_option(
            self.shadow_comp,
            "Shadow comp.",
            self.dflt_prf_params["shadow_comp"],
            self.shadow_comp.setValue,
            tool_tip="Specify black compensation. For positive values blacks are\n"
            "lifted with a film like curves. For negative values shadows are\n"
            "darkened without crushing any details.",
        )
