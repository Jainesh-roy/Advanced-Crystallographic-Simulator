from PySide6.QtWidgets import (
    QWidget,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QMessageBox,
)
import numpy as np
from core.material_presets import MATERIAL_PRESETS

from .plot_widget import PlotWidget

from core.forward_engine import (
    simulate_xrd_pattern,
    add_realistic_xrd_artifacts,
)
from .plot_widget import PlotWithToolbar



class ForwardTab(QWidget):

    def __init__(self):

        super().__init__()

        self.build_ui()

    def build_ui(self):

        main_layout = QHBoxLayout(self)

        controls = QFormLayout()

        # ----------------------------
        # Material
        # ----------------------------

        self.element = QComboBox()

        self.element.addItems([
            "Cu","Ni","Al","Ag","Au",
            "Fe","Cr","Mo","W","Po"
        ])

        self.structure = QComboBox()

        self.structure.addItems([
            "FCC",
            "BCC",
            "SC"
        ])

        # ----------------------------
        # Simulation Parameters
        # ----------------------------

        self.lattice = QDoubleSpinBox()
        self.lattice.setDecimals(4)
        self.lattice.setRange(1.0,10.0)
        self.lattice.setValue(3.615)

        self.wavelength = QDoubleSpinBox()
        self.wavelength.setDecimals(4)
        self.wavelength.setRange(0.5,3)
        self.wavelength.setValue(1.5406)

        self.crystallite = QDoubleSpinBox()
        self.crystallite.setRange(1,1000)
        self.crystallite.setValue(45)

        self.instrument = QDoubleSpinBox()
        self.instrument.setRange(0.01,2)
        self.instrument.setDecimals(3)
        self.instrument.setValue(0.15)

        self.profile = QComboBox()

        self.profile.addItems([
            "Lorentzian",
            "Gaussian"
        ])

        self.max_index = QComboBox()

        self.max_index.addItems([
            "4","5","6","7","8"
        ])

        # ----------------------------
        # Artifacts
        # ----------------------------

        self.noise = QDoubleSpinBox()

        self.noise.setDecimals(3)

        self.noise.setRange(0,0.5)

        self.noise.setValue(0.025)

        self.background = QDoubleSpinBox()

        self.background.setDecimals(3)

        self.background.setRange(0,1)

        self.background.setValue(0.04)

        self.seed = QComboBox()

        self.seed.addItems([
            "42","100","123","999"
        ])

        # ----------------------------

        controls.addRow("Element",self.element)

        controls.addRow("Structure",self.structure)

        controls.addRow("Lattice a (Å)",self.lattice)

        controls.addRow("Wavelength (Å)",self.wavelength)

        controls.addRow("Crystallite (nm)",self.crystallite)

        controls.addRow("Instrument FWHM",self.instrument)

        controls.addRow("Peak Profile",self.profile)

        controls.addRow("Max hkl",self.max_index)

        controls.addRow("Noise",self.noise)

        controls.addRow("Background",self.background)

        controls.addRow("Random Seed",self.seed)

        # ----------------------------

        self.generate_btn = QPushButton("Generate Pattern")

        self.save_csv_btn = QPushButton("Save CSV")

        self.save_plot_btn = QPushButton("Save Plot")

        self.generate_btn.clicked.connect(
            self.generate_pattern
        )
        self.save_csv_btn.clicked.connect(
            self.save_csv
        )

        self.save_plot_btn.clicked.connect(
            self.save_plot
        )

        # -------------------------------------------------------
        # Simulation Summary
        # -------------------------------------------------------

        self.summary = QLabel()
        self.summary.setStyleSheet("""
        QLabel{
            background: white;
            color: black;
            border: 1px solid gray;
            padding: 10px;
            font-family: Consolas;
            font-size: 10pt;
        }
        """)

        self.summary.setWordWrap(True)

        self.summary.setMinimumHeight(180)

        self.summary.setText(
            "Simulation Summary\n\n"
            "No simulation generated."
        )

        # -------------------------------------------------------
        # Left Layout
        # -------------------------------------------------------

        left_layout = QVBoxLayout()

        left_layout.addLayout(controls)

        left_layout.addSpacing(15)

        left_layout.addWidget(self.generate_btn)

        left_layout.addWidget(self.save_csv_btn)

        left_layout.addWidget(self.save_plot_btn)

        left_layout.addSpacing(20)

        left_layout.addWidget(self.summary)

        left_layout.addStretch()

        left = QWidget()

        left.setLayout(left_layout)

        # -------------------------------------------------------
        # Plot
        # -------------------------------------------------------

        self.plot = PlotWithToolbar()

        main_layout.addWidget(left,1)

        main_layout.addWidget(self.plot,3)
        self.element.currentTextChanged.connect(self.update_material)
        self.structure.currentTextChanged.connect(self.update_material_defaults)

        self.update_material()
        # ==========================================================
        # Generate Pattern
        # ==========================================================

    def update_material(self):

        element = self.element.currentText()

        self.structure.blockSignals(True)
        self.structure.clear()

        valid_structures = [
            structure
            for (mat, structure) in MATERIAL_PRESETS.keys()
            if mat == element
        ]

        self.structure.addItems(valid_structures)

        self.structure.blockSignals(False)

        self.update_material_defaults()

    def update_material_defaults(self):

        key = (
            self.element.currentText(),
            self.structure.currentText()
        )

        if key not in MATERIAL_PRESETS:
            return

        preset = MATERIAL_PRESETS[key]

        self.lattice.setValue(
            preset["lattice_a"]
        )

        self.crystallite.setValue(
            preset["crystallite"]
        )

    def generate_pattern(self):

        try:

            element = self.element.currentText()
            lattice = self.structure.currentText()


            wavelength = self.wavelength.value()

            lattice_parameter = self.lattice.value()

            crystallite_size = self.crystallite.value()

            instrument_fwhm = self.instrument.value()

            peak_profile = self.profile.currentText()

            max_index = int(self.max_index.currentText())

            noise = self.noise.value()

            background = self.background.value()

            seed = int(self.seed.currentText())

            theta, intensity, peaks  = simulate_xrd_pattern(

                element_symbol=element,

                element_b=None,

                composition_x=None,

                lattice_type=lattice,

                lattice_parameter_a=lattice_parameter,

                wavelength_angstrom=wavelength,

                two_theta_start_deg=20,

                two_theta_stop_deg=90,

                two_theta_step_deg=0.02,

                peak_profile_function=peak_profile,

                instrument_fwhm_deg=instrument_fwhm,

                crystallite_size_nm=crystallite_size,

                max_index=max_index,

                b_iso=0.5,
            )

            intensity = add_realistic_xrd_artifacts(

                theta,

                intensity,

                noise_fraction=noise,

                background_fraction=background,

                random_seed=seed,
            )

            self.current_theta = theta
            self.current_intensity = intensity
            self.current_peaks = peaks
            print(len(self.current_peaks))

            # -------------------------------------------------------
            # Simulation Summary
            # -------------------------------------------------------

            highest_peak = theta[np.argmax(intensity)]

            maximum_intensity = np.max(intensity)

            self.summary.setText(

                f"""
            Simulation Summary

            Material : {element}

            Structure : {lattice}

            Lattice Parameter : {lattice_parameter:.4f} Å

            Wavelength : {wavelength:.4f} Å

            Crystallite Size : {crystallite_size:.1f} nm

            Peak Profile : {peak_profile}

            Instrument FWHM : {instrument_fwhm:.3f}°

            Maximum HKL : {max_index}

            Noise Fraction : {noise:.3f}

            Background : {background:.3f}

            Highest Peak : {highest_peak:.2f}°

            Maximum Intensity : {maximum_intensity:.2f}

            Total Data Points : {len(theta)}
            """
            )

            # -------------------------------------------------------
            # Plot
            # -------------------------------------------------------
            print("Updating Summary...")
            self.plot.plot(
                theta,
                intensity,
                peaks=self.current_peaks,
                title=f"{element} ({lattice}) XRD Pattern",
            )

        except Exception as e:

            print("\n========== Simulation Error ==========")
            print(e)
            print("======================================\n")


    def save_csv(self):

        if not hasattr(self, "current_theta"):

            QMessageBox.warning(
                self,
                "No Data",
                "Generate a diffraction pattern first."
            )
            return

        filename, _ = QFileDialog.getSaveFileName(

            self,

            "Save XRD Pattern",

            "",

            "CSV Files (*.csv)"
        )

        if not filename:
            return

        data = np.column_stack(

            (
                self.current_theta,
                self.current_intensity,
            )
        )

        np.savetxt(

            filename,

            data,

            delimiter=",",

            header="TwoTheta,Intensity",

            comments="",

            fmt="%.6f",
        )

        QMessageBox.information(

            self,

            "Saved",

            f"Pattern saved successfully.\n\n{filename}"
        )

    def save_plot(self):

        if not hasattr(self, "current_theta"):

            QMessageBox.warning(
                self,
                "No Plot",
                "Generate a diffraction pattern first."
            )
            return

        filename, _ = QFileDialog.getSaveFileName(

            self,

            "Save Plot",

            "",

            "PNG Image (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )

        if not filename:
            return

        self.plot.figure.savefig(

            filename,

            dpi=300,

            bbox_inches="tight"
        )

        QMessageBox.information(

            self,

            "Saved",

            f"Plot saved successfully.\n\n{filename}"
        )