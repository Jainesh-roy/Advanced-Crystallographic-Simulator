from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT,
)
from PySide6.QtWidgets import QWidget, QVBoxLayout
import numpy as np


class PlotWidget(FigureCanvas):

    def __init__(self):

        self.figure = Figure(figsize=(8, 5))

        super().__init__(self.figure)

        self.ax = self.figure.add_subplot(111)

        self.ax.set_xlabel(r"2θ (Degrees)")

        self.ax.set_ylabel("Intensity (a.u.)")

        self.ax.grid(True)

        self.figure.tight_layout()

    def plot(self, theta, intensity, title="XRD Pattern", peaks=None):

        self.ax.clear()

        # Main Pattern
        self.ax.plot(
            theta,
            intensity,
            linewidth=2.0,
            color="#0055ff",
        )

        # ---------------------------------------------------------
        # Draw Bragg Reflection Markers
        # ---------------------------------------------------------

        if peaks is not None:

            ymax = intensity.max()

            for peak in peaks:

                x = peak.two_theta_deg

                idx = np.argmin(np.abs(theta - x))

                y = intensity[idx]

                # small red marker
                self.ax.plot(
                    x,
                    y,
                    color="red",
                    marker="^",
                    markersize=8,
                    markeredgecolor="black",
                    markerfacecolor="red",
                )
                

                # HKL label
                offset = 0.06 * ymax
                self.ax.text(
                    x,
                    y + offset,
                    f"({peak.h}{peak.k}{peak.l})",
                    fontsize=8,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    color="darkred",
                )
            self.ax.grid(
                True,
                which="major",
                linestyle="--",
                linewidth=0.8,
                alpha=0.45,
            )
            
            self.ax.minorticks_on()
            
            self.ax.grid(
                True,
                which="minor",
                 alpha=0.15,
            )

        # Titles
        self.ax.set_title(
            title,
            fontsize=18,
            fontweight="bold",
            pad=12,
        )

        self.ax.set_xlabel(
            r"2θ (Degrees)",
            fontsize=13,
        )

        self.ax.set_ylabel(
            "Intensity (a.u.)",
            fontsize=13,
        )

        # Grid
        self.ax.grid(
            True,
            which="major",
            linestyle="--",
            alpha=0.5,
        )

        self.ax.minorticks_on()

        self.ax.grid(
            True,
            which="minor",
            alpha=0.15,
        )

        self.ax.tick_params(
            axis="both",
            labelsize=11,
        )

        self.ax.ticklabel_format(
            axis="y",
            style="plain",
        )

        self.ax.margins(x=0.01)

        self.figure.tight_layout()

        self.draw()

    def clear(self):

        self.ax.clear()

        self.ax.set_xlabel(r"2θ (Degrees)")

        self.ax.set_ylabel("Intensity (a.u.)")

        self.ax.grid(True)

        self.draw()


class PlotWithToolbar(QWidget):

    def __init__(self):

        super().__init__()

        layout = QVBoxLayout(self)

        self.canvas = PlotWidget()

        self.toolbar = NavigationToolbar2QT(
            self.canvas,
            self,
        )

        layout.addWidget(self.toolbar)

        layout.addWidget(self.canvas)

    def plot(self, *args, **kwargs):
        self.canvas.plot(*args, **kwargs)

    def clear(self):
        self.canvas.clear()

    @property
    def figure(self):
        return self.canvas.figure