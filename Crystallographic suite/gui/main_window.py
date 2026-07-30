from PySide6.QtCore import Qt

from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .forward_tab import ForwardTab
from .predictor_tab import PredictorTab


class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Advanced Crystallographic Simulator"
        )

        self.resize(1400, 850)

        self.build_ui()

    def build_ui(self):

        central = QWidget()

        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        title = QLabel(
            "Advanced Crystallographic Simulator"
        )

        title.setAlignment(Qt.AlignCenter)

        title.setStyleSheet(
            """
            font-size:28px;
            font-weight:bold;
            padding:15px;
            """
        )

        layout.addWidget(title)

        self.tabs = QTabWidget()

        self.forward_tab = ForwardTab()

        self.predictor_tab = PredictorTab()

        self.tabs.addTab(
            self.forward_tab,
            "Forward Simulation",
        )

        self.tabs.addTab(
            self.predictor_tab,
            "AI Material Identification",
        )

        layout.addWidget(self.tabs)

        self.setStatusBar(QStatusBar())

        self.statusBar().showMessage("Ready")