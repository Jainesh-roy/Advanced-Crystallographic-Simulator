from PySide6.QtWidgets import (
    QWidget,
    QPushButton,
    QLabel,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
)

from .plot_widget import PlotWithToolbar
from core.ml.predictor import predict
import pandas as pd
from core.ml.predictor import predict, load_csv


class PredictorTab(QWidget):

    def __init__(self):

        super().__init__()

        self.csv_path = None

        self.build_ui()
        self.browse_btn.clicked.connect(self.browse_csv)
        self.predict_btn.clicked.connect(self.run_prediction)
        self.clear_btn.clicked.connect(self.clear_prediction)

    # ==========================================================
    # UI
    # ==========================================================

    def build_ui(self):

        main_layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()

        # -------------------------------------------------
        # Left Panel
        # -------------------------------------------------

        left_layout = QVBoxLayout()

        self.browse_btn = QPushButton(
            "📂 Browse CSV"
        )

        self.predict_btn = QPushButton(
            "▶ Predict"
        )

        self.clear_btn = QPushButton(
            "🗑 Clear"
        )

        left_layout.addWidget(self.browse_btn)
        left_layout.addWidget(self.predict_btn)
        left_layout.addWidget(self.clear_btn)
        left_layout.addStretch()

        # -------------------------------------------------
        # Plot
        # -------------------------------------------------

        self.plot = PlotWithToolbar()

        top_layout.addLayout(left_layout, 1)

        top_layout.addWidget(self.plot, 4)

        # -------------------------------------------------
        # Prediction Summary
        # -------------------------------------------------

        self.summary = QLabel()

        self.summary.setText(
            "Prediction Summary\n\n"
            "No prediction yet."
        )

        self.summary.setMinimumHeight(150)

        self.summary.setStyleSheet("""

        QLabel{

            background:white;

            color:black;

            border:1px solid gray;

            padding:10px;

            font-family:Consolas;

        }

        """)

        # -------------------------------------------------
        # Prediction Table
        # -------------------------------------------------

        self.table = QTableWidget()

        self.table.setColumnCount(3)

        self.table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Material",
                "Confidence",
            ]
        )

        self.table.setRowCount(5)

        # -------------------------------------------------

        main_layout.addLayout(top_layout)

        main_layout.addWidget(self.summary)

        main_layout.addWidget(self.table)

    def browse_csv(self):

        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Open CSV",
            "",
            "CSV Files (*.csv)"
        )

        if not file_name:
            return

        self.csv_path = file_name

        theta, intensity = load_csv(file_name)

        self.plot.plot(theta, intensity)

    def run_prediction(self):

        if self.csv_path is None:

            return

        result = predict(self.csv_path)

        self.show_prediction(result)

    def show_prediction(self, result):

        self.summary.setText(

            f"""Prediction Summary

    Material : {result["material"]}

    Confidence : {result["confidence"]*100:.2f} %

    Status : Prediction Complete
    """
        )

        predictions = result["top_predictions"][:5]

        self.table.setRowCount(len(predictions))

        for row, item in enumerate(predictions):

            self.table.setItem(
                row,
                0,
                QTableWidgetItem(str(row + 1))
            )

            self.table.setItem(
                row,
                1,
                QTableWidgetItem(item["material"])
            )

            self.table.setItem(
                row,
                2,
                QTableWidgetItem(
                    f"{item['confidence']*100:.2f}%"
                )
            )

    def clear_prediction(self):

        self.csv_path = None

        self.summary.setText(
            "Prediction Summary\n\nNo prediction yet."
        )

        self.table.clearContents()

        self.plot.clear()