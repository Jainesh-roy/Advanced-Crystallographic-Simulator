"""
Image Processing Utilities
--------------------------
Convert an XRD graph image into
(theta, intensity) arrays.
"""

from pathlib import Path

import cv2
import numpy as np


class XRDImageProcessor:

    def __init__(self):

        self.image = None

    # -----------------------------------------------------
    # Load Image
    # -----------------------------------------------------

    def load_image(self, image_path):

        self.image = cv2.imread(str(image_path))

        if self.image is None:

            raise ValueError(
                f"Cannot open image:\n{image_path}"
            )

        return self.image

    # -----------------------------------------------------
    # Show Image
    # -----------------------------------------------------

    def show_image(self, title="Image"):

        if self.image is None:
            raise ValueError("No image loaded.")

        cv2.imshow(title, self.image)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    # -----------------------------------------------------
    # Convert to Grayscale
    # -----------------------------------------------------

    def to_gray(self):

        if self.image is None:
            raise ValueError("Load an image first.")

        gray = cv2.cvtColor(
            self.image,
            cv2.COLOR_BGR2GRAY,
        )

        return gray

    # -----------------------------------------------------
    # Edge Detection
    # -----------------------------------------------------

    def detect_edges(self):

        gray = self.to_gray()

        edges = cv2.Canny(
            gray,
            50,
            150,
        )

        return edges

    # -----------------------------------------------------
    # Detect Plot Region
    # -----------------------------------------------------

    def detect_plot_region(self):

        gray = self.to_gray()

        edges = cv2.Canny(gray, 50, 150)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        largest = max(
            contours,
            key=cv2.contourArea,
        )

        x, y, w, h = cv2.boundingRect(largest)

        cropped = self.image[
            y:y+h,
            x:x+w,
        ]

        return cropped

    # -----------------------------------------------------
    # Binary Image
    # -----------------------------------------------------

    def binary(self):

        gray = cv2.cvtColor(
            self.detect_plot_region(),
            cv2.COLOR_BGR2GRAY,
        )

        _, binary = cv2.threshold(
            gray,
            220,
            255,
            cv2.THRESH_BINARY_INV,
        )

        return binary