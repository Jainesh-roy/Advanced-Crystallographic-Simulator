from pathlib import Path

import cv2

from core.ml.image_processor import XRDImageProcessor


PROJECT_ROOT = Path(__file__).resolve().parent.parent

image_path = PROJECT_ROOT / "Images" / "test.png"

processor = XRDImageProcessor()

img = processor.load_image(image_path)

binary = processor.binary()

cv2.imshow(
    "Binary",
    binary,
)

cv2.waitKey(0)

cv2.destroyAllWindows()