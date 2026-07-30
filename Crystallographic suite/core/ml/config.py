"""
Global configuration for the XRD AI project.
"""

from pathlib import Path
import numpy as np

# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

STRUCTURE_DIR = PROJECT_ROOT / "structures"

DATASET_DIR = PROJECT_ROOT / "dataset"

MODEL_DIR = PROJECT_ROOT / "models"

DATASET_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ==========================================================
# XRD PARAMETERS
# ==========================================================

TWO_THETA_MIN = 20.0
TWO_THETA_MAX = 90.0
STEP_SIZE = 0.02

THETA_GRID = np.arange(
    TWO_THETA_MIN,
    TWO_THETA_MAX + STEP_SIZE,
    STEP_SIZE,
)

# ==========================================================
# DATASET PARAMETERS
# ==========================================================

SAMPLES_PER_MATERIAL = 1000

TRAIN_SPLIT = 0.70
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# ==========================================================
# DATA AUGMENTATION
# ==========================================================

NOISE_STD_RANGE = (2, 20)

SHIFT_RANGE = (-0.08, 0.08)

SIGMA_RANGE = (0.06, 0.16)

BACKGROUND_LEVEL_RANGE = (10, 40)

BACKGROUND_SLOPE_RANGE = (0.00, 0.30)

INTENSITY_SCALE_RANGE = (500, 5000)

# ==========================================================
# RANDOM SEED
# ==========================================================

RANDOM_SEED = 42