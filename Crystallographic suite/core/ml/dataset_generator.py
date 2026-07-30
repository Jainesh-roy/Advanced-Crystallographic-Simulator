"""
dataset_generator.py

Generates PyTorch datasets from Materials Project crystal structures.
"""

from pathlib import Path

import numpy as np
import torch

from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

from sklearn.model_selection import train_test_split

from .config import *
from .augmentations import augment


# ==========================================================
# XRD Calculator
# ==========================================================

calculator = XRDCalculator(wavelength="CuKa")


# ==========================================================
# Theta Grid
# ==========================================================

THETA_GRID = np.arange(
    TWO_THETA_MIN,
    TWO_THETA_MAX + STEP_SIZE,
    STEP_SIZE,
)


# ==========================================================
# Load Structures
# ==========================================================

def load_structures():

    structures = {}

    cif_files = sorted(STRUCTURE_DIR.glob("*.cif"))

    if len(cif_files) == 0:
        raise RuntimeError(
            f"No CIF files found in {STRUCTURE_DIR}"
        )

    for cif in cif_files:

        label = cif.stem

        structure = Structure.from_file(cif)

        structures[label] = structure

    return structures


# ==========================================================
# Generate Continuous XRD Profile
# ==========================================================

def generate_profile(structure):

    pattern = calculator.get_pattern(structure)

    theta = np.asarray(pattern.x)

    intensity = np.asarray(pattern.y)

    profile = np.zeros_like(THETA_GRID)

    sigma = 0.10

    for peak_theta, peak_intensity in zip(theta, intensity):

        profile += peak_intensity * np.exp(
            -((THETA_GRID - peak_theta) ** 2)
            / (2 * sigma ** 2)
        )

    if profile.max() > 0:
        profile /= profile.max()

    return profile.astype(np.float32)


# ==========================================================
# Dataset Generation
# ==========================================================

def generate_dataset(samples_per_material=1000):

    structures = load_structures()

    X = []
    y = []

    material_names = sorted(structures.keys())

    label_map = {
        name: idx
        for idx, name in enumerate(material_names)
    }

    print()

    for material in material_names:

        print(f"Generating {material}...")

        base_profile = generate_profile(
            structures[material]
        )

        for _ in range(samples_per_material):

            profile = augment(base_profile.copy())

            X.append(profile)

            y.append(label_map[material])

    X = np.asarray(X, dtype=np.float32)

    y = np.asarray(y, dtype=np.int64)

    return X, y, material_names


# ==========================================================
# Save Dataset
# ==========================================================

def save_dataset(samples_per_material=1000):

    X, y, material_names = generate_dataset(
        samples_per_material
    )

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1 - TRAIN_SPLIT),
        random_state=RANDOM_SEED,
        stratify=y,
    )

    validation_fraction = VALIDATION_SPLIT / (
        VALIDATION_SPLIT + TEST_SPLIT
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - validation_fraction),
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    DATASET_DIR.mkdir(exist_ok=True)

    torch.save(
        {
            "X": torch.tensor(X_train),
            "y": torch.tensor(y_train),
            "materials": material_names,
            "theta": THETA_GRID,
        },
        DATASET_DIR / "train.pt",
    )

    torch.save(
        {
            "X": torch.tensor(X_val),
            "y": torch.tensor(y_val),
            "materials": material_names,
            "theta": THETA_GRID,
        },
        DATASET_DIR / "validation.pt",
    )

    torch.save(
        {
            "X": torch.tensor(X_test),
            "y": torch.tensor(y_test),
            "materials": material_names,
            "theta": THETA_GRID,
        },
        DATASET_DIR / "test.pt",
    )

    print("\nDataset Generated Successfully\n")

    print(f"Train Samples      : {len(X_train)}")
    print(f"Validation Samples : {len(X_val)}")
    print(f"Test Samples       : {len(X_test)}")


# ==========================================================
# Quick Test
# ==========================================================

def test():

    structures = load_structures()

    print("\nLoaded Structures\n")

    for name, structure in structures.items():

        profile = generate_profile(structure)

        print(
            f"{name:10s} -> "
            f"{structure.composition.reduced_formula:8s} "
            f"{profile.shape}"
        )


if __name__ == "__main__":

    test()

    # Uncomment to generate dataset
    save_dataset(samples_per_material=1000)