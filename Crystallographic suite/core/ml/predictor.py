"""
predictor.py

Predict material from an experimental XRD CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from scipy.interpolate import interp1d

from .cnn_model import XRDClassifier
from .config import *


# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================================
# Load trained model
# ==========================================================

checkpoint = torch.load(
    MODEL_DIR / "xrd_classifier.pth",
    weights_only=False,
    map_location=DEVICE,
)

materials = checkpoint["materials"]

model = XRDClassifier(
    num_classes=len(materials)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)

model.eval()


# ==========================================================
# Read CSV
# ==========================================================

def load_csv(csv_path):

    try:
        df = pd.read_csv(csv_path)

        if df.shape[1] < 2:
            raise ValueError

        theta = df.iloc[:, 0].to_numpy(dtype=np.float32)
        intensity = df.iloc[:, 1].to_numpy(dtype=np.float32)

    except Exception:

        data = np.loadtxt(
            csv_path,
            delimiter=",",
            dtype=np.float32
        )

        theta = data[:, 0]
        intensity = data[:, 1]

    return theta, intensity


# ==========================================================
# Preprocess
# ==========================================================

def preprocess(theta, intensity):

    interp = interp1d(
        theta,
        intensity,
        bounds_error=False,
        fill_value=0,
    )

    profile = interp(THETA_GRID)

    profile = np.clip(profile, 0, None)

    if profile.max() > 0:
        profile /= profile.max()

    profile = torch.tensor(
        profile,
        dtype=torch.float32
    )

    profile = profile.unsqueeze(0).unsqueeze(0)

    return profile.to(DEVICE)


# ==========================================================
# Prediction
# ==========================================================

def predict(csv_path):

    theta, intensity = load_csv(csv_path)

    x = preprocess(theta, intensity)

    with torch.no_grad():

        logits = model(x)

        probabilities = F.softmax(
            logits,
            dim=1
        )

    probabilities = probabilities.cpu().numpy()[0]

    prediction = np.argmax(probabilities)

    print("\nPrediction")
    print("-" * 40)

    order = np.argsort(probabilities)[::-1]

    top_predictions = []

    for idx in order:

        top_predictions.append(

            {
                "material": materials[idx],
                "confidence": float(probabilities[idx]),
            }

        )

    return {

        "material": materials[prediction],

        "confidence": float(probabilities[prediction]),

        "top_predictions": top_predictions,

        "theta": theta,

        "intensity": intensity,
    }


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    csv = input("CSV Path : ").strip()

    predict(csv)