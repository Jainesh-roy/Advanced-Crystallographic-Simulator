"""
augmentations.py

Realistic XRD data augmentation functions.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .config import *

rng = np.random.default_rng(RANDOM_SEED)


# ==========================================================
# Gaussian Noise
# ==========================================================

def add_gaussian_noise(profile):

    std = rng.uniform(*NOISE_STD_RANGE)

    noise = rng.normal(0, std / 1000.0, profile.shape)

    return np.clip(profile + noise, 0, None)


# ==========================================================
# Intensity Scaling
# ==========================================================

def scale_intensity(profile):

    scale = rng.uniform(0.7, 1.3)

    profile = profile * scale

    return profile


# ==========================================================
# Background
# ==========================================================

def add_background(profile):

    x = np.linspace(0, 1, len(profile))

    level = rng.uniform(0.00, 0.05)

    slope = rng.uniform(-0.05, 0.05)

    background = level + slope * x

    return profile + background


# ==========================================================
# Peak Broadening
# ==========================================================

def broaden_peaks(profile):

    sigma = rng.uniform(*SIGMA_RANGE)

    return gaussian_filter1d(profile, sigma=sigma)


# ==========================================================
# Weak Peak Suppression
# ==========================================================

def weaken_random_peaks(profile):

    mask = rng.random(profile.shape)

    profile = profile.copy()

    profile[mask < 0.01] *= rng.uniform(0.3, 0.8)

    return profile


# ==========================================================
# Zero Shift
# ==========================================================

def shift_profile(profile):

    shift = rng.uniform(*SHIFT_RANGE)

    bins = int(round(shift / STEP_SIZE))

    return np.roll(profile, bins)


# ==========================================================
# Normalize
# ==========================================================

def normalize(profile):

    profile = np.clip(profile, 0, None)

    maximum = profile.max()

    if maximum > 0:
        profile = profile / maximum

    return profile.astype(np.float32)


# ==========================================================
# Complete Augmentation Pipeline
# ==========================================================

def augment(profile):

    profile = broaden_peaks(profile)

    profile = scale_intensity(profile)

    profile = add_background(profile)

    profile = add_gaussian_noise(profile)

    profile = weaken_random_peaks(profile)

    profile = shift_profile(profile)

    profile = normalize(profile)

    return profile