"""
core/forwardengine.py

Role: Structural Theory & Mathematical Engine (forward physics)

Uses:
  - calculate_atomic_form_factor       (from core.structural_physics)
  - get_allowed_reflections            (from core.structural_physics)

Inputs (from main/orchestrator):
  - element_symbol        : e.g. "Cu", "Au"
  - lattice_type          : "SC", "BCC", "FCC"
  - lattice_parameter_a   : unit cell edge a in Angstroms (for cubic)
  - wavelength_angstrom   : X-ray wavelength in Angstroms (default 1.5406)
  - two_theta_start_deg   : start angle of 2θ scan
  - two_theta_stop_deg    : stop angle of 2θ scan
  - two_theta_step_deg    : step size of 2θ scan
  - peak_profile_function : "Gaussian", "Lorentzian", or "Pseudo-Voigt"
  - instrument_fwhm_deg   : instrumental broadening FWHM in degrees

Provides:
  - Vegard's law for composition
  - sin²θ ratio indexing helper
  - residual vector and least-squares cost S
  - simulated forward XRD profile on a 2θ grid
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Dict

import numpy as np

# IMPORT EXISTING PHYSICS
from core.structural_physics import (
    calculate_atomic_form_factor,
    get_allowed_reflections,
)

# -------------------------------------------------------------------------
# 1. Vegard's Law Implementation
# -------------------------------------------------------------------------

def vegards_law_lattice_parameter(
    x: float,
    a_A: float,
    a_B: float
) -> float:
    if not (0.0 <= x <= 1.0):
        raise ValueError("Composition x must lie in [0, 1].")
    return (1.0 - x) * a_A + x * a_B


# -------------------------------------------------------------------------
# 2. sin²θ Ratio Indexing Engine
# -------------------------------------------------------------------------

CubicLattice = Literal["SC", "BCC", "FCC"]

@dataclass
class CubicIndexingResult:
    lattice_type: CubicLattice
    scores: Dict[CubicLattice, float]
    sin2theta: np.ndarray
    peak_positions_deg: np.ndarray


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Input must be non-empty 1D array.")
    if values[0] <= 0:
        raise ValueError("First value must be positive for normalization.")
    return values / values[0]


def classify_cubic_from_peaks(
    two_theta_deg: Sequence[float],
    max_peaks: int = 5
) -> CubicIndexingResult:
    """
    Classify SC/BCC/FCC using sin²θ ratios of low-angle peaks.
    """
    two_theta_deg = np.asarray(two_theta_deg, dtype=float)
    if two_theta_deg.ndim != 1 or len(two_theta_deg) < 2:
        raise ValueError("Provide at least two peak positions.")

    two_theta_deg = two_theta_deg[:max_peaks]

    theta_deg = two_theta_deg / 2.0
    theta_rad = np.deg2rad(theta_deg)
    sin2theta = np.sin(theta_rad) ** 2

    exp_norm = _normalize(sin2theta)
    n = len(exp_norm)

    sc_seq = np.arange(1, n + 1)
    bcc_seq = 2 * np.arange(1, n + 1)
    fcc_base = np.array([3, 4, 8, 11, 12, 16, 19])
    fcc_seq = fcc_base[:n] if n <= len(fcc_base) else np.concatenate(
        [fcc_base, np.arange(1, n - len(fcc_base) + 1) + fcc_base[-1]]
    )

    sc_norm = _normalize(sc_seq)
    bcc_norm = _normalize(bcc_seq)
    fcc_norm = _normalize(fcc_seq)

    def mismatch(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum((a - b) ** 2))

    scores: Dict[CubicLattice, float] = {
        "SC": mismatch(exp_norm, sc_norm),
        "BCC": mismatch(exp_norm, bcc_norm),
        "FCC": mismatch(exp_norm, fcc_norm),
    }
    best_lattice: CubicLattice = min(scores, key=scores.get)  # type: ignore[arg-type]

    return CubicIndexingResult(
        lattice_type=best_lattice,
        scores=scores,
        sin2theta=sin2theta,
        peak_positions_deg=two_theta_deg,
    )


# -------------------------------------------------------------------------
# 3. Residual Vector and Objective Cost
# -------------------------------------------------------------------------

def residual_vector(
    y_exp: Sequence[float],
    y_sim: Sequence[float],
    weights: Sequence[float] | None = None
) -> np.ndarray:
    y_exp = np.asarray(y_exp, dtype=float)
    y_sim = np.asarray(y_sim, dtype=float)

    if y_exp.shape != y_sim.shape:
        raise ValueError("y_exp and y_sim must have same shape.")

    if weights is None:
        w = np.ones_like(y_exp)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != y_exp.shape:
            raise ValueError("weights must have same shape as y_exp.")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")

    return np.sqrt(w) * (y_exp - y_sim)


def least_squares_cost(
    y_exp: Sequence[float],
    y_sim: Sequence[float],
    weights: Sequence[float] | None = None
) -> float:
    r = residual_vector(y_exp, y_sim, weights)
    return float(np.dot(r, r))


# -------------------------------------------------------------------------
# 4. Peak Profile Functions
# -------------------------------------------------------------------------

def _gaussian_profile(
    two_theta_deg: np.ndarray,
    center_deg: float,
    fwhm_deg: float,
    intensity: float
) -> np.ndarray:
    sigma = fwhm_deg / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return intensity * np.exp(-0.5 * ((two_theta_deg - center_deg) / sigma) ** 2)


def _lorentzian_profile(
    two_theta_deg: np.ndarray,
    center_deg: float,
    fwhm_deg: float,
    intensity: float
) -> np.ndarray:
    gamma = fwhm_deg / 2.0
    return intensity * (gamma**2 / ((two_theta_deg - center_deg) ** 2 + gamma**2))


def _pseudo_voigt_profile(
    two_theta_deg: np.ndarray,
    center_deg: float,
    fwhm_deg: float,
    intensity: float,
    eta: float = 0.5
) -> np.ndarray:
    return (
        eta * _lorentzian_profile(two_theta_deg, center_deg, fwhm_deg, intensity)
        + (1.0 - eta) * _gaussian_profile(two_theta_deg, center_deg, fwhm_deg, intensity)
    )


# -------------------------------------------------------------------------
# 5. Forward Simulation using structural_physics inputs
# -------------------------------------------------------------------------

def simulate_xrd_pattern(
    element_symbol: str,
    lattice_type: str,
    lattice_parameter_a: float,
    wavelength_angstrom: float = 1.5406,
    two_theta_start_deg: float = 20.0,
    two_theta_stop_deg: float = 90.0,
    two_theta_step_deg: float = 0.02,
    peak_profile_function: Literal["Gaussian", "Lorentzian", "Pseudo-Voigt"] = "Lorentzian",
    instrument_fwhm_deg: float = 0.15,
    crystallite_size_nm: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Forward physics engine: generate a simulated XRD profile on a 2θ grid.

    NOTE: Uses get_allowed_reflections and calculate_atomic_form_factor
    from core.structural_physics, rather than reimplementing them.
    """
    two_theta_deg = np.arange(two_theta_start_deg, two_theta_stop_deg + two_theta_step_deg, two_theta_step_deg)
    intensity_total = np.zeros_like(two_theta_deg)

    planes = get_allowed_reflections(lattice_type)

    for h, k, l in planes:
        d_hkl = lattice_parameter_a / np.sqrt(h**2 + k**2 + l**2)

        arg = wavelength_angstrom / (2.0 * d_hkl)
        if arg >= 1.0:
            continue

        theta_rad = np.arcsin(arg)
        two_theta_peak_deg = np.degrees(2.0 * theta_rad)

        if two_theta_peak_deg < two_theta_start_deg or two_theta_peak_deg > two_theta_stop_deg:
            continue

        f_j = calculate_atomic_form_factor(element_symbol, np.array([theta_rad]), wavelength_angstrom)[0]
        peak_intensity = float(f_j**2)

        fwhm_micro_deg = 0.0
        if crystallite_size_nm is not None and crystallite_size_nm > 0.0:
            D_angstrom = crystallite_size_nm * 10.0
            kappa = 0.9
            beta_rad = (kappa * wavelength_angstrom) / (D_angstrom * np.cos(theta_rad))
            fwhm_micro_deg = np.degrees(beta_rad)

        fwhm_total_deg = np.sqrt(instrument_fwhm_deg**2 + fwhm_micro_deg**2)

        if peak_profile_function == "Gaussian":
            profile = _gaussian_profile(two_theta_deg, two_theta_peak_deg, fwhm_total_deg, peak_intensity)
        elif peak_profile_function == "Lorentzian":
            profile = _lorentzian_profile(two_theta_deg, two_theta_peak_deg, fwhm_total_deg, peak_intensity)
        else:
            profile = _pseudo_voigt_profile(two_theta_deg, two_theta_peak_deg, fwhm_total_deg, peak_intensity)

        intensity_total += profile

    return two_theta_deg, intensity_total
