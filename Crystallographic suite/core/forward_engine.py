# Physics simulation logic



"""
core/forwardengine.py

Structural theory & mathematical engine:
- Vegard's law lattice parameter interpolation
- sin^2(2θ) ratio-based cubic indexing helper
- Objective residual matrix for least-squares refinement

All functions are stateless and NumPy-friendly so they can be
composed with other modules (form factors, structure factors,
peak-profile engine, optimizers, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple, Dict

import numpy as np


# -------------------------------------------------------------------------
# 1. Vegard's Law Implementation
#    a_alloy = (1 − x) * a_A + x * a_B
# -------------------------------------------------------------------------

def vegards_law_lattice_parameter(
    x: float,
    a_A: float,
    a_B: float
) -> float:
    """
    Compute alloy lattice parameter using Vegard's law for a binary A_(1-x) B_x alloy.

    Parameters
    ----------
    x : float
        Mole fraction of species B in A_(1-x) B_x.
        Expected range: 0.0 <= x <= 1.0
    a_A : float
        Lattice parameter of pure end-member A (same units as output).
    a_B : float
        Lattice parameter of pure end-member B (same units as output).

    Returns
    -------
    float
        Lattice parameter of the alloy a_alloy.
    """
    if not (0.0 <= x <= 1.0):
        raise ValueError("Composition x must lie in [0, 1].")

    a_alloy = (1.0 - x) * a_A + x * a_B
    return a_alloy


# -------------------------------------------------------------------------
# 2. sin^2(2θ) Ratio Indexing Engine
#    Classify cubic system from peak positions
# -------------------------------------------------------------------------

CubicLattice = Literal["SC", "BCC", "FCC"]


@dataclass
class CubicIndexingResult:
    lattice_type: CubicLattice
    scores: Dict[CubicLattice, float]
    sin2theta: np.ndarray


def _normalize_ratios(values: np.ndarray) -> np.ndarray:
    """
    Normalize a positive array to unit first element: r_i = values_i / values_0.

    Used to compare experimental sin^2(2θ) ratios against ideal integer sequences.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("Input must be 1D.")
    if len(values) == 0:
        raise ValueError("Input array must be non-empty.")

    # Guard against zero or negative leading value
    v0 = values[0]
    if v0 <= 0:
        raise ValueError("First value must be positive for ratio normalization.")

    return values / v0


def classify_cubic_from_peaks(
    two_theta_deg: Sequence[float],
    wavelength: float,
    max_peaks: int | None = 5
) -> CubicIndexingResult:
    """
    Classify an unknown cubic lattice as SC, BCC, or FCC using sin^2(2θ) ratios.

    Parameters
    ----------
    two_theta_deg : Sequence[float]
        Peak center positions in 2θ (degrees), sorted ascending. These should
        be already baseline-corrected and identified by a peak-finder upstream.
    wavelength : float
        X-ray wavelength in same units as used elsewhere (e.g. Angstrom).
        Included for completeness; here we only use angle ratios, so λ cancels.
    max_peaks : int, optional
        Maximum number of lowest-angle peaks to use in the ratio comparison.
        Defaults to 5. If fewer peaks are provided, all will be used.

    Returns
    -------
    CubicIndexingResult
        lattice_type : "SC", "BCC", or "FCC" with smallest mismatch score.
        scores       : mapping from lattice type to residual score.
        sin2theta    : NumPy array of sin^2(2θ) used for the comparison.

    Notes
    -----
    Ideal cubic sequences (proportional to h^2 + k^2 + l^2):

    - SC  : 1, 2, 3, 4, 5, ...
    - BCC : 2, 4, 6, 8, 10, ... (h^2+k^2+l^2 even)
    - FCC : 3, 4, 8, 11, 12, ... (first few allowed sums)

    Here we compare *normalized* experimental ratios to normalized versions
    of these integer sequences using a simple least-squares mismatch.
    """
    # Convert to NumPy and trim to first few peaks
    two_theta_deg = np.asarray(two_theta_deg, dtype=float)
    if two_theta_deg.ndim != 1:
        raise ValueError("two_theta_deg must be 1D.")
    if len(two_theta_deg) < 2:
        raise ValueError("At least two peak positions are required.")

    if max_peaks is not None and max_peaks > 0:
        two_theta_deg = two_theta_deg[:max_peaks]

    # Compute sin^2(2θ) in radians; wavelength cancels for relative indexing
    two_theta_rad = np.deg2rad(two_theta_deg)
    sin2theta = np.sin(two_theta_rad) ** 2  # this is sin^2(2θ)

    # Normalize experimental ratios
    exp_ratios = _normalize_ratios(sin2theta)

    n = len(exp_ratios)

    # Ideal integer sequences (first n values)
    sc_seq = np.arange(1, n + 1)                     # 1, 2, 3, ...
    bcc_seq = 2 * np.arange(1, n + 1)                # 2, 4, 6, ...
    fcc_base = np.array([3, 4, 8, 11, 12, 16, 19])   # extendable if needed
    if n > len(fcc_base):
        # Simple extension by approximate spacing; can be refined later
        extra = np.arange(1, n - len(fcc_base) + 1) + fcc_base[-1]
        fcc_seq = np.concatenate([fcc_base, extra])
    else:
        fcc_seq = fcc_base[:n]

    # Normalize ideal sequences
    sc_norm = _normalize_ratios(sc_seq)
    bcc_norm = _normalize_ratios(bcc_seq)
    fcc_norm = _normalize_ratios(fcc_seq)

    # Compute simple least-squares mismatch between experimental and each ideal
    def mismatch(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum((a - b) ** 2))

    scores: Dict[CubicLattice, float] = {
        "SC": mismatch(exp_ratios, sc_norm),
        "BCC": mismatch(exp_ratios, bcc_norm),
        "FCC": mismatch(exp_ratios, fcc_norm),
    }

    best_lattice: CubicLattice = min(scores, key=scores.get)  # type: ignore[arg-type]

    return CubicIndexingResult(
        lattice_type=best_lattice,
        scores=scores,
        sin2theta=sin2theta,
    )


# -------------------------------------------------------------------------
# 3. Objective Refinement Matrix
#    Least-squares cost S = Σ w_i * |y_exp - y_sim|^2
# -------------------------------------------------------------------------

def residual_vector(
    y_exp: Sequence[float],
    y_sim: Sequence[float],
    weights: Sequence[float] | None = None
) -> np.ndarray:
    """
    Compute weighted residual vector r_i = sqrt(w_i) * (y_exp_i - y_sim_i).

    Parameters
    ----------
    y_exp : Sequence[float]
        Experimental intensity array after cleaning/baseline subtraction.
    y_sim : Sequence[float]
        Simulated intensity array evaluated on the same x-grid as y_exp.
    weights : Sequence[float], optional
        Non-negative weights w_i (e.g. 1/σ_i^2). If None, all weights = 1.

    Returns
    -------
    np.ndarray
        Residual vector r suitable for passing into least-squares optimizers:
        S = ||r||^2 = Σ w_i * (y_exp_i - y_sim_i)^2
    """
    y_exp = np.asarray(y_exp, dtype=float)
    y_sim = np.asarray(y_sim, dtype=float)

    if y_exp.shape != y_sim.shape:
        raise ValueError("y_exp and y_sim must have the same shape.")

    if weights is None:
        w = np.ones_like(y_exp)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != y_exp.shape:
            raise ValueError("weights must have the same shape as y_exp.")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")

    resid = y_exp - y_sim
    r = np.sqrt(w) * resid
    return r


def least_squares_cost(
    y_exp: Sequence[float],
    y_sim: Sequence[float],
    weights: Sequence[float] | None = None
) -> float:
    """
    Compute scalar least-squares objective S = Σ w_i * (y_exp_i - y_sim_i)^2.

    Parameters
    ----------
    y_exp : Sequence[float]
        Experimental intensity array after cleaning/baseline subtraction.
    y_sim : Sequence[float]
        Simulated intensity array evaluated on the same x-grid as y_exp.
    weights : Sequence[float], optional
        Non-negative weights w_i (e.g. 1/σ_i^2). If None, all weights = 1.

    Returns
    -------
    float
        Scalar cost S suitable for monitoring optimization progress.
    """
    r = residual_vector(y_exp, y_sim, weights=weights)
    return float(np.dot(r, r))
