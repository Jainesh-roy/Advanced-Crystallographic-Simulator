"""
core/forwardengine.py

Structural theory & mathematical engine:
- Vegard's law lattice parameter interpolation
- sin^2(2θ) ratio-based cubic indexing
- Objective residual matrix for least-squares
- Core XRD synthetic profile simulation
"""

import numpy as np
import periodictable
from typing import Literal, Sequence, Dict
from dataclasses import dataclass

# 1. Vegard's Law Implementation
def vegards_law_lattice_parameter(x: float, a_A: float, a_B: float) -> float:
    """Compute alloy lattice parameter using Vegard's law."""
    if not (0.0 <= x <= 1.0):
        raise ValueError("Composition x must lie in [0, 1].")
    return (1.0 - x) * a_A + x * a_B


# 2. Cubic Indexing Core
CubicLattice = Literal["SC", "BCC", "FCC"]

@dataclass
class CubicIndexingResult:
    lattice_type: CubicLattice
    scores: Dict[CubicLattice, float]
    sin2theta: np.ndarray

def _normalize_ratios(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values / values[0]

def classify_cubic_from_peaks(two_theta_deg: Sequence[float], max_peaks: int = 5) -> CubicIndexingResult:
    two_theta_deg = np.asarray(two_theta_deg, dtype=float)[:max_peaks]
    sin2theta = np.sin(np.deg2rad(two_theta_deg) / 2.0) ** 2 
    
    exp_ratios = _normalize_ratios(sin2theta)
    n = len(exp_ratios)
    
    sc_seq = np.arange(1, n + 1)
    bcc_seq = 2 * np.arange(1, n + 1)
    fcc_seq = np.array([3, 4, 8, 11, 12, 16, 19])[:n]
    
    scores = {
        "SC": float(np.sum((exp_ratios - _normalize_ratios(sc_seq)) ** 2)),
        "BCC": float(np.sum((exp_ratios - _normalize_ratios(bcc_seq)) ** 2)),
        "FCC": float(np.sum((exp_ratios - _normalize_ratios(fcc_seq)) ** 2)),
    }
    best_lattice = min(scores, key=scores.get)
    return CubicIndexingResult(lattice_type=best_lattice, scores=scores, sin2theta=sin2theta)


# 3. Residual Matrix
def least_squares_cost(y_exp: Sequence[float], y_sim: Sequence[float], weights: Sequence[float] = None) -> float:
    """Objective residual tracking function S = wi |y_exp - y_sim|^2"""
    y_exp, y_sim = np.asarray(y_exp), np.asarray(y_sim)
    w = np.ones_like(y_exp) if weights is None else np.asarray(weights)
    r = np.sqrt(w) * (y_exp - y_sim)
    return float(np.dot(r, r))


# 4. Multi-Physics Forward Profile Generator
def calculate_atomic_form_factor(element_symbol: str, theta_rad: np.ndarray, wavelength: float = 1.5406) -> np.ndarray:
    element = getattr(periodictable, element_symbol)
    q_vector = (4 * np.pi * np.sin(theta_rad)) / wavelength
    return np.vectorize(element.xray.f0)(q_vector)

def get_allowed_reflections(lattice_type: str) -> list:
    allowed_planes = []
    for h in range(5):
        for k in range(5):
            for l in range(5):
                if h == 0 and k == 0 and l == 0: continue
                if lattice_type == "SC": allowed_planes.append((h, k, l))
                elif lattice_type == "BCC" and (h + k + l) % 2 == 0: allowed_planes.append((h, k, l))
                elif lattice_type == "FCC" and (h % 2 == k % 2 == l % 2): allowed_planes.append((h, k, l))
    return allowed_planes

def lorentzian_profile(two_theta: np.ndarray, peak_center: float, fwhm: float, intensity: float) -> np.ndarray:
    gamma = fwhm / 2.0
    return intensity * (gamma**2 / ((two_theta - peak_center)**2 + gamma**2))

def simulate_xrd_pattern(lattice_type: str, a_nm: float, element: str, 
                         crystallite_nm: float, wavelength: float = 1.5406,
                         theta_min: float = 20, theta_max: float = 100, step: float = 0.05):
    two_theta = np.arange(theta_min, theta_max, step)
    intensity_total = np.zeros_like(two_theta)
    
    for h, k, l in get_allowed_reflections(lattice_type):
        d_spacing = (a_nm * 10.0) / np.sqrt(h**2 + k**2 + l**2)
        if d_spacing <= wavelength / 2.0: continue
            
        theta_rad = np.arcsin(wavelength / (2.0 * d_spacing))
        peak_center_deg = np.degrees(2.0 * theta_rad)
        if not (theta_min <= peak_center_deg <= theta_max): continue
            
        f_j = calculate_atomic_form_factor(element, np.array([theta_rad]), wavelength)[0]
        peak_intensity = f_j**2
        
        # Microstructural Peak Broadening (Scherrer Equation)
        fwhm_rad = (0.9 * wavelength) / (crystallite_nm * 10.0 * np.cos(theta_rad))
        fwhm_deg = np.degrees(fwhm_rad)
        
        intensity_total += lorentzian_profile(two_theta, peak_center_deg, fwhm_deg, peak_intensity)
        
    return two_theta, intensity_total
