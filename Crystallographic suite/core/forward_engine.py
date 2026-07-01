"""
Forward XRD simulation engine for cubic SC/BCC/FCC crystals.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np

from core.structural_physics import (
    calculate_atomic_form_factor,
    calculate_structure_factor,
    get_allowed_reflections,
)

PeakProfile = Literal["Gaussian", "Lorentzian"]


@dataclass
class XRDPeak:
    h: int
    k: int
    l: int
    multiplicity: int
    d_spacing_angstrom: float
    theta_deg: float
    two_theta_deg: float
    form_factor: float
    structure_factor_abs: float
    lorentz_polarization: float
    debye_waller: float
    relative_intensity: float
    fwhm_deg: float


def vegards_law_lattice_parameter(x: float, a_A: float, a_B: float) -> float:
    """
    Vegard's law for A(1-x)B(x) binary substitutional solid solutions.
    """
    if not (0.0 <= x <= 1.0):
        raise ValueError("Composition x must lie in [0, 1].")
    if a_A <= 0 or a_B <= 0:
        raise ValueError("Endpoint lattice parameters must be positive.")
    return (1.0 - x) * a_A + x * a_B


def effective_form_factor(
    element_a: str,
    theta_rad: np.ndarray,
    wavelength_angstrom: float,
    composition_x: float | None = None,
    element_b: str | None = None,
) -> np.ndarray:
    """
    Atomic form factor for a pure element or A(1-x)B(x) solid solution.
    """
    f_a = calculate_atomic_form_factor(element_a, theta_rad, wavelength_angstrom)
    if composition_x is None:
        return f_a
    if element_b is None:
        raise ValueError("element_b is required when composition_x is provided.")
    if not (0.0 <= composition_x <= 1.0):
        raise ValueError("composition_x must lie in [0, 1].")

    f_b = calculate_atomic_form_factor(element_b, theta_rad, wavelength_angstrom)
    return (1.0 - composition_x) * f_a + composition_x * f_b


def _cubic_multiplicity(plane: tuple[int, int, int]) -> int:
    nonzero_count = sum(index != 0 for index in plane)
    unique_permutations = set(itertools.permutations(plane, 3))
    return len(unique_permutations) * (2**nonzero_count)


def _representative_peak_families(lattice_type: str, max_index: int) -> list[tuple[int, int, int, int]]:
    grouped: dict[int, set[tuple[int, int, int]]] = {}
    for plane in get_allowed_reflections(lattice_type, max_index=max_index):
        grouped.setdefault(sum(index**2 for index in plane), set()).add(tuple(sorted(plane, reverse=True)))

    families = []
    for hkl_sum, planes in grouped.items():
        h, k, l = max(planes)
        multiplicity = sum(_cubic_multiplicity(plane) for plane in planes)
        families.append((hkl_sum, h, k, l, multiplicity))

    families.sort(key=lambda family: family[0])
    return [(h, k, l, multiplicity) for _, h, k, l, multiplicity in families]


def _scherrer_fwhm_deg(
    theta_rad: float,
    wavelength_angstrom: float,
    crystallite_size_nm: float | None,
    instrument_fwhm_deg: float,
) -> float:
    if instrument_fwhm_deg < 0:
        raise ValueError("instrument_fwhm_deg must be non-negative.")
    if crystallite_size_nm is None:
        return instrument_fwhm_deg
    if crystallite_size_nm <= 0:
        raise ValueError("crystallite_size_nm must be positive.")

    crystallite_size_angstrom = crystallite_size_nm * 10.0
    beta_rad = (0.9 * wavelength_angstrom) / (crystallite_size_angstrom * np.cos(theta_rad))
    beta_deg = float(np.degrees(beta_rad))
    return float(np.sqrt(instrument_fwhm_deg**2 + beta_deg**2))


def lorentz_polarization_factor(theta_rad: float) -> float:
    """
    Powder XRD Lorentz-polarization correction for unpolarized lab radiation.
    """
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)
    cos_2theta = np.cos(2.0 * theta_rad)
    if sin_theta <= 0.0 or cos_theta <= 0.0:
        return 0.0
    return float((1.0 + cos_2theta**2) / (sin_theta**2 * cos_theta))


def debye_waller_factor(theta_rad: float, wavelength_angstrom: float, b_iso: float) -> float:
    """
    Isotropic Debye-Waller intensity damping term.
    """
    if b_iso < 0:
        raise ValueError("b_iso must be non-negative.")
    return float(np.exp(-2.0 * b_iso * (np.sin(theta_rad) / wavelength_angstrom) ** 2))


def calculate_xrd_peaks(
    element_symbol: str,
    lattice_type: str,
    lattice_parameter_a: float,
    wavelength_angstrom: float = 1.5406,
    two_theta_start_deg: float = 20.0,
    two_theta_stop_deg: float = 90.0,
    instrument_fwhm_deg: float = 0.15,
    crystallite_size_nm: float | None = None,
    max_index: int = 6,
    element_b: str | None = None,
    composition_x: float | None = None,
    b_iso: float = 0.5,
) -> list[XRDPeak]:
    """
    Calculate Bragg peak positions and relative intensities for cubic XRD.
    Intensity is modeled as multiplicity * |F|^2 * LP(theta) * Debye-Waller.
    """
    if lattice_parameter_a <= 0:
        raise ValueError("lattice_parameter_a must be positive.")
    if wavelength_angstrom <= 0:
        raise ValueError("wavelength_angstrom must be positive.")
    if two_theta_stop_deg <= two_theta_start_deg:
        raise ValueError("two_theta_stop_deg must be greater than two_theta_start_deg.")
    if max_index <= 0:
        raise ValueError("max_index must be positive.")

    peaks: list[XRDPeak] = []
    for h, k, l, multiplicity in _representative_peak_families(lattice_type, max_index):
        d_spacing = lattice_parameter_a / np.sqrt(h**2 + k**2 + l**2)
        bragg_arg = wavelength_angstrom / (2.0 * d_spacing)
        if bragg_arg >= 1.0:
            continue

        theta_rad = float(np.arcsin(bragg_arg))
        theta_deg = float(np.degrees(theta_rad))
        two_theta_deg = 2.0 * theta_deg
        if not (two_theta_start_deg <= two_theta_deg <= two_theta_stop_deg):
            continue

        form_factor = float(
            effective_form_factor(
                element_symbol,
                np.array([theta_rad]),
                wavelength_angstrom,
                composition_x=composition_x,
                element_b=element_b,
            )[0]
        )
        structure_factor_abs = float(
            abs(calculate_structure_factor(lattice_type, h, k, l, form_factor))
        )
        lp_factor = lorentz_polarization_factor(theta_rad)
        dw_factor = debye_waller_factor(theta_rad, wavelength_angstrom, b_iso)
        intensity = multiplicity * structure_factor_abs**2 * lp_factor * dw_factor
        fwhm_deg = _scherrer_fwhm_deg(
            theta_rad,
            wavelength_angstrom,
            crystallite_size_nm,
            instrument_fwhm_deg,
        )

        peaks.append(
            XRDPeak(
                h=h,
                k=k,
                l=l,
                multiplicity=multiplicity,
                d_spacing_angstrom=float(d_spacing),
                theta_deg=theta_deg,
                two_theta_deg=two_theta_deg,
                form_factor=form_factor,
                structure_factor_abs=structure_factor_abs,
                lorentz_polarization=lp_factor,
                debye_waller=dw_factor,
                relative_intensity=float(intensity),
                fwhm_deg=fwhm_deg,
            )
        )

    return sorted(peaks, key=lambda peak: peak.two_theta_deg)


def _gaussian_profile(
    two_theta_deg: np.ndarray,
    center_deg: float,
    fwhm_deg: float,
    intensity: float,
) -> np.ndarray:
    sigma = fwhm_deg / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return (intensity / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((two_theta_deg - center_deg) / sigma) ** 2)


def _lorentzian_profile(
    two_theta_deg: np.ndarray,
    center_deg: float,
    fwhm_deg: float,
    intensity: float,
) -> np.ndarray:
    gamma = fwhm_deg / 2.0
    return intensity * (gamma / np.pi) / ((two_theta_deg - center_deg) ** 2 + gamma**2)


def simulate_xrd_pattern(
    element_symbol: str,
    lattice_type: str,
    lattice_parameter_a: float,
    wavelength_angstrom: float = 1.5406,
    two_theta_start_deg: float = 20.0,
    two_theta_stop_deg: float = 90.0,
    two_theta_step_deg: float = 0.02,
    peak_profile_function: PeakProfile = "Lorentzian",
    instrument_fwhm_deg: float = 0.15,
    crystallite_size_nm: float | None = None,
    max_index: int = 6,
    element_b: str | None = None,
    composition_x: float | None = None,
    b_iso: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a continuous simulated XRD profile on a 2theta grid.
    """
    if two_theta_step_deg <= 0:
        raise ValueError("two_theta_step_deg must be positive.")

    two_theta_deg = np.arange(
        two_theta_start_deg,
        two_theta_stop_deg + two_theta_step_deg,
        two_theta_step_deg,
    )
    intensity_total = np.zeros_like(two_theta_deg)

    peaks = calculate_xrd_peaks(
        element_symbol=element_symbol,
        lattice_type=lattice_type,
        lattice_parameter_a=lattice_parameter_a,
        wavelength_angstrom=wavelength_angstrom,
        two_theta_start_deg=two_theta_start_deg,
        two_theta_stop_deg=two_theta_stop_deg,
        instrument_fwhm_deg=instrument_fwhm_deg,
        crystallite_size_nm=crystallite_size_nm,
        max_index=max_index,
        element_b=element_b,
        composition_x=composition_x,
        b_iso=b_iso,
    )

    for peak in peaks:
        if peak_profile_function == "Gaussian":
            profile = _gaussian_profile(
                two_theta_deg,
                peak.two_theta_deg,
                peak.fwhm_deg,
                peak.relative_intensity,
            )
        elif peak_profile_function == "Lorentzian":
            profile = _lorentzian_profile(
                two_theta_deg,
                peak.two_theta_deg,
                peak.fwhm_deg,
                peak.relative_intensity,
            )
        else:
            raise ValueError("peak_profile_function must be 'Gaussian' or 'Lorentzian'.")

        intensity_total += profile

    return two_theta_deg, intensity_total
