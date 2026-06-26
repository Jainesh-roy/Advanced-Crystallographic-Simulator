"""
core/signal_processor.py

1. Apply Savitzky-Golay filter: high-frequency noise removal
2. Calculate Scherrer broadening: crystallite size - FWHM in radians
3. Calculate zero point shift: systematic angular offset detection
4. Apply zero point correction: global 2θ axis recalibration

"""

import numpy as np
from scipy.signal import savgol_filter


#_______________________________________________________________________________________________________________
# 1. SAVITZKY-GOLAY NOISE FILTER


def apply_savitzky_golay_filter(
    raw_intensity: np.ndarray,
    window_size: int = 15,
    poly_order: int = 4
) -> np.ndarray:
    
    """
    Applies a local polynomial least-squares smoothing convolution
    to eliminate high-frequency electronics or background noise.

    Returns - 
    smoothed_intensity: np.ndarray
    Noise-filtered intensity array, same shape as raw_intensity.
    Negative artefacts introduced by the convolution are clipped to 0.

    Notes -
         1. Larger window_size - smoother curve, but starts eating peak width
         2. Higher poly_order - better peak fidelity, less noise removal
    """
    
    if window_size % 2 == 0:
        raise ValueError(
            f"window_size must be an odd integer, got {window_size}. "
            "Try {window_size + 1}."
        )
    if poly_order >= window_size:
        raise ValueError(
            f"poly_order ({poly_order}) must be strictly less than "
            f"window_size ({window_size})."
        )
    if len(raw_intensity) < window_size:
        raise ValueError(
            f"Array length ({len(raw_intensity)}) is shorter than "
            f"window_size ({window_size}). Reduce window_size."
        )

    
    smoothed_intensity = savgol_filter(
        raw_intensity,
        window_length=window_size,
        polyorder=poly_order
    )

    
    smoothed_intensity = np.clip(smoothed_intensity, a_min=0.0, a_max=None)

    return smoothed_intensity


#__________________________________________________________________________________________________________________
# 2. SCHERRER BROADENING: crystallite size - beta array


def calculate_scherrer_broadening(
    crystallite_d: float,
    theta_rad: np.ndarray,
    wavelength: float = 1.5406,
    kappa: float = 0.9
) -> np.ndarray:
    """
    Transforms targeted crystallite domain sizes into continuous,
    broadened profile array values (beta) expressed in radians.
 
    Implements the Scherrer Equation:
    β = (κ * λ) / (D * cos θ)

    Notes -
    Unit consistency check:
        λ is in Angstrom,  D is in nm - convert D to Angstrom internally:
        D_Angstrom = crystallite_d * 10.0
  """

    if crystallite_d <= 0:
        raise ValueError(
            f"crystallite_d must be positive (got {crystallite_d} nm)."
        )
    if wavelength <= 0:
        raise ValueError(
            f"wavelength must be positive (got {wavelength} Angstrom)."
        )

    
    D_angstrom = crystallite_d * 10.0   # 1 nm = 10 Angstrom

    # β (radians) = κλ / (D * cosθ) -- Scherrer equation
    fwhm_rad = (kappa * wavelength) / (D_angstrom * np.cos(theta_rad))

    return fwhm_rad


#__________________________________________________________________________________________________________
# 3. ZERO-POINT SHIFT CALCULATOR


def calculate_zero_point_shift(
    measured_peaks_deg: np.ndarray,
    reference_peaks_deg: np.ndarray
) -> float:
    """
    Calculates the systematic angular offset between a set of measured
    peak positions and their known reference (standard) positions.
    """
    
    measured_peaks_deg  = np.asarray(measured_peaks_deg,  dtype=float)
    reference_peaks_deg = np.asarray(reference_peaks_deg, dtype=float)

    if measured_peaks_deg.shape != reference_peaks_deg.shape:
        raise ValueError(
            f"Array shape mismatch: measured has {measured_peaks_deg.shape}, "
            f"reference has {reference_peaks_deg.shape}. "
            "Both must be 1-D arrays of the same length."
        )
    if measured_peaks_deg.size == 0:
        raise ValueError("Peak arrays must not be empty.")

    differences = measured_peaks_deg - reference_peaks_deg
    shift_deg   = float(np.mean(differences))

    return shift_deg


#____________________________________________________________________________________________________
# 4. ZERO-POINT CORRECTION - apply the shift to the full 2θ axis


def apply_zero_point_correction(
    two_theta_deg: np.ndarray,
    shift_deg: float
) -> np.ndarray:
    """
    Subtracts the calculated systematic offset from the entire 2θ axis,
    returning a corrected angle array aligned to the true diffraction geometry.

    Notes - 
    The correction is a simple global subtraction - no interpolation or
    resampling is performed, so the array length and step size are unchanged.
    """
    corrected_two_theta = two_theta_deg - shift_deg
    return corrected_two_theta


#______________________________________________________________________________________________________
# Run the full signal-processing chain in one call


def run_signal_pipeline(
    two_theta_deg: np.ndarray,
    raw_intensity: np.ndarray,
    measured_peaks_deg: np.ndarray = None,
    reference_peaks_deg: np.ndarray = None,
    window_size: int = 15,
    poly_order: int = 4
) -> dict:
    
    # Step 1: Smooth
    smoothed = apply_savitzky_golay_filter(raw_intensity, window_size, poly_order)

    # Step 2: Zero-point correction
    shift = 0.0
    corrected_two_theta = two_theta_deg.copy()

    if measured_peaks_deg is not None and reference_peaks_deg is not None:
        shift = calculate_zero_point_shift(measured_peaks_deg, reference_peaks_deg)
        corrected_two_theta = apply_zero_point_correction(two_theta_deg, shift)

    return {
        'smoothed_intensity'  : smoothed,
        'corrected_two_theta'  : corrected_two_theta,
        'zero_point_shift_deg': shift
    }
