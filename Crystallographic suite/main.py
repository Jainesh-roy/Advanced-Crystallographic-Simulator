# The orchestrator (combines everyone's work)
# main.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

# Adjust these imports to match your actual package structure:
from core.signal_processor import run_signal_pipeline
from core.structural_physics import calculate_atomic_form_factor, get_allowed_reflections
from core.forwardengine import (
    classify_cubic_from_peaks,
    simulate_xrd_pattern,
    least_squares_cost,
)


def load_xrd_text_file(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a raw XRD text file with two columns:
    column 1 -> 2theta in degrees
    column 2 -> intensity
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Try whitespace-delimited first, then fall back to comma
    try:
        data = np.loadtxt(file_path, delimiter=None)
    except ValueError:
        data = np.loadtxt(file_path, delimiter=",")

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("File must contain at least two numeric columns: 2theta, intensity.")

    two_theta_deg = np.asarray(data[:, 0], dtype=float)
    intensity = np.asarray(data[:, 1], dtype=float)

    if two_theta_deg.shape != intensity.shape:
        raise ValueError("Angle and intensity arrays must have the same length.")

    return two_theta_deg, intensity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="XRD cubic lattice identification and forward simulation orchestrator."
    )

    parser.add_argument("file", help="Path to raw XRD text file (2theta, intensity).")
    parser.add_argument(
        "--wavelength",
        type=float,
        default=1.5406,  # Cu Kα in Angstroms (matches structural_physics / forwardengine)
        help="X-ray wavelength in Angstroms."
    )
    parser.add_argument(
        "--element",
        type=str,
        default="Cu",
        help="Element symbol for structure factor simulation (e.g., Cu, Fe)."
    )
    parser.add_argument(
        "--lattice-a",
        type=float,
        default=3.615,
        help="Initial lattice parameter a in Angstroms for forward simulation (cubic)."
    )
    parser.add_argument(
        "--crystallite",
        type=float,
        default=20.0,
        help="Crystallite size in nm for Scherrer broadening."
    )
    parser.add_argument(
        "--sg-window",
        type=int,
        default=15,
        help="Savitzky-Golay window size."
    )
    parser.add_argument(
        "--sg-order",
        type=int,
        default=4,
        help="Savitzky-Golay polynomial order."
    )
    parser.add_argument(
        "--prominence",
        type=float,
        default=20.0,
        help="Minimum prominence for peak detection."
    )
    parser.add_argument(
        "--width",
        type=float,
        default=2.0,
        help="Minimum peak width (in index units) for peak detection."
    )
    parser.add_argument(
        "--distance",
        type=int,
        default=5,
        help="Minimum distance (in index units) between peaks."
    )
    parser.add_argument(
        "--max-indexing-peaks",
        type=int,
        default=5,
        help="Maximum number of lowest-angle peaks used for cubic indexing."
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # 1. Load raw data
    # ----------------------------------------------------------------------
    try:
        two_theta_deg, raw_intensity = load_xrd_text_file(args.file)
    except Exception as exc:
        print(f"Failed to load data: {exc}")
        return 1

    # ----------------------------------------------------------------------
    # 2. Run signal-processing pipeline (Savitzky-Golay + zero-point)
    # ----------------------------------------------------------------------
    signal_result = run_signal_pipeline(
        two_theta_deg=two_theta_deg,
        raw_intensity=raw_intensity,
        measured_peaks_deg=None,
        reference_peaks_deg=None,
        window_size=args.sg_window,
        poly_order=args.sg_order,
    )

    smoothed_intensity = signal_result["smoothed_intensity"]
    corrected_two_theta = signal_result["corrected_two_theta"]
    zero_shift = signal_result["zero_point_shift_deg"]

    # For this main orchestration, treat smoothed_intensity as “cleaned” data.
    corrected_intensity = smoothed_intensity

    # ----------------------------------------------------------------------
    # 3. Automated peak extraction with scipy.signal.find_peaks
    # ----------------------------------------------------------------------
    peak_indices, peak_props = find_peaks(
        corrected_intensity,
        prominence=args.prominence,
        width=args.width,
        distance=args.distance,
    )

    peak_positions_deg = corrected_two_theta[peak_indices]
    peak_intensities = corrected_intensity[peak_indices]

    if len(peak_positions_deg) < 2:
        print("Not enough peaks detected for indexing; adjust peak parameters.")
        return 1

    # Sort peaks by angle and take the lowest-angle subset for indexing
    sort_idx = np.argsort(peak_positions_deg)
    peak_positions_sorted = peak_positions_deg[sort_idx]
    peak_intensities_sorted = peak_intensities[sort_idx]
    peak_positions_for_indexing = peak_positions_sorted[:args.max_indexing_peaks]

    # ----------------------------------------------------------------------
    # 4. Lattice classification using forwardengine.classify_cubic_from_peaks
    # ----------------------------------------------------------------------
    indexing_result = classify_cubic_from_peaks(
        two_theta_deg=peak_positions_for_indexing,
        max_peaks=args.max_indexing_peaks,
    )

    predicted_lattice = indexing_result.lattice_type
    lattice_scores = indexing_result.scores

    # ----------------------------------------------------------------------
    # 5. Forward simulation for the predicted lattice
    #    NOTE: updated to use simulate_xrd_pattern's new signature
    #          (wavelength_angstrom, two_theta_* names).
    # ----------------------------------------------------------------------
    sim_two_theta, sim_intensity = simulate_xrd_pattern(
        element_symbol=args.element,
        lattice_type=predicted_lattice,
        lattice_parameter_a=args.lattice_a,
        wavelength_angstrom=args.wavelength,
        two_theta_start_deg=float(corrected_two_theta.min()),
        two_theta_stop_deg=float(corrected_two_theta.max()),
        two_theta_step_deg=float(corrected_two_theta[1] - corrected_two_theta[0]),
        peak_profile_function="Lorentzian",   # or make this a CLI argument later
        instrument_fwhm_deg=0.15,             # could be exposed as CLI as well
        crystallite_size_nm=args.crystallite,
    )

    # Interpolate simulated intensity onto the experimental grid
    sim_on_exp_grid = np.interp(corrected_two_theta, sim_two_theta, sim_intensity)

    # Normalize both patterns for fair comparison
    if np.max(corrected_intensity) > 0:
        corrected_norm = corrected_intensity / np.max(corrected_intensity)
    else:
        corrected_norm = corrected_intensity

    if np.max(sim_on_exp_grid) > 0:
        sim_norm = sim_on_exp_grid / np.max(sim_on_exp_grid)
    else:
        sim_norm = sim_on_exp_grid

    # ----------------------------------------------------------------------
    # 6. Compute least-squares mismatch
    # ----------------------------------------------------------------------
    S = least_squares_cost(corrected_norm, sim_norm)

    # ----------------------------------------------------------------------
    # 7. Print consolidated analysis verdict
    # ----------------------------------------------------------------------
    print("\n--- XRD Cubic Lattice Analysis ---")
    print(f"Data file             : {args.file}")
    print(f"Number of data points : {len(two_theta_deg)}")
    print(f"Zero-point shift      : {zero_shift:.4f} deg")
    print(f"Detected peaks        : {len(peak_positions_deg)}")

    print("\nFirst few peak positions (2θ deg):")
    print(", ".join(f"{p:.3f}" for p in peak_positions_sorted[:args.max_indexing_peaks]))

    print("\nCubic classification scores (lower is better):")
    for lattice, score in lattice_scores.items():
        print(f"  {lattice}: {score:.6e}")

    print(f"\nPredicted lattice type: {predicted_lattice}")
    print(f"Initial lattice parameter a (Å): {args.lattice_a:.4f}")
    print(f"Crystallite size (nm): {args.crystallite:.2f}")

    print(f"\nLeast-squares mismatch S between cleaned data and simulated profile: {S:.6e}")
    print("------------------------------------------")

    return 0


if __name__ == "__main__":
    sys.exit(main())
