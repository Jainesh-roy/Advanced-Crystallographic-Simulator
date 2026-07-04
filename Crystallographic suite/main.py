from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from core.forward_engine import (
    add_realistic_xrd_artifacts,
    calculate_xrd_peaks,
    simulate_xrd_pattern,
    vegards_law_lattice_parameter,
)

MATERIAL_PRESETS = {
    ("Cu", "FCC"): {"lattice_a": 3.615, "crystallite": 45.0},
    ("Ni", "FCC"): {"lattice_a": 3.524, "crystallite": 45.0},
    ("Al", "FCC"): {"lattice_a": 4.049, "crystallite": 50.0},
    ("Ag", "FCC"): {"lattice_a": 4.086, "crystallite": 45.0},
    ("Au", "FCC"): {"lattice_a": 4.078, "crystallite": 45.0},
    ("Fe", "BCC"): {"lattice_a": 2.866, "crystallite": 50.0},
    ("Cr", "BCC"): {"lattice_a": 2.884, "crystallite": 50.0},
    ("W", "BCC"): {"lattice_a": 3.165, "crystallite": 55.0},
    ("Mo", "BCC"): {"lattice_a": 3.147, "crystallite": 55.0},
    ("Po", "SC"): {"lattice_a": 3.359, "crystallite": 40.0},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1 cubic XRD forward simulator for SC/BCC/FCC crystals."
    )

    parser.add_argument("--crystal-name", default="User crystal", help="Display name for the crystal/material.")
    parser.add_argument("--element", default="Cu", help="Primary element symbol, e.g. Cu, Fe, Au.")
    parser.add_argument("--element-b", default=None, help="Optional B element for A(1-x)B(x) solid solutions.")
    parser.add_argument("--composition-x", type=float, default=None, help="Optional B fraction x for A(1-x)B(x).")
    parser.add_argument("--lattice-type", choices=["SC", "BCC", "FCC"], default="FCC", help="Cubic lattice type.")
    parser.add_argument("--wavelength", type=float, default=1.5406, help="X-ray wavelength in Angstroms.")
    parser.add_argument("--two-theta-start", type=float, default=20.0, help="Start angle for simulated 2theta scan.")
    parser.add_argument("--two-theta-stop", type=float, default=90.0, help="Stop angle for simulated 2theta scan.")
    parser.add_argument("--two-theta-step", type=float, default=0.02, help="Step size for simulated 2theta scan.")
    parser.add_argument("--profile", choices=["Gaussian", "Lorentzian"], default="Lorentzian", help="Peak profile.")
    parser.add_argument("--instrument-fwhm", type=float, default=0.15, help="Instrumental FWHM broadening in degrees.")
    parser.add_argument("--b-iso", type=float, default=0.5, help="Isotropic Debye-Waller B factor in Angstrom^2.")
    parser.add_argument("--max-index", type=int, default=6, help="Maximum Miller index used for generated reflections.")
    parser.add_argument("--noise-level", type=float, default=0.025, help="Noise level applied to displayed/saved graph.")
    parser.add_argument("--background-level", type=float, default=0.04, help="Background level applied to displayed/saved graph.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for repeatable noisy graph generation.")
    parser.add_argument("--output-profile", default=None, help="Optional CSV path for simulated profile output.")
    parser.add_argument("--output-plot", default=None, help="Optional PNG/PDF path for the XRD plot.")

    return parser


def _lookup_preset(element: str, lattice_type: str) -> dict[str, float]:
    key = (element, lattice_type)
    if key not in MATERIAL_PRESETS:
        supported = ", ".join(f"{el}-{lat}" for el, lat in sorted(MATERIAL_PRESETS))
        raise ValueError(
            f"No built-in preset for {element}-{lattice_type}. "
            f"Supported presets: {supported}."
        )
    return MATERIAL_PRESETS[key]


def resolve_material_parameters(args: argparse.Namespace) -> tuple[float, float]:
    preset_a = _lookup_preset(args.element, args.lattice_type)
    crystallite_size = preset_a["crystallite"]

    if args.composition_x is None:
        return preset_a["lattice_a"], crystallite_size

    if args.element_b is None:
        raise ValueError("--element-b is required with --composition-x.")

    preset_b = _lookup_preset(args.element_b, args.lattice_type)
    lattice_a = vegards_law_lattice_parameter(
        args.composition_x,
        preset_a["lattice_a"],
        preset_b["lattice_a"],
    )
    crystallite_size = (1.0 - args.composition_x) * preset_a["crystallite"] + args.composition_x * preset_b["crystallite"]
    return lattice_a, crystallite_size


def save_profile(path: str, two_theta: np.ndarray, intensity: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_path,
        np.column_stack([two_theta, intensity]),
        delimiter=",",
        header="two_theta_deg,intensity",
        comments="",
    )

def save_xrd_plot(
    path: str,
    two_theta: np.ndarray,
    intensity: np.ndarray,
    max_intensity: float = 2.5,
):
    """
    Save the simulated XRD profile as a publication-style plot.
    """
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if np.max(intensity) > 0:
        intensity_norm = intensity / np.max(intensity) * max_intensity
    else:
        intensity_norm = intensity

    plt.figure(figsize=(10, 6))

    plt.plot(
        two_theta,
        intensity_norm,
        color="black",
        linewidth=1.5,
    )

    plt.xlabel(r"2$\theta$ (degrees)", fontsize=14)
    plt.ylabel("Intensity (a.u.)", fontsize=14)

    plt.xlim(float(two_theta.min()), float(two_theta.max()))
    plt.ylim(0, max_intensity)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(False)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        lattice_a, crystallite_size = resolve_material_parameters(args)
        sim_two_theta, sim_intensity = simulate_xrd_pattern(
            element_symbol=args.element,
            element_b=args.element_b,
            composition_x=args.composition_x,
            lattice_type=args.lattice_type,
            lattice_parameter_a=lattice_a,
            wavelength_angstrom=args.wavelength,
            two_theta_start_deg=args.two_theta_start,
            two_theta_stop_deg=args.two_theta_stop,
            two_theta_step_deg=args.two_theta_step,
            peak_profile_function=args.profile,
            instrument_fwhm_deg=args.instrument_fwhm,
            crystallite_size_nm=crystallite_size,
            max_index=args.max_index,
            b_iso=args.b_iso,
        )
        observed_intensity = add_realistic_xrd_artifacts(
            sim_two_theta,
            sim_intensity,
            noise_fraction=args.noise_level,
            background_fraction=args.background_level,
            random_seed=args.random_seed,
        )
        
        # Added peak_profile_function=args.profile to ensure math matches the simulation
        peaks = calculate_xrd_peaks(
            element_symbol=args.element,
            element_b=args.element_b,
            composition_x=args.composition_x,
            lattice_type=args.lattice_type,
            lattice_parameter_a=lattice_a,
            wavelength_angstrom=args.wavelength,
            two_theta_start_deg=args.two_theta_start,
            two_theta_stop_deg=args.two_theta_stop,
            instrument_fwhm_deg=args.instrument_fwhm,
            crystallite_size_nm=crystallite_size,
            max_index=args.max_index,
            b_iso=args.b_iso,
            peak_profile_function=args.profile, 
        )
    except Exception as exc:
        print(f"Failed to simulate XRD pattern: {exc}")
        return 1

    print("\n--- Phase 1: Cubic XRD Forward Simulation ---")
    print(f"Crystal name          : {args.crystal_name}")
    print(f"Composition           : {args.element}" + (f"(1-x){args.element_b}(x), x={args.composition_x:.4f}" if args.composition_x is not None else ""))
    print(f"Lattice type          : {args.lattice_type}")
    print(f"Internal assumed a    : {lattice_a:.4f} Angstrom")
    print(f"Wavelength            : {args.wavelength:.4f} Angstrom")
    print(f"Internal crystallite  : {crystallite_size:.2f} nm")
    print(f"Instrument FWHM       : {args.instrument_fwhm:.4f} deg")
    print(f"Debye-Waller B_iso    : {args.b_iso:.4f} Angstrom^2")
    print(f"Graph noise level     : {args.noise_level:.4f}")
    print(f"Graph background      : {args.background_level:.4f}")
    print(f"Peak profile          : {args.profile}")
    print(f"2theta range          : {args.two_theta_start:.2f} to {args.two_theta_stop:.2f} deg")
    print(f"2theta step           : {args.two_theta_step:.4f} deg")
    print(f"Max Miller index      : {args.max_index}")
    print(f"Allowed peak families : {len(peaks)}")

    if peaks:
        max_intensity = max(peak.relative_intensity for peak in peaks)
        print("\nSimulated peak table:")
        print(" h k l | mult | 2theta(deg) | d(A)     | f0       | |F|      | LP       | DW       | rel.int(%) | FWHM(deg)")
        print("-------|------|-------------|----------|----------|----------|----------|----------|------------|----------")
        for peak in peaks:
            rel_percent = 100.0 * peak.relative_intensity / max_intensity if max_intensity > 0 else 0.0
            print(
                f" {peak.h:1d} {peak.k:1d} {peak.l:1d} |"
                f" {peak.multiplicity:4d} |"
                f" {peak.two_theta_deg:11.3f} |"
                f" {peak.d_spacing_angstrom:8.4f} |"
                f" {peak.form_factor:8.3f} |"
                f" {peak.structure_factor_abs:8.3f} |"
                f" {peak.lorentz_polarization:8.3f} |"
                f" {peak.debye_waller:8.3f} |"
                f" {rel_percent:10.2f} |"
                f" {peak.fwhm_deg:8.4f}"
            )
    else:
        print("\nNo peaks fall inside the selected 2theta range.")

    if args.output_profile:
        save_profile(args.output_profile, sim_two_theta, observed_intensity)
        print(f"\nSaved simulated profile: {args.output_profile}")

    if args.output_plot:
        save_xrd_plot(args.output_plot, sim_two_theta, observed_intensity)
        print(f"Saved XRD plot: {args.output_plot}")

    print("---------------------------------------------")
    return 0


if __name__ == "__main__":
    sys.exit(main())
