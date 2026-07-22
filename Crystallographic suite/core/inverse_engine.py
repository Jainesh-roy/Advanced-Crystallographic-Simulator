# Back-calculation & refinement

"""
1. Load experimental data: reads raw .csv/.txt from Data/Experimental/
2. Load reference pattern: reads theoretical data from Data/Simulated/
3. Interpolate to a common grid: aligns experimental and reference arrays
4. Compute Rwp: weighted profile residual scoring
5. Match peaks to reference: compares detected peaks against the library
6. Identify phases: master pipeline function (SOP contract)
7. Save metrics: exports results to outputs/Metrics/
"""

import os
import json
import numpy as np
from signal_processor import clean_profile


# _______________________________________________________________________________________________________________
# 1. LOAD EXPERIMENTAL DATA

def load_experimental_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Experimental data file not found: {file_path}\n"
            f"Place input files in Data/Experimental/"
        )

    try:
        data = np.loadtxt(file_path, comments="#", delimiter=None)
        if data.ndim != 2 or data.shape[1] < 2:
            data = np.loadtxt(file_path, comments="#", delimiter=",")
    except Exception as e:
        raise ValueError(f"Could not parse {file_path}: {e}")

    two_theta_deg = data[:, 0].astype(float)
    raw_intensity = data[:, 1].astype(float)

    return two_theta_deg, raw_intensity


# _______________________________________________________________________________________________________________
# 2. LOAD REFERENCE PATTERN

def load_reference_pattern(simulated_path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(simulated_path):
        raise FileNotFoundError(
            f"Reference pattern not found: {simulated_path}\n"
            f"Run the Forward Engine first to generate Data/Simulated/ files."
        )

    data = np.loadtxt(simulated_path, comments="#", delimiter=",")
    return data[:, 0].astype(float), data[:, 1].astype(float)


# _______________________________________________________________________________________________________________
# 3. INTERPOLATE TO COMMON GRID

def _interpolate_to_common_grid(
    theta_exp: np.ndarray,
    intensity_exp: np.ndarray,
    theta_ref: np.ndarray,
    intensity_ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_interp = np.interp(
        theta_exp,
        theta_ref,
        intensity_ref,
        left=0.0,
        right=0.0
    )
    return theta_exp, intensity_exp, ref_interp


# _______________________________________________________________________________________________________________
# 4. COMPUTE Rwp

def compute_rwp(
    y_exp: np.ndarray,
    y_sim: np.ndarray,
    weights: np.ndarray = None
) -> float:
    y_exp = np.asarray(y_exp, dtype=float)
    y_sim = np.asarray(y_sim, dtype=float)

    if y_exp.shape != y_sim.shape:
        raise ValueError(
            f"Shape mismatch: y_exp {y_exp.shape} vs y_sim {y_sim.shape}"
        )

    if weights is None:
        weights = 1.0 / np.maximum(y_exp, 1.0)

    weights = np.asarray(weights, dtype=float)

    numerator = np.sum(weights * (y_exp - y_sim) ** 2)
    denominator = np.sum(weights * y_exp ** 2)

    if denominator == 0:
        raise ValueError(
            "Denominator in Rwp is zero — experimental intensity array is all zeros."
        )

    return float(np.sqrt(numerator / denominator))


# _______________________________________________________________________________________________________________
# 5. PEAK SEARCH OPTIMIZATION (ANAND)

def extract_experimental_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    min_relative_height: float = 0.05,
    min_distance_points: int = 5,
    min_prominence: float | None = None,
    slope_eps: float = 1e-12
) -> dict:
    two_theta = np.asarray(two_theta, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    if two_theta.ndim != 1 or intensity.ndim != 1:
        raise ValueError("two_theta and intensity must be 1D arrays")
    if len(two_theta) != len(intensity):
        raise ValueError("two_theta and intensity must have the same length")
    if len(two_theta) < 3:
        return {
            "indices": np.array([], dtype=int),
            "positions": np.array([], dtype=float),
            "intensities": np.array([], dtype=float),
            "relative_intensities": np.array([], dtype=float)
        }

    intensity = np.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    intensity = np.clip(intensity, a_min=0.0, a_max=None)

    max_intensity = float(np.max(intensity))
    if max_intensity <= 0:
        return {
            "indices": np.array([], dtype=int),
            "positions": np.array([], dtype=float),
            "intensities": np.array([], dtype=float),
            "relative_intensities": np.array([], dtype=float)
        }

    relative = intensity / max_intensity
    if min_prominence is None:
        min_prominence = 0.02 * max_intensity

    first_derivative = np.diff(intensity)
    candidate_indices = []

    for i in range(1, len(intensity) - 1):
        rising_before = first_derivative[i - 1] > slope_eps
        falling_after = first_derivative[i] <= slope_eps
        passes_height = relative[i] >= min_relative_height

        if rising_before and falling_after and passes_height:
            left = max(0, i - min_distance_points)
            right = min(len(intensity), i + min_distance_points + 1)

            left_min = np.min(intensity[left:i + 1]) if i > left else intensity[i]
            right_min = np.min(intensity[i:right]) if right > i else intensity[i]
            local_prominence = intensity[i] - max(left_min, right_min)

            if local_prominence >= min_prominence:
                candidate_indices.append(i)

    if not candidate_indices:
        return {
            "indices": np.array([], dtype=int),
            "positions": np.array([], dtype=float),
            "intensities": np.array([], dtype=float),
            "relative_intensities": np.array([], dtype=float)
        }

    candidate_indices = np.array(candidate_indices, dtype=int)
    order = np.argsort(intensity[candidate_indices])[::-1]
    selected = []

    for idx in candidate_indices[order]:
        if not selected or all(abs(idx - kept) >= min_distance_points for kept in selected):
            selected.append(int(idx))

    selected = np.array(sorted(selected), dtype=int)

    return {
        "indices": selected,
        "positions": two_theta[selected],
        "intensities": intensity[selected],
        "relative_intensities": relative[selected]
    }


# _______________________________________________________________________________________________________________
# 6. MATCH PEAKS TO REFERENCE

def match_peaks_to_reference(
    exp_peak_positions: np.ndarray,
    simulated_dir: str = "Data/Simulated/",
    tolerance_deg: float = 0.3
) -> list[dict]:
    if not os.path.exists(simulated_dir):
        raise FileNotFoundError(
            f"Simulated directory not found: {simulated_dir}\n"
            "Run the Forward Engine to populate Data/Simulated/."
        )

    reference_files = [
        f for f in os.listdir(simulated_dir)
        if f.endswith(".csv") or f.endswith(".txt")
    ]

    if not reference_files:
        raise FileNotFoundError(
            f"No reference pattern files found in {simulated_dir}"
        )

    results = []

    for ref_file in reference_files:
        ref_path = os.path.join(simulated_dir, ref_file)
        phase_name = os.path.splitext(ref_file)[0]

        try:
            ref_theta, ref_intensity = load_reference_pattern(ref_path)

            ref_peak_data = extract_experimental_peaks(
                ref_theta,
                ref_intensity,
                min_relative_height=0.05,
                min_distance_points=3,
                min_prominence=0.0
            )
            ref_peaks = ref_peak_data["positions"]

            if len(ref_peaks) == 0:
                threshold = 0.05 * ref_intensity.max() if ref_intensity.max() > 0 else np.inf
                ref_peak_idx = np.where(ref_intensity >= threshold)[0]
                ref_peaks = ref_theta[ref_peak_idx]

            matched = 0
            used_ref_indices = set()

            for exp_pos in exp_peak_positions:
                if len(ref_peaks) == 0:
                    break

                diffs = np.abs(ref_peaks - exp_pos)
                closest_idx = int(np.argmin(diffs))

                if diffs[closest_idx] <= tolerance_deg and closest_idx not in used_ref_indices:
                    used_ref_indices.add(closest_idx)
                    matched += 1

            score = matched / len(ref_peaks) if len(ref_peaks) > 0 else 0.0

            results.append({
                "phase_name": phase_name,
                "match_score": round(score, 4),
                "matched_peaks": matched,
                "total_ref_peaks": len(ref_peaks)
            })

        except Exception as e:
            results.append({
                "phase_name": phase_name,
                "match_score": 0.0,
                "matched_peaks": 0,
                "total_ref_peaks": 0,
                "error": str(e)
            })

    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results


# _______________________________________________________________________________________________________________
# 7. IDENTIFY PHASES

def identify_phases(file_path: str) -> dict:
    try:
        two_theta, raw_intensity = load_experimental_data(file_path)

        two_theta_clean, cleaned_intensity = clean_profile(two_theta, raw_intensity)

        peak_data = extract_experimental_peaks(
            two_theta_clean,
            cleaned_intensity,
            min_relative_height=0.05,
            min_distance_points=5,
            min_prominence=0.02 * np.max(cleaned_intensity) if np.max(cleaned_intensity) > 0 else 0.0
        )
        exp_peak_positions = peak_data["positions"]

        match_results = match_peaks_to_reference(exp_peak_positions)

        rwp_value = 1.0
        best_match = match_results[0] if match_results else None

        if best_match and best_match["match_score"] > 0:
            ref_file = f"Data/Simulated/{best_match['phase_name']}.csv"
            if os.path.exists(ref_file):
                ref_theta, ref_intensity = load_reference_pattern(ref_file)
                _, exp_grid, ref_grid = _interpolate_to_common_grid(
                    two_theta_clean,
                    cleaned_intensity,
                    ref_theta,
                    ref_intensity
                )
                rwp_value = compute_rwp(exp_grid, ref_grid)

        top_matches = [m for m in match_results if m["match_score"] > 0.1][:3]
        total_score = sum(m["match_score"] for m in top_matches) or 1.0

        identified_phases = [
            {
                "name": m["phase_name"],
                "fraction": round(m["match_score"] / total_score, 3)
            }
            for m in top_matches
        ]

        return {
            "success": True,
            "R_wp": round(rwp_value, 6),
            "identified_phases": identified_phases,
            "detected_peaks": {
                "positions": peak_data["positions"].tolist(),
                "intensities": peak_data["intensities"].tolist(),
                "relative_intensities": peak_data["relative_intensities"].tolist()
            },
            "cleaned_profile": {
                "two_theta": two_theta_clean.tolist(),
                "intensity": cleaned_intensity.tolist()
            },
            "match_details": match_results
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "R_wp": None,
            "identified_phases": [],
            "cleaned_profile": None,
            "match_details": []
        }


# _______________________________________________________________________________________________________________
# 8. SAVE METRICS

def save_metrics(results: dict, run_id: str = "run_001") -> str:
    os.makedirs("outputs/Metrics", exist_ok=True)
    out_path = f"outputs/Metrics/{run_id}_refinement_report.json"

    results_to_save = {k: v for k, v in results.items() if k != "cleaned_profile"}

    with open(out_path, "w") as f:
        json.dump(results_to_save, f, indent=2)

    return out_path
