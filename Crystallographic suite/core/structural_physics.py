import numpy as np
import periodictable


def calculate_atomic_form_factor(
    element_symbol: str,
    theta_rad: np.ndarray,
    wavelength: float = 1.5406,
) -> np.ndarray:
    """
    Compute the angle-dependent atomic scattering factor f0.
    """
    try:
        element = getattr(periodictable, element_symbol)
    except AttributeError as exc:
        raise ValueError(f"Element '{element_symbol}' is not a valid periodic table symbol.") from exc

    q_vector = (4 * np.pi * np.sin(theta_rad)) / wavelength
    return np.vectorize(element.xray.f0)(q_vector)


def get_allowed_reflections(lattice_type: str, max_index: int = 4) -> list[tuple[int, int, int]]:
    """
    Return allowed cubic reflections according to SC/BCC/FCC selection rules.
    """
    if lattice_type not in {"SC", "BCC", "FCC"}:
        raise ValueError("lattice_type must be one of: SC, BCC, FCC.")

    allowed_planes = []
    for h in range(max_index + 1):
        for k in range(max_index + 1):
            for l in range(max_index + 1):
                if h == 0 and k == 0 and l == 0:
                    continue

                if lattice_type == "SC":
                    allowed_planes.append((h, k, l))
                elif lattice_type == "BCC" and (h + k + l) % 2 == 0:
                    allowed_planes.append((h, k, l))
                elif lattice_type == "FCC" and (h % 2 == k % 2 == l % 2):
                    allowed_planes.append((h, k, l))

    return allowed_planes


def calculate_structure_factor(
    lattice_type: str,
    h: int,
    k: int,
    l: int,
    atomic_form_factor: float,
) -> complex:
    """
    Calculate F for monoatomic SC/BCC/FCC cubic bases.
    """
    if lattice_type == "SC":
        return complex(atomic_form_factor)
    if lattice_type == "BCC":
        return atomic_form_factor * (1 + np.exp(1j * np.pi * (h + k + l)))
    if lattice_type == "FCC":
        return atomic_form_factor * (
            1
            + np.exp(1j * np.pi * (h + k))
            + np.exp(1j * np.pi * (h + l))
            + np.exp(1j * np.pi * (k + l))
        )
    raise ValueError("lattice_type must be one of: SC, BCC, FCC.")
