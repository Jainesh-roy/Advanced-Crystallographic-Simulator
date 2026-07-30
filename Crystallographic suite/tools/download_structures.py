"""
download_structures.py

Downloads crystal structures from the Materials Project
and saves them as CIF files in the structures/ folder.

Requirements:
    pip install mp-api pymatgen
"""

from pathlib import Path

from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter

# ---------------------------------------------------------------------
# PUT YOUR API KEY HERE
# ---------------------------------------------------------------------

API_KEY = "tmKKwbFdrcZub1jH9OVZRzoiraZ9ubgn"

# ---------------------------------------------------------------------
# Materials to download
# ---------------------------------------------------------------------

MATERIALS = [
    "Cu",
    "Al",
    "Fe",
    "Ag",
    "Au",
]

# ---------------------------------------------------------------------

OUTPUT_DIR = Path("structures")
OUTPUT_DIR.mkdir(exist_ok=True)


def download_material(mpr, formula):
    print(f"\nSearching {formula}...")

    docs = mpr.materials.summary.search(
        formula=formula,
        is_stable=True,
        fields=[
            "material_id",
            "formula_pretty",
            "structure",
            "energy_above_hull",
        ],
    )

    if len(docs) == 0:
        print(f"No stable structure found for {formula}")
        return

    # Choose the lowest energy-above-hull structure
    docs = sorted(
        docs,
        key=lambda d: (
            d.energy_above_hull
            if d.energy_above_hull is not None
            else float("inf")
        ),
    )

    doc = docs[0]

    structure = doc.structure

    output_file = OUTPUT_DIR / f"{formula}.cif"

    CifWriter(structure).write_file(output_file)

    print(f"Downloaded {formula}")
    print(f"Material ID : {doc.material_id}")
    print(f"Formula     : {doc.formula_pretty}")
    print(f"Saved to    : {output_file}")


def main():
    with MPRester(API_KEY) as mpr:
        for formula in MATERIALS:
            try:
                download_material(mpr, formula)
            except Exception as e:
                print(f"Failed for {formula}: {e}")


if __name__ == "__main__":
    main()