#Lattice and form factors

import numpy as np
import periodictable

def calculate_atomic_form_factor(element_symbol: str, theta_rad: np.ndarray, wavelength: float = 1.5406) -> np.ndarray:
    """
    Computes the angle-dependent atomic scattering factor (f_0) automatically
    using the periodictable library's native quantum scattering engine.
    """
    try:
        # Fetch the official element object registry
        element = getattr(periodictable, element_symbol)
    except AttributeError:
        raise ValueError(f"Element '{element_symbol}' is not a valid periodic table symbol.")
    
    # Calculate the scattering vector magnitude Q = 4 * pi * sin(theta) / lambda
    q_vector = (4 * np.pi * np.sin(theta_rad)) / wavelength
    
    # Use the library's built-in vector-safe form factor calculator
    # We apply np.vectorize to allow the library to evaluate entire arrays seamlessly
    f0_calculator = np.vectorize(element.xray.f0)
    f_j = f0_calculator(q_vector)
    
    return f_j

def get_allowed_reflections(lattice_type: str) -> list:
    """
    Evaluates systematic absences using structural selection mask rules 
    and returns permitted (h, k, l) planes up to a maximum index threshold of 4.
    """
    allowed_planes = []
    
    # Loop over all reasonable combinations of Miller Indices
    for h in range(5):
        for k in range(5):
            for l in range(5):
                # The origin (0,0,0) represents no physical lattice plane intersection
                if h == 0 and k == 0 and l == 0:
                    continue
                
                # Check structural type conditions
                if lattice_type == "SC":
                    allowed_planes.append((h, k, l))
                    
                elif lattice_type == "BCC":
                    # Mask Condition: The sum of indices must be even
                    if (h + k + l) % 2 == 0:
                        allowed_planes.append((h, k, l))
                        
                elif lattice_type == "FCC":
                    # Mask Condition: Parity must be completely unmixed (all even or all odd)
                    if (h % 2 == k % 2 == l % 2):
                        allowed_planes.append((h, k, l))
                        
    return allowed_planes


# test code

if __name__ == "__main__":
    print("--- Testing Structure Physics Module ---")
    
    fcc_planes = get_allowed_reflections("FCC")
    print(f"Total allowed FCC reflections isolated: {len(fcc_planes)}")
    print(f"First few allowed FCC planes: {fcc_planes[:5]}")
    
    sample_angle = np.radians(22.0)  
    f_cu = calculate_atomic_form_factor("Cu", np.array([sample_angle]))
    
    # Grab the very first index value from your array output
    print(f"Copper atomic scattering power at 22 deg: {float(f_cu[0]):.3f}")

if __name__ == "__main__":
    print("--- Testing Professional Open-Source Integrated Physics Module ---")
    
    # Test 1: Check standard FCC planes collection
    fcc_planes = get_allowed_reflections("FCC")
    print(f"Total allowed FCC reflections isolated: {len(fcc_planes)}")
    
    # Test 2: Evaluate ANY arbitrary element seamlessly without updating dictionary maps
    sample_angle = np.radians(22.0)
    
    # Let's check Copper (Cu)
    f_cu = calculate_atomic_form_factor("Cu", np.array([sample_angle]))
    print(f"Copper atomic scattering factor at 22 deg: {float(f_cu[0]):.3f}")
    
    # Let's check Gold (Au) instantly!
    f_au = calculate_atomic_form_factor("Au", np.array([sample_angle]))
    print(f"Gold atomic scattering factor at 22 deg: {float(f_au[0]):.3f}")