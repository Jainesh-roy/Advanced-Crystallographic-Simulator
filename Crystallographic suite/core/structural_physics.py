#Lattice and form factors

import numpy as np

def calculate_atomic_form_factor(element_symbol: str, theta_rad: np.ndarray, wavelength: float = 1.5406) -> np.ndarray:
    """
    Computes the angle-dependent atomic scattering factor (f_j) using 
    the 9-parameter analytical Cromer-Mann Gaussian approximation constants.
    """
    coefficients = {
        "Cu": [13.338, 3.582, 7.168, 0.247, 5.616, 11.397, 1.674, 64.813, 1.191],
        "Ni": [12.838, 3.829, 6.810, 0.259, 5.565, 12.167, 1.595, 67.433, 1.171]
    }
    
    if element_symbol not in coefficients:
        raise ValueError(f"Element '{element_symbol}' constants not defined.")
        
    p = coefficients[element_symbol]
    
    # Calculate s^2 = (sin(theta) / lambda)^2
    s_sq = (np.sin(theta_rad) / wavelength) ** 2
    
    # Extract coefficients into distinct individual scalar variables
    a1, b1 = p[0], p[1]
    a2, b2 = p[2], p[3]
    a3, b3 = p[4], p[5]
    a4, b4 = p[6], p[7]
    c_val  = p[8]

    # Calculate each term using the distinct variables
    term1 = a1 * np.exp(-b1 * s_sq)
    term2 = a2 * np.exp(-b2 * s_sq)
    term3 = a3 * np.exp(-b3 * s_sq)
    term4 = a4 * np.exp(-b4 * s_sq)
    
    f_j = term1 + term2 + term3 + term4 + c_val
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