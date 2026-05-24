"""
Example script: design a waverider, analyse it, report and plot.

Run from src/:
    python main.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from Waverider import Waverider

def build_waverider() -> Waverider:
    """Construct and analyse the project baseline waverider."""
    print("Making Waverider geometry...")
    # Note: to play with the upper trailing edge parameters, see:
    # https://www.desmos.com/calculator/ertbddykms

    # The waverider which maximizes inviscid L/D
    wv_inviscid_ld = Waverider(
        M1 = 6,           # Freestream Mach number
        gamma = 1.4,      # Ratio of specific heats
        min_height = 3,   # [m]
        min_area = 100,   # [m^2]
        min_volume = 250, # [m^3]
        beta = 13.791,    # Input: shock angle [degrees]
        R1_frac = 0.348,  # Input: roughly controls height
        W2_frac = 0.605,  # Input: roughly controls width
        n_shape = 4.215,  # Input: roughly controls roundness
        N = 500,          # Resolution of the leading edge
        N_l = 30,         # Resulution of the upper, lower surfaces
    )

    # The waverider which maximizes viscous L/D
    wv_viscous_ld = Waverider(
        M1 = 6,           # Freestream Mach number
        gamma = 1.4,      # Ratio of specific heats
        min_height = 3,   # [m]
        min_area = 100,   # [m^2]
        min_volume = 250, # [m^3]
        beta = 11.3967,    # Input: shock angle [degrees]
        R1_frac = 0.835,  # Input: roughly controls height
        W2_frac = 0.36,  # Input: roughly controls width
        n_shape = 1.04,  # Input: roughly controls roundness
        N = 500,          # Resolution of the leading edge
        N_l = 30,         # Resulution of the upper, lower surfaces
    )

    # The waverider which minimizes volume/(L/D) (i.e., optimizes for thrust-to-weight)
    wv_viscous_thrust = Waverider(
        M1 = 6,           # Freestream Mach number
        gamma = 1.4,      # Ratio of specific heats
        min_height = 3,   # [m]
        min_area = 100,   # [m^2]
        min_volume = 250, # [m^3]
        beta = 12.513,    # Input: shock angle [degrees]
        R1_frac = 0.419,  # Input: roughly controls height
        W2_frac = 0.657,  # Input: roughly controls width
        n_shape = 1.136,  # Input: roughly controls roundness
        N = 500,          # Resolution of the leading edge
        N_l = 30,         # Resulution of the upper, lower surfaces
    )

    # Choose the waverider to run with
    wv = wv_viscous_thrust

    wv.aerothermodynamics(
        T_inf = 216.65,   # K   (~20 km standard atmosphere)
        p_inf = 5474.9,   # Pa  (~20 km standard atmosphere)
        T_allow = 2500.0, # K   (refractory composite limit)
        emissivity = 0.9, # [-] (typical for high-temp composites)
        safety_factor = 1.5, # [-] (safety factor for the bluntness sizing)
        resample = 200,   # per-streamline resampling resolution for the boundary layer integration
        n_theta = 20,     # number of polar angle samples for Taylor-Maccoll)
    )
    return wv

def main() -> None:
    wv = build_waverider()

    # Plotting requires pyvista to be installed
    output_dir = "../runs/demo/"
    wv.plot(output_dir)

    # Print
    wv.report()

    # Uncomment to show an interactive 3D plot of the geometry
    wv.interactive()


if __name__ == "__main__":
    main()
