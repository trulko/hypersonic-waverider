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
    wv_inviscid = Waverider(
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
    wv_viscous = Waverider(
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

    # Choose the waverider
    wv = wv_viscous
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
