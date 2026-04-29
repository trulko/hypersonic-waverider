"""
Example script: design a waverider, analyse it, report and plot.

Run from src/:
    python main.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from waverider import Waverider

print("Making Waverider geometry...")
# Note: to play with the upper trailing edge parameters, see:
# https://www.desmos.com/calculator/ertbddykms
wv = Waverider(
    M1 = 6,
    gamma = 1.4,
    beta = 16.5,
    L = 20, # m
    R1_frac = 0.2,
    W2_frac = 0.8,
    n_shape = 1.0,
)

print("Analyzing aerothermodynamics...")
wv.aerothermodynamics(
    T_inf = 216.65,   # K   (~20 km standard atmosphere)
    p_inf = 5474.9,   # Pa  (~20 km standard atmosphere)
    T_allow = 2500.0, # K   (refractory composite limit)
    emissivity = 0.9, # [-] (typical for high-temp composites)
    safety_factor = 1.5, # [-] (safety factor for the bluntness sizing)
    resample=200, # per-streamline resampling resolution for the boundary layer integration
    n_theta=20, # number of polar angle samples for Taylor--Maccoll
)

# Plotting requires pyvista to be installed
output_dir = "demo"
wv.plot(output_dir)

# Print
wv.report()

# Uncomment to show an interactive 3D plot of the geometry
# wv.interactive()