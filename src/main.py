"""
Example script: design a waverider and save geometry plots.

Run from src/:
    python main.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from waverider import design_waverider

geom = design_waverider(
    M1=8,
    gamma=1.4,
    beta=16.5,
    L=2.0,
    N=500,
    N_l=12,
    N_up=10,
    output_dir="runs/M8_beta16.5",
)

sc = geom["shock_conditions"]
print(f"M2            = {sc['M2']:.4f}")
print(f"theta         = {sc['theta_deg']:.4f} deg")
print(f"cone angle    = {sc['cone_half_angle_deg']:.4f} deg")
print(f"Plots saved to runs/M8_beta16.5/plots/")
