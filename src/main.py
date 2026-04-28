"""
Example script: design a waverider and save geometry plots.

Run from src/:
    python main.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from waverider import design_waverider
from mesh_panelization import (
    panelize_geometry,
    panelization_volume,
    panelization_wetted_area,
    plot_panelization,
    plot_scalar_field,
)
from aerodynamics import compute_forces, compute_pressure

output_dir = "runs/M6_beta16.5"
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

# Given the input parameters, design the optimal waverider geometry
geom = design_waverider(
    M1=6,
    gamma=1.4,
    beta=20,
    L=20, # meters
    N=500,
    N_l=50,
    N_up=25,
    output_dir=output_dir,
)

# Make the computational mesh over the waverider geometry
lower_mesh, upper_mesh = panelize_geometry(geom)

# Compute some mesh statistics
wetted = panelization_wetted_area(lower_mesh, upper_mesh)
volume = panelization_volume(lower_mesh, upper_mesh)
n_tri  = lower_mesh["triangles"].shape[0] + upper_mesh["triangles"].shape[0]

# Compute inviscid aerodynamic forces
forces = compute_forces(geom, lower_mesh)

# Report
sc = geom["shock_conditions"]
print(f"\nShock conditions")
print(f"  M2            = {sc['M2']:.4f}")
print(f"  theta         = {sc['theta_deg']:.4f} deg")
print(f"  cone angle    = {sc['cone_half_angle_deg']:.4f} deg")
print(f"\nMesh statistics")
print(f"  Triangles     = {n_tri}  ({lower_mesh['triangles'].shape[0]} lower, {upper_mesh['triangles'].shape[0]} upper)")
print(f"  Wetted area   = {wetted:.3f} m^2")
print(f"  Volume (approx) = {volume:.3f} m^3")
print(f"\nAerodynamic coefficients")
print(f"  p02/p1        = {forces['p02_over_p1']:.4f}")
print(f"  Cp mean       = {forces['Cp'].mean():.4f}  range [{forces['Cp'].min():.4f}, {forces['Cp'].max():.4f}]")
print(f"  CL            = {forces['CL']:.4f}")
print(f"  CD            = {forces['CD']:.4f}")
print(f"  L/D           = {forces['L_over_D']:.4f}")

# Make a nice plot of the geometry
mesh_plot_path = os.path.join(output_dir, "plots", "mesh.png")
plot_panelization(lower_mesh, upper_mesh, save_path=mesh_plot_path, show=False)
print(f"Mesh plot saved to {mesh_plot_path}")

# Plot the lower-surface pressure coefficient
pressure = compute_pressure(geom, lower_mesh)
pressure_plot_path = os.path.join(output_dir, "plots", "pressure_cp.png")
plot_scalar_field(
    lower_mesh,
    pressure["Cp"],
    title="Lower-surface pressure coefficient (Cp)",
    cmap="viridis",
    colorbar_label="Cp",
    save_path=pressure_plot_path,
    show=False,
)
print(f"Pressure plot saved to {pressure_plot_path}")
