"""
Example script: design a waverider and save geometry plots.

Run from src/:
    python main.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from waverider import design_waverider
from mesh_panelization import (
    panelize_geometry,
    panelization_volume,
    panelization_wetted_area,
    plot_scalar_field,
)
from aerodynamics import compute_inviscid_forces, compute_pressure
from boundary_layer import compute_skin_friction, skin_friction_on_mesh

output_dir = "runs/M6_beta16.5"
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

# Given the input parameters, design the optimal waverider geometry
geom = design_waverider(
    M1=6,
    gamma=1.2, # lower gamma exaggerates reaction effects
    beta=14,
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
inviscid_forces = compute_inviscid_forces(geom, lower_mesh)

# Viscous skin-friction (Walz integral method along lower- and upper-surface streamlines)
viscous_forces = compute_skin_friction(
    geom, lower_mesh, upper_mesh,
    T_inf=216.65, p_inf=5474.9,   # ~20 km standard atmosphere
    T_w=1000.0,                    # constant wall temperature (K)
)

# Total drag coefficient (inviscid + viscous)
CD_total = inviscid_forces['CD'] + viscous_forces['CDf']
LD_total = inviscid_forces['CL'] / CD_total if abs(CD_total) > 1e-12 else float('inf')

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
print(f"  p02/p1        = {inviscid_forces['p02_over_p1']:.4f}")
print(f"  Cp mean       = {inviscid_forces['Cp'].mean():.4f}  range [{inviscid_forces['Cp'].min():.4f}, {inviscid_forces['Cp'].max():.4f}]")
print(f"  CL            = {inviscid_forces['CL']:.4f}")
print(f"  CD (inviscid) = {inviscid_forces['CD']:.4f}")
print(f"  CDf lower     = {viscous_forces['CDf_lower']:.4f}  (D_f = {viscous_forces['D_lower']:.1f} N)")
print(f"  CDf upper     = {viscous_forces['CDf_upper']:.4f}  (D_f = {viscous_forces['D_upper']:.1f} N)")
print(f"  CDf total     = {viscous_forces['CDf']:.4f}  (D_f = {viscous_forces['D_friction']:.1f} N)")
print(f"  CD total      = {CD_total:.4f}")
print(f"  L/D inviscid  = {inviscid_forces['L_over_D']:.4f}")
print(f"  L/D w/ visc.  = {LD_total:.4f}")

# Plot the lower-surface pressure coefficient
pressure = compute_pressure(geom, lower_mesh)
pressure_plot_path = os.path.join(output_dir, "plots", "pressure_cp.png")
plot_scalar_field(
    lower_mesh = lower_mesh,
    lower_field = pressure["Cp"],
    upper_mesh = upper_mesh,
    upper_field = np.zeros(upper_mesh["triangles"].shape[0]), # dummy zero field for upper surface
    title="Lower-surface pressure coefficient (Cp)",
    cmap="viridis",
    colorbar_label="Cp",
    save_path=pressure_plot_path,
    show=False,
)
print(f"Pressure plot saved to {pressure_plot_path}")

# Plot the skin friction coefficient
cf_lo  = skin_friction_on_mesh(viscous_forces, lower_mesh, geom, "lower", "cf")
cf_up  = skin_friction_on_mesh(viscous_forces, upper_mesh, geom, "upper", "cf")
cf_all = np.concatenate([cf_lo, cf_up])
cf_plot_path = os.path.join(output_dir, "plots", "skin_friction_cf.png")
plot_scalar_field(
    lower_mesh = lower_mesh, lower_field = np.log10(cf_lo),
    upper_mesh = upper_mesh, upper_field = np.log10(cf_up),
    title="Skin-friction coefficient $c_f$ (Walz)",
    cmap="magma",
    colorbar_label=r"$\log_{10}(c_f) = \log_{10}(2\tau_w / (\rho_e u_e^2))$",
    vmin=float(np.log10(cf_all.min())), vmax=float(np.percentile(np.log10(cf_all), 98)),
    save_path=cf_plot_path, show=False,
)
print(f"Skin friction (cf) plot saved to {cf_plot_path}")

# Plot the momentum boundary layer thickness
d2_lo  = skin_friction_on_mesh(viscous_forces, lower_mesh, geom, "lower", "delta2")
d2_up  = skin_friction_on_mesh(viscous_forces, upper_mesh, geom, "upper", "delta2")
d2_plot_path = os.path.join(output_dir, "plots", "momentum_thickness.png")
plot_scalar_field(
    lower_mesh = lower_mesh, lower_field = d2_lo * 1e3,
    upper_mesh = upper_mesh, upper_field = d2_up * 1e3,
    title=r"Momentum thickness $\delta_2$ [mm]",
    cmap="viridis",
    colorbar_label=r"$\delta_2$ [mm]",
    save_path=d2_plot_path, show=True,
)
print(f"Momentum thickness plot saved to {d2_plot_path}")
