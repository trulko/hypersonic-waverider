"""
Public API for the osculating-cone waverider design tool.

Usage
-----
from waverider import design_waverider

geom = design_waverider(M1=8, gamma=1.4, beta=16.5, L=2.0)
"""

import os
import numpy as np

from oblique_shock import Oblique_Shock
from taylor_maccoll_sol import Taylor_Maccoll
from TE_Formation import TEG
from streamline_tracing import TRACE


def design_waverider(
    M1: float,
    gamma: float,
    beta: float,
    L: float = 2.0,
    N: int = 500,
    N_l: int = 12,
    N_up: int = 10,
    R1_frac: float = 0.2,
    W2_frac: float = 0.8,
    output_dir: str | None = None,
) -> dict:
    """
    Design an osculating-cone waverider for given freestream conditions.

    Parameters
    ----------
    M1      : Freestream Mach number.
    gamma   : Ratio of specific heats.
    beta    : Conical shock half-angle (degrees).
    L       : Vehicle length (m or non-dimensional).
    N       : Leading-edge resolution (>=500 recommended).
    N_l     : Number of lower-surface streamlines.
    N_up    : Number of upper-surface lines.
    R1_frac : Trailing-edge R1 as a fraction of Rs  (default 0.2).
    W2_frac : Trailing-edge W2 as a fraction of Rs  (default 0.8).
    output_dir : Directory for saved plots; None disables plot output.

    Returns
    -------
    dict with keys:
        "leading_edge"    : {"x", "y", "z"} arrays along the leading edge.
        "trailing_edge"   : {"x", "y", "z"} arrays along the trailing edge.
        "lower_surface"   : list of streamline dicts {"curve", "mirrored_curve"}.
        "upper_surface"   : list of {"x", "y", "z"} free-stream lines.
        "baseplane_curve" : {"x", "y", "z"} interpolated lower-surface TE.
        "shock_conditions": {"M2", "theta_deg", "beta_rad", "theta_rad",
                             "Vr_i", "V_theta_i", "cone_half_angle_deg"}.
        "parameters"      : echo of all input parameters plus derived Rs, a, b, c.
    """
    beta_rad = np.radians(beta)
    Rs = L * np.tan(beta_rad)
    R1 = R1_frac * Rs
    W2 = W2_frac * Rs

    a = -R1
    b = 2 * (R1 - np.sqrt(Rs**2 - W2**2)) / W2**2
    c = (np.sqrt(Rs**2 - W2**2) - R1) / W2**4

    def z_func(y):
        return a + b * y**2 + c * y**4

    # --- oblique shock initial conditions ---
    os_solver = Oblique_Shock()
    M2, theta_deg, beta_rad_out, theta_rad = os_solver.sub_1(M1, gamma, beta)
    Vr_i, V_theta_i = os_solver.initial_nondimensioned_conditions(M1, gamma, beta)

    # --- cone half-angle from Taylor-Maccoll ---
    tm = Taylor_Maccoll(gamma)
    cone_angle = tm.cone_half_angle(beta_rad, Vr_i, V_theta_i)

    # --- set up output directory if requested ---
    plot_dir = None
    if output_dir is not None:
        plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)

    # --- base-plane plot ---
    teg = TEG(gamma)
    teg.plot_baseplane(
        z_func, Rs, L, N, beta_rad, Vr_i, V_theta_i,
        save_path=os.path.join(plot_dir, "baseplane.png") if plot_dir else None,
    )

    # --- streamline tracing & geometry plot ---
    tracer = TRACE(gamma)
    geometry = tracer.tracing_module(z_func, Rs, L, N, N_l, N_up, Vr_i, V_theta_i)

    _, _, _, X_p, Y_p, Z_p, *_ = tracer.projection_module(z_func, Rs, L, N)
    X_b = np.full_like(Y_p, L)

    tracer.plot_geometry(
        geometry, X_p, Y_p, Z_p, X_b, Y_p, Z_p,
        save_path=os.path.join(plot_dir, "waverider_3d.png") if plot_dir else None,
    )

    geometry["shock_conditions"] = {
        "M2": float(M2),
        "theta_deg": float(theta_deg),
        "beta_rad": float(beta_rad_out),
        "theta_rad": float(theta_rad),
        "Vr_i": float(Vr_i),
        "V_theta_i": float(V_theta_i),
        "cone_half_angle_deg": float(np.degrees(cone_angle)),
    }
    geometry["parameters"] = {
        "M1": M1, "gamma": gamma, "beta": beta, "L": L,
        "N": N, "N_l": N_l, "N_up": N_up,
        "Rs": Rs, "R1": R1, "W2": W2, "a": a, "b": b, "c": c,
    }

    return geometry
