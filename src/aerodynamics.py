"""
Aerodynamic force integration for osculating-cone waveriders.

The lower surface lives in the Taylor-Maccoll conical flow field.  Pressure
depends only on the polar angle θ = arctan(√(y²+z²) / x) from the cone axis.
The upper surface is assumed to be at freestream pressure (Cp = 0).

Public API
----------
compute_forces(geom, lower_mesh) -> dict
"""

import numpy as np
from scipy.interpolate import interp1d
from taylor_maccoll_sol import Taylor_Maccoll


def _tm_vsq_profile(gamma, beta_rad, Vr_i, V_theta_i, n=4000):
    """Solve T-M from shock to cone surface; return (theta, V'^2) arrays."""
    tm = Taylor_Maccoll(gamma)
    theta_range = np.linspace(beta_rad, 1e-8, n)
    sol = tm.tracing_solver(Vr_i, V_theta_i, [beta_rad, 1e-8], theta_range)
    Vsq = sol.y[0] ** 2 + sol.y[1] ** 2   # Vr'^2 + Vtheta'^2
    return sol.t, Vsq                       # theta descending from beta


def compute_pressure(geom, lower_mesh):
    """
    Return Taylor-Maccoll pressure over the lower surface mesh.

    Parameters
    ----------
    geom        : dict returned by design_waverider
    lower_mesh  : dict returned by panelize_geometry (lower surface only)

    Returns
    -------
    dict with keys
        'Cp'          : ndarray (N_tri,) pressure coefficient per triangle
        'p_over_p1'   : ndarray (N_tri,) p/p1 at triangle centroids
        'p02_over_p1' : float, post-shock stagnation pressure ratio
        'theta'       : ndarray (N_tri,) polar angle at centroids (rad)
    """
    sc     = geom["shock_conditions"]
    params = geom["parameters"]
    M1         = params["M1"]
    gamma      = params["gamma"]
    beta_rad   = sc["beta_rad"]
    Vr_i       = sc["Vr_i"]
    V_theta_i  = sc["V_theta_i"]
    M2         = sc["M2"]

    # ── T-M V'^2(θ) profile ─────────────────────────────────────────────────
    theta_tm, Vsq_tm = _tm_vsq_profile(gamma, beta_rad, Vr_i, V_theta_i)
    # theta_tm is descending; flip so interp1d gets ascending x
    Vsq_interp = interp1d(theta_tm[::-1], Vsq_tm[::-1],
                          kind="linear", bounds_error=False,
                          fill_value=(Vsq_tm[-1], Vsq_tm[0]))

    # ── Post-shock stagnation pressure ratio p02/p1 ──────────────────────────
    # Oblique shock: normal Mach component
    Mn1 = M1 * np.sin(beta_rad)
    p2_p1   = (2 * gamma * Mn1**2 - (gamma - 1)) / (gamma + 1)
    p02_p2  = (1 + (gamma - 1) / 2 * M2**2) ** (gamma / (gamma - 1))
    p02_p1  = p02_p2 * p2_p1

    # ── Pressure coefficient at each triangle centroid ───────────────────────
    cents = lower_mesh["centroids"]          # (N, 3)
    xc, yc, zc = cents[:, 0], cents[:, 1], cents[:, 2]
    theta_c = np.arctan2(np.sqrt(yc**2 + zc**2), xc)

    Vsq_c    = Vsq_interp(theta_c)
    p_over_p1 = p02_p1 * (1.0 - Vsq_c) ** (gamma / (gamma - 1))
    q1_factor = gamma / 2.0 * M1**2          # q1 = q1_factor * p1
    Cp = (p_over_p1 - 1.0) / q1_factor

    return {
        "Cp":          Cp,
        "p_over_p1":   p_over_p1,
        "p02_over_p1": float(p02_p1),
        "theta":       theta_c,
    }


def compute_inviscid_forces(geom, lower_mesh):
    """Return inviscid force coefficients using the lower-surface pressure."""
    pressure = compute_pressure(geom, lower_mesh)
    Cp = pressure["Cp"]

    # Planform (reference) area from x-y projection
    tris = lower_mesh["triangles"]
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    cross_z = ((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
             - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0]))
    A_xy = np.abs(cross_z) / 2.0
    S_ref = float(A_xy.sum())

    # Pressure force: dF = -(p - p_inf) n_hat dA
    norms = lower_mesh["normals"]
    areas = lower_mesh["areas"]

    dCF = -Cp[:, None] * norms * (areas / S_ref)[:, None]
    CF = dCF.sum(axis=0)

    # Sign conventions (vehicle flies in +x, lift perpendicular = +z upward)
    CL = float(CF[2])
    CD = float(CF[0])
    LD = CL / CD if abs(CD) > 1e-12 else float("inf")

    return {
        "CL":            CL,
        "CD":            CD,
        "L_over_D":      LD,
        "Cp":            Cp,
        "planform_area": S_ref,
        "p02_over_p1":   pressure["p02_over_p1"],
    }
