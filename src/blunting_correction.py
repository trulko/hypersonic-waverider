import numpy as np

SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann constant [W/m^2/K^4]

def stagnation_point_heating_sutton_graves(rho_inf, V_inf, R_n, k_sg=1.83e-4):
    """
    Sutton-Graves stagnation-point convective heating estimate for air.

    Parameters
    ----------
    rho_inf : float
        Freestream density [kg/m^3]
    V_inf : float
        Freestream speed [m/s]
    R_n : float
        Nose / leading-edge radius [m]
    k_sg : float
        Sutton-Graves coefficient for air in SI units

    Returns
    -------
    q_conv : float
        Convective heat flux [W/m^2]
    """
    if R_n <= 0.0:
        raise ValueError("R_n must be positive.")

    return k_sg * np.sqrt(rho_inf / R_n) * V_inf**3


def equilibrium_wall_temperature(rho_inf, V_inf, R_n,
                                 emissivity=0.85,
                                 T_bg=300.0,
                                 k_sg=1.83e-4):
    """
    Radiative-equilibrium wall temperature from Sutton-Graves heating.

    Solves:
        q_conv = eps * sigma * (T_eq^4 - T_bg^4)

    Parameters
    ----------
    rho_inf : float
        Freestream density [kg/m^3]
    V_inf : float
        Freestream speed [m/s]
    R_n : float
        Nose / leading-edge radius [m]
    emissivity : float
        Surface emissivity [-]
    T_bg : float
        Background radiation temperature [K]
    k_sg : float
        Sutton-Graves coefficient

    Returns
    -------
    T_eq : float
        Radiative-equilibrium wall temperature [K]
    """
    q_conv = stagnation_point_heating_sutton_graves(rho_inf, V_inf, R_n, k_sg=k_sg)
    T_eq = (T_bg**4 + q_conv / (emissivity * SIGMA_SB))**0.25
    return T_eq


def minimum_blunting_radius(rho_inf, V_inf, T_allow,
                            emissivity=0.85,
                            T_bg=300.0,
                            k_sg=1.83e-4,
                            safety_factor=1.5):
    """
    Minimum constant blunt radius so that stagnation-point radiative-equilibrium
    temperature does not exceed T_allow.

    Uses:
        q_conv = k_sg * sqrt(rho_inf / R_n) * V_inf^3
        q_rad  = eps * sigma * (T_allow^4 - T_bg^4)

    Set q_conv <= q_rad and solve for R_n.

    Parameters
    ----------
    rho_inf : float
        Freestream density [kg/m^3]
    V_inf : float
        Freestream speed [m/s]
    T_allow : float
        Maximum allowable wall temperature [K]
    emissivity : float
        Surface emissivity [-]
    T_bg : float
        Background radiation temperature [K]
    k_sg : float
        Sutton-Graves coefficient
    safety_factor : float
        >1 makes the radius more conservative

    Returns
    -------
    result : dict
        {
            "R_min": minimum radius [m],
            "q_allow": allowable radiative heat flux [W/m^2]
        }
    """
    q_allow = emissivity * SIGMA_SB * (T_allow**4 - T_bg**4)
    if q_allow <= 0.0:
        raise ValueError("T_allow must be greater than T_bg.")

    R_min = rho_inf * ((safety_factor * k_sg * V_inf**3) / q_allow)**2

    return {
        "R_min": float(R_min),
        "q_allow": float(q_allow),
    }

def cpmax_modified_newtonian(M, gamma=1.4):
    """
    Modified-Newtonian maximum pressure coefficient based on a normal-shock
    stagnation-pressure construction.

    Parameters
    ----------
    M : float or array
        Upstream Mach number
    gamma : float
        Ratio of specific heats

    Returns
    -------
    Cp_max : float or array
    """
    M = np.asarray(M, dtype=float)
    Cp = np.zeros_like(M)

    sup = M > 1.0
    Ms = M[sup]

    if Ms.size == 0:
        return Cp.item() if Cp.ndim == 0 else Cp

    # Static pressure ratio across normal shock
    p2_p1 = 1.0 + (2.0 * gamma / (gamma + 1.0)) * (Ms**2 - 1.0)

    # Downstream Mach number after normal shock
    M2_sq = (1.0 + 0.5 * (gamma - 1.0) * Ms**2) / (gamma * Ms**2 - 0.5 * (gamma - 1.0))

    # Isentropic stagnation pressure ratio from downstream state
    p02_p2 = (1.0 + 0.5 * (gamma - 1.0) * M2_sq)**(gamma / (gamma - 1.0))

    # Stagnation pressure behind shock relative to upstream static pressure
    p02_p1 = p2_p1 * p02_p2

    # Dynamic pressure / upstream static pressure
    q_over_p1 = 0.5 * gamma * Ms**2

    Cp[sup] = (p02_p1 - 1.0) / q_over_p1

    return Cp.item() if Cp.ndim == 0 else Cp


def blunt_leading_edge_force(geom, R_n, rho_inf, V_inf, M1, gamma, S_ref):
    """
    Modified-Newtonian pressure force on a blunt (cylindrical) leading edge.

    The leading edge of the waverider is treated as a swept circular cylinder of
    radius R_n.  For each discretised LE segment with unit tangent t_hat, we
    decompose the freestream into components parallel and perpendicular to the
    LE.  The local sweep is sin(Lambda) = t_hat . u_inf so that the perpendicular
    Mach number is M_perp = M1 cos(Lambda).

    Around the cross-section (a circle of radius R_n in the plane normal to
    t_hat) the modified-Newtonian distribution is Cp(phi) = Cp_max(M_perp) *
    cos^2(Lambda) * cos^2(phi), with phi measured from the stagnation line in
    the perpendicular plane.  Closed-form integration over the windward half
    (-pi/2, pi/2) gives a resultant force per unit LE length

        dF/ds = (4/3) q_inf R_n Cp_max(M_perp) cos^2(Lambda) * e_perp,

    where e_perp = (u_inf - sin(Lambda) t_hat) / cos(Lambda) is the unit vector
    along the freestream component perpendicular to the LE (windward direction).

    Parameters
    ----------
    geom : dict
        Output of design_waverider; uses geom["leading_edge"] which spans the
        full LE (both halves, y in [-y_up, +y_up]).
    R_n : float
        Blunt-LE radius [m].
    rho_inf, V_inf : float
        Freestream density [kg/m^3] and speed [m/s].
    M1, gamma : float
        Freestream Mach and ratio of specific heats.
    S_ref : float
        Reference (planform) area [m^2] for non-dimensionalisation.

    Returns
    -------
    dict with keys:
        "F"         : ndarray (3,)  total force vector on blunt LE [N]
        "dCL"       : float         lift-coefficient increment
        "dCD"       : float         drag-coefficient increment
        "R_n"       : float         radius used [m]
        "Lambda_deg": ndarray (N-1,) local sweep angle per segment [deg]
        "Cp_max"    : ndarray (N-1,) Cp_max(M_perp) per segment
    """
    le = geom["leading_edge"]
    pts = np.column_stack([np.asarray(le["x"], dtype=float),
                           np.asarray(le["y"], dtype=float),
                           np.asarray(le["z"], dtype=float)])

    diffs = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    valid = seg_len > 1e-15
    t_hat = np.zeros_like(diffs)
    t_hat[valid] = diffs[valid] / seg_len[valid, None]

    # Sweep: sin(Lambda) = t_hat . u_inf = t_x
    sinL = t_hat[:, 0]
    cos2L = np.clip(1.0 - sinL**2, 0.0, 1.0)
    cosL = np.sqrt(cos2L)

    # Perpendicular freestream component direction (unit vector e_perp)
    u_perp = np.zeros_like(diffs)
    u_perp[:, 0] = 1.0
    u_perp -= sinL[:, None] * t_hat
    e_perp = np.zeros_like(u_perp)
    nz = cosL > 1e-12
    e_perp[nz] = u_perp[nz] / cosL[nz, None]

    # Perpendicular Mach and Cp_max(M_perp)
    M_perp = M1 * cosL
    Cp_max = cpmax_modified_newtonian(M_perp, gamma=gamma)
    if np.ndim(Cp_max) == 0:
        Cp_max = np.full_like(cosL, float(Cp_max))

    q_inf = 0.5 * rho_inf * V_inf**2

    # Force per unit LE length (magnitude along e_perp)
    f_mag = (4.0 / 3.0) * q_inf * R_n * cos2L * Cp_max  # (N-1,)

    # Force on each segment
    F_seg = (f_mag * seg_len)[:, None] * e_perp  # (N-1, 3)
    F_total = F_seg.sum(axis=0)

    dCD = float(F_total[0] / (q_inf * S_ref))
    dCL = float(F_total[2] / (q_inf * S_ref))

    return {
        "F":          F_total,
        "dCL":        dCL,
        "dCD":        dCD,
        "R_n":        float(R_n),
        "Lambda_deg": np.degrees(np.arcsin(np.clip(sinL, -1.0, 1.0))),
        "Cp_max":     Cp_max,
    }
