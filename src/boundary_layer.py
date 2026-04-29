"""
Laminar boundary-layer / skin-friction module.

Implements Walz's integral method (compressible, two-equation: Z, W) along
each lower-surface streamline of an osculating-cone waverider, then sums the
shear stress to a friction-drag coefficient.

Public API
----------
compute_skin_friction(geom, lower_mesh=None, ...) -> dict
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq

from taylor_maccoll_sol import Taylor_Maccoll


def _trapz_compat(y, x):
    """Use trapezoid integration across NumPy versions."""
    trapz_fn = getattr(np, "trapezoid", None)
    if trapz_fn is None:
        trapz_fn = np.trapz
    return trapz_fn(y, x)


# ---------------------------------------------------------------------------
# Gas / fluid helpers
# ---------------------------------------------------------------------------

R_AIR = 287.05  # J/(kg K)

_SUTH_T0 = 273.15
_SUTH_MU0 = 1.716e-5
_SUTH_S = 110.4


def sutherland_mu(T):
    T = np.asarray(T, dtype=float)
    return _SUTH_MU0 * (T / _SUTH_T0) ** 1.5 * (_SUTH_T0 + _SUTH_S) / (T + _SUTH_S)


# ---------------------------------------------------------------------------
# Walz auxiliary functions
# ---------------------------------------------------------------------------
#
# All formulas follow the report (Section "Viscous shear calculation").  The
# starred quantities (W*, etc.) are the *incompressible* shape parameters; the
# unstarred ones are their compressible counterparts.  Given Me, theta_tilde,
# and a candidate W*, the helpers return every Walz coefficient we need.

W_STAR_MIN = 1.515 + 1e-6
W_STAR_MAX = 1.99


def _walz_aux(Wstar, Me, th_tilde, gamma, r):
    """Compute (a, b, H12, H, betau, chi, beta_, psi, W) given W*."""
    x = Wstar - 1.515
    a = 1.7261 * x ** 0.7158
    H12 = 4.0306 - 4.2845 * x ** 0.3886
    betau = 0.1564 + 2.1921 * x ** 1.70

    rgM2 = r * (gamma - 1) / 2.0 * Me ** 2

    # Compressibility correction for beta
    one_pl = 1.0 + rgM2 * (1.16 * Wstar - 1.072 - th_tilde * (2.0 * Wstar - 2.581))
    one_pr = 1.0 + rgM2 * (1.0 - th_tilde)
    # Guard against tiny negative due to roundoff
    one_pl = max(one_pl, 1e-12)
    one_pr = max(one_pr, 1e-12)
    chi = (one_pl ** 0.7) * (one_pr ** -0.7)
    beta_ = betau * chi

    # ψ from W* using its definition:  ψ12, ψ0  →  ψ
    delta1u_over_delta = 0.420 - x ** (0.424 * Wstar)
    g = 0.324 + 0.336 * x ** 0.555
    psi0 = 0.0144 * (2.0 - Wstar) * (2.0 - th_tilde) ** 0.8
    psi12 = ((2.0 - delta1u_over_delta) / Wstar) * th_tilde \
          + ((1.0 - delta1u_over_delta) / (Wstar * g)) * (1.0 - th_tilde)
    Me2 = Me ** 2
    psi = 1.0 + (psi12 - 1.0) * Me2 / (Me2 + (1.0 / psi12) / psi0)

    # Compressible W and shape factor
    W = Wstar * psi
    b = 1.0 + rgM2 * (W - th_tilde) * (2.0 - W)
    H = b * H12 + rgM2 * (W - th_tilde)

    return dict(a=a, b=b, H12=H12, H=H, betau=betau, chi=chi, beta_=beta_,
                psi=psi, W=W, g=g, psi0=psi0, psi12=psi12)


def _solve_Wstar(W, Me, th_tilde, gamma, r):
    """Given compressible W, find W* such that W*ψ(W*) = W."""
    def f(Ws):
        return _walz_aux(Ws, Me, th_tilde, gamma, r)["W"] - W
    fa = f(W_STAR_MIN)
    fb = f(W_STAR_MAX)
    if fa * fb > 0:
        # fall back: clamp to whichever endpoint is closer
        return W_STAR_MIN if abs(fa) < abs(fb) else W_STAR_MAX
    return brentq(f, W_STAR_MIN, W_STAR_MAX, xtol=1e-9)


# ---------------------------------------------------------------------------
# Walz integration along a streamline
# ---------------------------------------------------------------------------

def integrate_walz(s, ue, Te, rho_e, mu_w, T_w, T_aw, gamma, Pr=0.72,
                   Z0=None, W0=None):
    """
    Integrate Walz two-equation method along arc length s.

    Parameters
    ----------
    s       : (N,) monotonically increasing arc length, s[0] >= 0.
    ue, Te  : (N,) edge velocity / temperature.
    rho_e   : (N,) edge density.
    mu_w    : (N,) wall viscosity (μ at T_w).
    T_w     : (N,) wall temperature (constant or varying).
    T_aw    : (N,) adiabatic wall temperature.
    gamma   : ratio of specific heats.
    Pr      : Prandtl number.
    Z0, W0  : optional initial values; defaults to Blasius-like start.

    Returns
    -------
    dict with keys
        'Z'        : (N,) Z(s)
        'W'        : (N,) W(s)
        'Wstar'    : (N,)
        'delta2'   : (N,) momentum thickness
        'tau_w'    : (N,) wall shear  (Pa)
        'cf'       : (N,) skin-friction coefficient based on edge q (2τ_w/ρ_e u_e²)
    """
    s = np.asarray(s, dtype=float)
    n = s.size
    ue = np.asarray(ue, dtype=float)
    Te = np.asarray(Te, dtype=float)
    rho_e = np.asarray(rho_e, dtype=float)
    mu_w = np.asarray(mu_w, dtype=float)
    T_w = np.broadcast_to(np.asarray(T_w, dtype=float), s.shape)
    T_aw = np.asarray(T_aw, dtype=float)

    r = np.sqrt(Pr)

    # Edge Mach number from a^2 = γ R T:
    Me = ue / np.sqrt(gamma * R_AIR * Te)

    # θ̃(s)
    denom = T_aw - Te
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    th_tilde = (T_aw - T_w) / denom

    # Build splines (linear) for ue, μ_w (we need d(ue)/ds, d(μw)/ds)
    # Smoothed central differences via numpy gradient is fine here.
    due_ds = np.gradient(ue, s)
    dmu_ds = np.gradient(mu_w, s)

    # Self-similar (Blasius-like) initial conditions at s[0] using local
    # quantities.  For zero pressure gradient: W = 2β/a, Z = (2a/b) s.
    if W0 is None or Z0 is None:
        # Iterate W* to find self-consistent W = 2β/a
        Ws_guess = 1.572  # Blasius incompressible W*
        for _ in range(40):
            aux = _walz_aux(Ws_guess, Me[0], th_tilde[0], gamma, r)
            W_target = 2.0 * aux["beta_"] / aux["a"]
            Ws_new = _solve_Wstar(W_target, Me[0], th_tilde[0], gamma, r)
            if abs(Ws_new - Ws_guess) < 1e-8:
                Ws_guess = Ws_new
                break
            Ws_guess = Ws_new
        aux = _walz_aux(Ws_guess, Me[0], th_tilde[0], gamma, r)
        s0 = max(s[0], 1e-6)
        Z0_ = (2.0 * aux["a"] / aux["b"]) * s0
        W0_ = aux["W"]
    else:
        Z0_, W0_ = Z0, W0

    # Interpolators for the RHS (linear)
    ue_i = interp1d(s, ue, kind="linear", bounds_error=False,
                    fill_value=(ue[0], ue[-1]))
    Te_i = interp1d(s, Te, kind="linear", bounds_error=False,
                    fill_value=(Te[0], Te[-1]))
    Me_i = interp1d(s, Me, kind="linear", bounds_error=False,
                    fill_value=(Me[0], Me[-1]))
    rhoe_i = interp1d(s, rho_e, kind="linear", bounds_error=False,
                      fill_value=(rho_e[0], rho_e[-1]))
    muw_i = interp1d(s, mu_w, kind="linear", bounds_error=False,
                     fill_value=(mu_w[0], mu_w[-1]))
    th_i = interp1d(s, th_tilde, kind="linear", bounds_error=False,
                    fill_value=(th_tilde[0], th_tilde[-1]))
    due_i = interp1d(s, due_ds, kind="linear", bounds_error=False,
                     fill_value=(due_ds[0], due_ds[-1]))
    dmu_i = interp1d(s, dmu_ds, kind="linear", bounds_error=False,
                     fill_value=(dmu_ds[0], dmu_ds[-1]))

    def rhs(ss, y):
        Z, W = y
        if Z <= 0:
            Z = 1e-20
        ue_l = float(ue_i(ss)); Me_l = float(Me_i(ss))
        muw_l = float(muw_i(ss)); th_l = float(th_i(ss))
        due_l = float(due_i(ss)); dmu_l = float(dmu_i(ss))

        Wstar = _solve_Wstar(W, Me_l, th_l, gamma, r)
        aux = _walz_aux(Wstar, Me_l, th_l, gamma, r)

        # F1 with viscosity-gradient term: n=1 if Tw varies (we always include it)
        log_ue_grad = due_l / max(ue_l, 1e-12)
        log_mu_grad = dmu_l / max(muw_l, 1e-12)
        if abs(log_ue_grad) < 1e-12:
            F1 = 3.0 + 2.0 * aux["H"] - Me_l ** 2  # n→0 effectively
        else:
            F1 = 3.0 + 2.0 * aux["H"] - Me_l ** 2 + log_mu_grad / log_ue_grad
        F2 = 2.0 * aux["a"] / aux["b"]
        F3 = 1.0 - aux["H"] + r * (gamma - 1.0) * Me_l ** 2 * (1.0 - th_l / max(W, 1e-12))
        F4 = (2.0 * aux["beta_"] - aux["a"] * W) / aux["b"]

        dZ = F2 - log_ue_grad * F1 * Z
        dW = F4 / Z - log_ue_grad * F3 * W
        return [dZ, dW]

    sol = solve_ivp(rhs, (s[0], s[-1]), [Z0_, W0_], t_eval=s,
                    method="RK45", rtol=1e-7, atol=1e-10, max_step=(s[-1] - s[0]) / 20)
    if not sol.success:
        raise RuntimeError(f"Walz integration failed: {sol.message}")

    Z = sol.y[0]
    W = sol.y[1]

    # Recover dimensional quantities
    Wstar_arr = np.array([_solve_Wstar(W[i], Me[i], th_tilde[i], gamma, r)
                          for i in range(n)])
    a_arr = 1.7261 * (Wstar_arr - 1.515) ** 0.7158
    delta2 = np.sqrt(np.maximum(Z, 0.0) * mu_w / (rho_e * ue))
    tau_w = a_arr * mu_w * ue / np.maximum(delta2, 1e-30)
    cf = 2.0 * tau_w / (rho_e * ue ** 2)

    return dict(Z=Z, W=W, Wstar=Wstar_arr, delta2=delta2, tau_w=tau_w, cf=cf)


# ---------------------------------------------------------------------------
# Edge conditions along a streamline (from Taylor-Maccoll)
# ---------------------------------------------------------------------------

def _tm_profiles(geom, n_theta=4000):
    """Return interpolators for V'^2(θ), evaluated on the conical flow."""
    sc = geom["shock_conditions"]
    gamma = geom["parameters"]["gamma"]
    beta_rad = sc["beta_rad"]
    Vr_i = sc["Vr_i"]; Vt_i = sc["V_theta_i"]

    tm = Taylor_Maccoll(gamma)
    theta_range = np.linspace(beta_rad, 1e-8, n_theta)
    sol = tm.tracing_solver(Vr_i, Vt_i, [beta_rad, 1e-8], theta_range)
    th = sol.t[::-1]
    Vr = sol.y[0][::-1]
    Vt = sol.y[1][::-1]
    Vsq = Vr ** 2 + Vt ** 2

    Vsq_of_theta = interp1d(th, Vsq, kind="linear",
                            bounds_error=False, fill_value=(Vsq[0], Vsq[-1]))
    Vr_of_theta = interp1d(th, Vr, kind="linear",
                           bounds_error=False, fill_value=(Vr[0], Vr[-1]))
    Vt_of_theta = interp1d(th, Vt, kind="linear",
                           bounds_error=False, fill_value=(Vt[0], Vt[-1]))
    return Vsq_of_theta, Vr_of_theta, Vt_of_theta


def edge_conditions_along(points, geom, T_inf, p_inf):
    """
    For a streamline (N,3) of Cartesian points in the conical flowfield,
    return arrays (s, ue, Te, rho_e, p_e, Me) at each point.

    Edge values come from Taylor-Maccoll; conical flow is isentropic with
    constant total enthalpy and constant post-shock total pressure.
    """
    pts = np.asarray(points, dtype=float)
    M1 = geom["parameters"]["M1"]
    gamma = geom["parameters"]["gamma"]
    beta_rad = geom["shock_conditions"]["beta_rad"]
    M2 = geom["shock_conditions"]["M2"]

    # Total / post-shock-total quantities (constant in the conical region)
    T0 = T_inf * (1.0 + (gamma - 1.0) / 2.0 * M1 ** 2)
    a_inf = np.sqrt(gamma * R_AIR * T_inf)
    V_inf = M1 * a_inf
    Vmax = np.sqrt(2.0 * gamma * R_AIR * T0 / (gamma - 1.0))

    # Post-shock stagnation pressure (same as in aerodynamics.py)
    Mn1 = M1 * np.sin(beta_rad)
    p2_p1 = (2 * gamma * Mn1 ** 2 - (gamma - 1)) / (gamma + 1)
    p02_p2 = (1 + (gamma - 1) / 2 * M2 ** 2) ** (gamma / (gamma - 1))
    p02 = p02_p2 * p2_p1 * p_inf

    Vsq_of_theta, _, _ = _tm_profiles(geom)

    # Polar angle from cone axis at each point
    x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]
    rho_cyl = np.sqrt(y ** 2 + z ** 2)
    theta = np.arctan2(rho_cyl, x)

    # Clamp inside the TM domain
    theta_clip = np.clip(theta, 1e-7, beta_rad - 1e-7)
    Vsq = Vsq_of_theta(theta_clip)            # (V/Vmax)^2
    Vsq = np.clip(Vsq, 1e-12, 1.0 - 1e-9)

    # Local edge velocity and Mach
    ue = np.sqrt(Vsq) * Vmax
    # Energy: T = T0 (1 - V'^2)
    Te = T0 * (1.0 - Vsq)
    Me = ue / np.sqrt(gamma * R_AIR * Te)
    # Isentropic from post-shock stagnation
    pe = p02 * (1.0 - Vsq) ** (gamma / (gamma - 1.0))
    rhoe = pe / (R_AIR * Te)

    # Arc length
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])

    return dict(s=s, ue=ue, Te=Te, Me=Me, pe=pe, rhoe=rhoe,
                V_inf=V_inf, T0=T0, Vmax=Vmax, p02=p02)


# ---------------------------------------------------------------------------
# Per-streamline skin-friction and drag
# ---------------------------------------------------------------------------

def upper_streamline_skin_friction(x_LE, y, z, L, T_inf, p_inf, T_w,
                                   gamma, M1, Pr=0.72, resample=200):
    """
    Walz on a single upper-surface streamline.

    The upper surface is a flat top exposed to the freestream: edge conditions
    are uniform (u_e=V_inf, T_e=T_inf, p_e=p_inf, M_e=M_1).  The streamline
    runs from (x_LE, y, z) to (L, y, z), so dx/ds = 1.
    """
    s_total = float(L - x_LE)
    if s_total <= 0:
        return None
    s = np.linspace(0.0, s_total, resample)
    a_inf = np.sqrt(gamma * R_AIR * T_inf)
    V_inf = M1 * a_inf
    rho_inf = p_inf / (R_AIR * T_inf)

    ue = np.full(resample, V_inf)
    Te = np.full(resample, T_inf)
    rhoe = np.full(resample, rho_inf)
    Tw_arr = np.full(resample, float(T_w))
    mu_w = sutherland_mu(Tw_arr)
    cp = gamma * R_AIR / (gamma - 1.0)
    r = np.sqrt(Pr)
    T_aw = Te + r * ue ** 2 / (2.0 * cp)

    res = integrate_walz(s, ue, Te, rhoe, mu_w, Tw_arr, T_aw,
                         gamma=gamma, Pr=Pr)

    pts = np.empty((resample, 3))
    pts[:, 0] = x_LE + s
    pts[:, 1] = y
    pts[:, 2] = z

    res.update(dict(s=s, pts=pts, ue=ue, Te=Te,
                    Me=ue / np.sqrt(gamma * R_AIR * Te), rhoe=rhoe,
                    T_aw=T_aw, mu_w=mu_w, V_inf=V_inf,
                    y=float(y), z=float(z), x_LE=float(x_LE)))
    return res


def streamline_skin_friction(points, geom, T_inf, p_inf, T_w, Pr=0.72,
                             resample=200):
    """Run Walz on one (N,3) streamline; return per-station tau_w(s)."""
    pts = np.asarray(points, dtype=float)
    # Resample uniformly in arc length for stable derivatives
    edge0 = edge_conditions_along(pts, geom, T_inf, p_inf)
    s_raw = edge0["s"]
    if s_raw[-1] <= 0:
        return None
    s_u = np.linspace(s_raw[0], s_raw[-1], resample)
    pts_u = np.empty((resample, 3))
    for j in range(3):
        pts_u[:, j] = np.interp(s_u, s_raw, pts[:, j])

    edge = edge_conditions_along(pts_u, geom, T_inf, p_inf)
    gamma = geom["parameters"]["gamma"]
    Tw_arr = np.full_like(edge["Te"], float(T_w))
    cp = gamma * R_AIR / (gamma - 1.0)
    r = np.sqrt(Pr)
    T_aw = edge["Te"] + r * edge["ue"] ** 2 / (2.0 * cp)
    mu_w = sutherland_mu(Tw_arr)

    res = integrate_walz(edge["s"], edge["ue"], edge["Te"], edge["rhoe"],
                         mu_w, Tw_arr, T_aw, gamma, Pr=Pr)
    res.update(dict(s=edge["s"], pts=pts_u, ue=edge["ue"], Te=edge["Te"],
                    Me=edge["Me"], rhoe=edge["rhoe"], T_aw=T_aw, mu_w=mu_w,
                    V_inf=edge["V_inf"]))
    return res


def _strip_drag(streamlines):
    """Sum streamwise friction force from a list of per-streamline results.

    Each streamline owns a strip whose width is half the Euclidean distance
    to each neighbour (full neighbour width if the streamline is at the
    outer edge).  Returns total Σ ∫ τ_w (dx/ds) w(s) ds.
    """
    n_sl = len(streamlines)
    D_total = 0.0
    for k, res in enumerate(streamlines):
        pts = res["pts"]
        s = res["s"]
        tau = res["tau_w"]
        dx_ds = np.gradient(pts[:, 0], s)

        widths = np.zeros_like(s)
        contribs = 0
        for k_nb in (k - 1, k + 1):
            if 0 <= k_nb < n_sl:
                pts_nb = streamlines[k_nb]["pts"]
                s_nb = streamlines[k_nb]["s"]
                u = s / max(s[-1], 1e-30)
                s_nb_at = u * s_nb[-1]
                nb_x = np.interp(s_nb_at, s_nb, pts_nb[:, 0])
                nb_y = np.interp(s_nb_at, s_nb, pts_nb[:, 1])
                nb_z = np.interp(s_nb_at, s_nb, pts_nb[:, 2])
                d = np.sqrt((pts[:, 0] - nb_x) ** 2
                          + (pts[:, 1] - nb_y) ** 2
                          + (pts[:, 2] - nb_z) ** 2)
                widths += 0.5 * d
                contribs += 1
        if contribs == 0:
            continue
        if contribs == 1:
            widths *= 2.0
        integrand = tau * dx_ds * widths
        D_total += _trapz_compat(integrand, s)
    return D_total


def compute_skin_friction(geom, lower_mesh=None, upper_mesh=None,
                          T_inf=216.65, p_inf=5474.9,
                          T_w=1000.0, Pr=0.72, resample=200):
    """
    Compute skin-friction drag on the lower *and* upper surfaces of the
    waverider using Walz's integral method along each surface streamline.

    The lower surface uses Taylor--Maccoll edge conditions; the upper (flat)
    surface is exposed to uniform freestream conditions.

    Parameters
    ----------
    geom        : dict from design_waverider().
    lower_mesh  : optional dict from panelize_geometry(); used to define the
                  reference (planform) area, identical to compute_inviscid_forces.
    upper_mesh  : optional dict from panelize_geometry(); not required but
                  accepted for symmetry with the inviscid API.
    T_inf, p_inf: freestream static conditions (K, Pa).
    T_w         : wall temperature (K).  Constant by default.
    Pr          : Prandtl number.
    resample    : per-streamline resampling resolution.

    Returns
    -------
    dict with keys:
        'CDf'              : total friction drag coefficient (lower + upper).
        'CDf_lower', 'CDf_upper' : per-surface contributions.
        'D_friction'       : total dimensional friction drag (N) in +x.
        'D_lower', 'D_upper'     : per-surface contributions.
        'lower_streamlines', 'upper_streamlines' : per-streamline result lists.
        'q_inf', 'S_ref', 'rho_inf', 'V_inf'.
    """
    del upper_mesh  # accepted for API symmetry; not needed for the integration
    M1 = geom["parameters"]["M1"]
    gamma = geom["parameters"]["gamma"]
    L = geom["parameters"]["L"]
    a_inf = np.sqrt(gamma * R_AIR * T_inf)
    V_inf = M1 * a_inf
    rho_inf = p_inf / (R_AIR * T_inf)
    q_inf = 0.5 * rho_inf * V_inf ** 2

    if lower_mesh is not None:
        tris = lower_mesh["triangles"]
        v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
        cross_z = ((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
                   - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0]))
        S_ref = float(np.abs(cross_z).sum() / 2.0)
    else:
        S_ref = 1.0

    # ----- lower surface -----
    lower_streamlines = []
    for ls in geom["lower_surface"]:
        crv = np.array(ls["curve"], dtype=float)
        if crv.shape[0] < 3:
            continue
        res = streamline_skin_friction(crv, geom, T_inf, p_inf, T_w,
                                       Pr=Pr, resample=resample)
        if res is None:
            continue
        lower_streamlines.append(res)

    # ----- upper surface -----
    upper_streamlines = []
    for us in geom["upper_surface"]:
        x_arr = np.asarray(us["x"], dtype=float)
        x_LE = float(x_arr[0])
        res = upper_streamline_skin_friction(
            x_LE, us["y"], us["z"], L,
            T_inf, p_inf, T_w, gamma=gamma, M1=M1,
            Pr=Pr, resample=resample,
        )
        if res is None:
            continue
        upper_streamlines.append(res)

    # Strip drag on each surface, doubled for the mirrored half
    D_lower = 2.0 * _strip_drag(lower_streamlines)
    D_upper = 2.0 * _strip_drag(upper_streamlines)
    D_total = D_lower + D_upper

    return dict(
        CDf=float(D_total / (q_inf * S_ref)),
        CDf_lower=float(D_lower / (q_inf * S_ref)),
        CDf_upper=float(D_upper / (q_inf * S_ref)),
        D_friction=float(D_total),
        D_lower=float(D_lower),
        D_upper=float(D_upper),
        lower_streamlines=lower_streamlines,
        upper_streamlines=upper_streamlines,
        # backwards-compatible alias
        streamlines=lower_streamlines,
        q_inf=float(q_inf),
        S_ref=float(S_ref),
        rho_inf=float(rho_inf),
        V_inf=float(V_inf),
    )


# ---------------------------------------------------------------------------
# Mapping streamline results onto the mesh
# ---------------------------------------------------------------------------

def skin_friction_on_mesh(friction, mesh, geom, surface="lower", field="cf"):
    """
    Project a per-streamline boundary-layer quantity onto a triangulated mesh
    for plotting.

    For the *lower* surface, each centroid is associated with the streamline
    whose osculating-plane azimuth φ is closest, and the field is interpolated
    by polar angle θ from the cone axis.  For the *upper* (flat) surface,
    each centroid is matched to the upper streamline with the closest
    (y, z) seed and the field is interpolated by streamwise distance
    s = x − x_LE.

    Parameters
    ----------
    friction : dict from compute_skin_friction.
    mesh     : dict from panelize_geometry (lower_mesh or upper_mesh).
    geom     : dict from design_waverider.
    surface  : 'lower' or 'upper'.
    field    : 'cf', 'tau_w', or 'delta2'.
    """
    n_tri = mesh["triangles"].shape[0]
    cents = mesh["centroids"]
    xc, yc, zc = cents[:, 0], cents[:, 1], cents[:, 2]

    if surface == "lower":
        streamlines = friction.get("lower_streamlines", friction.get("streamlines", []))
        if not streamlines:
            return np.zeros(n_tri)

        phi_list = []
        theta_tables = []
        val_tables = []
        for res in streamlines:
            pts = res["pts"]
            x = pts[:, 0]; y = pts[:, 1]; z = pts[:, 2]
            mid = pts.shape[0] // 2
            phi_list.append(np.arctan2(y[mid], -z[mid]))
            theta_pt = np.arctan2(np.sqrt(y ** 2 + z ** 2), x)
            order = np.argsort(theta_pt)
            theta_tables.append(theta_pt[order])
            val_tables.append(np.asarray(res[field])[order])
        phi_arr = np.array(phi_list)

        phi_c = np.arctan2(np.abs(yc), -zc)
        theta_c = np.arctan2(np.sqrt(yc ** 2 + zc ** 2), xc)

        out = np.zeros(n_tri)
        for i in range(n_tri):
            k = int(np.argmin(np.abs(phi_arr - phi_c[i])))
            out[i] = np.interp(theta_c[i], theta_tables[k], val_tables[k])
        return out

    elif surface == "upper":
        streamlines = friction.get("upper_streamlines", [])
        if not streamlines:
            return np.zeros(n_tri)

        # Each upper streamline has fixed (y, z) and runs over s = x - x_LE.
        ys = np.array([res["y"] for res in streamlines])
        zs = np.array([res["z"] for res in streamlines])
        x_LE = np.array([res["x_LE"] for res in streamlines])
        s_tables = [np.asarray(res["s"]) for res in streamlines]
        v_tables = [np.asarray(res[field]) for res in streamlines]

        # Match centroid (y, z) to the closest upper streamline (use |y|
        # because upper surface is mirrored about y=0).
        out = np.zeros(n_tri)
        for i in range(n_tri):
            d2 = (ys - abs(yc[i])) ** 2 + (zs - zc[i]) ** 2
            k = int(np.argmin(d2))
            s_local = max(xc[i] - x_LE[k], 0.0)
            out[i] = np.interp(s_local, s_tables[k], v_tables[k])
        return out

    else:
        raise ValueError("surface must be 'lower' or 'upper'")


# ---------------------------------------------------------------------------
# Stand-alone tests
# ---------------------------------------------------------------------------

def _test_flat_plate():
    """Sanity check: zero-pressure-gradient laminar BL → δ_2 ∝ √s, c_f Re_x^0.5 const."""
    print("[boundary_layer] Flat-plate test (M_e = 0.1, low-speed)")
    n = 200
    s = np.linspace(1e-3, 1.0, n)
    ue = np.full(n, 30.0)            # m/s
    Te = np.full(n, 300.0)
    pe = np.full(n, 101325.0)
    rhoe = pe / (R_AIR * Te)
    Tw = np.full(n, 300.0)
    mu_w = sutherland_mu(Tw)
    cp = 1.4 * R_AIR / 0.4
    r = np.sqrt(0.72)
    T_aw = Te + r * ue ** 2 / (2 * cp)

    res = integrate_walz(s, ue, Te, rhoe, mu_w, Tw, T_aw, gamma=1.4, Pr=0.72)

    # Blasius: δ_2 / x = 0.664 / sqrt(Re_x), Re_x = ρ u x / μ
    Re_x = rhoe * ue * s / mu_w
    blasius_d2 = 0.664 * s / np.sqrt(np.maximum(Re_x, 1.0))
    blasius_cf = 0.664 / np.sqrt(np.maximum(Re_x, 1.0))

    err_d2 = np.abs(res["delta2"][-1] / blasius_d2[-1] - 1.0)
    err_cf = np.abs(res["cf"][-1] / blasius_cf[-1] - 1.0)
    print(f"   x=1m  delta2: walz={res['delta2'][-1]:.4e}  blasius={blasius_d2[-1]:.4e}  "
          f"rel-err={err_d2:.3f}")
    print(f"   x=1m  c_f   : walz={res['cf'][-1]:.4e}  blasius={blasius_cf[-1]:.4e}  "
          f"rel-err={err_cf:.3f}")


if __name__ == "__main__":
    _test_flat_plate()
