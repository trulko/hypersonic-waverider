"""
Surface mesh (triangulation) of a waverider geometry for pressure-force integration.

Public API
----------
panelize_geometry(geom)            -> lower_mesh, upper_mesh
panelization_wetted_area(lo, up)   -> float  (m^2)
panelization_volume(lo, up)        -> float  (m^3, approximate)
plot_panelization(lo, up, ...)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample_curve(pts, n):
    """Resample a (M, 3) curve to n evenly-spaced points by arc length."""
    pts = np.asarray(pts, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seg)])
    total = cumlen[-1]
    if total < 1e-30:
        return np.tile(pts[0], (n, 1))
    t = np.linspace(0.0, total, n)
    out = np.empty((n, 3))
    for j in range(3):
        out[:, j] = np.interp(t, cumlen, pts[:, j])
    return out


def _strip_triangles(s1, s2):
    """Triangulate the quad strip between two n-point curves.

    Each quad (s1[i], s1[i+1], s2[i+1], s2[i]) is split into two triangles:
        T1 = (s1[i], s1[i+1], s2[i+1])
        T2 = (s1[i], s2[i+1], s2[i])

    Returns
    -------
    tris   : (2*(n-1), 3, 3)  vertex coordinates
    norms  : (2*(n-1), 3)     raw (non-oriented) unit normals
    areas  : (2*(n-1),)       triangle areas
    """
    n = s1.shape[0]
    tris  = np.empty((2 * (n - 1), 3, 3))
    norms = np.empty((2 * (n - 1), 3))
    areas = np.empty(2 * (n - 1))
    k = 0
    for i in range(n - 1):
        A, B, C, D = s1[i], s1[i + 1], s2[i + 1], s2[i]
        for v0, v1, v2 in ((A, B, C), (A, C, D)):
            e1, e2 = v1 - v0, v2 - v0
            cross = np.cross(e1, e2)
            area  = 0.5 * np.linalg.norm(cross)
            tris[k]  = (v0, v1, v2)
            norms[k] = cross / (2.0 * area) if area > 1e-30 else cross
            areas[k] = area
            k += 1
    return tris, norms, areas


def _orient_outward(tris, norms, areas, interior_pt):
    """Flip any normals that point toward interior_pt rather than away from it."""
    centroids = tris.mean(axis=1)          # (N, 3)
    out_vec   = centroids - interior_pt    # points from interior toward face
    dot = (out_vec * norms).sum(axis=1)
    norms = norms.copy()
    norms[dot < 0] *= -1
    return norms


def _mesh_half_surface(curves, n_pts):
    """Build triangulated strips for an ordered list of resampled curves."""
    all_t, all_n, all_a = [], [], []
    for i in range(len(curves) - 1):
        t, n, a = _strip_triangles(curves[i], curves[i + 1])
        all_t.append(t)
        all_n.append(n)
        all_a.append(a)
    return np.vstack(all_t), np.vstack(all_n), np.concatenate(all_a)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def panelize_geometry(geom):
    """Build lower and upper surface triangle meshes from a waverider geometry dict.

    The mesh for each surface is a structured triangulation derived directly from
    the N_l / N_up streamlines stored in *geom*.  No additional resolution
    parameters are needed.

    Parameters
    ----------
    geom : dict
        Returned by ``design_waverider``.  Must contain keys
        ``"lower_surface"``, ``"upper_surface"``, and ``"parameters"``.

    Returns
    -------
    lower_mesh, upper_mesh : dict, each with keys

        ``'triangles'`` : ndarray (N_tri, 3, 3)
            Vertex coordinates ``[triangle_index, vertex_index, xyz]``.
        ``'normals'``   : ndarray (N_tri, 3)
            Outward unit normals (pointing away from vehicle interior).
        ``'areas'``     : ndarray (N_tri,)
            Triangle areas in the same units as geometry coordinates.
        ``'centroids'`` : ndarray (N_tri, 3)
            Triangle centroids, convenient for per-panel pressure evaluation.

    Notes
    -----
    For pressure-force integration use::

        F = sum_i  p_i * mesh['areas'][i] * mesh['normals'][i]

    where ``p_i`` is the gauge pressure evaluated at ``mesh['centroids'][i]``.
    """
    params = geom["parameters"]
    N_l    = params["N_l"]
    N_up   = params["N_up"]
    L      = params["L"]

    # ── Lower surface ────────────────────────────────────────────────────────
    # Streamlines ordered: index 0 ≈ symmetry plane, index N_l-1 = outer edge.
    pos_lo = [_resample_curve(ls["curve"],          N_l) for ls in geom["lower_surface"]]
    neg_lo = [_resample_curve(ls["mirrored_curve"], N_l) for ls in geom["lower_surface"]]

    # Strip winding: inner→outer in y.  For pos-y strips Δy>0 → raw N_z>0 (inward);
    # flip to get outward.  For neg-y strips Δy<0 → raw N_z<0 (outward); keep.
    lt, ln, la = _mesh_half_surface(pos_lo, N_l);  ln = -ln
    mt, mn, ma = _mesh_half_surface(neg_lo, N_l)

    lower_tris  = np.vstack([lt, mt])
    lower_norms = np.vstack([ln, mn])
    lower_areas = np.concatenate([la, ma])

    # ── Upper surface ────────────────────────────────────────────────────────
    def _upper_pts(us, sign_y):
        x = np.asarray(us["x"])
        y = np.full_like(x, sign_y * float(us["y"]))
        z = np.full_like(x, float(us["z"]))
        return _resample_curve(np.column_stack([x, y, z]), N_up)

    pos_up = [_upper_pts(us,  1.0) for us in geom["upper_surface"]]
    neg_up = [_upper_pts(us, -1.0) for us in geom["upper_surface"]]

    # For pos-y strips Δy>0 → raw N_z>0 (outward for upper); keep.
    # For neg-y strips Δy<0 → raw N_z<0 (inward);  flip.
    ut, un, ua = _mesh_half_surface(pos_up, N_up)
    vt, vn, va = _mesh_half_surface(neg_up, N_up);  vn = -vn

    upper_tris  = np.vstack([ut, vt])
    upper_norms = np.vstack([un, vn])
    upper_areas = np.concatenate([ua, va])

    # Orient normals consistently outward using a point inside the volume
    lower_centroids = lower_tris.mean(axis=1)
    upper_centroids = upper_tris.mean(axis=1)
    interior_pt = 0.5 * (lower_centroids.mean(axis=0) + upper_centroids.mean(axis=0))
    lower_norms = _orient_outward(lower_tris, lower_norms, lower_areas, interior_pt)
    upper_norms = _orient_outward(upper_tris, upper_norms, upper_areas, interior_pt)

    def _build(tris, norms, areas):
        return {
            "triangles": tris,
            "normals":   norms,
            "areas":     areas,
            "centroids": tris.mean(axis=1),
        }

    return _build(lower_tris, lower_norms, lower_areas), \
           _build(upper_tris, upper_norms, upper_areas)


def panelization_wetted_area(lower_mesh, upper_mesh):
    """Return total wetted area (lower + upper surfaces)."""
    return float(lower_mesh["areas"].sum() + upper_mesh["areas"].sum())


def panelization_volume(lower_mesh, upper_mesh):
    """Volume enclosed between lower and upper surfaces.

    For each triangle, project a prism up to the z=0 plane.  The prism volume
    is A_xy × (|z1|+|z2|+|z3|)/3, which is exact for a linearly-varying wedge.
    Subtracting upper from lower gives the vehicle thickness integral.
    """
    def _proj_vol(mesh):
        tris = mesh["triangles"]             # (N, 3, 3)
        v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
        cross_z = (v1[:, 0]-v0[:, 0])*(v2[:, 1]-v0[:, 1]) \
                - (v1[:, 1]-v0[:, 1])*(v2[:, 0]-v0[:, 0])
        A_xy  = np.abs(cross_z) / 2.0
        z_avg = np.abs(tris[:, :, 2]).sum(axis=1) / 3.0
        return float((A_xy * z_avg).sum())

    return _proj_vol(lower_mesh) - _proj_vol(upper_mesh)

def plot_scalar_field(lower_mesh, lower_field,
                      upper_mesh=None, upper_field=None,
                      title="Scalar field on mesh",
                      cmap="viridis",
                      colorbar_label="Value",
                      vmin=None, vmax=None,
                      lower_alpha=None, upper_alpha=None,
                      save_path=None, show=True,
                      ax=None, return_fig_ax=False, norm=None):
    """Plot a scalar field over a triangular surface mesh.

    Parameters
    ----------
    lower_mesh    : dict from ``panelize_geometry``
    lower_field   : ndarray (N_tri,) scalar value per triangle on lower surface
    upper_mesh    : dict from ``panelize_geometry`` (optional)
    upper_field   : ndarray (N_tri,) scalar value per triangle on upper surface (optional)
    title         : plot title
    cmap          : matplotlib colormap name
    colorbar_label: label for the colorbar
    vmin, vmax    : optional color limits
    lower_alpha   : transparency for lower mesh (0–1); if None, uses 1.0 unless
                    an upper mesh is provided, then defaults to 0.8
    upper_alpha   : transparency for upper mesh (0–1); if None, defaults to 0.8
    save_path     : file path to save figure; ``None`` skips saving
    show          : call ``plt.show()`` if True
    """
    lower_field = np.asarray(lower_field, dtype=float)
    if lower_field.shape[0] != lower_mesh["triangles"].shape[0]:
        raise ValueError("lower_field must match number of triangles")
    
    if upper_mesh is not None and upper_field is not None:
        upper_field = np.asarray(upper_field, dtype=float)
        if upper_field.shape[0] != upper_mesh["triangles"].shape[0]:
            raise ValueError("upper_field must match number of triangles")

    if lower_alpha is None: lower_alpha = 1.0
    if upper_alpha is None: upper_alpha = 1.0

    if not (0.0 <= lower_alpha <= 1.0):
        raise ValueError("lower_alpha must be between 0 and 1")
    if not (0.0 <= upper_alpha <= 1.0):
        raise ValueError("upper_alpha must be between 0 and 1")

    if ax is None:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if norm is None:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap,15)

    facecolors_lower = cmap_obj(norm(lower_field))
    facecolors_lower[:, 3] = lower_alpha
    verts = [tri for tri in lower_mesh["triangles"]]
    poly = Poly3DCollection(verts, facecolors=facecolors_lower,
                            edgecolor="k", linewidth=0.1)
    ax.add_collection3d(poly)

    if upper_mesh is not None and upper_field is not None:
        facecolors_upper = cmap_obj(norm(upper_field))
        facecolors_upper[:, 3] = upper_alpha
        verts = [tri for tri in upper_mesh["triangles"]]
        poly = Poly3DCollection(verts, facecolors=facecolors_upper,
                                edgecolor="k", linewidth=0.1)
        ax.add_collection3d(poly)

    # Axis limits from mesh vertices
    all_verts = lower_mesh["triangles"].reshape(-1, 3)
    if upper_mesh is not None:
        all_verts = np.vstack([all_verts, upper_mesh["triangles"].reshape(-1, 3)])
    for i, lbl in enumerate(("x", "y", "z")):
        lo, hi = all_verts[:, i].min(), all_verts[:, i].max()
        pad = 0.05 * (hi - lo) if (hi - lo) > 0 else 1.0
        getattr(ax, f"set_{lbl}lim")(lo - pad, hi + pad)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.set_box_aspect([1, 1, 1])
    ax.set_aspect("equal")

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array(lower_field)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.75, pad=0.15)
    cbar.set_label(colorbar_label)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if return_fig_ax:
        return fig, ax
    if show:
        plt.show()
    plt.close(fig)


def _vsq_to_field(vsq, field, gamma, M1):
    """Convert V'^2 to a chosen scalar field. T0 references upstream stagnation."""
    vsq = np.clip(vsq, 0.0, 0.999999)
    Mloc = np.sqrt(2.0 / (gamma - 1.0) * vsq / (1.0 - vsq))
    if field == "mach":
        return Mloc, "Mach number $M$"
    if field == "temperature":
        # T/T_inf = (1 - V'^2) * (1 + (gamma-1)/2 * M1^2)
        T_ratio = (1.0 - vsq) * (1.0 + 0.5 * (gamma - 1.0) * M1**2)
        return T_ratio, r"$T/T_\infty$"
    if field == "density":
        # Through isentropic conical flow downstream of the shock, using
        # T/T0 = 1-V'^2 and p02 (constant in conical region) so rho/rho02 = (1-V'^2)^(1/(g-1)).
        # For the freestream branch (used outside captured region), the same
        # conversion gives rho/rho_inf = (1 - V'^2)/(1 - V'_inf^2) up to constants;
        # we just plot rho/rho_inf-equivalent via Mach-derived (1-V'^2)^(1/(g-1)).
        rho_ratio = (1.0 - vsq) ** (1.0 / (gamma - 1.0))
        # normalise so freestream (M1) maps to 1
        Vsq_inf = 1.0 / (2.0 / ((gamma - 1.0) * M1**2) + 1.0)
        rho_inf = (1.0 - Vsq_inf) ** (1.0 / (gamma - 1.0))
        return rho_ratio / rho_inf, r"$\rho/\rho_\infty$ (isentropic proxy)"
    raise ValueError(f"unknown field '{field}'")


def plot_flowfield_slices(geom, lower_mesh, upper_mesh,
                          x_planes=None, n_grid=180,
                          field="mach",
                          cmap="jet",
                          n_levels=24,
                          slice_alpha=1.0,
                          title=None,
                          save_path=None, show=True):
    """Plot the waverider body coloured by a flowfield scalar, with several
    cutting planes that show CFD-style filled contours of the inviscid
    Taylor-Maccoll/freestream solution and the captured shock arc.

    Parameters
    ----------
    geom         : dict from ``design_waverider``.
    lower_mesh   : dict from ``panelize_geometry`` (lower surface).
    upper_mesh   : dict from ``panelize_geometry`` (upper surface).
    x_planes     : sequence of axial stations (m); defaults to fractions of L.
    n_grid       : grid resolution per side on each slice.
    field        : 'mach', 'temperature', or 'density'.
    cmap         : matplotlib colormap name.
    n_levels     : number of contour levels per slice.
    slice_alpha  : alpha for the filled contours.
    title        : plot title; auto-generated if None.
    save_path    : file path to save figure.
    show         : call plt.show() if True.
    """
    from scipy.interpolate import interp1d
    import matplotlib.tri as mtri
    from aerodynamics import _tm_vsq_profile

    params = geom["parameters"]
    sc = geom["shock_conditions"]
    M1       = params["M1"]
    gamma    = params["gamma"]
    L        = params["L"]
    Rs       = params["Rs"]
    beta_rad = sc["beta_rad"]
    Vr_i     = sc["Vr_i"]
    V_theta_i = sc["V_theta_i"]

    # ── T-M V'^2(theta) profile and freestream value ───────────────────────
    theta_tm, Vsq_tm = _tm_vsq_profile(gamma, beta_rad, Vr_i, V_theta_i)
    Vsq_interp = interp1d(theta_tm[::-1], Vsq_tm[::-1], kind="linear",
                          bounds_error=False,
                          fill_value=(Vsq_tm[-1], Vsq_tm[0]))
    Vsq_inf = 1.0 / (2.0 / ((gamma - 1.0) * M1**2) + 1.0)

    # ── Body surface field for the 3-D mesh ────────────────────────────────
    cents = lower_mesh["centroids"]
    theta_lo = np.arctan2(np.sqrt(cents[:, 1]**2 + cents[:, 2]**2), cents[:, 0])
    Vsq_lo = Vsq_interp(theta_lo)
    f_lower, cbar_label = _vsq_to_field(Vsq_lo, field, gamma, M1)
    f_upper_val, _      = _vsq_to_field(np.full(1, Vsq_inf), field, gamma, M1)
    f_upper = np.full(upper_mesh["triangles"].shape[0], float(f_upper_val[0]))

    # Color scale: span from cone-surface field through to freestream so the
    # post-shock layer and the freestream rectangle both fit the colorbar.
    f_shock, _ = _vsq_to_field(Vsq_tm[:1], field, gamma, M1)   # at θ=β  (post-shock)
    f_cone,  _ = _vsq_to_field(Vsq_tm[-1:], field, gamma, M1)  # at θ=θc (cone surface)
    f_inf,   _ = _vsq_to_field(np.full(1, Vsq_inf), field, gamma, M1)
    f_min = min(float(f_shock[0]), float(f_cone[0]), float(f_inf[0]))
    f_max = max(float(f_shock[0]), float(f_cone[0]), float(f_inf[0]))

    if title is None:
        title = f"Inviscid flowfield slices: {field}"

    # Two-slope normalisation: the post-shock T-M layer occupies the lower
    # half of the colormap (so its narrow variation is visible), and the
    # freestream sits at the top half.
    f_post = float(f_shock[0])
    if f_min < f_post < f_max:
        norm = colors.TwoSlopeNorm(vmin=f_min, vcenter=f_post, vmax=f_max)
    else:
        norm = colors.Normalize(vmin=f_min, vmax=f_max)
    cmap_o = plt.get_cmap(cmap,15)
    inf_rgba = list(cmap_o(norm(float(f_inf[0]))))

    # ── Render the 3-D body via plot_scalar_field, get axes back ───────────
    fig, ax = plot_scalar_field(
        lower_mesh, f_lower,
        upper_mesh=upper_mesh, upper_field=f_upper,
        title=title, cmap=cmap, colorbar_label=cbar_label,
        vmin=f_min, vmax=f_max, norm=norm,
        lower_alpha=0.55, upper_alpha=0.25,
        save_path=None, show=False,
        return_fig_ax=True,
    )

    # ── Leading edge (used to determine captured azimuth at each slice) ────
    le = geom["leading_edge"]
    X_LE = np.asarray(le["x"], dtype=float)
    Y_LE = np.asarray(le["y"], dtype=float)
    Z_LE = np.asarray(le["z"], dtype=float)
    # Azimuth around +x using the streamline_tracing convention
    # (y = r sinθ sinφ, z = -r sinθ cosφ  →  φ = arctan2(y, -z))
    phi_LE = np.arctan2(Y_LE, -Z_LE)

    if x_planes is None: x_planes = [0.45 * L, 0.7 * L, 1.0 * L]


    # ── Loop over slice planes ─────────────────────────────────────────────
    for x_p in x_planes:
        # Captured azimuth range at this x: |φ| ≤ φ_max(x_p)
        cap = X_LE <= x_p + 1e-12
        if not cap.any():
            continue
        phi_max = float(np.max(np.abs(phi_LE[cap])))
        Rs_xp   = x_p * np.tan(beta_rad)

        # Body lower-surface trace at x=x_p: one (y,z) per lower streamline.
        # Streamlines lie in osculating planes (constant φ_k), so we can build
        # r_body(φ) directly.
        phi_body, rho_body = [], []
        for ls in geom["lower_surface"]:
            crv = np.asarray(ls["curve"], dtype=float)
            xs = crv[:, 0]
            if xs.min() - 1e-9 <= x_p <= xs.max() + 1e-9:
                idx = np.argsort(xs)
                ys = float(np.interp(x_p, xs[idx], crv[idx, 1]))
                zs = float(np.interp(x_p, xs[idx], crv[idx, 2]))
                phi_k = float(np.arctan2(ys, -zs))
                rho_k = float(np.hypot(ys, zs))
                phi_body.append(phi_k)
                rho_body.append(rho_k)
        if len(phi_body) < 2:
            continue
        phi_body = np.asarray(phi_body)
        rho_body = np.asarray(rho_body)
        # Mirror to negative-φ side
        phi_full = np.concatenate([-phi_body[::-1], phi_body])
        rho_full = np.concatenate([ rho_body[::-1], rho_body])
        # Sort by φ and deduplicate
        order = np.argsort(phi_full)
        phi_full = phi_full[order]
        rho_full = rho_full[order]
        _, uniq = np.unique(phi_full, return_index=True)
        phi_full = phi_full[uniq]
        rho_full = rho_full[uniq]
        rbody_interp = interp1d(phi_full, rho_full, kind="linear",
                                bounds_error=False, fill_value=Rs_xp)

        # Body upper-surface trace at x=x_p: LE break points (y_LE, z_LE) of
        # streamlines whose LE is upstream of x_p (constant (y,z) along upper
        # ruled lines from LE to TE).
        cap_le = X_LE <= x_p + 1e-12
        phi_up = phi_LE[cap_le]
        rho_up = np.hypot(Y_LE[cap_le], Z_LE[cap_le])
        order_u = np.argsort(phi_up)
        phi_up = phi_up[order_u]
        rho_up = rho_up[order_u]
        _, uu = np.unique(phi_up, return_index=True)
        phi_up = phi_up[uu]
        rho_up = rho_up[uu]
        rupper_interp = interp1d(phi_up, rho_up, kind="linear",
                                 bounds_error=False, fill_value=Rs_xp)

        # 2-D grid in the (y, z) plane at x = x_p
        extent = 1.15 * Rs
        y_lin = np.linspace(-extent, extent, n_grid)
        z_lin = np.linspace(-extent, extent/5, n_grid)
        Y, Z  = np.meshgrid(y_lin, z_lin)
        rho_g = np.hypot(Y, Z)
        theta_g = np.arctan2(rho_g, x_p)
        phi_g   = np.arctan2(Y, -Z)

        rbody_g = rbody_interp(phi_g)
        Vsq_grid  = Vsq_interp(theta_g)
        F_grid, _ = _vsq_to_field(Vsq_grid, field, gamma, M1)

        # Single Poly3DCollection covering the whole plane: each triangle is
        # coloured at the freestream value, or at the T-M field if inside the
        # captured shock layer.  Body-interior triangles are dropped.
        yy = Y.ravel(); zz = Z.ravel(); ff = F_grid.ravel()
        triang = mtri.Triangulation(yy, zz)
        tris = triang.triangles
        yc = yy[tris].mean(axis=1)
        zc = zz[tris].mean(axis=1)
        rc = np.hypot(yc, zc)
        phi_c = np.arctan2(yc, -zc)
        rb_c  = rbody_interp(phi_c)

        ru_c = rupper_interp(phi_c)
        in_capt = (np.abs(phi_c) <= phi_max) & (rc >= rb_c) & (rc <= Rs_xp)
        in_body = (np.abs(phi_c) <= phi_max) & (rc >= ru_c) & (rc < rb_c)
        keep = ~in_body
        if keep.any():
            tris_k = tris[keep]
            f_per  = np.where(in_capt[keep],
                              ff[tris_k].mean(axis=1),
                              float(f_inf[0]))
            face = cmap_o(norm(f_per))
            face[:, 3] = slice_alpha
            verts3d = np.empty((tris_k.shape[0], 3, 3))
            verts3d[:, :, 0] = x_p
            verts3d[:, :, 1] = yy[tris_k]
            verts3d[:, :, 2] = zz[tris_k]
            ax.add_collection3d(Poly3DCollection(
                verts3d, facecolors=face, edgecolor="none"))

        # Captured shock arc as thick black line
        # n_arc = 200
        # phi_arc = np.linspace(-phi_max, phi_max, n_arc)
        # y_arc =  Rs_xp * np.sin(phi_arc)
        # z_arc = -Rs_xp * np.cos(phi_arc)
        # x_arc = np.full_like(y_arc, x_p)
        # ax.plot(x_arc, y_arc, z_arc,
        #         color="black", linewidth=1.0, zorder=20)

    # Refresh axis limits to include slice extents
    extent = 1.15 * Rs
    ax.set_xlim(0.0, L * 1.02)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)
    ax.set_axis_off()
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    try:
        ax.set_box_aspect((L, 2 * extent, 2 * extent))
    except Exception:
        pass
    ax.view_init(elev=-12, azim=-65)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
